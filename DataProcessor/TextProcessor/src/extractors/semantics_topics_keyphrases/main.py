from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import os
import hashlib
from pathlib import Path

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.schemas.models import VideoDocument
from src.core.text_utils import normalize_whitespace
from src.core.model_registry import get_model_with_meta
from src.core.path_utils import default_artifacts_dir
from src.extractors.semantics_topics_keyphrases.topics_db import (
    load_or_build_prompt_embeddings,
    load_topics_db,
    resolve_topics_db,
)


class SemanticTopicExtractor(BaseExtractor):
    VERSION = "2.0.0"

    def __init__(
        self,
        device: str | None = "cpu",
        artifacts_dir: str | None = None,
        enabled: bool = True,
        # global topics retrieval
        enable_topic_distribution: bool = True,
        topics_db_spec_name: str = "topics_taxonomy_v1",
        model_name: str = "intfloat/multilingual-e5-large",
        top_k_topics: int = 5,
        top_k_slots: int = 5,
        temperature: float = 0.07,
        # keyphrases
        enable_keyphrases: bool = True,
        max_keyphrases: int = 10,
        keyphrase_slots: int = 10,
        max_keyphrase_len_chars: int = 64,
        export_keyphrases_mode: str = "none",  # none|raw|hashed
        enable_keyphrase_embeddings: bool = True,
        # style flags
        enable_style_flags: bool = True,
        style_instruction_words_ru: Optional[List[str]] = None,
        style_audience_words_ru: Optional[List[str]] = None,
        style_cta_words_ru: Optional[List[str]] = None,
        # transcript policy
        allow_legacy_transcripts: bool = False,
        transcript_source_policy: str = "asr_only",  # asr_only|asr_then_legacy|legacy_only
        max_text_chars: int = 20000,
        # prompt embeddings cache policy (NOT result_store)
        cache_enabled: bool = True,
        cache_ttl_s: float = 7 * 24 * 3600.0,
        cache_max_total_mb: int = 512,
    ) -> None:
        super().__init__()
        # device is injected by MainProcessor via devices_config; default to CPU for determinism.
        self.device = str(device or "cpu")
        self.artifacts_dir = default_artifacts_dir() if artifacts_dir is None else Path(str(artifacts_dir)).expanduser().resolve()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.enabled = bool(enabled)

        self.enable_topic_distribution = bool(enable_topic_distribution)
        self.topics_db_spec_name = str(topics_db_spec_name or "topics_taxonomy_v1")
        self.model_name = str(model_name)
        self.top_k_topics = int(max(1, top_k_topics))
        self.top_k_slots = int(max(1, top_k_slots))
        self.temperature = float(max(temperature, 1e-6))

        self.enable_keyphrases = bool(enable_keyphrases)
        self.max_keyphrases = int(max(0, max_keyphrases))
        self.keyphrase_slots = int(max(0, keyphrase_slots))
        self.max_keyphrase_len_chars = int(max(8, max_keyphrase_len_chars))
        self.export_keyphrases_mode = str(export_keyphrases_mode or "none").strip().lower()
        self.enable_keyphrase_embeddings = bool(enable_keyphrase_embeddings)

        self.enable_style_flags = bool(enable_style_flags)
        self.allow_legacy_transcripts = bool(allow_legacy_transcripts)
        self.transcript_source_policy = str(transcript_source_policy or "asr_only").strip().lower()
        self.max_text_chars = int(max(0, max_text_chars))
        self.cache_enabled = bool(cache_enabled)
        self.cache_ttl_s = float(cache_ttl_s)
        self.cache_max_total_mb = int(cache_max_total_mb)

        # Heuristic dictionaries (configurable)
        self.style_instruction_words_ru = style_instruction_words_ru or ["нажмите", "сделайте", "кликните", "откройте", "выберите"]
        self.style_audience_words_ru = style_audience_words_ru or ["вы", "ты", "тебя", "вас", "твой", "ваш"]
        self.style_cta_words_ru = style_cta_words_ru or ["подпишитесь", "лайк", "комментарий", "ставьте лайк", "подписывайтесь", "переходите"]

    @staticmethod
    def _safe_asr_text(doc: VideoDocument) -> str:
        asr = getattr(doc, "asr", None)
        if not isinstance(asr, dict):
            return ""
        segs = asr.get("segments")
        if not isinstance(segs, list) or not segs:
            return ""
        parts: List[str] = []
        for s in segs:
            if isinstance(s, dict):
                parts.append(str(s.get("text") or ""))
        return normalize_whitespace(" ".join(parts))

    def _safe_legacy_transcripts_text(self, doc: VideoDocument) -> str:
        if not self.allow_legacy_transcripts:
            return ""
        tr = getattr(doc, "transcripts", None) or {}
        if not isinstance(tr, dict):
            return ""
        full = " ".join([str(tr.get(k, "")) for k in ("whisper", "youtube_auto") if tr.get(k)])
        return normalize_whitespace(full)

    @staticmethod
    def _safe_join_artifacts_dir(base_dir: "Any", relpath: str) -> "Any":
        base = base_dir.expanduser().resolve()
        cand = (base / relpath).resolve()
        if not (cand == base or base in cand.parents):
            raise RuntimeError("semantics_topics_keyphrases: relpath escapes artifacts_dir")
        return cand

    @staticmethod
    def _hash01(s: str) -> int:
        h = hashlib.sha256(s.encode("utf-8")).digest()
        return int(h[0])

    @staticmethod
    def _tokenize_words(text: str) -> List[str]:
        import re

        # letters/numbers with unicode support via \w but exclude '_' by post-filter
        tokens = re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", text.lower())
        return [t for t in tokens if len(t) >= 2]

    def _extract_keyphrases_simple(self, text: str) -> List[Tuple[str, float]]:
        """
        Deterministic lightweight keyphrase scoring (no external deps).
        Not YAKE; we document YAKE as higher-quality future option.
        """
        if not text or self.max_keyphrases <= 0:
            return []
        tokens = self._tokenize_words(text)
        if len(tokens) < 5:
            return []

        # small bilingual stoplist (extend later)
        stop = {
            "и","а","но","что","это","как","в","во","на","по","к","ко","из","у","за","для","про","о","об","от","до","же","ли","мы","вы","ты","он","она","они",
            "the","a","an","and","or","but","to","of","in","on","for","with","as","is","are","was","were","be","this","that","it","you","we","they",
        }

        # candidate n-grams (1..3)
        max_n = 3
        candidates: Dict[str, float] = {}
        positions: Dict[str, int] = {}
        for i in range(len(tokens)):
            for n in range(1, max_n + 1):
                if i + n > len(tokens):
                    break
                ng = tokens[i : i + n]
                if any(t in stop for t in ng):
                    continue
                phrase = " ".join(ng).strip()
                if not phrase:
                    continue
                if len(phrase) > self.max_keyphrase_len_chars:
                    continue
                candidates[phrase] = candidates.get(phrase, 0.0) + 1.0
                positions.setdefault(phrase, i)

        if not candidates:
            return []

        # score: tf * (1 / (1 + first_position)) * length_bonus
        scored: List[Tuple[str, float]] = []
        for ph, tf in candidates.items():
            pos = positions.get(ph, 0)
            length_bonus = 1.0 + 0.1 * (len(ph.split()) - 1)
            score = float(tf) * (1.0 / (1.0 + float(pos))) * length_bonus
            scored.append((ph, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.max_keyphrases]

    @staticmethod
    def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
        z = x.astype(np.float32) / float(max(temperature, 1e-6))
        z = z - np.max(z)
        e = np.exp(z)
        d = float(np.sum(e)) + 1e-9
        return (e / d).astype(np.float32)

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time
        import logging
        import torch

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        logger = logging.getLogger(self.__class__.__name__)

        if not self.enabled:
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            features_flat: Dict[str, Any] = {
                "tp_topics_present": 0.0,
                "tp_topics_disabled_by_policy": 1.0,
            }
            return {
                "device": self.device,
                "version": self.VERSION,
                "model_version": "disabled",
                "system": {
                    "pre_init": sys_before,
                    "post_init": sys_before,
                    "post_process": sys_after,
                    "peaks": {"ram_peak_mb": int(max(mem_before, mem_after) / 1024 / 1024), "gpu_peak_mb": 0},
                },
                "timings_s": {"total": round(total_s, 3)},
                "result": {"features_flat": features_flat},
                "error": None,
            }

        # source-of-truth transcript
        asr_text = self._safe_asr_text(doc)
        legacy_tr = self._safe_legacy_transcripts_text(doc)
        title = normalize_whitespace(str(getattr(doc, "title", "") or ""))
        description = normalize_whitespace(str(getattr(doc, "description", "") or ""))

        # Combine available text (privacy: do not persist)
        if self.transcript_source_policy == "asr_only":
            transcript = asr_text
        elif self.transcript_source_policy == "legacy_only":
            transcript = legacy_tr
        elif self.transcript_source_policy == "asr_then_legacy":
            transcript = asr_text or legacy_tr
        else:
            raise RuntimeError("SemanticTopicExtractor: invalid transcript_source_policy (expected asr_only|asr_then_legacy|legacy_only)")

        full_text = normalize_whitespace(" ".join([transcript, title, description]).strip())
        if self.max_text_chars and len(full_text) > self.max_text_chars:
            full_text = full_text[: self.max_text_chars]

        # empty semantics: no text at all
        if not full_text:
            logger.info("SemanticTopicExtractor: empty (no text available).")
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return {
                "device": self.device,
                "version": self.VERSION,
                "model_version": self.model_name,
                "system": {
                    "pre_init": sys_before,
                    "post_init": sys_before,
                    "post_process": sys_after,
                    "peaks": {"ram_peak_mb": int(max(mem_before, mem_after) / 1024 / 1024), "gpu_peak_mb": 0},
                },
                "timings_s": {"total": round(total_s, 3)},
                "result": {
                    "features_flat": {
                        "tp_topics_present": 0.0,
                        "tp_topics_text_chars": 0.0,
                        "tp_topics_has_asr": 0.0,
                        "tp_topics_has_title": 0.0,
                        "tp_topics_has_description": 0.0,
                    }
                },
                "error": None,
            }

        # Model (dp_models, offline)
        use_fp16 = "cuda" in self.device
        model, model_weights_digest, model_version = get_model_with_meta(self.model_name, device=self.device, fp16=use_fp16)

        # --- Topics retrieval ---
        top_topic_ids: List[int] = []
        top_topic_scores: List[float] = []
        top_topic_probs: List[float] = []
        entropy = float("nan")

        if self.enable_topic_distribution:
            db_path, db_weights_digest, db_version = resolve_topics_db(self.topics_db_spec_name)
            db = load_topics_db(db_path, version=str(db_version))

            def _encode_texts(texts: List[str]) -> np.ndarray:
                with torch.no_grad():
                    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)  # type: ignore[attr-defined]
                return np.asarray(vecs, dtype=np.float32)

            prompt_emb = load_or_build_prompt_embeddings(
                db,
                db_path=db_path,
                db_digest=str(db_weights_digest),
                model_name=self.model_name,
                model_weights_digest=str(model_weights_digest),
                encode_fn=_encode_texts,
                cache_enabled=self.cache_enabled,
                cache_ttl_s=self.cache_ttl_s,
                cache_max_total_mb=self.cache_max_total_mb,
            )

            with torch.no_grad():
                q = model.encode([full_text], convert_to_numpy=True, normalize_embeddings=True)  # type: ignore[attr-defined]
            qv = np.asarray(q, dtype=np.float32).reshape(-1)

            # cosine via dot product on normalized vectors
            scores = prompt_emb @ qv.reshape(-1, 1)
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)

            # aggregate prompts -> topic score = max prompt score per topic id (stable, recall-friendly)
            by_topic: Dict[int, float] = {}
            for tid, sc in zip(db.prompt_topic_ids, scores.tolist()):
                cur = by_topic.get(int(tid))
                if cur is None or float(sc) > float(cur):
                    by_topic[int(tid)] = float(sc)

            pairs = sorted(by_topic.items(), key=lambda x: x[1], reverse=True)
            pairs = pairs[: max(self.top_k_topics, 1)]
            top_topic_ids = [int(t) for t, _ in pairs]
            top_topic_scores = [float(s) for _, s in pairs]

            probs = self._softmax(np.asarray(top_topic_scores, dtype=np.float32), self.temperature) if top_topic_scores else np.zeros((0,), dtype=np.float32)
            top_topic_probs = [float(x) for x in probs.tolist()]
            if top_topic_probs:
                entropy = float(-np.sum([p * np.log(p + 1e-9) for p in top_topic_probs]))

            # In-memory registry for downstream extractors (no raw text, no persisted paths).
            # This is intentionally not part of `features_flat` to avoid complex/nested outputs in NPZ,
            # but it enables deterministic inter-extractor communication within a single run.
            try:
                tp = getattr(doc, "tp_artifacts", None)
                if not isinstance(tp, dict):
                    tp = {}
                    setattr(doc, "tp_artifacts", tp)
                tp.setdefault("topics", {})
                tp["topics"]["topk_distribution"] = {
                    "k": int(self.top_k_topics),
                    "temperature": float(self.temperature),
                    "topic_ids": [int(x) for x in top_topic_ids],
                    "topic_scores": [float(x) for x in top_topic_scores],
                    "topic_probs": [float(x) for x in top_topic_probs],
                    "entropy_topk": float(entropy) if np.isfinite(entropy) else None,
                    "model_name": str(self.model_name),
                    "model_version": str(model_version),
                    "model_weights_digest": str(model_weights_digest),
                    "topics_db_spec_name": str(self.topics_db_spec_name),
                    "topics_db_weights_digest": str(db_weights_digest),
                }
            except Exception:
                pass

        # --- Keyphrases ---
        keyphrases_scored: List[Tuple[str, float]] = []
        if self.enable_keyphrases:
            keyphrases_scored = self._extract_keyphrases_simple(full_text)

        keyphrases_list = [p for p, _ in keyphrases_scored]
        keyphrases_scores = [float(s) for _, s in keyphrases_scored]

        # Optional raw export (privacy gated)
        raw_export: Dict[str, Any] = {}
        if self.export_keyphrases_mode not in ("none", "raw", "hashed"):
            raise RuntimeError("SemanticTopicExtractor: invalid export_keyphrases_mode (expected none|raw|hashed)")
        if self.export_keyphrases_mode == "raw" and keyphrases_list:
            raw_export["tp_topics_keyphrases_raw"] = keyphrases_list[: self.max_keyphrases]

        # Keyphrase embeddings artifact (per-run) + in-memory relpath
        kp_relpath = None
        kp_dim = 0
        if self.enable_keyphrase_embeddings and keyphrases_list:
            with torch.no_grad():
                kp = model.encode(keyphrases_list, convert_to_numpy=True, normalize_embeddings=True)  # type: ignore[attr-defined]
            arr = np.asarray(kp, dtype=np.float32)
            kp_dim = int(arr.shape[1]) if arr.ndim == 2 else 0
            # per-run fixed artifact name (no raw-derived fingerprints in filename)
            out_name = "tp_topics_keyphrase_embeddings.npy"
            out_path = self._safe_join_artifacts_dir(self.artifacts_dir, out_name)
            tmp = str(out_path) + ".tmp"
            try:
                with open(tmp, "wb") as f:
                    np.save(f, arr.astype(np.float32))
                os.replace(tmp, str(out_path))
                kp_relpath = out_name
                # in-memory registry for downstream (no paths in result)
                try:
                    tp = getattr(doc, "tp_artifacts", None)
                    if not isinstance(tp, dict):
                        tp = {}
                        setattr(doc, "tp_artifacts", tp)
                    tp.setdefault("topics", {})
                    tp["topics"]["keyphrase_embeddings"] = {
                        "relpath": str(kp_relpath),
                        "count": int(arr.shape[0]),
                        "dim": int(kp_dim),
                        "model_name": self.model_name,
                        "model_version": str(model_version),
                        "weights_digest": str(model_weights_digest),
                    }
                except Exception:
                    pass
            except Exception:
                kp_relpath = None

        # --- Style flags (heuristics proxy, configurable) ---
        faq_like_question_count = 0
        instructional_language_flag = False
        audience_addressing_flag = False
        call_to_action_flag = False
        if self.enable_style_flags:
            try:
                faq_like_question_count = int(full_text.count("?"))
                lt = full_text.lower()
                instructional_language_flag = any(w in lt for w in self.style_instruction_words_ru)
                audience_addressing_flag = any(w in lt for w in self.style_audience_words_ru)
                call_to_action_flag = any(w in lt for w in self.style_cta_words_ru)
            except Exception:
                pass

        # --- features_flat ---
        # --- features_flat ---
        features_flat: Dict[str, Any] = {
            "tp_topics_present": 1.0,
            "tp_topics_disabled_by_policy": 0.0,
            "tp_topics_text_chars": float(len(full_text)),
            "tp_topics_has_asr": 1.0 if bool(asr_text) else 0.0,
            "tp_topics_has_title": 1.0 if bool(title) else 0.0,
            "tp_topics_has_description": 1.0 if bool(description) else 0.0,
            "tp_topics_enable_topic_distribution": 1.0 if self.enable_topic_distribution else 0.0,
            "tp_topics_enable_keyphrases": 1.0 if self.enable_keyphrases else 0.0,
            "tp_topics_enable_keyphrase_embeddings": 1.0 if self.enable_keyphrase_embeddings else 0.0,
            "tp_topics_export_keyphrases_mode_raw": 1.0 if self.export_keyphrases_mode == "raw" else 0.0,
            "tp_topics_export_keyphrases_mode_hashed": 1.0 if self.export_keyphrases_mode == "hashed" else 0.0,
            "tp_topics_export_keyphrases_mode_none": 1.0 if self.export_keyphrases_mode == "none" else 0.0,
            "tp_topics_enable_style_flags": 1.0 if self.enable_style_flags else 0.0,
            "tp_topics_allow_legacy_transcripts": 1.0 if self.allow_legacy_transcripts else 0.0,
            "tp_topics_top_k_topics": float(int(self.top_k_topics)),
            "tp_topics_top_k_slots": float(int(self.top_k_slots)),
            "tp_topics_temperature": float(self.temperature),
            "tp_topics_entropy_topk": float(entropy) if np.isfinite(entropy) else float("nan"),
            "tp_topics_entropy_topk_norm": float(entropy / np.log(len(top_topic_probs))) if (top_topic_probs and len(top_topic_probs) > 1 and np.isfinite(entropy)) else float("nan"),
            "tp_topics_perplexity_topk": float(np.exp(entropy)) if np.isfinite(entropy) else float("nan"),
            "tp_topics_topic_top1_id": float(top_topic_ids[0]) if top_topic_ids else float("nan"),
            "tp_topics_topic_top1_score": float(top_topic_scores[0]) if top_topic_scores else float("nan"),
            "tp_topics_topic_top1_prob": float(top_topic_probs[0]) if top_topic_probs else float("nan"),
            "tp_topics_keyphrases_count": float(len(keyphrases_list)),
            "tp_topics_keyphrases_dim": float(kp_dim) if kp_dim else float("nan"),
            "tp_topics_style_faq_qmarks": float(int(faq_like_question_count)),
            "tp_topics_style_instructional_flag": 1.0 if instructional_language_flag else 0.0,
            "tp_topics_style_audience_flag": 1.0 if audience_addressing_flag else 0.0,
            "tp_topics_style_cta_flag": 1.0 if call_to_action_flag else 0.0,
        }

        # fixed slots (stable schema)
        for i in range(self.top_k_slots):
            s = float(top_topic_scores[i]) if i < len(top_topic_scores) else float("nan")
            p = float(top_topic_probs[i]) if i < len(top_topic_probs) else float("nan")
            tid = float(top_topic_ids[i]) if i < len(top_topic_ids) else float("nan")
            features_flat[f"tp_topics_topic_top{i+1}_id"] = tid
            features_flat[f"tp_topics_topic_top{i+1}_score"] = s
            features_flat[f"tp_topics_topic_top{i+1}_prob"] = p

        # privacy-safe keyphrase hashed slots (stable)
        for i in range(int(self.keyphrase_slots)):
            features_flat[f"tp_topics_kp_top{i+1}_present"] = 0.0
            features_flat[f"tp_topics_kp_top{i+1}_hash01"] = float("nan")
            features_flat[f"tp_topics_kp_top{i+1}_len"] = float("nan")
        if self.export_keyphrases_mode == "hashed":
            for i in range(min(int(self.keyphrase_slots), len(keyphrases_list))):
                kp = keyphrases_list[i]
                features_flat[f"tp_topics_kp_top{i+1}_present"] = 1.0
                features_flat[f"tp_topics_kp_top{i+1}_hash01"] = float(self._hash01(kp))
                features_flat[f"tp_topics_kp_top{i+1}_len"] = float(len(kp))

        # keyphrase score summaries
        if keyphrases_scores:
            features_flat["tp_topics_keyphrase_score_top1"] = float(keyphrases_scores[0])
            features_flat["tp_topics_keyphrase_score_mean"] = float(np.mean(np.asarray(keyphrases_scores, dtype=np.float32)))
        else:
            features_flat["tp_topics_keyphrase_score_top1"] = float("nan")
            features_flat["tp_topics_keyphrase_score_mean"] = float("nan")

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        return {
            "device": self.device,
            "version": self.VERSION,
            "model_version": str(model_version),
            "system": {
                "pre_init": sys_before,
                "post_init": sys_before,
                "post_process": sys_after,
                "peaks": {
                    "ram_peak_mb": int(max(mem_before, mem_after) / 1024 / 1024),
                    "gpu_peak_mb": 0,
                },
            },
            "timings_s": {"total": round(total_s, 3)},
            "result": {"features_flat": features_flat, **raw_export},
            "error": None,
        }


