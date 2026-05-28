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

SCHEMA_MAX_TOPIC_SLOTS = 8
SCHEMA_MAX_KP_SLOTS = 16


def _build_features_flat_keys() -> Tuple[str, ...]:
    keys: List[str] = [
        "tp_topics_present",
        "tp_topics_disabled_by_policy",
        "tp_topics_emit_extra_metrics_enabled",
        "tp_topics_text_chars",
        "tp_topics_has_asr",
        "tp_topics_has_title",
        "tp_topics_has_description",
        "tp_topics_enable_topic_distribution",
        "tp_topics_enable_keyphrases",
        "tp_topics_enable_keyphrase_embeddings",
        "tp_topics_export_keyphrases_mode_raw",
        "tp_topics_export_keyphrases_mode_hashed",
        "tp_topics_export_keyphrases_mode_none",
        "tp_topics_enable_style_flags",
        "tp_topics_allow_legacy_transcripts",
        "tp_topics_transcript_source_policy_asr_only",
        "tp_topics_transcript_source_policy_asr_then_legacy",
        "tp_topics_transcript_source_policy_legacy_only",
        "tp_topics_schema_topic_slots_max",
        "tp_topics_top_k_slots_requested",
        "tp_topics_top_k_slots",
        "tp_topics_top_k_slots_clamped",
        "tp_topics_schema_kp_slots_max",
        "tp_topics_keyphrase_slots_requested",
        "tp_topics_keyphrase_slots",
        "tp_topics_keyphrase_slots_clamped",
        "tp_topics_top_k_topics",
        "tp_topics_temperature",
        "tp_topics_entropy_topk",
        "tp_topics_entropy_topk_norm",
        "tp_topics_perplexity_topk",
        "tp_topics_keyphrases_count",
        "tp_topics_keyphrases_dim",
        "tp_topics_style_faq_qmarks",
        "tp_topics_style_instructional_flag",
        "tp_topics_style_audience_flag",
        "tp_topics_style_cta_flag",
    ]
    for i in range(1, SCHEMA_MAX_TOPIC_SLOTS + 1):
        keys.extend(
            [
                f"tp_topics_topic_top{i}_id",
                f"tp_topics_topic_top{i}_score",
                f"tp_topics_topic_top{i}_prob",
            ]
        )
    for i in range(1, SCHEMA_MAX_KP_SLOTS + 1):
        keys.extend(
            [
                f"tp_topics_kp_top{i}_present",
                f"tp_topics_kp_top{i}_hash01",
                f"tp_topics_kp_top{i}_len",
            ]
        )
    keys.extend(
        [
            "tp_topics_keyphrase_score_top1",
            "tp_topics_keyphrase_score_mean",
            "tp_topics_extra_model_load_ms",
            "tp_topics_extra_topics_pipeline_ms",
            "tp_topics_extra_keyphrases_encode_ms",
            "tp_topics_extra_topics_db_digest_u24",
            "tp_topics_extra_model_digest_u24",
        ]
    )
    return tuple(keys)


_FEATURES_FLAT_KEYS = _build_features_flat_keys()


class SemanticTopicExtractor(BaseExtractor):
    VERSION = "2.1.0"

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
        emit_extra_metrics: bool = False,
    ) -> None:
        super().__init__()
        init_sys_before = system_snapshot()
        init_mem_before = process_memory_bytes()

        self.device = str(device or "cpu")
        self.artifacts_dir = default_artifacts_dir() if artifacts_dir is None else Path(str(artifacts_dir)).expanduser().resolve()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.enabled = bool(enabled)
        self.emit_extra_metrics = bool(emit_extra_metrics)

        self.enable_topic_distribution = bool(enable_topic_distribution)
        self.topics_db_spec_name = str(topics_db_spec_name or "topics_taxonomy_v1")
        self.model_name = str(model_name)
        self.top_k_topics = int(max(1, top_k_topics))
        t_req = int(max(1, int(top_k_slots)))
        self.top_k_slots_requested = t_req
        self.top_k_slots = min(t_req, SCHEMA_MAX_TOPIC_SLOTS)
        self.top_k_slots_clamped = bool(t_req > SCHEMA_MAX_TOPIC_SLOTS)
        self.temperature = float(max(temperature, 1e-6))

        self.enable_keyphrases = bool(enable_keyphrases)
        self.max_keyphrases = int(max(0, max_keyphrases))
        kp_req = int(max(0, int(keyphrase_slots)))
        self.keyphrase_slots_requested = kp_req
        self.keyphrase_slots = min(kp_req, SCHEMA_MAX_KP_SLOTS) if kp_req > 0 else 0
        self.keyphrase_slots_clamped = bool(kp_req > SCHEMA_MAX_KP_SLOTS)
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

        self.style_instruction_words_ru = style_instruction_words_ru or ["нажмите", "сделайте", "кликните", "откройте", "выберите"]
        self.style_audience_words_ru = style_audience_words_ru or ["вы", "ты", "тебя", "вас", "твой", "ваш"]
        self.style_cta_words_ru = style_cta_words_ru or ["подпишитесь", "лайк", "комментарий", "ставьте лайк", "подписывайтесь", "переходите"]

        init_sys_after = system_snapshot()
        init_mem_after = process_memory_bytes()
        self._init_metrics: Dict[str, Any] = {
            "pre_init": init_sys_before,
            "post_init": init_sys_after,
            "ram_peak_bytes": max(init_mem_before, init_mem_after),
        }

    def _gpu_peak_mb(self, sys_after: Any) -> int:
        def _g(snap: Any) -> int:
            try:
                g = (snap or {}).get("gpu") or {}
                arr = g.get("gpus") or []
                return max([int(x.get("memory_used_mb", 0)) for x in arr] or [0])
            except Exception:
                return 0

        return max(
            _g(self._init_metrics.get("pre_init")),
            _g(self._init_metrics.get("post_init")),
            _g(sys_after),
        )

    @staticmethod
    def _digest_u24_hex(d: str) -> float:
        if not isinstance(d, str) or len(d) < 6:
            return float("nan")
        try:
            return float(int(d[:6], 16))
        except Exception:
            return float("nan")

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

        stop = {
            "и", "а", "но", "что", "это", "как", "в", "во", "на", "по", "к", "ко", "из", "у", "за", "для", "про", "о", "об", "от", "до", "же", "ли", "мы", "вы", "ты", "он", "она", "они",
            "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with", "as", "is", "are", "was", "were", "be", "this", "that", "it", "you", "we", "they",
        }

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

    @staticmethod
    def _pack_features_flat(values: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        nan = float("nan")
        for k in _FEATURES_FLAT_KEYS:
            if k not in values:
                raise KeyError(f"SemanticTopicExtractor: missing features_flat key {k!r}")
            v = values[k]
            if v is None:
                out[k] = nan
            elif isinstance(v, (bool, np.bool_)):
                out[k] = float(bool(v))
            else:
                out[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else float("nan")
        return out

    def _config_fragment(self) -> Dict[str, Any]:
        pol = self.transcript_source_policy
        return {
            "tp_topics_emit_extra_metrics_enabled": float(bool(self.emit_extra_metrics)),
            "tp_topics_enable_topic_distribution": float(bool(self.enable_topic_distribution)),
            "tp_topics_enable_keyphrases": float(bool(self.enable_keyphrases)),
            "tp_topics_enable_keyphrase_embeddings": float(bool(self.enable_keyphrase_embeddings)),
            "tp_topics_export_keyphrases_mode_raw": 1.0 if self.export_keyphrases_mode == "raw" else 0.0,
            "tp_topics_export_keyphrases_mode_hashed": 1.0 if self.export_keyphrases_mode == "hashed" else 0.0,
            "tp_topics_export_keyphrases_mode_none": 1.0 if self.export_keyphrases_mode == "none" else 0.0,
            "tp_topics_enable_style_flags": float(bool(self.enable_style_flags)),
            "tp_topics_allow_legacy_transcripts": 1.0 if self.allow_legacy_transcripts else 0.0,
            "tp_topics_transcript_source_policy_asr_only": 1.0 if pol == "asr_only" else 0.0,
            "tp_topics_transcript_source_policy_asr_then_legacy": 1.0 if pol == "asr_then_legacy" else 0.0,
            "tp_topics_transcript_source_policy_legacy_only": 1.0 if pol == "legacy_only" else 0.0,
            "tp_topics_schema_topic_slots_max": float(SCHEMA_MAX_TOPIC_SLOTS),
            "tp_topics_top_k_slots_requested": float(int(self.top_k_slots_requested)),
            "tp_topics_top_k_slots": float(int(self.top_k_slots)),
            "tp_topics_top_k_slots_clamped": 1.0 if self.top_k_slots_clamped else 0.0,
            "tp_topics_schema_kp_slots_max": float(SCHEMA_MAX_KP_SLOTS),
            "tp_topics_keyphrase_slots_requested": float(int(self.keyphrase_slots_requested)),
            "tp_topics_keyphrase_slots": float(int(self.keyphrase_slots)),
            "tp_topics_keyphrase_slots_clamped": 1.0 if self.keyphrase_slots_clamped else 0.0,
            "tp_topics_top_k_topics": float(int(self.top_k_topics)),
            "tp_topics_temperature": float(self.temperature),
        }

    def _nan_topic_slots(self, d: Dict[str, Any]) -> None:
        nan = float("nan")
        for i in range(1, SCHEMA_MAX_TOPIC_SLOTS + 1):
            d[f"tp_topics_topic_top{i}_id"] = nan
            d[f"tp_topics_topic_top{i}_score"] = nan
            d[f"tp_topics_topic_top{i}_prob"] = nan

    def _nan_kp_slots(self, d: Dict[str, Any]) -> None:
        nan = float("nan")
        for i in range(1, SCHEMA_MAX_KP_SLOTS + 1):
            d[f"tp_topics_kp_top{i}_present"] = 0.0
            d[f"tp_topics_kp_top{i}_hash01"] = nan
            d[f"tp_topics_kp_top{i}_len"] = nan

    def _apply_extra_nans(self, d: Dict[str, Any]) -> None:
        nan = float("nan")
        for k in (
            "tp_topics_extra_model_load_ms",
            "tp_topics_extra_topics_pipeline_ms",
            "tp_topics_extra_keyphrases_encode_ms",
            "tp_topics_extra_topics_db_digest_u24",
            "tp_topics_extra_model_digest_u24",
        ):
            d[k] = nan

    def _fill_extra_metrics(
        self,
        d: Dict[str, Any],
        *,
        model_load_ms: float,
        topics_ms: float,
        kp_encode_ms: float,
        db_digest: str,
        model_digest: str,
    ) -> None:
        self._apply_extra_nans(d)
        if not self.emit_extra_metrics:
            return
        d["tp_topics_extra_model_load_ms"] = float(model_load_ms) if model_load_ms == model_load_ms else float("nan")
        d["tp_topics_extra_topics_pipeline_ms"] = float(topics_ms) if topics_ms == topics_ms else float("nan")
        d["tp_topics_extra_keyphrases_encode_ms"] = float(kp_encode_ms) if kp_encode_ms == kp_encode_ms else float("nan")
        d["tp_topics_extra_topics_db_digest_u24"] = self._digest_u24_hex(db_digest) if db_digest else float("nan")
        d["tp_topics_extra_model_digest_u24"] = self._digest_u24_hex(model_digest) if model_digest else float("nan")

    def _build_return(
        self,
        *,
        features_flat: Dict[str, Any],
        raw_export: Dict[str, Any],
        sys_after: Any,
        mem_before: int,
        mem_after: int,
        total_s: float,
        model_name: Optional[str],
        model_version: Optional[str],
        weights_digest: Optional[str],
    ) -> Dict[str, Any]:
        gpu_peak_mb = self._gpu_peak_mb(sys_after)
        return {
            "device": self.device,
            "version": self.VERSION,
            "model_name": model_name,
            "model_version": model_version,
            "weights_digest": weights_digest,
            "system": {
                "pre_init": self._init_metrics.get("pre_init"),
                "post_init": self._init_metrics.get("post_init"),
                "post_process": sys_after,
                "peaks": {
                    "ram_peak_mb": int(max(self._init_metrics.get("ram_peak_bytes", 0), mem_before, mem_after) / 1024 / 1024),
                    "gpu_peak_mb": int(gpu_peak_mb),
                },
            },
            "timings_s": {"total": round(total_s, 3)},
            "result": {"features_flat": self._pack_features_flat(features_flat), **raw_export},
            "error": None,
        }

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time
        import logging
        import torch

        t0 = time.perf_counter()
        mem_before = process_memory_bytes()
        logger = logging.getLogger(self.__class__.__name__)
        nan = float("nan")
        raw_export: Dict[str, Any] = {}

        if not self.enabled:
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            d: Dict[str, Any] = {
                "tp_topics_present": 0.0,
                "tp_topics_disabled_by_policy": 1.0,
                "tp_topics_text_chars": 0.0,
                "tp_topics_has_asr": 0.0,
                "tp_topics_has_title": 0.0,
                "tp_topics_has_description": 0.0,
                "tp_topics_entropy_topk": nan,
                "tp_topics_entropy_topk_norm": nan,
                "tp_topics_perplexity_topk": nan,
                "tp_topics_keyphrases_count": 0.0,
                "tp_topics_keyphrases_dim": nan,
                "tp_topics_style_faq_qmarks": 0.0,
                "tp_topics_style_instructional_flag": 0.0,
                "tp_topics_style_audience_flag": 0.0,
                "tp_topics_style_cta_flag": 0.0,
            }
            d.update(self._config_fragment())
            self._nan_topic_slots(d)
            self._nan_kp_slots(d)
            d["tp_topics_keyphrase_score_top1"] = nan
            d["tp_topics_keyphrase_score_mean"] = nan
            self._fill_extra_metrics(d, model_load_ms=nan, topics_ms=nan, kp_encode_ms=nan, db_digest="", model_digest="")
            return self._build_return(
                features_flat=d,
                raw_export=raw_export,
                sys_after=sys_after,
                mem_before=mem_before,
                mem_after=mem_after,
                total_s=total_s,
                model_name=None,
                model_version=None,
                weights_digest=None,
            )

        asr_text = self._safe_asr_text(doc)
        legacy_tr = self._safe_legacy_transcripts_text(doc)
        title = normalize_whitespace(str(getattr(doc, "title", "") or ""))
        description = normalize_whitespace(str(getattr(doc, "description", "") or ""))

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

        if not full_text:
            logger.info("SemanticTopicExtractor: empty (no text available).")
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            d = {
                "tp_topics_present": 0.0,
                "tp_topics_disabled_by_policy": 0.0,
                "tp_topics_text_chars": 0.0,
                "tp_topics_has_asr": 1.0 if bool(asr_text) else 0.0,
                "tp_topics_has_title": 1.0 if bool(title) else 0.0,
                "tp_topics_has_description": 1.0 if bool(description) else 0.0,
                "tp_topics_entropy_topk": nan,
                "tp_topics_entropy_topk_norm": nan,
                "tp_topics_perplexity_topk": nan,
                "tp_topics_keyphrases_count": 0.0,
                "tp_topics_keyphrases_dim": nan,
                "tp_topics_style_faq_qmarks": 0.0,
                "tp_topics_style_instructional_flag": 0.0,
                "tp_topics_style_audience_flag": 0.0,
                "tp_topics_style_cta_flag": 0.0,
            }
            d.update(self._config_fragment())
            self._nan_topic_slots(d)
            self._nan_kp_slots(d)
            d["tp_topics_keyphrase_score_top1"] = nan
            d["tp_topics_keyphrase_score_mean"] = nan
            self._fill_extra_metrics(d, model_load_ms=nan, topics_ms=nan, kp_encode_ms=nan, db_digest="", model_digest="")
            return self._build_return(
                features_flat=d,
                raw_export=raw_export,
                sys_after=sys_after,
                mem_before=mem_before,
                mem_after=mem_after,
                total_s=total_s,
                model_name=None,
                model_version=None,
                weights_digest=None,
            )

        model_load_ms = nan
        topics_ms = nan
        kp_encode_ms = nan
        db_weights_digest_for_extra = ""

        t_m0 = time.perf_counter()
        use_fp16 = "cuda" in self.device
        model, model_weights_digest, model_version = get_model_with_meta(self.model_name, device=self.device, fp16=use_fp16)
        model_load_ms = (time.perf_counter() - t_m0) * 1000.0
        model_digest_str = str(model_weights_digest or "")

        top_topic_ids: List[int] = []
        top_topic_scores: List[float] = []
        top_topic_probs: List[float] = []
        entropy = float("nan")

        if self.enable_topic_distribution:
            t_tp0 = time.perf_counter()
            db_path, db_weights_digest, db_version = resolve_topics_db(self.topics_db_spec_name)
            db_weights_digest_for_extra = str(db_weights_digest or "")
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

            scores = prompt_emb @ qv.reshape(-1, 1)
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)

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

            topics_ms = (time.perf_counter() - t_tp0) * 1000.0

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

        keyphrases_scored: List[Tuple[str, float]] = []
        if self.enable_keyphrases:
            keyphrases_scored = self._extract_keyphrases_simple(full_text)

        keyphrases_list = [p for p, _ in keyphrases_scored]
        keyphrases_scores = [float(s) for _, s in keyphrases_scored]

        if self.export_keyphrases_mode not in ("none", "raw", "hashed"):
            raise RuntimeError("SemanticTopicExtractor: invalid export_keyphrases_mode (expected none|raw|hashed)")
        if self.export_keyphrases_mode == "raw" and keyphrases_list:
            raw_export["tp_topics_keyphrases_raw"] = keyphrases_list[: self.max_keyphrases]

        kp_relpath = None
        kp_dim = 0
        if self.enable_keyphrase_embeddings and keyphrases_list:
            t_k0 = time.perf_counter()
            with torch.no_grad():
                kp = model.encode(keyphrases_list, convert_to_numpy=True, normalize_embeddings=True)  # type: ignore[attr-defined]
            kp_encode_ms = (time.perf_counter() - t_k0) * 1000.0
            arr = np.asarray(kp, dtype=np.float32)
            kp_dim = int(arr.shape[1]) if arr.ndim == 2 else 0
            out_name = "tp_topics_keyphrase_embeddings.npy"
            out_path = self._safe_join_artifacts_dir(self.artifacts_dir, out_name)
            tmp = str(out_path) + ".tmp"
            try:
                with open(tmp, "wb") as f:
                    np.save(f, arr.astype(np.float32))
                os.replace(tmp, str(out_path))
                kp_relpath = out_name
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
        else:
            kp_encode_ms = nan

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

        d = {
            "tp_topics_present": 1.0,
            "tp_topics_disabled_by_policy": 0.0,
            "tp_topics_text_chars": float(len(full_text)),
            "tp_topics_has_asr": 1.0 if bool(asr_text) else 0.0,
            "tp_topics_has_title": 1.0 if bool(title) else 0.0,
            "tp_topics_has_description": 1.0 if bool(description) else 0.0,
            "tp_topics_entropy_topk": float(entropy) if np.isfinite(entropy) else nan,
            "tp_topics_entropy_topk_norm": float(entropy / np.log(len(top_topic_probs))) if (top_topic_probs and len(top_topic_probs) > 1 and np.isfinite(entropy)) else nan,
            "tp_topics_perplexity_topk": float(np.exp(entropy)) if np.isfinite(entropy) else nan,
            "tp_topics_keyphrases_count": float(len(keyphrases_list)),
            "tp_topics_keyphrases_dim": float(kp_dim) if kp_dim else nan,
            "tp_topics_style_faq_qmarks": float(int(faq_like_question_count)),
            "tp_topics_style_instructional_flag": 1.0 if instructional_language_flag else 0.0,
            "tp_topics_style_audience_flag": 1.0 if audience_addressing_flag else 0.0,
            "tp_topics_style_cta_flag": 1.0 if call_to_action_flag else 0.0,
        }
        d.update(self._config_fragment())

        for i in range(1, SCHEMA_MAX_TOPIC_SLOTS + 1):
            if i <= self.top_k_slots and i <= len(top_topic_ids):
                d[f"tp_topics_topic_top{i}_id"] = float(top_topic_ids[i - 1])
                d[f"tp_topics_topic_top{i}_score"] = float(top_topic_scores[i - 1])
                d[f"tp_topics_topic_top{i}_prob"] = float(top_topic_probs[i - 1])
            else:
                d[f"tp_topics_topic_top{i}_id"] = nan
                d[f"tp_topics_topic_top{i}_score"] = nan
                d[f"tp_topics_topic_top{i}_prob"] = nan

        self._nan_kp_slots(d)
        if self.export_keyphrases_mode == "hashed":
            cap = min(int(self.keyphrase_slots), len(keyphrases_list))
            for i in range(cap):
                kp = keyphrases_list[i]
                d[f"tp_topics_kp_top{i+1}_present"] = 1.0
                d[f"tp_topics_kp_top{i+1}_hash01"] = float(self._hash01(kp))
                d[f"tp_topics_kp_top{i+1}_len"] = float(len(kp))

        if keyphrases_scores:
            d["tp_topics_keyphrase_score_top1"] = float(keyphrases_scores[0])
            d["tp_topics_keyphrase_score_mean"] = float(np.mean(np.asarray(keyphrases_scores, dtype=np.float32)))
        else:
            d["tp_topics_keyphrase_score_top1"] = nan
            d["tp_topics_keyphrase_score_mean"] = nan

        self._fill_extra_metrics(
            d,
            model_load_ms=model_load_ms,
            topics_ms=topics_ms,
            kp_encode_ms=kp_encode_ms,
            db_digest=db_weights_digest_for_extra,
            model_digest=model_digest_str,
        )

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        return self._build_return(
            features_flat=d,
            raw_export=raw_export,
            sys_after=sys_after,
            mem_before=mem_before,
            mem_after=mem_after,
            total_s=total_s,
            model_name=self.model_name,
            model_version=str(model_version),
            weights_digest=str(model_weights_digest),
        )
