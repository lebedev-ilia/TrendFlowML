from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.model_registry import get_model_with_meta
from src.core.path_utils import default_artifacts_dir
from src.core.text_utils import normalize_whitespace
from src.schemas.models import VideoDocument


class QAEmbeddingPairsExtractor(BaseExtractor):
    """
    Извлекает вопросоподобные фразы из транскрипта и считает их эмбеддинги.
    """

    VERSION = "1.1.0"
    DEFAULT_RU_WORDS = ["кто", "что", "где", "когда", "почему", "зачем", "как", "какой", "какая", "какие", "сколько"]
    DEFAULT_EN_WORDS = ["who", "what", "where", "when", "why", "how", "which"]

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        artifacts_dir: str | None = None,
        device: Optional[str] = "cpu",
        fp16: bool = True,
        batch_size: int = 64,
        enabled: bool = True,
        allow_legacy_transcripts: bool = False,
        transcript_source_policy: str = "asr_only",  # asr_only|asr_then_legacy|legacy_only
        # feature gating (config/UI)
        use_title: bool = True,
        use_description: bool = True,
        use_transcript: bool = True,
        use_comments: bool = True,
        # question extraction policy
        question_langs: List[str] | str = "ru,en",
        question_words_ru: Optional[List[str]] = None,
        question_words_en: Optional[List[str]] = None,
        min_chars_per_question: int = 8,
        max_question_chars: int = 240,
        dedup_questions: bool = True,
        # cost control
        max_questions_total: int = 128,
        max_questions_per_source: int = 64,
        max_comments: int = 200,
        max_chars_per_comment: int = 280,
        max_transcript_chars: int = 200_000,
        require_min_questions: int = 0,
        # optional sub-artifacts (privacy-safe)
        write_question_hashes_artifact: bool = False,
        write_question_source_ids_artifact: bool = False,
        # extra metrics
        emit_extra_metrics: bool = False,
    ) -> None:
        self.model_name = model_name
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.device = str(device or "cpu")
        self.fp16 = fp16 and ("cuda" in self.device)
        self.batch_size = batch_size
        self.enabled = bool(enabled)
        self.allow_legacy_transcripts = bool(allow_legacy_transcripts)
        self.transcript_source_policy = str(transcript_source_policy or "asr_only").strip().lower()
        self.use_title = bool(use_title)
        self.use_description = bool(use_description)
        self.use_transcript = bool(use_transcript)
        self.use_comments = bool(use_comments)
        self.min_chars_per_question = int(min_chars_per_question)
        self.max_question_chars = int(max_question_chars)
        self.dedup_questions = bool(dedup_questions)
        self.max_questions_total = int(max_questions_total)
        self.max_questions_per_source = int(max_questions_per_source)
        self.max_comments = int(max_comments)
        self.max_chars_per_comment = int(max_chars_per_comment)
        self.max_transcript_chars = int(max_transcript_chars)
        self.require_min_questions = int(max(0, require_min_questions))
        self.write_question_hashes_artifact = bool(write_question_hashes_artifact)
        self.write_question_source_ids_artifact = bool(write_question_source_ids_artifact)
        self.emit_extra_metrics = bool(emit_extra_metrics)

        # langs parsing
        if isinstance(question_langs, str):
            self.question_langs = [s.strip().lower() for s in question_langs.split(",") if s.strip()]
        else:
            self.question_langs = [str(s).strip().lower() for s in question_langs if str(s).strip()]

        ru = question_words_ru if isinstance(question_words_ru, list) and question_words_ru else list(self.DEFAULT_RU_WORDS)
        en = question_words_en if isinstance(question_words_en, list) and question_words_en else list(self.DEFAULT_EN_WORDS)
        self._question_re = self._build_question_regex(ru_words=ru, en_words=en, langs=self.question_langs)

        self._model, self._weights_digest, self._model_version = get_model_with_meta(self.model_name, self.device, self.fp16)

    @staticmethod
    def _safe_join_artifacts_dir(base_dir: Path, relpath: str) -> Path:
        base = base_dir.expanduser().resolve()
        cand = (base / relpath).resolve()
        if not (cand == base or base in cand.parents):
            raise RuntimeError("qa_embedding_pairs_extractor: relpath escapes artifacts_dir")
        return cand

    @staticmethod
    def _stable_template(
        *,
        enabled: bool,
        allow_legacy_transcripts: bool,
        transcript_source_policy: str,
        use_title: bool,
        use_description: bool,
        use_transcript: bool,
        use_comments: bool,
        require_min_questions: int,
        max_questions_total: int,
        max_questions_per_source: int,
        max_comments: int,
        max_transcript_chars: int,
        min_chars_per_question: int,
        max_question_chars: int,
        dedup_questions: bool,
        write_question_hashes_artifact: bool,
        write_question_source_ids_artifact: bool,
    ) -> Dict[str, float]:
        return {
            "tp_qa_present": 0.0,
            "tp_qa_disabled_by_policy": 0.0,
            "tp_qa_enabled": float(bool(enabled)),
            "tp_qa_num_questions": 0.0,
            "tp_qa_embedding_dim": float("nan"),
            "tp_qa_q_title": 0.0,
            "tp_qa_q_description": 0.0,
            "tp_qa_q_transcript": 0.0,
            "tp_qa_q_comments": 0.0,
            "tp_qa_allow_legacy_transcripts": 1.0 if allow_legacy_transcripts else 0.0,
            "tp_qa_transcript_source_policy_asr_only": 1.0 if transcript_source_policy == "asr_only" else 0.0,
            "tp_qa_transcript_source_policy_asr_then_legacy": 1.0 if transcript_source_policy == "asr_then_legacy" else 0.0,
            "tp_qa_transcript_source_policy_legacy_only": 1.0 if transcript_source_policy == "legacy_only" else 0.0,
            "tp_qa_use_title": 1.0 if use_title else 0.0,
            "tp_qa_use_description": 1.0 if use_description else 0.0,
            "tp_qa_use_transcript": 1.0 if use_transcript else 0.0,
            "tp_qa_use_comments": 1.0 if use_comments else 0.0,
            "tp_qa_require_min_questions": float(int(require_min_questions)),
            "tp_qa_max_questions_total": float(int(max_questions_total)),
            "tp_qa_max_questions_per_source": float(int(max_questions_per_source)),
            "tp_qa_max_comments": float(int(max_comments)),
            "tp_qa_max_transcript_chars": float(int(max_transcript_chars)),
            "tp_qa_min_chars_per_question": float(int(min_chars_per_question)),
            "tp_qa_max_question_chars": float(int(max_question_chars)),
            "tp_qa_dedup_questions": 1.0 if dedup_questions else 0.0,
            "tp_qa_write_question_hashes_artifact_enabled": 1.0 if write_question_hashes_artifact else 0.0,
            "tp_qa_write_question_source_ids_artifact_enabled": 1.0 if write_question_source_ids_artifact else 0.0,
            "tp_qa_hashes_written": 0.0,
            "tp_qa_source_ids_written": 0.0,
            "tp_qa_questions_per_min": float("nan"),
            "tp_qa_questions_per_1k_chars": float("nan"),
            "tp_qa_mean_cosine_to_centroid": float("nan"),
            "tp_qa_mean_cosine_to_centroid_present": 0.0,
        }

    @staticmethod
    def _build_question_regex(ru_words: Sequence[str], en_words: Sequence[str], langs: Sequence[str]) -> re.Pattern:
        words: List[str] = []
        if "ru" in langs:
            words.extend([w for w in ru_words if isinstance(w, str) and w.strip()])
        if "en" in langs:
            words.extend([w for w in en_words if isinstance(w, str) and w.strip()])
        # If no words are configured, we still require '?' and accept nothing (safe).
        if not words:
            return re.compile(r"^\b\B$")  # matches nothing
        # Unicode-aware "whole word" boundaries: avoid partial matches inside words.
        # Use negative lookaround on \w to work for both latin/cyrillic.
        escaped = [re.escape(w.casefold()) for w in words]
        inner = "|".join(sorted(set(escaped), key=len, reverse=True))
        return re.compile(rf"(?i)(?<!\w)({inner})(?!\w)")

    def _extract_questions_from_text(self, text: str) -> List[str]:
        """
        Extract question-like segments that end with '?' or '？'.
        Keeps the question mark, so downstream filters are consistent.
        """
        t = normalize_whitespace(text)
        if not t:
            return []
        # Split by question marks but keep them.
        # Example: "Hello? How are you?" -> ["Hello?", "How are you?"]
        parts = re.split(r"(?<=[\?\uFF1F])\s+", t)
        out: List[str] = []
        for p in parts:
            p = normalize_whitespace(p)
            if not p:
                continue
            if not (p.endswith("?") or p.endswith("\uFF1F")):
                continue
            if self.max_question_chars and len(p) > self.max_question_chars:
                p = p[: self.max_question_chars].rstrip()
                # Ensure it still looks like a question.
                if not (p.endswith("?") or p.endswith("\uFF1F")):
                    p = p.rstrip(" .!…") + "?"
            if self.min_chars_per_question and len(p) < self.min_chars_per_question:
                continue
            # require interrogative word (configurable)
            if not self._question_re.search(p.casefold()):
                continue
            out.append(p)
        return out

    @staticmethod
    def _canonical_question(text: str) -> str:
        t = normalize_whitespace(text).casefold()
        t = t.replace("\uFF1F", "?")
        # Strip trailing punctuation except '?'.
        while len(t) > 0 and t[-1] in ".!…":
            t = t[:-1].rstrip()
        return t

    def _encode(self, sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, 0), dtype=np.float32)
        out: List[np.ndarray] = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            with torch.no_grad():
                raw = self._model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
            raw = np.asarray(raw, dtype=np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out.append(raw / norms)
        return np.vstack(out)

    def _write_npy_atomic(self, out_path: Path, arr: np.ndarray) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        try:
            with open(tmp_path, "wb") as f:
                np.save(f, np.asarray(arr))
            os.replace(tmp_path, out_path)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        features_flat = self._stable_template(
            enabled=self.enabled,
            allow_legacy_transcripts=self.allow_legacy_transcripts,
            transcript_source_policy=self.transcript_source_policy,
            use_title=self.use_title,
            use_description=self.use_description,
            use_transcript=self.use_transcript,
            use_comments=self.use_comments,
            require_min_questions=self.require_min_questions,
            max_questions_total=self.max_questions_total,
            max_questions_per_source=self.max_questions_per_source,
            max_comments=self.max_comments,
            max_transcript_chars=self.max_transcript_chars,
            min_chars_per_question=self.min_chars_per_question,
            max_question_chars=self.max_question_chars,
            dedup_questions=self.dedup_questions,
            write_question_hashes_artifact=self.write_question_hashes_artifact,
            write_question_source_ids_artifact=self.write_question_source_ids_artifact,
        )

        if not self.enabled:
            features_flat["tp_qa_disabled_by_policy"] = 1.0
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return {
                "device": self.device,
                "version": self.VERSION,
                "model_version": self._model_version,
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
                "result": {"features_flat": features_flat},
                "error": None,
            }

        # Собираем источники: title, description, transcript, comments
        sources: Dict[str, List[str]] = {"title": [], "description": [], "transcript": [], "comments": []}

        # title
        if self.use_title and getattr(doc, "title", None):
            sources["title"].append(normalize_whitespace(str(doc.title)))
        # description
        if self.use_description and getattr(doc, "description", None):
            sources["description"].append(normalize_whitespace(str(doc.description)))
        # transcript
        if self.use_transcript:
            asr = getattr(doc, "asr", None)
            joined_asr = ""
            if isinstance(asr, dict) and isinstance(asr.get("segments"), list) and asr.get("segments"):
                seg_texts = []
                for s in asr.get("segments") or []:
                    if isinstance(s, dict):
                        seg_texts.append(str(s.get("text") or ""))
                joined_asr = normalize_whitespace(" ".join(seg_texts))

            joined_legacy = ""
            if self.allow_legacy_transcripts:
                tr = getattr(doc, "transcripts", {}) or {}
                if isinstance(tr, dict):
                    tx = []
                    if tr.get("whisper"):
                        tx.append(str(tr.get("whisper")))
                    if tr.get("youtube_auto"):
                        tx.append(str(tr.get("youtube_auto")))
                    if tx:
                        joined_legacy = normalize_whitespace(" ".join(tx))

            if self.transcript_source_policy == "asr_only":
                joined = joined_asr
            elif self.transcript_source_policy == "legacy_only":
                joined = joined_legacy
            elif self.transcript_source_policy == "asr_then_legacy":
                joined = joined_asr or joined_legacy
            else:
                raise RuntimeError("QAEmbeddingPairsExtractor: invalid transcript_source_policy (expected asr_only|asr_then_legacy|legacy_only)")

            if joined:
                if self.max_transcript_chars and len(joined) > self.max_transcript_chars:
                    joined = joined[: self.max_transcript_chars].rstrip()
                sources["transcript"].append(joined)
        # comments
        if self.use_comments:
            cmts = getattr(doc, "comments", []) or []
            # deterministic cost control: first-K comments with truncation
            for i, c in enumerate(cmts):
                if i >= max(self.max_comments, 0):
                    break
                try:
                    txt = normalize_whitespace(str(getattr(c, "text", "") or ""))
                except Exception:
                    txt = ""
                if not txt:
                    continue
                if self.max_chars_per_comment and len(txt) > self.max_chars_per_comment:
                    txt = txt[: self.max_chars_per_comment].rstrip()
                sources["comments"].append(txt)

        # Извлекаем вопросы по каждому источнику
        candidate_texts: List[str] = []
        candidate_sources: List[str] = []
        per_source_counts: Dict[str, int] = {"title": 0, "description": 0, "transcript": 0, "comments": 0}
        dedup_set = set()

        # deterministic order: title -> description -> transcript -> comments
        for key in ["title", "description", "transcript", "comments"]:
            added_for_source = 0
            for block in sources[key]:
                qs = self._extract_questions_from_text(block)
                for q in qs:
                    if self.dedup_questions:
                        canon = self._canonical_question(q)
                        if canon in dedup_set:
                            continue
                        dedup_set.add(canon)
                    candidate_texts.append(q)
                    candidate_sources.append(key)
                    per_source_counts[key] += 1
                    added_for_source += 1
                    if self.max_questions_per_source and added_for_source >= self.max_questions_per_source:
                        break
                if self.max_questions_per_source and added_for_source >= self.max_questions_per_source:
                    break
            if self.max_questions_total and len(candidate_texts) >= self.max_questions_total:
                candidate_texts = candidate_texts[: self.max_questions_total]
                candidate_sources = candidate_sources[: self.max_questions_total]
                break

        num_q = len(candidate_texts)
        if num_q <= 0:
            # valid empty: do not write any artifacts or tp_artifacts entries
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0

            # stable schema is already present; fill derived rates
            dur = getattr(doc, "audio_duration_sec", None)
            try:
                dur_f = float(dur) if dur is not None else float("nan")
            except Exception:
                dur_f = float("nan")
            features_flat["tp_qa_questions_per_min"] = float("nan") if not np.isfinite(dur_f) or dur_f <= 0 else 0.0

            return {
                "device": self.device,
                "version": self.VERSION,
                "model_version": self._model_version,
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
                "result": {"features_flat": features_flat},
                "error": None,
            }

        if self.require_min_questions and int(num_q) < int(self.require_min_questions):
            raise RuntimeError(
                f"QAEmbeddingPairsExtractor: require_min_questions not met: num_questions={int(num_q)} require_min_questions={int(self.require_min_questions)}"
            )

        embs = self._encode(candidate_texts)

        # Сохраняем как артефакт матрицу вопросов (N×D) и мета по источникам
        out_path = self.artifacts_dir / "qa_question_embeddings.npy"
        try:
            self._write_npy_atomic(out_path, embs.astype(np.float32))
        except Exception as e:
            raise RuntimeError(f"QAEmbeddingPairsExtractor: failed to write embeddings artifact: {out_path}") from e

        # privacy-safe optional artifacts (no raw text)
        hashes_relpath = None
        sources_relpath = None
        hashes_written = False
        source_ids_written = False
        if self.write_question_hashes_artifact or self.write_question_source_ids_artifact:
            # stable per-question hash (casefolded canonical question)
            q_hashes = np.asarray(
                [hashlib.sha256(self._canonical_question(q).encode("utf-8")).hexdigest()[:24] for q in candidate_texts],
                dtype=object,
            )
            # deterministic vocab
            source_vocab = ["title", "description", "transcript", "comments"]
            source_to_id = {s: i for i, s in enumerate(list(source_vocab))}
            q_source_ids = np.asarray([source_to_id.get(s, -1) for s in candidate_sources], dtype=np.int16)

            if self.write_question_hashes_artifact:
                hp = self.artifacts_dir / "qa_question_hashes.npy"
                try:
                    self._write_npy_atomic(hp, q_hashes)
                    hashes_relpath = hp.name
                    hashes_written = True
                except Exception as e:
                    raise RuntimeError(f"QAEmbeddingPairsExtractor: failed to write hashes artifact: {hp}") from e
            if self.write_question_source_ids_artifact:
                sp = self.artifacts_dir / "qa_question_source_ids.npy"
                try:
                    self._write_npy_atomic(sp, q_source_ids)
                    sources_relpath = sp.name
                    source_ids_written = True
                except Exception as e:
                    raise RuntimeError(f"QAEmbeddingPairsExtractor: failed to write source_ids artifact: {sp}") from e

        # In-memory registry for downstream (no JSON sidecars, no absolute paths in result/NPZ).
        try:
            tp = getattr(doc, "tp_artifacts", None)
            if not isinstance(tp, dict):
                tp = {}
                setattr(doc, "tp_artifacts", tp)
            tp.setdefault("qa", {})
            tp["qa"]["question_embeddings"] = {
                "relpath": out_path.name,
                "num_questions": int(num_q),
                "embedding_dim": int(embs.shape[1]) if embs.size > 0 else 0,
                "per_source_counts": dict(per_source_counts),
                "model_name": self.model_name,
                "model_version": self._model_version,
                "weights_digest": self._weights_digest,
                "hashes_relpath": hashes_relpath,
                "source_ids_relpath": sources_relpath,
                "source_vocab": ["title", "description", "transcript", "comments"],
            }
        except Exception:
            pass

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        features_flat["tp_qa_present"] = 1.0
        features_flat["tp_qa_num_questions"] = float(int(num_q))
        features_flat["tp_qa_embedding_dim"] = float(int(embs.shape[1]) if embs.size > 0 else 0)
        features_flat["tp_qa_q_title"] = float(int(per_source_counts.get("title", 0)))
        features_flat["tp_qa_q_description"] = float(int(per_source_counts.get("description", 0)))
        features_flat["tp_qa_q_transcript"] = float(int(per_source_counts.get("transcript", 0)))
        features_flat["tp_qa_q_comments"] = float(int(per_source_counts.get("comments", 0)))
        features_flat["tp_qa_hashes_written"] = 1.0 if hashes_written else 0.0
        features_flat["tp_qa_source_ids_written"] = 1.0 if source_ids_written else 0.0

        # derived rates (gated by emit_extra_metrics)
        dur = getattr(doc, "audio_duration_sec", None)
        try:
            dur_f = float(dur) if dur is not None else float("nan")
        except Exception:
            dur_f = float("nan")
        if self.emit_extra_metrics:
            features_flat["tp_qa_questions_per_min"] = float("nan") if not np.isfinite(dur_f) or dur_f <= 0 else float(num_q) / (dur_f / 60.0)
            total_chars = sum(len(x) for x in candidate_texts) if candidate_texts else 0
            features_flat["tp_qa_questions_per_1k_chars"] = float("nan") if total_chars <= 0 else float(num_q) / (float(total_chars) / 1000.0)
            # dispersion: mean cosine to centroid (privacy-safe)
            if embs.size > 0 and embs.ndim == 2 and embs.shape[0] >= 2:
                c = np.mean(embs, axis=0)
                cn = np.linalg.norm(c)
                if cn > 0:
                    c = c / cn
                    sims = embs @ c.reshape(-1, 1)
                    features_flat["tp_qa_mean_cosine_to_centroid"] = float(np.mean(sims))
                    features_flat["tp_qa_mean_cosine_to_centroid_present"] = 1.0
            # keep comment truncation config visible
            features_flat["tp_qa_max_chars_per_comment"] = float(int(self.max_chars_per_comment))

        return {
            "device": self.device,
            "version": self.VERSION,
            "model_version": self._model_version,
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
            "result": {"features_flat": features_flat},
            "error": None,
        }


