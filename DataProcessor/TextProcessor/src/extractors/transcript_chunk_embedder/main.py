"""
TranscriptChunkEmbedder — извлекает эмбеддинги по чанкам из транскрипта.

Особенности (согласованы со стилем проекта):
- Разбиение транскрипта на чанки (по предложениям с overlap);
- L2-нормализация эмбеддингов;
- Кеширование векторов и меты (атомарная запись *.tmp.npy → .npy);
- Метрики: pre_init/post_init/post_process, peaks.ram_peak_mb и peaks.gpu_peak_mb;
- Timings в секундах: timings_s { total };
- Privacy policy: не возвращает raw тексты и абсолютные пути к артефактам в result (пути живут только in-memory в `doc.tp_artifacts`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from dp_models.manager import ModelManager
from src.core.base_extractor import BaseExtractor
from src.core.metrics import process_memory_bytes, system_snapshot
from src.core.model_registry import get_model
from src.core.path_utils import default_artifacts_dir, default_cache_dir
from src.core.text_utils import normalize_whitespace
from src.schemas.models import VideoDocument


class TranscriptChunkEmbedder(BaseExtractor):
    """
    A-policy transcript chunk embedder.

    - Strict model + tokenizer via dp_models (no-network, reproducible, weights_digest in keys)
    - Token-aware chunking (shared_tokenizer_v1)
    - Deterministic per-run artifact names (no content hashes in filenames)
    - Valid empty semantics (NaNs + *_present flags)
    """

    VERSION = "1.2.0"

    _SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        tokenizer_spec_name: str = "shared_tokenizer_v1",
        cache_dir: Optional[str] = None,
        cache_enabled: bool = False,
        cache_ttl_days: Optional[float] = 30.0,
        cache_max_items: Optional[int] = 50_000,
        cache_max_bytes: Optional[int] = 5_000_000_000,
        cache_cleanup_on_init: bool = True,
        cache_cleanup_max_seconds: float = 0.25,
        artifacts_dir: Optional[str] = None,
        device: Optional[str] = "cpu",
        fp16: bool = True,
        batch_size: int = 64,
        # Source policy / feature gating
        use_asr: bool = True,
        use_youtube_auto: bool = False,
        require_asr: bool = False,
        require_any_source: bool = False,
        # Chunking / cost controls
        max_chunk_tokens_model: int = 256,
        overlap_ratio: float = 0.15,
        max_chunks_total: int = 256,
        emit_confidence_metrics: bool = True,
        emit_extra_metrics: bool = False,
    ) -> None:
        self.model_name = model_name
        self.tokenizer_spec_name = str(tokenizer_spec_name or "shared_tokenizer_v1")
        # No-network policy: cache/artifacts live under TextProcessor folder by default (configurable via env).
        base_cache = default_cache_dir() / "transcript_embed"
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else base_cache
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.batch_size = batch_size
        self.cache_enabled = bool(cache_enabled)
        self.cache_ttl_days = float(cache_ttl_days) if cache_ttl_days is not None else None
        self.cache_max_items = int(cache_max_items) if cache_max_items is not None else None
        self.cache_max_bytes = int(cache_max_bytes) if cache_max_bytes is not None else None
        self.cache_cleanup_on_init = bool(cache_cleanup_on_init)
        self.cache_cleanup_max_seconds = float(cache_cleanup_max_seconds)

        self.use_asr = bool(use_asr)
        self.use_youtube_auto = bool(use_youtube_auto)
        self.require_asr = bool(require_asr)
        self.require_any_source = bool(require_any_source)

        self.max_chunk_tokens_model = int(max_chunk_tokens_model)
        self.overlap_ratio = float(overlap_ratio)
        self.max_chunks_total = int(max_chunks_total)
        self.emit_confidence_metrics = bool(emit_confidence_metrics)
        self.emit_extra_metrics = bool(emit_extra_metrics)

        if self.max_chunk_tokens_model <= 0:
            raise RuntimeError("TranscriptChunkEmbedder: max_chunk_tokens_model must be > 0")
        if not (0.0 <= self.overlap_ratio < 1.0):
            raise RuntimeError("TranscriptChunkEmbedder: overlap_ratio must be in [0, 1)")
        if self.max_chunks_total <= 0:
            raise RuntimeError("TranscriptChunkEmbedder: max_chunks_total must be > 0")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.device = str(device or "cpu")
        self.fp16 = fp16 and ("cuda" in self.device)

        # metrics: init snapshots
        init_sys_before = system_snapshot()
        init_mem_before = process_memory_bytes()

        # Resolve model + tokenizer via dp_models (fail-fast).
        try:
            from dp_models import get_global_model_manager  # type: ignore

            mm = get_global_model_manager()
            spec = mm.get_spec(model_name=str(self.model_name))
            _d, _p, _rt, _eng, weights_digest, _arts = mm.resolve(spec)
            self.weights_digest = str(weights_digest or "unknown")
            self.model_version = str(getattr(spec, "model_version", "unknown") or "unknown")
        except Exception as e:
            raise RuntimeError(f"TranscriptChunkEmbedder | dp_models resolve failed for model_name={self.model_name!r}: {e}") from e

        self._load_tokenizer_strict()
        self._load_model()

        if self.cache_cleanup_on_init and self.cache_enabled:
            self._cleanup_cache_best_effort()

        init_sys_after = system_snapshot()
        init_mem_after = process_memory_bytes()
        self._init_metrics: Dict[str, Any] = {
            "pre_init": init_sys_before,
            "post_init": init_sys_after,
            "ram_peak_bytes": max(init_mem_before, init_mem_after),
        }

    def _load_model(self) -> None:
        # No-fallback policy: if requested device/model can't load, fail-fast.
        self.model = get_model(self.model_name, self.device, self.fp16)

    # release_resources removed: models are shared via registry and persist

    def _load_tokenizer_strict(self) -> None:
        """
        Strictly load shared tokenizer via dp_models (no fallback).
        """
        try:
            from dp_models import get_global_model_manager  # type: ignore

            mm = get_global_model_manager()
        except Exception:
            mm = ModelManager()

        spec = mm.get_spec(model_name=self.tokenizer_spec_name)
        _d, _p, _rt, _eng, _wd, artifacts = mm.resolve(spec)
        tok_path = list(artifacts.values())[0] if artifacts else None
        if not tok_path:
            raise RuntimeError(f"TranscriptChunkEmbedder: tokenizer artifacts are empty: {self.tokenizer_spec_name}")
        try:
            from tokenizers import Tokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(f"TranscriptChunkEmbedder: python package 'tokenizers' is required: {e}") from e
        self._tokenizer = Tokenizer.from_file(tok_path)

    @staticmethod
    def _hash_text(text: str, key: str, source: str) -> str:
        payload = (str(key) + "||" + str(source) + "||" + normalize_whitespace(text)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _transcript_id_from_token_ids(model_key: str, source: str, token_ids: Optional[List[int]]) -> Optional[str]:
        """
        Privacy-safe stable transcript_id derived from token IDs (preferred).
        """
        if not token_ids:
            return None
        try:
            payload = json.dumps(
                {"model_key": str(model_key), "source": str(source), "token_ids": [int(x) for x in token_ids]},
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
            return hashlib.sha256(payload).hexdigest()
        except Exception:
            return None

    def _get_sources(self, doc: VideoDocument) -> Dict[str, str]:
        """
        Production source-of-truth: AudioProcessor provides ASR in doc.asr.
        Mapping policy (v1):
        - doc.asr.segments -> treated as "whisper" source (primary)
        - doc.transcripts["youtube_auto"] -> optional secondary (legacy)
        """
        out: Dict[str, str] = {}

        if self.use_asr:
            asr = getattr(doc, "asr", None)
            if isinstance(asr, dict):
                segs = asr.get("segments") or []
                if isinstance(segs, list):
                    parts: List[str] = []
                    for s in segs:
                        if isinstance(s, dict):
                            t = normalize_whitespace(s.get("text"))
                            if t:
                                parts.append(t)
                    text = " ".join(parts).strip()
                    if text:
                        out["whisper"] = text
            if self.require_asr and "whisper" not in out:
                raise RuntimeError("TranscriptChunkEmbedder: require_asr=True but doc.asr.segments is missing/empty")

        if self.use_youtube_auto:
            transcripts_dict = getattr(doc, "transcripts", {}) or {}
            if isinstance(transcripts_dict, dict) and transcripts_dict.get("youtube_auto"):
                t2 = normalize_whitespace(transcripts_dict.get("youtube_auto", ""))
                if t2:
                    out["youtube_auto"] = t2

        if self.require_any_source and not out:
            raise RuntimeError("TranscriptChunkEmbedder: require_any_source=True but no enabled transcript sources are available")

        return out

    @classmethod
    def _get_asr_segments_with_conf(cls, doc: VideoDocument) -> List[Tuple[str, Optional[float]]]:
        asr = getattr(doc, "asr", None)
        if not isinstance(asr, dict):
            return []
        segs = asr.get("segments") or []
        if not isinstance(segs, list):
            return []
        out: List[Tuple[str, Optional[float]]] = []
        for s in segs:
            if not isinstance(s, dict):
                continue
            t = normalize_whitespace(s.get("text"))
            if not t:
                continue
            conf = s.get("confidence")
            c = float(conf) if isinstance(conf, (int, float)) else None
            out.append((t, c))
        return out

    def _count_tokens(self, text: str) -> int:
        ids = self._tokenizer.encode(text).ids  # type: ignore[attr-defined]
        return int(len(ids))

    def _chunk_by_asr_segments(self, segs: List[Tuple[str, Optional[float]]]) -> Tuple[List[str], List[Optional[float]]]:
        """
        Build chunks from ASR segments to preserve confidence mapping.
        Returns (chunk_texts, chunk_confidences_mean_or_none).
        """
        chunks: List[str] = []
        confs: List[Optional[float]] = []
        buf: List[str] = []
        buf_confs: List[float] = []
        tok_count = 0
        target = int(self.max_chunk_tokens_model)
        overlap_tokens = int(target * float(self.overlap_ratio))

        def flush() -> None:
            nonlocal buf, buf_confs, tok_count
            if not buf:
                return
            text = " ".join(buf).strip()
            if text:
                chunks.append(text)
                confs.append(float(sum(buf_confs) / len(buf_confs)) if buf_confs else None)
            # overlap: keep tail segments that roughly fit overlap_tokens
            if overlap_tokens > 0 and buf:
                new_buf: List[str] = []
                new_confs: List[float] = []
                new_tok = 0
                for t, c in reversed(list(zip(buf, buf_confs or [0.0] * len(buf)))):
                    tt = self._count_tokens(t)
                    if new_tok + tt > overlap_tokens:
                        break
                    new_buf.append(t)
                    new_confs.append(float(c))
                    new_tok += tt
                new_buf.reverse()
                new_confs.reverse()
                buf = new_buf
                buf_confs = new_confs
                tok_count = new_tok
            else:
                buf = []
                buf_confs = []
                tok_count = 0

        for t, conf in segs:
            tt = self._count_tokens(t)
            if tok_count + tt > target and buf:
                flush()
            buf.append(t)
            if conf is not None:
                buf_confs.append(float(conf))
            tok_count += tt

        flush()
        # Cost cap
        if len(chunks) > self.max_chunks_total:
            chunks = chunks[: self.max_chunks_total]
            confs = confs[: self.max_chunks_total]
        return chunks, confs

    @classmethod
    def _sent_split(cls, text: str) -> List[str]:
        # Lightweight sentence split without nltk (no downloads).
        t = normalize_whitespace(text)
        if not t:
            return []
        parts = cls._SENT_SPLIT_RE.split(t)
        return [p.strip() for p in parts if p and p.strip()]

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Token-aware chunking using shared tokenizer (dp_models).
        Sentence-guided to keep readability, but token counting is exact w.r.t tokenizer.
        """
        sents = self._sent_split(text)
        if not sents:
            return []

        chunks: List[str] = []
        buf: List[str] = []
        tok_count = 0
        target = int(self.max_chunk_tokens_model)
        overlap_tokens = int(target * float(self.overlap_ratio))

        def flush() -> None:
            nonlocal buf, tok_count
            if not buf:
                return
            chunk_text = " ".join(buf).strip()
            if chunk_text:
                chunks.append(chunk_text)
            if len(chunks) >= self.max_chunks_total:
                buf = []
                tok_count = 0
                return
            # overlap: keep tail sentences that fit overlap_tokens
            if overlap_tokens > 0 and buf:
                new_buf: List[str] = []
                new_tok = 0
                for s in reversed(buf):
                    st = self._count_tokens(s)
                    if new_tok + st > overlap_tokens:
                        break
                    new_buf.append(s)
                    new_tok += st
                new_buf.reverse()
                buf = new_buf
                tok_count = new_tok
            else:
                buf = []
                tok_count = 0

        for sent in sents:
            st = self._count_tokens(sent)
            if st <= 0:
                continue
            if tok_count + st > target and buf:
                flush()
                if len(chunks) >= self.max_chunks_total:
                    break
            buf.append(sent)
            tok_count += st

        if buf and len(chunks) < self.max_chunks_total:
            chunk_text = " ".join(buf).strip()
            if chunk_text:
                chunks.append(chunk_text)
        return chunks[: self.max_chunks_total]

    def _encode_chunks(self, chunks: List[str]) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            if torch is None:
                raise RuntimeError("TranscriptChunkEmbedder: torch is required to run sentence-transformers in-process")
            with torch.no_grad():
                raw = self.model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
            raw = np.asarray(raw, dtype=np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normed = raw / norms
            all_embeddings.append(normed)
        return np.vstack(all_embeddings) if all_embeddings else np.zeros((0, 0), dtype=np.float32)

    def _gpu_used_mb(self, snap: Any) -> int:
        try:
            g = (snap or {}).get("gpu") or {}
            arr = g.get("gpus") or []
            return max([int(x.get("memory_used_mb", 0)) for x in arr] or [0])
        except Exception:
            return 0

    def _cache_paths(self, transcript_id: str) -> Tuple[Path, Path]:
        """
        Cache key is privacy-safe and content-addressed (hash/token-ids derived id) + weights_digest.
        """
        key = hashlib.sha256((self.weights_digest + "||" + str(transcript_id)).encode("utf-8")).hexdigest()
        return (self.cache_dir / f"{key}.npy", self.cache_dir / f"{key}.meta.json")

    def _cleanup_cache_best_effort(self) -> None:
        """
        Best-effort TTL/size cleanup with time budget.
        """
        t0 = time.perf_counter()
        if not self.cache_dir.exists():
            return
        try:
            items = list(self.cache_dir.glob("*.meta.json"))
        except Exception:
            return

        now = time.time()
        # TTL
        if self.cache_ttl_days is not None:
            ttl_s = float(self.cache_ttl_days) * 86400.0
        else:
            ttl_s = None

        def too_old(p: Path) -> bool:
            if ttl_s is None:
                return False
            try:
                return (now - p.stat().st_mtime) > ttl_s
            except Exception:
                return False

        # Remove old metas (and matching .npy)
        for meta in items:
            if time.perf_counter() - t0 > self.cache_cleanup_max_seconds:
                return
            if too_old(meta):
                try:
                    vec = meta.with_suffix("").with_suffix(".npy")
                    if vec.exists():
                        vec.unlink(missing_ok=True)  # type: ignore[arg-type]
                    meta.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass

        # Enforce max_items/max_bytes (approximate)
        try:
            metas = list(self.cache_dir.glob("*.meta.json"))
        except Exception:
            return
        recs: List[Tuple[float, int, Path]] = []
        total_bytes = 0
        for meta in metas:
            if time.perf_counter() - t0 > self.cache_cleanup_max_seconds:
                return
            try:
                st = meta.stat()
                vec = meta.with_suffix("").with_suffix(".npy")
                vb = vec.stat().st_size if vec.exists() else 0
                b = int(st.st_size + vb)
                total_bytes += b
                recs.append((float(st.st_mtime), b, meta))
            except Exception:
                continue
        recs.sort(key=lambda t: t[0])  # oldest first
        while True:
            if time.perf_counter() - t0 > self.cache_cleanup_max_seconds:
                return
            too_many = self.cache_max_items is not None and len(recs) > int(self.cache_max_items)
            too_big = self.cache_max_bytes is not None and total_bytes > int(self.cache_max_bytes)
            if not (too_many or too_big):
                break
            if not recs:
                break
            _mt, b, meta = recs.pop(0)
            try:
                vec = meta.with_suffix("").with_suffix(".npy")
                if vec.exists():
                    vec.unlink(missing_ok=True)  # type: ignore[arg-type]
                meta.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            total_bytes -= int(b)

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        started = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()
        error: Optional[str] = None

        texts_by_source = self._get_sources(doc)

        if not any(t.strip() for t in texts_by_source.values()):
            return {
                "device": self.device,
                "version": self.VERSION,
                "system": {
                    "pre_init": self._init_metrics.get("pre_init"),
                    "post_init": self._init_metrics.get("post_init"),
                    "post_process": sys_before,
                    "peaks": {
                        "ram_peak_mb": int(max(self._init_metrics.get("ram_peak_bytes", 0), mem_before) / 1024 / 1024),
                        "gpu_peak_mb": int(self._gpu_used_mb(sys_before)),
                    },
                },
                "timings_s": {"total": 0.0},
                "result": {
                    "features_flat": {
                        "tp_tchunk_present": 0.0,
                        "tp_tchunk_sources_count": 0.0,
                        "tp_tchunk_whisper_present": 0.0,
                        "tp_tchunk_youtube_auto_present": 0.0,
                        "tp_tchunk_whisper_chunks": 0.0,
                        "tp_tchunk_youtube_chunks": 0.0,
                        "tp_tchunk_embedding_dim": float("nan"),
                        "tp_tchunk_conf_present": 0.0,
                        "tp_tchunk_conf_mean": float("nan"),
                        "tp_tchunk_conf_min": float("nan"),
                        "tp_tchunk_conf_max": float("nan"),
                    }
                },
                "error": None,
            }

        results_by_source: Dict[str, Any] = {}
        tp_art = getattr(doc, "tp_artifacts", None)
        if not isinstance(tp_art, dict):
            tp_art = {}
            setattr(doc, "tp_artifacts", tp_art)
        # Canonical registry
        tp_art.setdefault("transcripts", {})
        # Legacy registry (kept for downstream compatibility)
        tp_art.setdefault("transcript_chunks", {})

        # Process each available source independently
        for source_key, source_text in texts_by_source.items():
            if not source_text.strip():
                continue

            # Stable transcript_id: prefer token IDs if available; else fallback to text hash (privacy-safe: only hash).
            token_ids = None
            try:
                tti = getattr(doc, "transcripts_token_ids", {}) or {}
                if isinstance(tti, dict):
                    token_ids = tti.get(source_key)
                    if not isinstance(token_ids, list):
                        token_ids = None
            except Exception:
                token_ids = None

            transcript_id = (
                self._transcript_id_from_token_ids(self.weights_digest, source_key, token_ids)
                or self._hash_text(source_text, self.weights_digest, source_key)
            )
            _ = transcript_id  # id kept for in-memory registry only

            # Deterministic per-run artifacts (no hashes in filenames)
            artifacts_vec_path = self.artifacts_dir / f"transcript_{source_key}_chunk_embeddings.npy"

            # Optional cache (outside result_store). Default off.
            cached_ok = False
            cache_vec_path, cache_meta_path = self._cache_paths(transcript_id)
            if self.cache_enabled and cache_meta_path.exists() and cache_vec_path.exists():
                try:
                    meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
                    vectors = np.load(cache_vec_path)
                    vectors = np.asarray(vectors, dtype=np.float32)
                    tmp_vec = artifacts_vec_path.with_suffix(".tmp.npy")
                    np.save(tmp_vec, vectors)
                    tmp_vec.replace(artifacts_vec_path)
                    results_by_source[source_key] = {
                        "n_chunks": int(meta.get("n_chunks", 0)),
                        "embedding_dim": int(meta.get("embedding_dim", 0)),
                        "conf_present": float(meta.get("conf_present", 0.0) or 0.0),
                        "conf_mean": float(meta.get("conf_mean")) if meta.get("conf_mean") is not None else float("nan"),
                        "conf_min": float(meta.get("conf_min")) if meta.get("conf_min") is not None else float("nan"),
                        "conf_max": float(meta.get("conf_max")) if meta.get("conf_max") is not None else float("nan"),
                    }
                    cached_ok = True
                except Exception:
                    cached_ok = False

            if not cached_ok:
                chunk_conf: List[Optional[float]] = []
                if source_key == "whisper":
                    segs = self._get_asr_segments_with_conf(doc)
                else:
                    segs = []
                if segs:
                    chunks, chunk_conf = self._chunk_by_asr_segments(segs)
                else:
                    chunks = self._split_into_chunks(source_text)
                embeddings = self._encode_chunks(chunks)

                # save vectors to artifacts (atomic tmp → final)
                tmp_vec = artifacts_vec_path.with_suffix(".tmp.npy")
                np.save(tmp_vec, embeddings.astype(np.float32))
                tmp_vec.replace(artifacts_vec_path)

                conf_vals = [float(c) for c in chunk_conf if isinstance(c, (int, float))]
                conf_present = 1.0 if conf_vals else 0.0
                conf_mean = float(sum(conf_vals) / len(conf_vals)) if conf_vals else float("nan")
                conf_min = float(min(conf_vals)) if conf_vals else float("nan")
                conf_max = float(max(conf_vals)) if conf_vals else float("nan")

                if self.cache_enabled:
                    # Privacy-safe cache meta: no raw text.
                    meta = {
                        "source": source_key,
                        "model_name": self.model_name,
                        "model_version": self.model_version,
                        "weights_digest": self.weights_digest,
                        "device": self.device,
                        "n_chunks": int(len(chunks)),
                        "embedding_dim": int(embeddings.shape[1]) if embeddings.size > 0 else 0,
                        "conf_present": float(conf_present),
                        "conf_mean": None if math.isnan(conf_mean) else float(conf_mean),
                        "conf_min": None if math.isnan(conf_min) else float(conf_min),
                        "conf_max": None if math.isnan(conf_max) else float(conf_max),
                    }
                    tmp_meta = cache_meta_path.with_suffix(".tmp.json")
                    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
                    os.replace(str(tmp_meta), str(cache_meta_path))
                    tmp_c = cache_vec_path.with_suffix(".tmp.npy")
                    np.save(tmp_c, embeddings.astype(np.float32))
                    tmp_c.replace(cache_vec_path)

                results_by_source[source_key] = {
                    "n_chunks": len(chunks),
                    "embedding_dim": int(embeddings.shape[1]) if embeddings.size > 0 else 0,
                    "conf_present": float(conf_present),
                    "conf_mean": float(conf_mean),
                    "conf_min": float(conf_min),
                    "conf_max": float(conf_max),
                }

            # In-memory registry for downstream (no absolute paths in result/NPZ).
            rel = artifacts_vec_path.name
            # Canonical
            tp_art["transcripts"].setdefault(source_key, {})
            tp_art["transcripts"][source_key].update(
                {
                    "transcript_id": str(transcript_id),
                    "chunk_embeddings_relpath": str(rel),
                    "n_chunks": int(results_by_source[source_key].get("n_chunks", 0)),
                    "embedding_dim": int(results_by_source[source_key].get("embedding_dim", 0)),
                }
            )
            # Legacy
            tp_art["transcript_chunks"][source_key] = {
                "transcript_id": str(transcript_id),
                "embeddings_relpath": str(rel),
                "embeddings_path": str(rel),
                "n_chunks": int(results_by_source[source_key].get("n_chunks", 0)),
                "embedding_dim": int(results_by_source[source_key].get("embedding_dim", 0)),
            }

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = (time.perf_counter() - started)

        gpu_peak_mb = max(
            self._gpu_used_mb(self._init_metrics.get("pre_init")),
            self._gpu_used_mb(self._init_metrics.get("post_init")),
            self._gpu_used_mb(sys_after),
        )

        features_flat = {
            "tp_tchunk_present": 1.0,
            "tp_tchunk_sources_count": float(len(results_by_source)),
            "tp_tchunk_whisper_present": 1.0 if ("whisper" in results_by_source) else 0.0,
            "tp_tchunk_youtube_auto_present": 1.0 if ("youtube_auto" in results_by_source) else 0.0,
            "tp_tchunk_whisper_chunks": float(
                results_by_source.get("whisper", {}).get("n_chunks", 0) if isinstance(results_by_source.get("whisper"), dict) else 0
            ),
            "tp_tchunk_youtube_chunks": float(
                results_by_source.get("youtube_auto", {}).get("n_chunks", 0) if isinstance(results_by_source.get("youtube_auto"), dict) else 0
            ),
            "tp_tchunk_embedding_dim": float(
                results_by_source.get("whisper", {}).get("embedding_dim", float("nan"))
                if isinstance(results_by_source.get("whisper"), dict)
                else (
                    results_by_source.get("youtube_auto", {}).get("embedding_dim", float("nan"))
                    if isinstance(results_by_source.get("youtube_auto"), dict)
                    else float("nan")
                )
            ),
        }
        if self.emit_confidence_metrics:
            # confidence metrics only make sense for ASR-derived chunks (whisper)
            wd = results_by_source.get("whisper") if isinstance(results_by_source.get("whisper"), dict) else {}
            features_flat["tp_tchunk_conf_present"] = float(wd.get("conf_present", 0.0) or 0.0)
            features_flat["tp_tchunk_conf_mean"] = float(wd.get("conf_mean", float("nan")))
            features_flat["tp_tchunk_conf_min"] = float(wd.get("conf_min", float("nan")))
            features_flat["tp_tchunk_conf_max"] = float(wd.get("conf_max", float("nan")))
        else:
            features_flat["tp_tchunk_conf_present"] = 0.0
            features_flat["tp_tchunk_conf_mean"] = float("nan")
            features_flat["tp_tchunk_conf_min"] = float("nan")
            features_flat["tp_tchunk_conf_max"] = float("nan")
        if self.emit_extra_metrics:
            features_flat["tp_tchunk_batch_size"] = float(int(self.batch_size))
            features_flat["tp_tchunk_max_chunk_tokens_model"] = float(int(self.max_chunk_tokens_model))
            features_flat["tp_tchunk_overlap_ratio"] = float(self.overlap_ratio)
            features_flat["tp_tchunk_max_chunks_total"] = float(int(self.max_chunks_total))
            features_flat["tp_tchunk_cache_enabled"] = 1.0 if self.cache_enabled else 0.0

        return {
            "device": self.device,
            "version": self.VERSION,
            "model_version": str(self.model_version),
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
            "result": {"features_flat": features_flat},
            "error": None,
        }

    @property
    def supports_batch(self) -> bool:
        """TranscriptChunkEmbedder supports batch processing."""
        return True

    def extract_batch(self, docs: List[VideoDocument]) -> List[Dict[str, Any]]:
        """
        Batch processing: collect all chunks from all documents and sources,
        encode in batches, then distribute back per-document.
        """
        import time
        from collections import defaultdict

        started = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        n_docs = len(docs)
        if n_docs == 0:
            return []

        # Step 1: Collect chunks per document and source
        # Structure: doc_idx -> source_key -> (chunks, confidences, transcript_id, artifacts_path)
        doc_chunks_data: Dict[int, Dict[str, Tuple[List[str], List[Optional[float]], str, Path]]] = {}
        all_chunks_flat: List[str] = []
        chunk_to_doc_source: List[Tuple[int, str, int]] = []  # (doc_idx, source_key, chunk_idx_in_source)

        for doc_idx, doc in enumerate(docs):
            texts_by_source = self._get_sources(doc)
            doc_chunks_data[doc_idx] = {}

            for source_key, source_text in texts_by_source.items():
                if not source_text.strip():
                    continue

                # Get transcript_id (same logic as extract())
                token_ids = None
                try:
                    tti = getattr(doc, "transcripts_token_ids", {}) or {}
                    if isinstance(tti, dict):
                        token_ids = tti.get(source_key)
                        if not isinstance(token_ids, list):
                            token_ids = None
                except Exception:
                    token_ids = None

                transcript_id = (
                    self._transcript_id_from_token_ids(self.weights_digest, source_key, token_ids)
                    or self._hash_text(source_text, self.weights_digest, source_key)
                )

                # Get per-doc artifacts directory
                doc_artifacts_dir = getattr(doc, "_tp_artifacts_dir", None)
                if doc_artifacts_dir:
                    artifacts_vec_path = Path(doc_artifacts_dir) / f"transcript_{source_key}_chunk_embeddings.npy"
                else:
                    artifacts_vec_path = self.artifacts_dir / f"transcript_{source_key}_chunk_embeddings.npy"

                # Chunk the text
                chunk_conf: List[Optional[float]] = []
                if source_key == "whisper":
                    segs = self._get_asr_segments_with_conf(doc)
                else:
                    segs = []
                if segs:
                    chunks, chunk_conf = self._chunk_by_asr_segments(segs)
                else:
                    chunks = self._split_into_chunks(source_text)

                # Store mapping and collect chunks
                doc_chunks_data[doc_idx][source_key] = (chunks, chunk_conf, transcript_id, artifacts_vec_path)
                for chunk_idx, chunk in enumerate(chunks):
                    all_chunks_flat.append(chunk)
                    chunk_to_doc_source.append((doc_idx, source_key, chunk_idx))

        # Step 2: Batch encode all chunks
        t_enc0 = time.perf_counter()
        if all_chunks_flat:
            all_embeddings = self._encode_chunks(all_chunks_flat)  # (N_total_chunks, D)
        else:
            all_embeddings = np.zeros((0, 0), dtype=np.float32)
        t_enc_s = time.perf_counter() - t_enc0

        # Step 3: Process each document: distribute embeddings, save artifacts, build results
        results: List[Dict[str, Any]] = []
        global_chunk_idx = 0

        for doc_idx, doc in enumerate(docs):
            doc_t0 = time.perf_counter()
            doc_sys_before = system_snapshot()
            doc_mem_before = process_memory_bytes()

            texts_by_source = self._get_sources(doc)
            if not any(t.strip() for t in texts_by_source.values()):
                # Empty case
                sys_after = system_snapshot()
                mem_after = process_memory_bytes()
                total_s = time.perf_counter() - doc_t0
                results.append({
                    "device": self.device,
                    "version": self.VERSION,
                    "system": {
                        "pre_init": self._init_metrics.get("pre_init"),
                        "post_init": self._init_metrics.get("post_init"),
                        "post_process": sys_after,
                        "peaks": {
                            "ram_peak_mb": int(max(self._init_metrics.get("ram_peak_bytes", 0), doc_mem_before, mem_after) / 1024 / 1024),
                            "gpu_peak_mb": int(self._gpu_used_mb(sys_after)),
                        },
                    },
                    "timings_s": {"total": round(total_s, 3)},
                    "result": {
                        "features_flat": {
                            "tp_tchunk_present": 0.0,
                            "tp_tchunk_sources_count": 0.0,
                            "tp_tchunk_whisper_present": 0.0,
                            "tp_tchunk_youtube_auto_present": 0.0,
                            "tp_tchunk_whisper_chunks": 0.0,
                            "tp_tchunk_youtube_chunks": 0.0,
                            "tp_tchunk_embedding_dim": float("nan"),
                            "tp_tchunk_conf_present": 0.0,
                            "tp_tchunk_conf_mean": float("nan"),
                            "tp_tchunk_conf_min": float("nan"),
                            "tp_tchunk_conf_max": float("nan"),
                        }
                    },
                    "error": None,
                })
                continue

            # Initialize tp_artifacts
            tp_art = getattr(doc, "tp_artifacts", None)
            if not isinstance(tp_art, dict):
                tp_art = {}
                setattr(doc, "tp_artifacts", tp_art)
            tp_art.setdefault("transcripts", {})
            tp_art.setdefault("transcript_chunks", {})

            results_by_source: Dict[str, Any] = {}
            error: Optional[str] = None

            # Process each source for this document
            if doc_idx in doc_chunks_data:
                for source_key, (chunks, chunk_conf, transcript_id, artifacts_vec_path) in doc_chunks_data[doc_idx].items():
                    # Extract embeddings for this source's chunks
                    n_chunks = len(chunks)
                    if n_chunks == 0:
                        continue

                    source_embeddings = all_embeddings[global_chunk_idx:global_chunk_idx + n_chunks]
                    global_chunk_idx += n_chunks

                    # Save artifact
                    try:
                        artifacts_vec_path.parent.mkdir(parents=True, exist_ok=True)
                        tmp_vec = artifacts_vec_path.with_suffix(".tmp.npy")
                        np.save(tmp_vec, source_embeddings.astype(np.float32))
                        tmp_vec.replace(artifacts_vec_path)
                    except Exception as e:
                        error = f"artifact_save_error: {e}"

                    # Compute confidence stats
                    conf_vals = [float(c) for c in chunk_conf if isinstance(c, (int, float))]
                    conf_present = 1.0 if conf_vals else 0.0
                    conf_mean = float(sum(conf_vals) / len(conf_vals)) if conf_vals else float("nan")
                    conf_min = float(min(conf_vals)) if conf_vals else float("nan")
                    conf_max = float(max(conf_vals)) if conf_vals else float("nan")

                    results_by_source[source_key] = {
                        "n_chunks": n_chunks,
                        "embedding_dim": int(source_embeddings.shape[1]) if source_embeddings.size > 0 else 0,
                        "conf_present": float(conf_present),
                        "conf_mean": float(conf_mean),
                        "conf_min": float(conf_min),
                        "conf_max": float(conf_max),
                    }

                    # Update tp_artifacts
                    rel = artifacts_vec_path.name
                    tp_art["transcripts"].setdefault(source_key, {})
                    tp_art["transcripts"][source_key].update({
                        "transcript_id": str(transcript_id),
                        "chunk_embeddings_relpath": str(rel),
                        "n_chunks": int(n_chunks),
                        "embedding_dim": int(source_embeddings.shape[1]) if source_embeddings.size > 0 else 0,
                    })
                    # Legacy
                    tp_art["transcript_chunks"][source_key] = {
                        "transcript_id": str(transcript_id),
                        "embeddings_relpath": str(rel),
                        "embeddings_path": str(rel),
                        "n_chunks": int(n_chunks),
                        "embedding_dim": int(source_embeddings.shape[1]) if source_embeddings.size > 0 else 0,
                    }

            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = (time.perf_counter() - doc_t0)

            gpu_peak_mb = max(
                self._gpu_used_mb(self._init_metrics.get("pre_init")),
                self._gpu_used_mb(self._init_metrics.get("post_init")),
                self._gpu_used_mb(sys_after),
            )

            features_flat = {
                "tp_tchunk_present": 1.0 if results_by_source else 0.0,
                "tp_tchunk_sources_count": float(len(results_by_source)),
                "tp_tchunk_whisper_present": 1.0 if ("whisper" in results_by_source) else 0.0,
                "tp_tchunk_youtube_auto_present": 1.0 if ("youtube_auto" in results_by_source) else 0.0,
                "tp_tchunk_whisper_chunks": float(
                    results_by_source.get("whisper", {}).get("n_chunks", 0) if isinstance(results_by_source.get("whisper"), dict) else 0
                ),
                "tp_tchunk_youtube_chunks": float(
                    results_by_source.get("youtube_auto", {}).get("n_chunks", 0) if isinstance(results_by_source.get("youtube_auto"), dict) else 0
                ),
                "tp_tchunk_embedding_dim": float(
                    results_by_source.get("whisper", {}).get("embedding_dim", float("nan"))
                    if isinstance(results_by_source.get("whisper"), dict)
                    else (results_by_source.get("youtube_auto", {}).get("embedding_dim", float("nan")) if isinstance(results_by_source.get("youtube_auto"), dict) else float("nan"))
                ),
            }

            # Confidence metrics (from whisper if available)
            wd = results_by_source.get("whisper", {})
            if isinstance(wd, dict) and wd.get("conf_present", 0.0) > 0.5:
                features_flat["tp_tchunk_conf_present"] = float(wd.get("conf_present", 0.0) or 0.0)
                features_flat["tp_tchunk_conf_mean"] = float(wd.get("conf_mean", float("nan")))
                features_flat["tp_tchunk_conf_min"] = float(wd.get("conf_min", float("nan")))
                features_flat["tp_tchunk_conf_max"] = float(wd.get("conf_max", float("nan")))
            else:
                features_flat["tp_tchunk_conf_present"] = 0.0
                features_flat["tp_tchunk_conf_mean"] = float("nan")
                features_flat["tp_tchunk_conf_min"] = float("nan")
                features_flat["tp_tchunk_conf_max"] = float("nan")

            features_flat["tp_tchunk_batch_size"] = float(int(self.batch_size))
            features_flat["tp_tchunk_max_chunk_tokens_model"] = float(int(self.max_chunk_tokens_model))
            features_flat["tp_tchunk_overlap_ratio"] = float(self.overlap_ratio)
            features_flat["tp_tchunk_max_chunks_total"] = float(int(self.max_chunks_total))
            features_flat["tp_tchunk_cache_enabled"] = 1.0 if self.cache_enabled else 0.0

            results.append({
                "device": self.device,
                "version": self.VERSION,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "weights_digest": self.weights_digest,
                "system": {
                    "pre_init": self._init_metrics.get("pre_init"),
                    "post_init": self._init_metrics.get("post_init"),
                    "post_process": sys_after,
                    "peaks": {
                        "ram_peak_mb": int(max(self._init_metrics.get("ram_peak_bytes", 0), doc_mem_before, mem_after) / 1024 / 1024),
                        "gpu_peak_mb": int(gpu_peak_mb),
                    },
                },
                "timings_s": {
                    "encode": round(t_enc_s / n_docs, 3),  # Per-doc share of batch encoding
                    "total": round(total_s, 3),
                },
                "result": {"features_flat": features_flat},
                "error": error,
            })

        return results


