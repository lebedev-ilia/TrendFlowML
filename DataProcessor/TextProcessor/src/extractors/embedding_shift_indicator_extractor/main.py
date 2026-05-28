from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.path_utils import default_artifacts_dir

_FEATURES_FLAT_KEYS: Tuple[str, ...] = (
    "tp_embshift_present",
    "tp_embshift_disabled_by_policy",
    "tp_embshift_enabled",
    "tp_embshift_require_transcript_chunks_enabled",
    "tp_embshift_require_min_chunks",
    "tp_embshift_emit_extra_metrics_enabled",
    "tp_embshift_compute_shift_flag_enabled",
    "tp_embshift_compute_extra_cosines_enabled",
    "tp_embshift_n_chunks",
    "tp_embshift_n_window_chunks",
    "tp_embshift_dim",
    "tp_embshift_cosine_threshold",
    "tp_embshift_cosine_begin_end",
    "tp_embshift_shift_flag",
    "tp_embshift_margin",
    "tp_embshift_cosine_first_last",
    "tp_embshift_mean_cosine_last_to_start_window",
    "tp_embshift_source_used_whisper",
    "tp_embshift_source_used_youtube_auto",
    "tp_embshift_used_legacy_key_flag",
    "tp_embshift_unsafe_relpath_flag",
    "tp_embshift_chunk_embed_missing_flag",
    "tp_embshift_dim_mismatch_flag",
    "tp_embshift_zero_norm_flag",
    "tp_embshift_nan_inf_flag",
    "tp_embshift_load_ms",
    "tp_embshift_compute_ms",
)


class EmbeddingShiftIndicatorExtractor(BaseExtractor):
    VERSION = "1.3.0"

    def __init__(
        self,
        artifacts_dir: str | None = None,
        n_window_chunks: int = 2,
        cosine_threshold: float = 0.85,
        transcript_source_priority: Sequence[str] | str = ("whisper", "youtube_auto"),
        enabled: bool = True,
        require_transcript_chunks: bool = False,
        require_min_chunks: int = 2,
        compute_shift_flag: bool = True,
        compute_extra_cosines: bool = False,
        emit_extra_metrics: bool = False,
    ) -> None:
        init_sys_before = system_snapshot()
        init_mem_before = process_memory_bytes()

        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.n_window_chunks = int(max(1, n_window_chunks))
        self.cosine_threshold = float(cosine_threshold)
        self.enabled = bool(enabled)
        self.require_transcript_chunks = bool(require_transcript_chunks)
        self.require_min_chunks = int(max(1, require_min_chunks))
        self.compute_shift_flag = bool(compute_shift_flag)
        self.compute_extra_cosines = bool(compute_extra_cosines)
        self.emit_extra_metrics = bool(emit_extra_metrics)
        if isinstance(transcript_source_priority, str):
            pr = [p.strip() for p in transcript_source_priority.split(",") if p.strip()]
        else:
            pr = [str(p).strip() for p in transcript_source_priority if str(p).strip()]
        self.transcript_source_priority = pr or ["whisper", "youtube_auto"]

        init_sys_after = system_snapshot()
        init_mem_after = process_memory_bytes()
        self._init_metrics: Dict[str, Any] = {
            "pre_init": init_sys_before,
            "post_init": init_sys_after,
            "ram_peak_bytes": max(init_mem_before, init_mem_after),
        }

    @staticmethod
    def _safe_join_artifacts_dir(base_dir: Path, relpath: str) -> Path:
        base = base_dir.expanduser().resolve()
        cand = (base / relpath).resolve()
        if not (cand == base or base in cand.parents):
            raise RuntimeError("embedding_shift_indicator_extractor: relpath escapes artifacts_dir")
        return cand

    @staticmethod
    def _cosine_checked(a: np.ndarray, b: np.ndarray) -> tuple[float, bool, bool]:
        """Returns (cosine, zero_norm_flag, nan_inf_flag)."""
        if (not np.isfinite(a).all()) or (not np.isfinite(b).all()):
            return float("nan"), False, True
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 0.0 or nb <= 0.0:
            return float("nan"), True, False
        return float(np.dot(a, b) / (na * nb)), False, False

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
    def _pack_features_flat(values: Dict[str, Any]) -> Dict[str, Any]:
        nan = float("nan")
        out: Dict[str, Any] = {}
        for k in _FEATURES_FLAT_KEYS:
            if k not in values:
                raise KeyError(f"EmbeddingShiftIndicatorExtractor: missing features_flat key {k!r}")
            v = values[k]
            if v is None:
                out[k] = nan
            elif isinstance(v, (bool, np.bool_)):
                out[k] = float(bool(v))
            else:
                out[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else nan
        return out

    def _base_features_flat(self) -> Dict[str, Any]:
        nan = float("nan")
        return {
            "tp_embshift_present": 0.0,
            "tp_embshift_disabled_by_policy": 0.0,
            "tp_embshift_enabled": float(bool(self.enabled)),
            "tp_embshift_require_transcript_chunks_enabled": float(bool(self.require_transcript_chunks)),
            "tp_embshift_require_min_chunks": float(int(self.require_min_chunks)),
            "tp_embshift_emit_extra_metrics_enabled": float(bool(self.emit_extra_metrics)),
            "tp_embshift_compute_shift_flag_enabled": float(bool(self.compute_shift_flag)),
            "tp_embshift_compute_extra_cosines_enabled": float(bool(self.compute_extra_cosines)),
            "tp_embshift_n_chunks": 0.0,
            "tp_embshift_n_window_chunks": nan,
            "tp_embshift_dim": nan,
            "tp_embshift_cosine_threshold": float(self.cosine_threshold),
            "tp_embshift_cosine_begin_end": nan,
            "tp_embshift_shift_flag": nan,
            "tp_embshift_margin": nan,
            "tp_embshift_cosine_first_last": nan,
            "tp_embshift_mean_cosine_last_to_start_window": nan,
            "tp_embshift_source_used_whisper": 0.0,
            "tp_embshift_source_used_youtube_auto": 0.0,
            "tp_embshift_used_legacy_key_flag": 0.0,
            "tp_embshift_unsafe_relpath_flag": 0.0,
            "tp_embshift_chunk_embed_missing_flag": 0.0,
            "tp_embshift_dim_mismatch_flag": 0.0,
            "tp_embshift_zero_norm_flag": 0.0,
            "tp_embshift_nan_inf_flag": 0.0,
            "tp_embshift_load_ms": nan,
            "tp_embshift_compute_ms": nan,
        }

    def _maybe_set_load_ms(self, features_flat: Dict[str, Any], load_s: float) -> None:
        if self.emit_extra_metrics:
            features_flat["tp_embshift_load_ms"] = float(round(load_s * 1000.0, 3))

    def _maybe_set_compute_ms(self, features_flat: Dict[str, Any], compute_s: float) -> None:
        if self.emit_extra_metrics:
            features_flat["tp_embshift_compute_ms"] = float(round(compute_s * 1000.0, 3))

    def _build_return(
        self,
        *,
        features_flat: Dict[str, Any],
        sys_after: Any,
        mem_before: int,
        mem_after: int,
        total_s: float,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        gpu_peak_mb = self._gpu_peak_mb(sys_after)
        ram_peak = max(int(self._init_metrics.get("ram_peak_bytes", 0)), mem_before, mem_after)
        return {
            "device": "cpu",
            "version": self.VERSION,
            "model_name": None,
            "model_version": None,
            "weights_digest": None,
            "system": {
                "pre_init": self._init_metrics.get("pre_init"),
                "post_init": self._init_metrics.get("post_init"),
                "post_process": sys_after,
                "peaks": {
                    "ram_peak_mb": int(ram_peak / 1024 / 1024),
                    "gpu_peak_mb": int(gpu_peak_mb),
                },
            },
            "timings_s": {"total": round(total_s, 3)},
            "result": {"features_flat": self._pack_features_flat(features_flat)},
            "error": error,
        }

    def _resolve_chunk_relpath(
        self, doc: Any
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """Returns (relpath, source_used, used_legacy)."""
        tp = getattr(doc, "tp_artifacts", None)
        transcripts = tp.get("transcripts") if isinstance(tp, dict) else None
        tchunks = tp.get("transcript_chunks") if isinstance(tp, dict) else None
        rel: Optional[str] = None
        source_used: Optional[str] = None
        used_legacy = False
        seen: set[str] = set()
        for k in list(self.transcript_source_priority):
            if k in seen:
                continue
            seen.add(str(k))
            if isinstance(transcripts, dict):
                d2 = transcripts.get(k)
                if isinstance(d2, dict):
                    rel2 = d2.get("chunk_embeddings_relpath")
                    if isinstance(rel2, str) and rel2:
                        rel = rel2
                        source_used = str(k)
                        used_legacy = False
                        break
            if isinstance(tchunks, dict):
                d = tchunks.get(k)
                if isinstance(d, dict):
                    rel3 = d.get("embeddings_relpath") or d.get("embeddings_path")
                    if isinstance(rel3, str) and rel3:
                        rel = rel3
                        source_used = str(k)
                        used_legacy = True
                        break
        return rel, source_used, used_legacy

    def extract(self, doc: Any) -> Dict[str, Any]:
        t0 = time.perf_counter()
        mem_before = process_memory_bytes()
        features_flat = self._base_features_flat()
        error: Optional[str] = None

        def _finish(ff: Dict[str, Any]) -> Dict[str, Any]:
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return self._build_return(
                features_flat=ff,
                sys_after=sys_after,
                mem_before=mem_before,
                mem_after=mem_after,
                total_s=total_s,
                error=error,
            )

        if not self.enabled:
            features_flat["tp_embshift_disabled_by_policy"] = 1.0
            return _finish(features_flat)

        rel, source_used, used_legacy = self._resolve_chunk_relpath(doc)
        if source_used is not None:
            features_flat["tp_embshift_source_used_whisper"] = float(source_used == "whisper")
            features_flat["tp_embshift_source_used_youtube_auto"] = float(source_used == "youtube_auto")
            features_flat["tp_embshift_used_legacy_key_flag"] = float(bool(used_legacy))

        if not isinstance(rel, str) or not rel:
            if self.require_transcript_chunks:
                raise RuntimeError("embedding_shift_indicator_extractor: missing transcript chunk embeddings relpath in doc.tp_artifacts")
            return _finish(features_flat)

        try:
            p = self._safe_join_artifacts_dir(self.artifacts_dir, rel)
        except Exception:
            features_flat["tp_embshift_unsafe_relpath_flag"] = 1.0
            if self.require_transcript_chunks:
                raise RuntimeError("embedding_shift_indicator_extractor: unsafe transcript chunk embeddings relpath")
            return _finish(features_flat)

        if not p.exists():
            features_flat["tp_embshift_chunk_embed_missing_flag"] = 1.0
            if self.require_transcript_chunks:
                raise RuntimeError("embedding_shift_indicator_extractor: transcript chunk embeddings file not found in per-run artifacts")
            return _finish(features_flat)

        t_load0 = time.perf_counter()
        try:
            emb = np.load(p)
        except Exception:
            features_flat["tp_embshift_chunk_embed_missing_flag"] = 1.0
            if self.require_transcript_chunks:
                raise RuntimeError("embedding_shift_indicator_extractor: failed to load transcript chunk embeddings from per-run artifacts")
            self._maybe_set_load_ms(features_flat, time.perf_counter() - t_load0)
            return _finish(features_flat)
        load_s = time.perf_counter() - t_load0
        self._maybe_set_load_ms(features_flat, load_s)

        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if emb.ndim != 2 or int(emb.shape[0]) <= 0 or int(emb.shape[1]) <= 0:
            features_flat["tp_embshift_dim_mismatch_flag"] = 1.0
            if self.require_transcript_chunks:
                raise RuntimeError("embedding_shift_indicator_extractor: invalid embeddings matrix shape")
            return _finish(features_flat)

        if not np.isfinite(emb).all():
            features_flat["tp_embshift_nan_inf_flag"] = 1.0
            if self.require_transcript_chunks:
                raise RuntimeError("embedding_shift_indicator_extractor: embeddings contain NaN/inf")
            return _finish(features_flat)

        n_chunks = int(emb.shape[0])
        win = int(min(self.n_window_chunks, max(1, n_chunks // 2)))
        dim = int(emb.shape[1])
        features_flat["tp_embshift_n_chunks"] = float(int(n_chunks))
        features_flat["tp_embshift_n_window_chunks"] = float(int(win))
        features_flat["tp_embshift_dim"] = float(int(dim))

        if int(n_chunks) < int(self.require_min_chunks):
            if self.require_transcript_chunks:
                raise RuntimeError(
                    f"embedding_shift_indicator_extractor: not enough chunks: n_chunks={int(n_chunks)} require_min_chunks={int(self.require_min_chunks)}"
                )
            return _finish(features_flat)

        t_comp0 = time.perf_counter()
        start_emb = emb[:win].mean(axis=0)
        end_emb = emb[-win:].mean(axis=0)
        cosine_shift, zn, ni = self._cosine_checked(start_emb, end_emb)
        if zn:
            features_flat["tp_embshift_zero_norm_flag"] = 1.0
        if ni:
            features_flat["tp_embshift_nan_inf_flag"] = 1.0
        compute_s = time.perf_counter() - t_comp0
        self._maybe_set_compute_ms(features_flat, compute_s)

        margin = float(cosine_shift - float(self.cosine_threshold)) if cosine_shift == cosine_shift else float("nan")
        shift_flag = float("nan")
        if self.compute_shift_flag:
            if cosine_shift == cosine_shift:
                shift_flag = 1.0 if (cosine_shift < self.cosine_threshold) else 0.0
            else:
                shift_flag = float("nan")

        cosine_first_last = float("nan")
        mean_cos_last_to_start = float("nan")
        if self.compute_extra_cosines:
            c1, zn1, ni1 = self._cosine_checked(emb[0], emb[-1])
            cosine_first_last = float(c1)
            if zn1:
                features_flat["tp_embshift_zero_norm_flag"] = 1.0
            if ni1:
                features_flat["tp_embshift_nan_inf_flag"] = 1.0
            start_vec = start_emb
            cos_vals: list[float] = []
            for i in range(max(0, n_chunks - win), n_chunks):
                ci, zn2, ni2 = self._cosine_checked(emb[i], start_vec)
                if zn2:
                    features_flat["tp_embshift_zero_norm_flag"] = 1.0
                if ni2:
                    features_flat["tp_embshift_nan_inf_flag"] = 1.0
                if ci == ci:
                    cos_vals.append(float(ci))
            mean_cos_last_to_start = float(np.mean(np.asarray(cos_vals, dtype=np.float32))) if cos_vals else float("nan")

        if cosine_shift != cosine_shift:
            features_flat["tp_embshift_present"] = 0.0
        else:
            features_flat["tp_embshift_present"] = 1.0

        features_flat["tp_embshift_cosine_begin_end"] = float(cosine_shift)
        features_flat["tp_embshift_shift_flag"] = float(shift_flag)
        features_flat["tp_embshift_margin"] = float(margin)
        if self.compute_extra_cosines:
            features_flat["tp_embshift_cosine_first_last"] = float(cosine_first_last)
            features_flat["tp_embshift_mean_cosine_last_to_start_window"] = float(mean_cos_last_to_start)

        return _finish(features_flat)

