from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.path_utils import default_artifacts_dir


class EmbeddingShiftIndicatorExtractor(BaseExtractor):
    VERSION = "1.2.0"

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
        from pathlib import Path

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

    @staticmethod
    def _safe_join_artifacts_dir(base_dir: "Any", relpath: str) -> "Any":
        base = base_dir.expanduser().resolve()
        cand = (base / relpath).resolve()
        if not (cand == base or base in cand.parents):
            raise RuntimeError("embedding_shift_indicator_extractor: relpath escapes artifacts_dir")
        return cand

    @staticmethod
    def _cosine_checked(a: np.ndarray, b: np.ndarray) -> tuple[float, bool, bool]:
        """
        Returns (cosine, zero_norm_flag, nan_inf_flag).
        """
        if (not np.isfinite(a).all()) or (not np.isfinite(b).all()):
            return float("nan"), False, True
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 0.0 or nb <= 0.0:
            return float("nan"), True, False
        return float(np.dot(a, b) / (na * nb)), False, False

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        def _stable_template() -> Dict[str, float]:
            return {
                "tp_embshift_present": 0.0,
                "tp_embshift_disabled_by_policy": 0.0,
                "tp_embshift_enabled": float(bool(self.enabled)),
                "tp_embshift_require_transcript_chunks_enabled": float(bool(self.require_transcript_chunks)),
                "tp_embshift_require_min_chunks": float(int(self.require_min_chunks)),
                "tp_embshift_n_chunks": 0.0,
                "tp_embshift_n_window_chunks": float("nan"),
                "tp_embshift_dim": float("nan"),
                "tp_embshift_cosine_begin_end": float("nan"),
                "tp_embshift_shift_flag": float("nan"),
                "tp_embshift_cosine_threshold": float(self.cosine_threshold),
                "tp_embshift_margin": float("nan"),
                # extra cosines (gated)
                "tp_embshift_cosine_first_last": float("nan"),
                "tp_embshift_mean_cosine_last_to_start_window": float("nan"),
                "tp_embshift_compute_shift_flag_enabled": float(bool(self.compute_shift_flag)),
                "tp_embshift_compute_extra_cosines_enabled": float(bool(self.compute_extra_cosines)),
                # source flags
                "tp_embshift_source_used_whisper": 0.0,
                "tp_embshift_source_used_youtube_auto": 0.0,
                "tp_embshift_used_legacy_key_flag": 0.0,
                # safety flags
                "tp_embshift_unsafe_relpath_flag": 0.0,
                "tp_embshift_dim_mismatch_flag": 0.0,
                "tp_embshift_zero_norm_flag": 0.0,
                "tp_embshift_nan_inf_flag": 0.0,
                # timings
                "tp_embshift_load_ms": float("nan"),
                "tp_embshift_compute_ms": float("nan"),
            }

        features_flat = _stable_template()

        if not self.enabled:
            features_flat["tp_embshift_disabled_by_policy"] = 1.0
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return {
                "device": "cpu",
                "version": self.VERSION,
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

        # Deterministic source: read chunk embeddings via in-memory registry filled by TranscriptChunkEmbedder.
        tp = getattr(doc, "tp_artifacts", None)
        transcripts = tp.get("transcripts") if isinstance(tp, dict) else None
        tchunks = tp.get("transcript_chunks") if isinstance(tp, dict) else None
        rel = None
        source_used: Optional[str] = None
        used_legacy = False
        unsafe_relpath_flag = False
        if isinstance(tchunks, dict):
            seen = set()
            for k in list(self.transcript_source_priority):
                if k in seen:
                    continue
                seen.add(k)
                # Canonical first: tp_artifacts["transcripts"][source]["chunk_embeddings_relpath"]
                if isinstance(transcripts, dict):
                    d2 = transcripts.get(k)
                    if isinstance(d2, dict):
                        rel2 = d2.get("chunk_embeddings_relpath")
                        if isinstance(rel2, str) and rel2:
                            rel = rel2
                            source_used = str(k)
                            used_legacy = False
                            break
                # Legacy fallback: tp_artifacts["transcript_chunks"][source]["embeddings_relpath"/"embeddings_path"]
                d = tchunks.get(k)
                if isinstance(d, dict):
                    rel3 = d.get("embeddings_relpath") or d.get("embeddings_path")
                    if isinstance(rel3, str) and rel3:
                        rel = rel3
                        source_used = str(k)
                        used_legacy = True
                        break

        if source_used is not None:
            features_flat["tp_embshift_source_used_whisper"] = float(source_used == "whisper")
            features_flat["tp_embshift_source_used_youtube_auto"] = float(source_used == "youtube_auto")
            features_flat["tp_embshift_used_legacy_key_flag"] = float(bool(used_legacy))

        if not isinstance(rel, str) or not rel:
            if self.require_transcript_chunks:
                raise RuntimeError("embedding_shift_indicator_extractor: missing transcript chunk embeddings relpath in doc.tp_artifacts")
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return {
                "device": "cpu",
                "version": self.VERSION,
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

        try:
            p = self._safe_join_artifacts_dir(self.artifacts_dir, rel)
        except Exception:
            unsafe_relpath_flag = True
            features_flat["tp_embshift_unsafe_relpath_flag"] = 1.0
            if self.require_transcript_chunks:
                raise RuntimeError("embedding_shift_indicator_extractor: unsafe transcript chunk embeddings relpath")
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return {
                "device": "cpu",
                "version": self.VERSION,
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

        if not p.exists():
            if self.require_transcript_chunks:
                raise RuntimeError("embedding_shift_indicator_extractor: transcript chunk embeddings file not found in per-run artifacts")
            return {
                "device": "cpu",
                "version": self.VERSION,
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

        t_load0 = time.perf_counter()
        emb = np.load(p)
        emb = np.asarray(emb, dtype=np.float32)
        load_s = time.perf_counter() - t_load0
        features_flat["tp_embshift_load_ms"] = float(round(load_s * 1000.0, 3))

        # Strict shape validation
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if emb.ndim != 2 or int(emb.shape[0]) <= 0 or int(emb.shape[1]) <= 0:
            features_flat["tp_embshift_dim_mismatch_flag"] = 1.0
            if self.require_transcript_chunks:
                raise RuntimeError("embedding_shift_indicator_extractor: invalid embeddings matrix shape")
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return {
                "device": "cpu",
                "version": self.VERSION,
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

        if not np.isfinite(emb).all():
            features_flat["tp_embshift_nan_inf_flag"] = 1.0
            if self.require_transcript_chunks:
                raise RuntimeError("embedding_shift_indicator_extractor: embeddings contain NaN/inf")
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return {
                "device": "cpu",
                "version": self.VERSION,
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
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return {
                "device": "cpu",
                "version": self.VERSION,
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

        t_comp0 = time.perf_counter()
        start_emb = emb[:win].mean(axis=0)
        end_emb = emb[-win:].mean(axis=0)
        cosine_shift, zn, ni = self._cosine_checked(start_emb, end_emb)
        if zn:
            features_flat["tp_embshift_zero_norm_flag"] = 1.0
        if ni:
            features_flat["tp_embshift_nan_inf_flag"] = 1.0
        compute_s = time.perf_counter() - t_comp0
        features_flat["tp_embshift_compute_ms"] = float(round(compute_s * 1000.0, 3))
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
            # mean cosine between last window chunks and start window centroid
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

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        # If core cosine is invalid, treat as valid empty (no fake metrics)
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

        return {
            "device": "cpu",
            "version": self.VERSION,
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


