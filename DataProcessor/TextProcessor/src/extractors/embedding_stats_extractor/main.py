from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes

SCHEMA_MAX_TOPVAR_SLOTS = 8
# Keys under tp_artifacts["transcripts"] written by TranscriptChunkEmbedder (Audit v3 ASR path uses "whisper").
SCHEMA_TRANSCRIPT_SOURCES: Tuple[str, ...] = ("whisper", "youtube_auto")


def _build_features_flat_keys() -> Tuple[str, ...]:
    keys: List[str] = [
        "tp_embstats_present",
        "tp_embstats_enabled",
        "tp_embstats_disabled_by_policy",
        "tp_embstats_emit_extra_metrics_enabled",
        "tp_embstats_require_chunks_enabled",
        "tp_embstats_compute_topic_entropy_enabled",
        "tp_embstats_require_topic_distribution_enabled",
        "tp_embstats_schema_topvar_slots_max",
        "tp_embstats_top_k_slots_requested",
        "tp_embstats_top_k_slots",
        "tp_embstats_top_k_slots_clamped",
        "tp_embstats_min_chunks_required",
        "tp_embstats_topk",
        "tp_embstats_variance_ddof",
        "tp_embstats_n_chunks",
        "tp_embstats_dim",
        "tp_embstats_l2_variance",
        "tp_embstats_topic_entropy",
        "tp_embstats_topic_entropy_norm",
        "tp_embstats_topic_perplexity",
        "tp_embstats_topic_entropy_present",
        "tp_embstats_topic_probs_present",
        "tp_embstats_topic_probs_invalid_flag",
        "tp_embstats_used_legacy_key_flag",
        "tp_embstats_unsafe_relpath_flag",
        "tp_embstats_dim_mismatch_flag",
        "tp_embstats_nan_inf_flag",
        "tp_embstats_load_ms",
        "tp_embstats_compute_ms",
    ]
    for i in range(1, SCHEMA_MAX_TOPVAR_SLOTS + 1):
        keys.append(f"tp_embstats_topvar_{i}")
    for src in SCHEMA_TRANSCRIPT_SOURCES:
        keys.append(f"tp_embstats_source_used_{src}")
    return tuple(keys)


_FEATURES_FLAT_KEYS = _build_features_flat_keys()


class EmbeddingStatsExtractor(BaseExtractor):
    """
    Transcript chunk embedding matrix statistics (variance across chunks) + optional
    topic distribution entropy from tp_artifacts (semantics_topics_keyphrases).
    """

    VERSION = "1.2.0"

    def __init__(
        self,
        artifacts_dir: str | None = None,
        transcript_source_priority: List[str] | str | None = None,
        top_k_slots: int = 8,
        topk: int = 8,
        min_chunks_required: int = 2,
        variance_ddof: int = 0,
        enabled: bool = True,
        require_chunks: bool = False,
        compute_topic_entropy: bool = True,
        require_topic_distribution: bool = False,
        emit_extra_metrics: bool = False,
    ) -> None:
        from src.core.path_utils import default_artifacts_dir, default_cache_dir  # local import

        init_sys_before = system_snapshot()
        init_mem_before = process_memory_bytes()

        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.cache_dir = default_cache_dir() / "transcript_embed"

        if transcript_source_priority is None:
            raw_priority = ["whisper"]
        elif isinstance(transcript_source_priority, str):
            raw_priority = [s.strip() for s in transcript_source_priority.split(",") if s.strip()]
        else:
            raw_priority = [str(s).strip() for s in transcript_source_priority if str(s).strip()]

        canon = set(SCHEMA_TRANSCRIPT_SOURCES)
        self.transcript_source_priority = [s for s in raw_priority if s in canon]
        if not self.transcript_source_priority:
            self.transcript_source_priority = ["whisper"]

        t_req = int(max(0, int(top_k_slots)))
        self.top_k_slots_requested = t_req
        self.top_k_slots = min(t_req, SCHEMA_MAX_TOPVAR_SLOTS)
        self.top_k_slots_clamped = bool(t_req > SCHEMA_MAX_TOPVAR_SLOTS)

        self.topk = int(topk)
        self.min_chunks_required = int(min_chunks_required)
        self.variance_ddof = int(variance_ddof)
        self.enabled = bool(enabled)
        self.require_chunks = bool(require_chunks)
        self.compute_topic_entropy = bool(compute_topic_entropy)
        self.require_topic_distribution = bool(require_topic_distribution)
        self.emit_extra_metrics = bool(emit_extra_metrics)

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
    def _safe_join_artifacts_dir(base_dir: Path, relpath: str) -> Path:
        base = base_dir.expanduser().resolve()
        cand = (base / relpath).resolve()
        if not (cand == base or base in cand.parents):
            raise RuntimeError("embedding_stats_extractor: relpath escapes artifacts_dir")
        return cand

    @staticmethod
    def _pack_features_flat(values: Dict[str, Any]) -> Dict[str, Any]:
        nan = float("nan")
        out: Dict[str, Any] = {}
        for k in _FEATURES_FLAT_KEYS:
            if k not in values:
                raise KeyError(f"EmbeddingStatsExtractor: missing features_flat key {k!r}")
            v = values[k]
            if v is None:
                out[k] = nan
            elif isinstance(v, (bool, np.bool_)):
                out[k] = float(bool(v))
            else:
                out[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else nan
        return out

    def _config_numeric_fragment(self) -> Dict[str, float]:
        return {
            "tp_embstats_emit_extra_metrics_enabled": float(bool(self.emit_extra_metrics)),
            "tp_embstats_require_chunks_enabled": float(bool(self.require_chunks)),
            "tp_embstats_compute_topic_entropy_enabled": float(bool(self.compute_topic_entropy)),
            "tp_embstats_require_topic_distribution_enabled": float(bool(self.require_topic_distribution)),
            "tp_embstats_schema_topvar_slots_max": float(SCHEMA_MAX_TOPVAR_SLOTS),
            "tp_embstats_top_k_slots_requested": float(int(self.top_k_slots_requested)),
            "tp_embstats_top_k_slots": float(int(self.top_k_slots)),
            "tp_embstats_top_k_slots_clamped": 1.0 if self.top_k_slots_clamped else 0.0,
            "tp_embstats_min_chunks_required": float(int(self.min_chunks_required)),
            "tp_embstats_topk": float(int(self.topk)),
            "tp_embstats_variance_ddof": float(int(self.variance_ddof)),
        }

    def _empty_shell(
        self,
        *,
        enabled: bool,
        disabled_by_policy: bool,
        present: float,
    ) -> Dict[str, Any]:
        nan = float("nan")
        d: Dict[str, Any] = {
            "tp_embstats_present": float(present),
            "tp_embstats_enabled": float(bool(enabled)),
            "tp_embstats_disabled_by_policy": 1.0 if disabled_by_policy else 0.0,
            "tp_embstats_n_chunks": nan,
            "tp_embstats_dim": nan,
            "tp_embstats_l2_variance": nan,
            "tp_embstats_topic_entropy": nan,
            "tp_embstats_topic_entropy_norm": nan,
            "tp_embstats_topic_perplexity": nan,
            "tp_embstats_topic_entropy_present": 0.0,
            "tp_embstats_topic_probs_present": 0.0,
            "tp_embstats_topic_probs_invalid_flag": 0.0,
            "tp_embstats_used_legacy_key_flag": 0.0,
            "tp_embstats_unsafe_relpath_flag": 0.0,
            "tp_embstats_dim_mismatch_flag": 0.0,
            "tp_embstats_nan_inf_flag": 0.0,
            "tp_embstats_load_ms": nan,
            "tp_embstats_compute_ms": nan,
        }
        d.update(self._config_numeric_fragment())
        for i in range(1, SCHEMA_MAX_TOPVAR_SLOTS + 1):
            d[f"tp_embstats_topvar_{i}"] = nan
        for src in SCHEMA_TRANSCRIPT_SOURCES:
            d[f"tp_embstats_source_used_{src}"] = 0.0
        return d

    def _load_chunks(self, doc: Any) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str], bool, bool, float]:
        import time

        tp = getattr(doc, "tp_artifacts", None)
        transcripts = tp.get("transcripts") if isinstance(tp, dict) else None
        tchunks = tp.get("transcript_chunks") if isinstance(tp, dict) else None
        if (not isinstance(transcripts, dict)) and (not isinstance(tchunks, dict)):
            return None, None, None, False, False, float("nan")

        seen = set()
        for src in self.transcript_source_priority:
            if src in seen:
                continue
            seen.add(src)
            rel = None
            tid = None
            used_legacy = False
            if isinstance(transcripts, dict):
                d2 = transcripts.get(src)
                rel2 = d2.get("chunk_embeddings_relpath") if isinstance(d2, dict) else None
                if isinstance(rel2, str) and rel2:
                    rel = rel2
                    tid = d2.get("transcript_id") if isinstance(d2, dict) else None
                    used_legacy = False

            if rel is None and isinstance(tchunks, dict):
                d = tchunks.get(src)
                rel3 = d.get("embeddings_relpath") if isinstance(d, dict) else None
                if isinstance(rel3, str) and rel3:
                    rel = rel3
                    tid = d.get("transcript_id") if isinstance(d, dict) else None
                    used_legacy = True

            if not isinstance(rel, str) or not rel:
                continue

            try:
                p = self._safe_join_artifacts_dir(self.artifacts_dir, rel)
            except Exception:
                return None, None, None, used_legacy, True, float("nan")
            if not p.exists():
                continue

            t0 = time.perf_counter()
            try:
                m = np.load(p)
                m = np.asarray(m, dtype=np.float32)
                if m.ndim == 1:
                    m = m.reshape(1, -1)
                load_ms = float((time.perf_counter() - t0) * 1000.0)
                return m, (str(tid) if tid is not None else None), str(src), used_legacy, False, load_ms
            except Exception:
                continue

        return None, None, None, False, False, float("nan")

    def _variance_across_chunks(self, chunks: np.ndarray) -> Dict[str, Any]:
        if chunks.size == 0:
            return {"l2_variance": None, "topk_variances": []}
        if chunks.ndim != 2:
            return {"l2_variance": None, "topk_variances": []}
        if chunks.shape[0] < max(self.min_chunks_required, 1):
            return {"l2_variance": None, "topk_variances": []}
        var_vec = np.var(chunks, axis=0, ddof=max(self.variance_ddof, 0))
        l2_variance = float(np.linalg.norm(var_vec))
        k = max(min(self.topk, var_vec.shape[0] if var_vec.ndim == 1 else var_vec.size), 0)
        topk = np.sort(var_vec)[-k:] if k > 0 else np.asarray([], dtype=np.float32)
        return {"l2_variance": l2_variance, "topk_variances": [float(x) for x in topk.tolist()]}

    def _topic_mix_entropy(self, doc: Any) -> Dict[str, Any]:
        """
        Entropy over topic_probs from semantics_topics_keyphrases (upstream softmax/temperature).
        """
        tp = getattr(doc, "tp_artifacts", None)
        topics = tp.get("topics") if isinstance(tp, dict) else None
        dist = topics.get("topk_distribution") if isinstance(topics, dict) else None
        probs = dist.get("topic_probs") if isinstance(dist, dict) else None
        if not isinstance(probs, list) or not probs:
            return {"present": False, "invalid": False, "entropy": None, "entropy_norm": None, "perplexity": None}
        try:
            arr = np.asarray([float(x) for x in probs], dtype=np.float32).reshape(-1)
        except Exception:
            return {"present": True, "invalid": True, "entropy": None, "entropy_norm": None, "perplexity": None}
        if arr.size <= 0 or (not np.isfinite(arr).all()) or np.any(arr < 0.0):
            return {"present": True, "invalid": True, "entropy": None, "entropy_norm": None, "perplexity": None}
        s = float(np.sum(arr))
        if s <= 0.0:
            return {"present": True, "invalid": True, "entropy": None, "entropy_norm": None, "perplexity": None}
        p = arr / s
        eps = 1e-12
        entropy = -float(np.sum(p * np.log(p + eps)))
        ksz = int(p.size)
        entropy_norm = float(entropy / float(np.log(ksz))) if ksz > 1 else float("nan")
        perplexity = float(np.exp(entropy)) if entropy == entropy else float("nan")
        return {"present": True, "invalid": False, "entropy": entropy, "entropy_norm": entropy_norm, "perplexity": perplexity}

    def _build_return(
        self,
        *,
        features_flat: Dict[str, Any],
        sys_after: Any,
        mem_before: int,
        mem_after: int,
        total_s: float,
    ) -> Dict[str, Any]:
        gpu_peak_mb = self._gpu_peak_mb(sys_after)
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
                    "ram_peak_mb": int(max(self._init_metrics.get("ram_peak_bytes", 0), mem_before, mem_after) / 1024 / 1024),
                    "gpu_peak_mb": int(gpu_peak_mb),
                },
            },
            "timings_s": {"total": round(total_s, 3)},
            "result": {"features_flat": self._pack_features_flat(features_flat)},
            "error": None,
        }

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        mem_before = process_memory_bytes()
        nan = float("nan")

        if not self.enabled:
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            d = self._empty_shell(enabled=False, disabled_by_policy=True, present=0.0)
            return self._build_return(
                features_flat=d,
                sys_after=sys_after,
                mem_before=mem_before,
                mem_after=mem_after,
                total_s=total_s,
            )

        d = self._empty_shell(enabled=True, disabled_by_policy=False, present=0.0)

        chunks, _transcript_id, source_used, used_legacy, unsafe_rel, load_ms_raw = self._load_chunks(doc)
        d["tp_embstats_used_legacy_key_flag"] = 1.0 if used_legacy else 0.0
        d["tp_embstats_unsafe_relpath_flag"] = 1.0 if unsafe_rel else 0.0

        variance_block: Dict[str, Any] = {"l2_variance": None, "topk_variances": []}
        comp_start: Optional[float] = None

        if chunks is None:
            if self.require_chunks:
                raise RuntimeError("EmbeddingStatsExtractor: required transcript chunk embeddings not found.")
        else:
            if chunks.ndim != 2 or int(chunks.shape[0]) <= 0 or int(chunks.shape[1]) <= 0:
                d["tp_embstats_dim_mismatch_flag"] = 1.0
                if self.require_chunks:
                    raise RuntimeError("EmbeddingStatsExtractor: invalid chunk embeddings matrix shape.")
            elif not np.isfinite(chunks).all():
                d["tp_embstats_nan_inf_flag"] = 1.0
                if self.require_chunks:
                    raise RuntimeError("EmbeddingStatsExtractor: chunk embeddings contain NaN/inf.")
            elif int(chunks.shape[0]) < max(self.min_chunks_required, 1):
                if self.require_chunks:
                    raise RuntimeError("EmbeddingStatsExtractor: insufficient chunks for statistics.")
            else:
                d["tp_embstats_n_chunks"] = float(int(chunks.shape[0]))
                d["tp_embstats_dim"] = float(int(chunks.shape[1]))
                comp_start = time.perf_counter()
                variance_block = self._variance_across_chunks(chunks)

        topic_block: Dict[str, Any]
        if self.compute_topic_entropy:
            if comp_start is None:
                comp_start = time.perf_counter()
            topic_block = self._topic_mix_entropy(doc)
        else:
            topic_block = {"present": False, "invalid": False, "entropy": None, "entropy_norm": None, "perplexity": None}

        raw_compute_ms = float((time.perf_counter() - comp_start) * 1000.0) if comp_start is not None else nan
        if load_ms_raw == load_ms_raw:
            d["tp_embstats_load_ms"] = float(round(load_ms_raw, 3))
        d["tp_embstats_compute_ms"] = float(round(raw_compute_ms, 3)) if raw_compute_ms == raw_compute_ms else nan

        if not self.emit_extra_metrics:
            d["tp_embstats_load_ms"] = nan
            d["tp_embstats_compute_ms"] = nan

        l2v = variance_block.get("l2_variance")
        topk_vars = variance_block.get("topk_variances") if isinstance(variance_block.get("topk_variances"), list) else []
        topic_entropy = topic_block.get("entropy") if isinstance(topic_block, dict) else None
        topic_entropy_norm = topic_block.get("entropy_norm") if isinstance(topic_block, dict) else None
        topic_perplexity = topic_block.get("perplexity") if isinstance(topic_block, dict) else None
        topic_present = bool(topic_block.get("present")) if isinstance(topic_block, dict) else False
        topic_invalid = bool(topic_block.get("invalid")) if isinstance(topic_block, dict) else False

        d["tp_embstats_present"] = 1.0 if (l2v is not None) else 0.0
        d["tp_embstats_l2_variance"] = float(l2v) if l2v is not None else nan
        d["tp_embstats_topic_probs_present"] = 1.0 if topic_present else 0.0
        d["tp_embstats_topic_probs_invalid_flag"] = 1.0 if topic_invalid else 0.0
        if self.require_topic_distribution and self.compute_topic_entropy and (not topic_present or topic_invalid):
            raise RuntimeError("EmbeddingStatsExtractor: required topic distribution missing/invalid.")
        d["tp_embstats_topic_entropy"] = float(topic_entropy) if topic_entropy is not None else nan
        d["tp_embstats_topic_entropy_norm"] = float(topic_entropy_norm) if topic_entropy_norm is not None else nan
        d["tp_embstats_topic_perplexity"] = float(topic_perplexity) if topic_perplexity is not None else nan
        d["tp_embstats_topic_entropy_present"] = 1.0 if topic_entropy is not None else 0.0

        eff = max(int(self.top_k_slots), 0)
        for i in range(SCHEMA_MAX_TOPVAR_SLOTS):
            if i < eff and isinstance(topk_vars, list) and len(topk_vars) > i:
                d[f"tp_embstats_topvar_{i + 1}"] = float(topk_vars[-(i + 1)])
            else:
                d[f"tp_embstats_topvar_{i + 1}"] = nan

        for src in SCHEMA_TRANSCRIPT_SOURCES:
            d[f"tp_embstats_source_used_{src}"] = 1.0 if (source_used == src) else 0.0

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0
        return self._build_return(
            features_flat=d,
            sys_after=sys_after,
            mem_before=mem_before,
            mem_after=mem_after,
            total_s=total_s,
        )
