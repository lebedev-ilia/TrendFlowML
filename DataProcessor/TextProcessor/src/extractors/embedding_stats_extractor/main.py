from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes


class EmbeddingStatsExtractor(BaseExtractor):
    """
    18. embedding_variance_across_chunks: L2-норма дисперсии + top-k компонентных дисперсий
    19. embedding_topic_mix_entropy: энтропия top-K topic distribution (если доступно)
    """

    VERSION = "1.1.0"

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

        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        # Keep a cache root available for future, but do not read arbitrary JSON artifacts.
        # (ARTIFACTS_AND_SCHEMAS.md: arbitrary JSON artifacts are not source-of-truth.)
        self.cache_dir = default_cache_dir() / "transcript_embed"

        # priority policy
        if transcript_source_priority is None:
            self.transcript_source_priority = ["whisper", "youtube_auto"]
        elif isinstance(transcript_source_priority, str):
            self.transcript_source_priority = [s.strip() for s in transcript_source_priority.split(",") if s.strip()]
        else:
            self.transcript_source_priority = [str(s).strip() for s in transcript_source_priority if str(s).strip()]

        self.top_k_slots = int(top_k_slots)
        self.topk = int(topk)
        self.min_chunks_required = int(min_chunks_required)
        self.variance_ddof = int(variance_ddof)
        self.enabled = bool(enabled)
        self.require_chunks = bool(require_chunks)
        self.compute_topic_entropy = bool(compute_topic_entropy)
        self.require_topic_distribution = bool(require_topic_distribution)
        self.emit_extra_metrics = bool(emit_extra_metrics)

    @staticmethod
    def _safe_join_artifacts_dir(base_dir: Path, relpath: str) -> Path:
        base = base_dir.expanduser().resolve()
        cand = (base / relpath).resolve()
        if not (cand == base or base in cand.parents):
            raise RuntimeError("embedding_stats_extractor: relpath escapes artifacts_dir")
        return cand

    @staticmethod
    def _stable_template(
        *,
        enabled: bool,
        require_chunks: bool,
        compute_topic_entropy: bool,
        require_topic_distribution: bool,
        min_chunks_required: int,
        top_k_slots: int,
        topk: int,
        transcript_source_priority: List[str],
        variance_ddof: int,
    ) -> Dict[str, float]:
        features_flat: Dict[str, float] = {
            "tp_embstats_present": 0.0,
            "tp_embstats_enabled": float(bool(enabled)),
            "tp_embstats_disabled_by_policy": 0.0,
            "tp_embstats_require_chunks_enabled": float(bool(require_chunks)),
            "tp_embstats_compute_topic_entropy_enabled": float(bool(compute_topic_entropy)),
            "tp_embstats_require_topic_distribution_enabled": float(bool(require_topic_distribution)),
            "tp_embstats_min_chunks_required": float(int(min_chunks_required)),
            "tp_embstats_topk": float(int(topk)),
            "tp_embstats_top_k_slots": float(int(top_k_slots)),
            "tp_embstats_variance_ddof": float(int(variance_ddof)),
            "tp_embstats_n_chunks": float("nan"),
            "tp_embstats_dim": float("nan"),
            "tp_embstats_l2_variance": float("nan"),
            "tp_embstats_topic_entropy": float("nan"),
            "tp_embstats_topic_entropy_norm": float("nan"),
            "tp_embstats_topic_perplexity": float("nan"),
            "tp_embstats_topic_entropy_present": 0.0,
            "tp_embstats_topic_probs_present": 0.0,
            "tp_embstats_topic_probs_invalid_flag": 0.0,
            "tp_embstats_used_legacy_key_flag": 0.0,
            "tp_embstats_unsafe_relpath_flag": 0.0,
            "tp_embstats_dim_mismatch_flag": 0.0,
            "tp_embstats_nan_inf_flag": 0.0,
            "tp_embstats_load_ms": float("nan"),
            "tp_embstats_compute_ms": float("nan"),
        }
        # fixed slots for schema stability
        for i in range(max(int(top_k_slots), 0)):
            features_flat[f"tp_embstats_topvar_{i+1}"] = float("nan")
        # source tracking (stable)
        for src in transcript_source_priority:
            features_flat[f"tp_embstats_source_used_{src}"] = 0.0
        return features_flat

    def _load_chunks(self, doc: Any) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str], bool, bool, float]:
        """
        Returns: (chunks, transcript_id, source_used, used_legacy_key, unsafe_relpath_flag, load_ms)
        """
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
            # canonical: tp_artifacts["transcripts"][src]["chunk_embeddings_relpath"]
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

            # legacy fallback
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

            import time
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
        # ddof=0 by default (stable for small N); configurable via `variance_ddof`.
        var_vec = np.var(chunks, axis=0, ddof=max(self.variance_ddof, 0))
        l2_variance = float(np.linalg.norm(var_vec))
        k = max(min(self.topk, var_vec.shape[0] if var_vec.ndim == 1 else var_vec.size), 0)
        topk = np.sort(var_vec)[-k:] if k > 0 else np.asarray([], dtype=np.float32)
        return {"l2_variance": l2_variance, "topk_variances": [float(x) for x in topk.tolist()]}

    def _topic_mix_entropy(self, doc: Any) -> Dict[str, Any]:
        """
        Entropy over top-K topic distribution produced by `semantics_topics_keyphrases`.
        Expected location (in-memory only):
          doc.tp_artifacts["topics"]["topk_distribution"]["topic_probs"] -> List[float]
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
        k = int(p.size)
        entropy_norm = float(entropy / float(np.log(k))) if k > 1 else float("nan")
        perplexity = float(np.exp(entropy)) if entropy == entropy else float("nan")
        return {"present": True, "invalid": False, "entropy": entropy, "entropy_norm": entropy_norm, "perplexity": perplexity}

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        features_flat = self._stable_template(
            enabled=self.enabled,
            require_chunks=self.require_chunks,
            compute_topic_entropy=self.compute_topic_entropy,
            require_topic_distribution=self.require_topic_distribution,
            min_chunks_required=self.min_chunks_required,
            top_k_slots=self.top_k_slots,
            topk=self.topk,
            transcript_source_priority=self.transcript_source_priority,
            variance_ddof=self.variance_ddof,
        )

        if not self.enabled:
            features_flat["tp_embstats_disabled_by_policy"] = 1.0
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

        chunks, transcript_id, source_used, used_legacy, unsafe_rel, load_ms = self._load_chunks(doc)
        features_flat["tp_embstats_used_legacy_key_flag"] = 1.0 if used_legacy else 0.0
        features_flat["tp_embstats_unsafe_relpath_flag"] = 1.0 if unsafe_rel else 0.0
        if load_ms == load_ms:
            features_flat["tp_embstats_load_ms"] = float(round(load_ms, 3))

        variance_block: Dict[str, Any]
        topic_block: Dict[str, Any]

        if chunks is None:
            if self.require_chunks:
                raise RuntimeError("EmbeddingStatsExtractor: required transcript chunk embeddings not found.")
            variance_block = {"l2_variance": None, "topk_variances": []}
            topic_block = {"present": False, "invalid": False, "entropy": None, "entropy_norm": None, "perplexity": None}
        else:
            if chunks.ndim != 2 or int(chunks.shape[0]) <= 0 or int(chunks.shape[1]) <= 0:
                features_flat["tp_embstats_dim_mismatch_flag"] = 1.0
                if self.require_chunks:
                    raise RuntimeError("EmbeddingStatsExtractor: invalid chunk embeddings matrix shape.")
                variance_block = {"l2_variance": None, "topk_variances": []}
                topic_block = {"present": False, "invalid": False, "entropy": None, "entropy_norm": None, "perplexity": None}
            elif not np.isfinite(chunks).all():
                features_flat["tp_embstats_nan_inf_flag"] = 1.0
                if self.require_chunks:
                    raise RuntimeError("EmbeddingStatsExtractor: chunk embeddings contain NaN/inf.")
                variance_block = {"l2_variance": None, "topk_variances": []}
                topic_block = {"present": False, "invalid": False, "entropy": None, "entropy_norm": None, "perplexity": None}
            elif int(chunks.shape[0]) < max(self.min_chunks_required, 1):
                if self.require_chunks:
                    raise RuntimeError("EmbeddingStatsExtractor: insufficient chunks for statistics.")
                variance_block = {"l2_variance": None, "topk_variances": []}
                topic_block = {"present": False, "invalid": False, "entropy": None, "entropy_norm": None, "perplexity": None}
            else:
                features_flat["tp_embstats_n_chunks"] = float(int(chunks.shape[0]))
                features_flat["tp_embstats_dim"] = float(int(chunks.shape[1]))
                t_comp0 = time.perf_counter()
                variance_block = self._variance_across_chunks(chunks)
                if self.compute_topic_entropy:
                    topic_block = self._topic_mix_entropy(doc)
                else:
                    topic_block = {"present": False, "invalid": False, "entropy": None, "entropy_norm": None, "perplexity": None}
                features_flat["tp_embstats_compute_ms"] = float(round((time.perf_counter() - t_comp0) * 1000.0, 3))

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        # flatten to numeric scalars for NPZ export
        l2v = variance_block.get("l2_variance")
        topk_vars = variance_block.get("topk_variances") if isinstance(variance_block.get("topk_variances"), list) else []
        topic_entropy = topic_block.get("entropy") if isinstance(topic_block, dict) else None
        topic_entropy_norm = topic_block.get("entropy_norm") if isinstance(topic_block, dict) else None
        topic_perplexity = topic_block.get("perplexity") if isinstance(topic_block, dict) else None
        topic_present = bool(topic_block.get("present")) if isinstance(topic_block, dict) else False
        topic_invalid = bool(topic_block.get("invalid")) if isinstance(topic_block, dict) else False

        features_flat["tp_embstats_present"] = 1.0 if (l2v is not None) else 0.0
        features_flat["tp_embstats_l2_variance"] = float(l2v) if l2v is not None else float("nan")
        features_flat["tp_embstats_topic_probs_present"] = 1.0 if topic_present else 0.0
        features_flat["tp_embstats_topic_probs_invalid_flag"] = 1.0 if topic_invalid else 0.0
        if self.require_topic_distribution and self.compute_topic_entropy and (not topic_present or topic_invalid):
            raise RuntimeError("EmbeddingStatsExtractor: required topic distribution missing/invalid.")
        features_flat["tp_embstats_topic_entropy"] = float(topic_entropy) if topic_entropy is not None else float("nan")
        features_flat["tp_embstats_topic_entropy_norm"] = float(topic_entropy_norm) if topic_entropy_norm is not None else float("nan")
        features_flat["tp_embstats_topic_perplexity"] = float(topic_perplexity) if topic_perplexity is not None else float("nan")
        features_flat["tp_embstats_topic_entropy_present"] = 1.0 if topic_entropy is not None else 0.0

        # fixed slots for schema stability
        slots = max(int(self.top_k_slots), 0)
        for i in range(slots):
            v = float(topk_vars[-(i + 1)]) if (isinstance(topk_vars, list) and len(topk_vars) > i) else float("nan")
            features_flat[f"tp_embstats_topvar_{i+1}"] = v

        # source tracking
        for src in self.transcript_source_priority:
            features_flat[f"tp_embstats_source_used_{src}"] = 1.0 if (source_used == src) else 0.0

        # Extra metrics are already part of stable template; keep emit_extra_metrics for future additions.
        if not self.emit_extra_metrics:
            # No-op: keep keys stable.
            pass

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


