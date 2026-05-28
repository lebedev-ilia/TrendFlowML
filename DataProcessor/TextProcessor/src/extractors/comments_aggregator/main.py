"""
CommentsAggregationExtractor — агрегирует эмбеддинги комментариев (уже посчитанные) по стратегиям:
- weighted mean (веса: likes × authority × recency, если заданы)
- component-wise median

Совместим со структурой проекта:
- читает список комментариев из VideoDocument
- читает матрицу эмбеддингов детерминированно через `doc.tp_artifacts["embeddings"]["comments"]["relpath"]`
- сохраняет агрегаты в per-run artifacts и возвращает только features_flat + relpath через `doc.tp_artifacts`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.schemas.models import VideoDocument

# Stable key order for features_flat (canonical → legacy rows). Must match comments_aggregator_output_v1.json.
_FEATURES_FLAT_KEYS: Tuple[str, ...] = (
    # tp_commentsagg_* (core + gating + weights + safety + extra timings)
    "tp_commentsagg_present",
    "tp_commentsagg_count",
    "tp_commentsagg_dim",
    "tp_commentsagg_mean_std",
    "tp_commentsagg_median_std",
    "tp_commentsagg_compute_mean_enabled",
    "tp_commentsagg_compute_median_enabled",
    "tp_commentsagg_compute_std_enabled",
    "tp_commentsagg_write_artifacts_enabled",
    "tp_commentsagg_require_comment_embeddings_enabled",
    "tp_commentsagg_artifact_mean_written",
    "tp_commentsagg_artifact_median_written",
    "tp_commentsagg_weights_applied",
    "tp_commentsagg_weights_mask_likes",
    "tp_commentsagg_weights_mask_authority",
    "tp_commentsagg_weights_mask_recency",
    "tp_commentsagg_weights_align_present",
    "tp_commentsagg_weights_align_shape_ok",
    "tp_commentsagg_dim_mismatch_flag",
    "tp_commentsagg_unsafe_relpath_flag",
    "tp_commentsagg_agg_mean_ms",
    "tp_commentsagg_agg_median_ms",
    # legacy tp_comments_agg_*
    "tp_comments_agg_present",
    "tp_comments_agg_count",
    "tp_comments_agg_dim",
    "tp_comments_agg_mean_std",
    "tp_comments_agg_median_std",
    "tp_comments_agg_weights_applied",
    "tp_comments_agg_weights_mask_likes",
    "tp_comments_agg_weights_mask_authority",
    "tp_comments_agg_weights_mask_recency",
    "tp_comments_agg_compute_std",
    "tp_comments_agg_compute_mean",
    "tp_comments_agg_compute_median",
    # legacy tp_cagg_*
    "tp_cagg_present",
    "tp_cagg_count",
    "tp_cagg_dim",
    "tp_cagg_mean_std",
    "tp_cagg_median_std",
)


class CommentsAggregationExtractor(BaseExtractor):
    VERSION = "1.3.0"

    def __init__(
        self,
        artifacts_dir: str | None = None,
        model_name: str = "intfloat/multilingual-e5-large",
        compute_mean: bool = True,
        compute_median: bool = True,
        compute_std: bool = False,
        write_artifacts: bool = True,
        require_comment_embeddings: bool = False,
        emit_extra_metrics: bool = False,
    ) -> None:
        from src.core.path_utils import default_artifacts_dir  # local import to avoid cycles

        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = str(model_name)
        self.compute_mean = bool(compute_mean)
        self.compute_median = bool(compute_median)
        self.compute_std = bool(compute_std)
        self.write_artifacts = bool(write_artifacts)
        self.require_comment_embeddings = bool(require_comment_embeddings)
        self.emit_extra_metrics = bool(emit_extra_metrics)

        init_sys_before = system_snapshot()
        init_mem_before = process_memory_bytes()
        try:
            from dp_models import get_global_model_manager  # type: ignore

            mm = get_global_model_manager()
            spec = mm.get_spec(model_name=self.model_name)
            _d, _p, _rt, _eng, weights_digest, _arts = mm.resolve(spec)
            self.weights_digest = str(weights_digest or "unknown")
            self.model_version = str(getattr(spec, "model_version", "unknown") or "unknown")
        except Exception as e:
            raise RuntimeError(
                f"CommentsAggregationExtractor: dp_models resolve failed for model_name={self.model_name!r}: {e}"
            ) from e
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
        """
        Join artifacts_dir with relpath and forbid path traversal.
        """
        base = base_dir.expanduser().resolve()
        cand = (base / relpath).resolve()
        if not (cand == base or base in cand.parents):
            raise RuntimeError("comments_aggregator: relpath escapes artifacts_dir")
        return cand

    @staticmethod
    def _aggregate_weighted_mean(
        embs: np.ndarray,
        weights: Optional[np.ndarray],
        *,
        compute_std: bool,
    ) -> Dict[str, Any]:
        n, _ = embs.shape
        w = np.ones((n,), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32).reshape(-1)
        if w.shape[0] != n:
            w = np.ones((n,), dtype=np.float32)
        w = np.maximum(w, 0.0)
        if float(w.sum()) <= 0:
            w = np.ones((n,), dtype=np.float32)
        w = w / (float(w.sum()) + 1e-9)
        vec = np.average(embs, axis=0, weights=w)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        # More interpretable than std over all entries: per-dimension std (across rows), then mean.
        std_val = float(np.mean(np.std(embs, axis=0))) if compute_std else float("nan")
        return {"embedding": vec.astype(np.float32), "count": int(n), "std": std_val}

    @staticmethod
    def _aggregate_median(embs: np.ndarray, *, compute_std: bool) -> Dict[str, Any]:
        vec = np.median(embs, axis=0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        std_val = float(np.mean(np.std(embs, axis=0))) if compute_std else float("nan")
        return {"embedding": vec.astype(np.float32), "count": int(embs.shape[0]), "std": std_val}

    @staticmethod
    def _load_selected_indices(doc: VideoDocument, artifacts_dir: Path) -> Optional[np.ndarray]:
        tp = getattr(doc, "tp_artifacts", None)
        if not isinstance(tp, dict):
            return None
        c = tp.get("comments")
        if not isinstance(c, dict):
            return None
        rel = c.get("selected_indices_relpath")
        if not isinstance(rel, str) or not rel:
            return None
        try:
            p = CommentsAggregationExtractor._safe_join_artifacts_dir(artifacts_dir, rel)
        except Exception:
            return None
        if not p.exists():
            return None
        try:
            idx = np.asarray(np.load(p), dtype=np.int32).reshape(-1)
            return idx
        except Exception:
            return None

    def _pack_features_flat(self, values: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in _FEATURES_FLAT_KEYS:
            if k not in values:
                raise KeyError(f"comments_aggregator: missing features_flat key {k!r}")
            out[k] = float(values[k])
        return out

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        # Load embeddings deterministically via in-memory registry filled by CommentsEmbedder
        tp = getattr(doc, "tp_artifacts", None)
        emb = tp.get("embeddings") if isinstance(tp, dict) else None
        rel = emb.get("comments", {}).get("relpath") if isinstance(emb, dict) and isinstance(emb.get("comments"), dict) else None
        embs = None
        unsafe_relpath_flag = 0.0
        if isinstance(rel, str) and rel:
            try:
                p = self._safe_join_artifacts_dir(self.artifacts_dir, rel)
            except Exception:
                unsafe_relpath_flag = 1.0
                p = None
            if p is not None and p.exists():
                try:
                    embs = np.asarray(np.load(p), dtype=np.float32)
                except Exception:
                    embs = None

        # Validate matrix shape (N, D)
        invalid = embs is None or (not isinstance(embs, np.ndarray)) or embs.ndim != 2 or int(embs.shape[0]) <= 0 or int(embs.shape[1]) <= 0
        if invalid:
            if self.require_comment_embeddings:
                raise RuntimeError("CommentsAggregationExtractor: required comment embeddings missing or invalid shape")
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            gpu_peak_mb = self._gpu_peak_mb(sys_after)
            nan = float("nan")
            dim_mismatch = 1.0 if isinstance(embs, np.ndarray) else 0.0
            features_flat = self._pack_features_flat(
                {
                    "tp_commentsagg_present": 0.0,
                    "tp_commentsagg_count": 0.0,
                    "tp_commentsagg_dim": nan,
                    "tp_commentsagg_mean_std": nan,
                    "tp_commentsagg_median_std": nan,
                    "tp_commentsagg_compute_mean_enabled": float(self.compute_mean),
                    "tp_commentsagg_compute_median_enabled": float(self.compute_median),
                    "tp_commentsagg_compute_std_enabled": float(self.compute_std),
                    "tp_commentsagg_write_artifacts_enabled": float(self.write_artifacts),
                    "tp_commentsagg_require_comment_embeddings_enabled": float(self.require_comment_embeddings),
                    "tp_commentsagg_artifact_mean_written": 0.0,
                    "tp_commentsagg_artifact_median_written": 0.0,
                    "tp_commentsagg_weights_applied": 0.0,
                    "tp_commentsagg_weights_mask_likes": 0.0,
                    "tp_commentsagg_weights_mask_authority": 0.0,
                    "tp_commentsagg_weights_mask_recency": 0.0,
                    "tp_commentsagg_weights_align_present": 0.0,
                    "tp_commentsagg_weights_align_shape_ok": 0.0,
                    "tp_commentsagg_dim_mismatch_flag": float(dim_mismatch),
                    "tp_commentsagg_unsafe_relpath_flag": float(unsafe_relpath_flag),
                    "tp_commentsagg_agg_mean_ms": nan,
                    "tp_commentsagg_agg_median_ms": nan,
                    "tp_comments_agg_present": 0.0,
                    "tp_comments_agg_count": 0.0,
                    "tp_comments_agg_dim": nan,
                    "tp_comments_agg_mean_std": nan,
                    "tp_comments_agg_median_std": nan,
                    "tp_comments_agg_weights_applied": 0.0,
                    "tp_comments_agg_weights_mask_likes": 0.0,
                    "tp_comments_agg_weights_mask_authority": 0.0,
                    "tp_comments_agg_weights_mask_recency": 0.0,
                    "tp_comments_agg_compute_std": float(self.compute_std),
                    "tp_comments_agg_compute_mean": float(self.compute_mean),
                    "tp_comments_agg_compute_median": float(self.compute_median),
                    "tp_cagg_present": 0.0,
                    "tp_cagg_count": 0.0,
                    "tp_cagg_dim": nan,
                    "tp_cagg_mean_std": nan,
                    "tp_cagg_median_std": nan,
                }
            )
            return {
                "device": "cpu",
                "version": self.VERSION,
                "model_name": self.model_name,
                "model_version": str(self.model_version),
                "weights_digest": self.weights_digest,
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

        # Optional weights — only if we can align them via selected_indices from CommentsEmbedder.
        likes = getattr(doc, "comments_likes", None)
        authority = getattr(doc, "comments_authority", None)
        recency = getattr(doc, "comments_recency", None)
        idx = self._load_selected_indices(doc, self.artifacts_dir)
        weights_applied = 0.0
        weights_vec: Optional[np.ndarray] = None
        weights_mask_likes = 0.0
        weights_mask_authority = 0.0
        weights_mask_recency = 0.0
        weights_align_present = float(idx is not None and isinstance(idx, np.ndarray) and idx.size > 0)
        weights_align_shape_ok = 0.0

        def _arr_ok(a: Any) -> bool:
            return isinstance(a, list) and all(isinstance(x, (int, float)) for x in a)

        if idx is not None and idx.size == int(embs.shape[0]) and idx.size > 0:
            weights_align_shape_ok = 1.0
            w = np.ones((idx.size,), dtype=np.float32)
            if _arr_ok(likes):
                arr = np.asarray(likes, dtype=np.float32)
                if int(arr.size) > int(np.max(idx)):
                    w *= np.clip(arr[idx], 0.1, None)
                    weights_mask_likes = 1.0
            if _arr_ok(authority):
                arr = np.asarray(authority, dtype=np.float32)
                if int(arr.size) > int(np.max(idx)):
                    w *= np.clip(arr[idx], 0.1, None)
                    weights_mask_authority = 1.0
            if _arr_ok(recency):
                arr = np.asarray(recency, dtype=np.float32)
                if int(arr.size) > int(np.max(idx)):
                    w *= np.clip(arr[idx], 0.1, None)
                    weights_mask_recency = 1.0
            if (weights_mask_likes + weights_mask_authority + weights_mask_recency) > 0:
                weights_applied = 1.0
                weights_vec = w

        dim = int(embs.shape[1])

        mean_res = {"embedding": None, "count": int(embs.shape[0]), "std": float("nan")}
        med_res = {"embedding": None, "count": int(embs.shape[0]), "std": float("nan")}
        mean_s = float("nan")
        median_s = float("nan")

        if self.compute_mean:
            t_agg0 = time.perf_counter()
            mean_res = self._aggregate_weighted_mean(embs, weights_vec, compute_std=self.compute_std)
            mean_s = time.perf_counter() - t_agg0

        if self.compute_median:
            t_agg1 = time.perf_counter()
            med_res = self._aggregate_median(embs, compute_std=self.compute_std)
            median_s = time.perf_counter() - t_agg1

        # save artifacts (optional)
        mean_path = self.artifacts_dir / "comments_agg_mean.npy"
        med_path = self.artifacts_dir / "comments_agg_median.npy"
        mean_written = False
        median_written = False
        if self.write_artifacts:
            if self.compute_mean and isinstance(mean_res.get("embedding"), np.ndarray):
                tmp_mean = mean_path.with_suffix(".tmp.npy")
                np.save(tmp_mean, mean_res["embedding"])
                tmp_mean.replace(mean_path)
                mean_written = True
            if self.compute_median and isinstance(med_res.get("embedding"), np.ndarray):
                tmp_median = med_path.with_suffix(".tmp.npy")
                np.save(tmp_median, med_res["embedding"])
                tmp_median.replace(med_path)
                median_written = True

        # In-memory registry for downstream (no absolute paths in result/NPZ).
        try:
            tp2 = getattr(doc, "tp_artifacts", None)
            if not isinstance(tp2, dict):
                tp2 = {}
                setattr(doc, "tp_artifacts", tp2)
            tp2.setdefault("comments", {})
            tp2.setdefault("embeddings", {})  # legacy alias
            if mean_written and mean_path.exists():
                tp2["comments"]["agg_mean_relpath"] = mean_path.name
                tp2["embeddings"]["comments_agg_mean"] = {"relpath": mean_path.name, "kind": "vector", "dim": int(mean_res["embedding"].size)}
            if median_written and med_path.exists():
                tp2["comments"]["agg_median_relpath"] = med_path.name
                tp2["embeddings"]["comments_agg_median"] = {"relpath": med_path.name, "kind": "vector", "dim": int(med_res["embedding"].size)}
        except Exception:
            pass

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        present = float(
            bool(
                (self.compute_mean and isinstance(mean_res.get("embedding"), np.ndarray))
                or (self.compute_median and isinstance(med_res.get("embedding"), np.ndarray))
            )
        )
        nan = float("nan")
        mean_ms = nan
        median_ms = nan
        if self.emit_extra_metrics:
            mean_ms = float(mean_s * 1000.0) if self.compute_mean and mean_s == mean_s else nan
            median_ms = float(median_s * 1000.0) if self.compute_median and median_s == median_s else nan

        features_flat = self._pack_features_flat(
            {
                "tp_commentsagg_present": float(present),
                "tp_commentsagg_count": float(int(mean_res["count"])),
                "tp_commentsagg_dim": float(int(dim)),
                "tp_commentsagg_mean_std": float(mean_res["std"]) if self.compute_mean else nan,
                "tp_commentsagg_median_std": float(med_res["std"]) if self.compute_median else nan,
                "tp_commentsagg_compute_mean_enabled": float(self.compute_mean),
                "tp_commentsagg_compute_median_enabled": float(self.compute_median),
                "tp_commentsagg_compute_std_enabled": float(self.compute_std),
                "tp_commentsagg_write_artifacts_enabled": float(self.write_artifacts),
                "tp_commentsagg_require_comment_embeddings_enabled": float(self.require_comment_embeddings),
                "tp_commentsagg_artifact_mean_written": float(bool(mean_written)),
                "tp_commentsagg_artifact_median_written": float(bool(median_written)),
                "tp_commentsagg_weights_applied": float(weights_applied),
                "tp_commentsagg_weights_mask_likes": float(weights_mask_likes),
                "tp_commentsagg_weights_mask_authority": float(weights_mask_authority),
                "tp_commentsagg_weights_mask_recency": float(weights_mask_recency),
                "tp_commentsagg_weights_align_present": float(weights_align_present),
                "tp_commentsagg_weights_align_shape_ok": float(weights_align_shape_ok),
                "tp_commentsagg_dim_mismatch_flag": 0.0,
                "tp_commentsagg_unsafe_relpath_flag": float(unsafe_relpath_flag),
                "tp_commentsagg_agg_mean_ms": mean_ms,
                "tp_commentsagg_agg_median_ms": median_ms,
                "tp_comments_agg_present": float(present),
                "tp_comments_agg_count": float(int(mean_res["count"])),
                "tp_comments_agg_dim": float(int(dim)),
                "tp_comments_agg_mean_std": float(mean_res["std"]) if self.compute_mean else nan,
                "tp_comments_agg_median_std": float(med_res["std"]) if self.compute_median else nan,
                "tp_comments_agg_weights_applied": float(weights_applied),
                "tp_comments_agg_weights_mask_likes": float(weights_mask_likes),
                "tp_comments_agg_weights_mask_authority": float(weights_mask_authority),
                "tp_comments_agg_weights_mask_recency": float(weights_mask_recency),
                "tp_comments_agg_compute_std": float(self.compute_std),
                "tp_comments_agg_compute_mean": float(self.compute_mean),
                "tp_comments_agg_compute_median": float(self.compute_median),
                "tp_cagg_present": float(present),
                "tp_cagg_count": float(int(mean_res["count"])),
                "tp_cagg_dim": float(int(dim)),
                "tp_cagg_mean_std": float(mean_res["std"]) if self.compute_mean else nan,
                "tp_cagg_median_std": float(med_res["std"]) if self.compute_median else nan,
            }
        )

        gpu_peak_mb = self._gpu_peak_mb(sys_after)
        return {
            "device": "cpu",
            "version": self.VERSION,
            "model_name": self.model_name,
            "model_version": str(self.model_version),
            "weights_digest": self.weights_digest,
            "system": {
                "pre_init": self._init_metrics.get("pre_init"),
                "post_init": self._init_metrics.get("post_init"),
                "post_process": sys_after,
                "peaks": {
                    "ram_peak_mb": int(max(self._init_metrics.get("ram_peak_bytes", 0), mem_before, mem_after) / 1024 / 1024),
                    "gpu_peak_mb": int(gpu_peak_mb),
                },
            },
            "timings_s": {
                "mean": round(float(mean_s), 3) if mean_s == mean_s else float("nan"),
                "median": round(float(median_s), 3) if median_s == median_s else float("nan"),
                "total": round(total_s, 3),
            },
            "result": {"features_flat": features_flat},
            "error": None,
        }
