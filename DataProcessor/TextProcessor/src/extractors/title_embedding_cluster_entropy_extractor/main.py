from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dp_models.manager import ModelManager
from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.path_utils import default_artifacts_dir

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore


class TitleEmbeddingClusterEntropyExtractor(BaseExtractor):
    """
    A-policy: title embedding cluster entropy.

    - Strict model/assets loading via dp_models (fail-fast in __init__)
    - Deterministic input via doc.tp_artifacts["embeddings"]["title"]["relpath"]
    - Valid empty semantics (NaNs + *_present flags; no fake vectors)
    - Uses the shared semantic cluster taxonomy (semantic_clusters_v1) by default
    """

    VERSION = "1.2.0"

    def __init__(
        self,
        artifacts_dir: str | None = None,
        clusters_spec_name: str = "semantic_clusters_v1",
        top_k_slots: int = 5,
        temperature: float = 0.1,
        export_topk_distribution: bool = False,
        require_title_embedding: bool = False,
        require_faiss: bool = False,
        use_faiss: bool = True,
        emit_extra_metrics: bool = False,
        # Deprecated legacy arg (kept for config compat; bypassing dp_models is forbidden)
        clusters_path: str | None = None,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.clusters_spec_name = str(clusters_spec_name or "").strip()
        if not self.clusters_spec_name:
            raise RuntimeError("title_embedding_cluster_entropy_extractor: clusters_spec_name is required")

        self.top_k_slots = int(top_k_slots)
        if self.top_k_slots <= 0:
            raise RuntimeError("title_embedding_cluster_entropy_extractor: top_k_slots must be > 0")

        self.temperature = float(temperature)
        self.export_topk_distribution = bool(export_topk_distribution)
        self.require_title_embedding = bool(require_title_embedding)
        self.use_faiss = bool(use_faiss)
        self.require_faiss = bool(require_faiss)
        self.emit_extra_metrics = bool(emit_extra_metrics)

        if clusters_path is not None:
            raise RuntimeError(
                "title_embedding_cluster_entropy_extractor: legacy clusters_path is forbidden under A-policy. "
                "Use dp_models spec via clusters_spec_name."
            )

        self._pca: np.ndarray
        self._centroids: np.ndarray
        self._faiss_index: Any = None
        self._clusters_spec_version: str = "unknown"
        self._clusters_weights_digest: str = "unknown"
        self._cluster_db_version: str = "unknown"
        self._n_clusters: int = 0
        self._orig_dim: int = 0
        self._reduced_dim: int = 0

        self._load_assets_from_dp_models()

    def _load_assets_from_dp_models(self) -> None:
        try:
            from dp_models import get_global_model_manager  # type: ignore

            mm = get_global_model_manager()
        except Exception:
            mm = ModelManager()

        spec = mm.get_spec(model_name=self.clusters_spec_name)
        _d, _p, _rt, _eng, weights_digest, resolved = mm.resolve(spec)
        self._clusters_weights_digest = str(weights_digest or "unknown")
        self._clusters_spec_version = str(getattr(spec, "model_version", None) or "unknown")

        rp = spec.runtime_params if isinstance(spec.runtime_params, dict) else {}
        pca_rel = rp.get("pca_relpath")
        cent_rel = rp.get("centroids_relpath")
        self._cluster_db_version = str(rp.get("cluster_db_version") or "unknown")

        if not isinstance(pca_rel, str) or not pca_rel:
            raise RuntimeError(f"title_embedding_cluster_entropy_extractor: spec missing runtime_params.pca_relpath: {self.clusters_spec_name}")
        if not isinstance(cent_rel, str) or not cent_rel:
            raise RuntimeError(
                f"title_embedding_cluster_entropy_extractor: spec missing runtime_params.centroids_relpath: {self.clusters_spec_name}"
            )

        pca_path = resolved.get(pca_rel) or resolved.get(str(pca_rel))
        cent_path = resolved.get(cent_rel) or resolved.get(str(cent_rel))
        if not pca_path:
            raise RuntimeError(f"title_embedding_cluster_entropy_extractor: PCA artifact not resolved: {pca_rel}")
        if not cent_path:
            raise RuntimeError(f"title_embedding_cluster_entropy_extractor: centroids artifact not resolved: {cent_rel}")

        pca = np.load(pca_path)
        pca = np.asarray(pca, dtype=np.float32)
        if pca.ndim != 2:
            raise RuntimeError("title_embedding_cluster_entropy_extractor: pca.npy must be 2D (orig_dim, reduced_dim)")

        centroids = np.load(cent_path)
        centroids = np.asarray(centroids, dtype=np.float32)
        if centroids.ndim != 2:
            raise RuntimeError("title_embedding_cluster_entropy_extractor: centroids.npy must be 2D (n_clusters, reduced_dim)")
        if int(pca.shape[1]) != int(centroids.shape[1]):
            raise RuntimeError(
                f"title_embedding_cluster_entropy_extractor: reduced_dim mismatch between PCA and centroids: "
                f"pca_reduced_dim={int(pca.shape[1])} centroids_reduced_dim={int(centroids.shape[1])}"
            )
        if int(centroids.shape[0]) <= 0:
            raise RuntimeError("title_embedding_cluster_entropy_extractor: centroids.npy must contain at least 1 cluster")

        # L2-normalize centroids for cosine / inner product backend.
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        centroids = centroids / norms

        self._pca = pca
        self._centroids = centroids
        self._n_clusters = int(centroids.shape[0])
        self._orig_dim = int(pca.shape[0])
        self._reduced_dim = int(pca.shape[1])

        if self.use_faiss:
            if faiss is None:
                if self.require_faiss:
                    raise RuntimeError("title_embedding_cluster_entropy_extractor: faiss is required but not available")
            else:
                index = faiss.IndexFlatIP(int(self._reduced_dim))
                index.add(self._centroids.astype(np.float32))
                self._faiss_index = index

    @staticmethod
    def _l2n(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n > 0:
            return v / n
        return v

    @staticmethod
    def _entropy(p: np.ndarray) -> float:
        return float(-np.sum(p * np.log(p + 1e-9)))

    @staticmethod
    def _safe_join_artifacts_dir(base_dir: Path, relpath: str) -> Path:
        base = base_dir.expanduser().resolve()
        cand = (base / relpath).resolve()
        if not (cand == base or base in cand.parents):
            raise RuntimeError("title_embedding_cluster_entropy_extractor: embedding relpath escapes artifacts_dir")
        return cand

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        z = x / max(self.temperature, 1e-6)
        z = z - np.max(z)
        e = np.exp(z)
        d = np.sum(e) + 1e-9
        return (e / d).astype(np.float32)

    def _topk(self, sims: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (topk_idx, topk_scores) sorted by descending score.
        """
        k = min(int(k), int(sims.shape[0]))
        if k <= 0:
            return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
        # argpartition for speed + sort
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        scores = sims[idx].astype(np.float32, copy=False)
        order = np.argsort(-scores)
        return idx[order].astype(np.int32, copy=False), scores[order].astype(np.float32, copy=False)

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        features_flat: Dict[str, float] = {
            "tp_titleclent_present": 0.0,
            "tp_titleclent_entropy_raw": float("nan"),
            "tp_titleclent_entropy_norm": float("nan"),
            "tp_titleclent_perplexity": float("nan"),
            "tp_titleclent_top_k_slots": float(int(self.top_k_slots)),
            "tp_titleclent_top_k_used": float("nan"),
            "tp_titleclent_temperature": float(self.temperature),
            "tp_titleclent_dim_mismatch_flag": 0.0,
            "tp_titleclent_title_present": 0.0,
            "tp_titleclent_backend_faiss": 1.0 if (self._faiss_index is not None) else 0.0,
        }

        # Models must be loaded in __init__ (fail-fast). Guard anyway.
        if self._pca is None or self._centroids is None:  # type: ignore[truthy-bool]
            raise RuntimeError("title_embedding_cluster_entropy_extractor: models not loaded (this should have failed in __init__)")

        tp = getattr(doc, "tp_artifacts", None)
        emb = tp.get("embeddings") if isinstance(tp, dict) else None
        title_rel = emb.get("title", {}).get("relpath") if isinstance(emb, dict) and isinstance(emb.get("title"), dict) else None
        if isinstance(title_rel, str) and title_rel:
            features_flat["tp_titleclent_title_present"] = 1.0
        else:
            if self.require_title_embedding:
                raise RuntimeError("title_embedding_cluster_entropy_extractor: missing title embedding relpath in doc.tp_artifacts.embeddings.title")
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
                "result": {
                    "features_flat": features_flat,
                    "title_cluster_entropy_meta": {
                        "clusters_spec_name": self.clusters_spec_name,
                        "clusters_spec_version": self._clusters_spec_version,
                        "clusters_weights_digest": self._clusters_weights_digest,
                        "cluster_db_version": self._cluster_db_version,
                        "backend": "faiss_ip" if (self._faiss_index is not None) else "numpy_cosine",
                    },
                },
                "error": None,
            }

        p = self._safe_join_artifacts_dir(self.artifacts_dir, str(title_rel))
        if not p.exists():
            if self.require_title_embedding:
                raise RuntimeError("title_embedding_cluster_entropy_extractor: title embedding file not found in per-run artifacts")
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
                "result": {
                    "features_flat": features_flat,
                    "title_cluster_entropy_meta": {
                        "clusters_spec_name": self.clusters_spec_name,
                        "clusters_spec_version": self._clusters_spec_version,
                        "clusters_weights_digest": self._clusters_weights_digest,
                        "cluster_db_version": self._cluster_db_version,
                        "backend": "faiss_ip" if (self._faiss_index is not None) else "numpy_cosine",
                    },
                },
                "error": None,
            }

        title = np.load(p).astype(np.float32).reshape(-1)
        if int(title.shape[0]) != int(self._orig_dim):
            features_flat["tp_titleclent_dim_mismatch_flag"] = 1.0
            if self.require_title_embedding:
                raise RuntimeError(
                    f"title_embedding_cluster_entropy_extractor: embedding dim mismatch: embedding_dim={int(title.shape[0])} "
                    f"model_orig_dim={int(self._orig_dim)} (clusters_spec_name={self.clusters_spec_name})"
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
                "result": {
                    "features_flat": features_flat,
                    "title_cluster_entropy_meta": {
                        "clusters_spec_name": self.clusters_spec_name,
                        "clusters_spec_version": self._clusters_spec_version,
                        "clusters_weights_digest": self._clusters_weights_digest,
                        "cluster_db_version": self._cluster_db_version,
                        "backend": "faiss_ip" if (self._faiss_index is not None) else "numpy_cosine",
                    },
                },
                "error": None,
            }

        # Project to shared reduced space and normalize
        t_compute0 = time.perf_counter()
        reduced = title @ self._pca
        reduced = self._l2n(np.asarray(reduced, dtype=np.float32).reshape(-1)).reshape(1, -1)

        # cosine similarities: reduced(title) vs centroids (already normalized)
        want_k = min(int(self.top_k_slots), int(self._n_clusters))
        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(reduced.astype(np.float32), want_k)
            topk_idx = idx[0].astype(np.int32, copy=False)
            topk_scores = scores[0].astype(np.float32, copy=False)
            margin = float("nan")
            if self.emit_extra_metrics and want_k >= 2:
                margin = float(topk_scores[0] - topk_scores[1])
        else:
            sims = (reduced @ self._centroids.T).reshape(-1)
            topk_idx, topk_scores = self._topk(sims, want_k)
            margin = float("nan")
            if self.emit_extra_metrics and topk_scores.size >= 2:
                margin = float(topk_scores[0] - topk_scores[1])

        probs = self._softmax(topk_scores)
        ent = self._entropy(probs)
        denom = math.log(int(want_k)) if int(want_k) > 1 else 0.0
        ent_norm = float(ent / max(denom, 1e-12))
        perp = float(math.exp(ent))
        t_compute = time.perf_counter() - t_compute0

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        features_flat["tp_titleclent_present"] = 1.0
        features_flat["tp_titleclent_entropy_raw"] = float(ent)
        features_flat["tp_titleclent_entropy_norm"] = float(ent_norm)
        features_flat["tp_titleclent_perplexity"] = float(perp)
        features_flat["tp_titleclent_top_k_used"] = float(int(want_k))
        features_flat["tp_titleclent_distinct_clusters_topk"] = float(int(len(np.unique(topk_idx))))
        if self.emit_extra_metrics:
            features_flat["tp_titleclent_n_clusters"] = float(int(self._n_clusters))
            features_flat["tp_titleclent_model_orig_dim"] = float(int(self._orig_dim))
            features_flat["tp_titleclent_model_reduced_dim"] = float(int(self._reduced_dim))
            features_flat["tp_titleclent_margin_top2"] = float(margin)
            features_flat["tp_titleclent_compute_ms"] = float(round(t_compute * 1000.0, 3))

        out_payload: Dict[str, Any] = {
            "clusters_spec_name": self.clusters_spec_name,
            "clusters_spec_version": self._clusters_spec_version,
            "clusters_weights_digest": self._clusters_weights_digest,
            "cluster_db_version": self._cluster_db_version,
            "backend": "faiss_ip" if (self._faiss_index is not None) else "numpy_cosine",
        }

        if self.export_topk_distribution:
            # Fixed-slot export for UI/debug without raw text: ids + probs + scores
            ids: List[int] = [int(x) for x in topk_idx.tolist()]
            pr: List[float] = [float(x) for x in probs.tolist()]
            sc: List[float] = [float(x) for x in topk_scores.tolist()]
            out_payload["topk"] = {"k": int(want_k), "cluster_ids": ids, "probs": pr, "scores": sc}

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
            "result": {
                "features_flat": dict(features_flat),
                "title_cluster_entropy_meta": out_payload,
            },
            "error": None,
        }


