from __future__ import annotations

import json
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

def _l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


class SemanticClusterExtractor(BaseExtractor):
    """
    A-policy semantic cluster classifier:
    - Strict model loading via dp_models (fail-fast in __init__)
    - Deterministic input loading via doc.tp_artifacts["embeddings"][...]["relpath"]
    - Valid empty semantics (no fake vectors): missing optional inputs -> present=0 + NaNs + *_present flags
    """

    VERSION = "1.2.0"

    def __init__(
        self,
        artifacts_dir: str | None = None,
        clusters_spec_name: str = "semantic_clusters_v1",
        primary_source: str = "title",  # title|description|hashtag
        allow_fallback_sources: Optional[List[str]] = None,  # list[str] subset of {title,description,hashtag}
        require_primary_source: bool = False,
        require_embedding: bool = False,
        use_faiss: bool = True,
        require_faiss: bool = False,
        emit_extra_metrics: bool = False,
        # Deprecated legacy args (kept for config compatibility; A-policy forbids bypassing dp_models)
        cluster_model_path: str | None = None,
        pca_model_path: str | None = None,
        use_hdbscan: bool = False,
        source: str | None = None,
    ) -> None:
        if cluster_model_path is not None or pca_model_path is not None or use_hdbscan:
            raise RuntimeError(
                "semantic_cluster_extractor: legacy model paths / use_hdbscan are no longer supported under A-policy. "
                "Use dp_models spec (clusters_spec_name=...) and nearest-centroid classifier only."
            )
        # Back-compat: allow old `source` config key to map to primary_source if provided.
        if source is not None:
            primary_source = str(source)

        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.clusters_spec_name = str(clusters_spec_name or "").strip()
        if not self.clusters_spec_name:
            raise RuntimeError("semantic_cluster_extractor: clusters_spec_name is required")

        self.primary_source = str(primary_source or "").strip().lower()
        if self.primary_source not in ("title", "description", "hashtag"):
            raise RuntimeError("semantic_cluster_extractor: primary_source must be one of: title|description|hashtag")

        self.require_primary_source = bool(require_primary_source)
        self.require_embedding = bool(require_embedding)
        self.use_faiss = bool(use_faiss)
        self.require_faiss = bool(require_faiss)
        self.emit_extra_metrics = bool(emit_extra_metrics)

        self.allow_fallback_sources = self._normalize_fallback_sources(allow_fallback_sources)

        self._centroids: np.ndarray
        self._pca: np.ndarray
        self._faiss_index: Any = None

        self._clusters_spec_version: str = "unknown"
        self._clusters_weights_digest: str = "unknown"
        self._cluster_db_version: str = "unknown"
        self._n_clusters: int = 0
        self._orig_dim: int = 0
        self._reduced_dim: int = 0

        self._load_assets_from_dp_models()

    def _normalize_fallback_sources(self, allow_fallback_sources: Optional[List[str]]) -> List[str]:
        if allow_fallback_sources is None:
            if self.primary_source == "title":
                allow_fallback_sources = ["description", "hashtag"]
            elif self.primary_source == "description":
                allow_fallback_sources = ["title", "hashtag"]
            else:
                allow_fallback_sources = ["title", "description"]
        out: List[str] = []
        for s in allow_fallback_sources:
            ss = str(s).strip().lower()
            if not ss:
                continue
            if ss not in ("title", "description", "hashtag"):
                raise RuntimeError("semantic_cluster_extractor: allow_fallback_sources must be subset of title|description|hashtag")
            if ss == self.primary_source:
                continue
            if ss not in out:
                out.append(ss)
        return out

    def _load_assets_from_dp_models(self) -> None:
        """
        Strictly loads PCA + centroids + cluster dictionary via dp_models (offline + fail-fast).
        """
        # Use global manager if available (TextProcessor sets DP_MODELS_ROOT externally).
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
        dict_rel = rp.get("clusters_jsonl_relpath")
        self._cluster_db_version = str(rp.get("cluster_db_version") or "unknown")

        if not isinstance(pca_rel, str) or not pca_rel:
            raise RuntimeError(f"semantic_cluster_extractor: spec missing runtime_params.pca_relpath: {self.clusters_spec_name}")
        if not isinstance(cent_rel, str) or not cent_rel:
            raise RuntimeError(f"semantic_cluster_extractor: spec missing runtime_params.centroids_relpath: {self.clusters_spec_name}")
        if not isinstance(dict_rel, str) or not dict_rel:
            raise RuntimeError(f"semantic_cluster_extractor: spec missing runtime_params.clusters_jsonl_relpath: {self.clusters_spec_name}")

        pca_path = resolved.get(pca_rel) or resolved.get(str(pca_rel))
        cent_path = resolved.get(cent_rel) or resolved.get(str(cent_rel))
        dict_path = resolved.get(dict_rel) or resolved.get(str(dict_rel))
        if not pca_path:
            raise RuntimeError(f"semantic_cluster_extractor: PCA artifact not resolved: {pca_rel}")
        if not cent_path:
            raise RuntimeError(f"semantic_cluster_extractor: centroids artifact not resolved: {cent_rel}")
        if not dict_path:
            raise RuntimeError(f"semantic_cluster_extractor: clusters dictionary artifact not resolved: {dict_rel}")

        pca = np.load(pca_path)
        pca = np.asarray(pca, dtype=np.float32)
        if pca.ndim != 2:
            raise RuntimeError("semantic_cluster_extractor: pca.npy must be 2D (orig_dim, reduced_dim)")

        centroids = np.load(cent_path)
        centroids = np.asarray(centroids, dtype=np.float32)
        if centroids.ndim != 2:
            raise RuntimeError("semantic_cluster_extractor: centroids.npy must be 2D (n_clusters, reduced_dim)")

        if int(pca.shape[1]) != int(centroids.shape[1]):
            raise RuntimeError(
                f"semantic_cluster_extractor: reduced_dim mismatch between PCA and centroids: "
                f"pca_reduced_dim={int(pca.shape[1])} centroids_reduced_dim={int(centroids.shape[1])}"
            )

        # Validate dictionary (privacy-safe; contains only IDs/names/groups)
        # We do NOT store it in output; UI can load it from dp_models by spec name.
        n_dict = 0
        try:
            with open(dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise RuntimeError("clusters.jsonl line must be a JSON object")
                    if not isinstance(obj.get("cluster_id"), int):
                        raise RuntimeError("clusters.jsonl must have integer cluster_id")
                    n_dict += 1
        except Exception as e:
            raise RuntimeError(f"semantic_cluster_extractor: invalid clusters.jsonl: {e}") from e

        self._pca = pca
        self._centroids = _l2_normalize(centroids, axis=1)
        self._n_clusters = int(self._centroids.shape[0])
        self._orig_dim = int(self._pca.shape[0])
        self._reduced_dim = int(self._pca.shape[1])
        if self._n_clusters <= 0:
            raise RuntimeError("semantic_cluster_extractor: centroids.npy must contain at least 1 cluster")
        if n_dict and n_dict != self._n_clusters:
            # Not fatal, but likely a packaging error.
            raise RuntimeError(
                f"semantic_cluster_extractor: clusters.jsonl size mismatch: dict_lines={int(n_dict)} centroids_n={int(self._n_clusters)}"
            )

        if self.use_faiss:
            if faiss is None:
                if self.require_faiss:
                    raise RuntimeError("semantic_cluster_extractor: faiss is required but not available")
            else:
                dim = int(self._centroids.shape[1])
                index = faiss.IndexFlatIP(dim)
                index.add(self._centroids.astype("float32"))
                self._faiss_index = index

    @staticmethod
    def _safe_join_artifacts_dir(base_dir: Path, relpath: str) -> Path:
        """
        Join artifacts_dir with relpath and forbid path traversal.
        """
        base = base_dir.expanduser().resolve()
        cand = (base / relpath).resolve()
        if not (cand == base or base in cand.parents):
            raise RuntimeError("semantic_cluster_extractor: embedding relpath escapes artifacts_dir")
        return cand

    def _pick_embedding(self, doc: Any) -> Tuple[Optional[np.ndarray], str]:
        # Deterministic: read relpaths via doc.tp_artifacts filled by embedder extractors.
        tp = getattr(doc, "tp_artifacts", None)
        emb = tp.get("embeddings") if isinstance(tp, dict) else None

        order: List[str] = [self.primary_source]
        if not self.require_primary_source:
            order.extend(self.allow_fallback_sources)

        if not isinstance(emb, dict):
            return None, ""

        for key in order:
            d = emb.get(key)
            rel = d.get("relpath") if isinstance(d, dict) else None
            if not isinstance(rel, str) or not rel:
                continue
            p = self._safe_join_artifacts_dir(self.artifacts_dir, rel)
            if not p.exists():
                continue
            try:
                v = np.load(p)
                v = np.asarray(v, dtype=np.float32).reshape(-1)
                return v, key
            except Exception:
                continue
        return None, ""

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        features_flat: Dict[str, float] = {
            "tp_semclust_present": 0.0,
            "tp_semclust_id": float("nan"),
            "tp_semclust_similarity": float("nan"),
            "tp_semclust_distance": float("nan"),
            "tp_semclust_dim_mismatch_flag": 0.0,
            "tp_semclust_fallback_used": 0.0,
            "tp_semclust_backend_faiss": 1.0 if (self._faiss_index is not None) else 0.0,
            # input presence flags (best-effort, based on relpath existence)
            "tp_semclust_title_present": 0.0,
            "tp_semclust_description_present": 0.0,
            "tp_semclust_hashtag_present": 0.0,
            # source-used one-hot (always present, even for empty)
            "tp_semclust_source_title": 0.0,
            "tp_semclust_source_description": 0.0,
            "tp_semclust_source_hashtag": 0.0,
        }

        tp = getattr(doc, "tp_artifacts", None)
        emb = tp.get("embeddings") if isinstance(tp, dict) else None
        if isinstance(emb, dict):
            for k in ("title", "description", "hashtag"):
                d = emb.get(k)
                rel = d.get("relpath") if isinstance(d, dict) else None
                if isinstance(rel, str) and rel:
                    features_flat[f"tp_semclust_{k}_present"] = 1.0

        # Models must be loaded in __init__ (fail-fast). Guard anyway.
        if self._pca is None or self._centroids is None:  # type: ignore[truthy-bool]
            raise RuntimeError("semantic_cluster_extractor: models not loaded (this should have failed in __init__)")

        vec, detected = self._pick_embedding(doc)
        if vec is None:
            if self.require_embedding:
                raise RuntimeError(
                    "semantic_cluster_extractor: required embedding is missing. "
                    f"primary_source={self.primary_source!r} require_primary_source={self.require_primary_source}"
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
                    "semantic_cluster_meta": {
                        "clusters_spec_name": self.clusters_spec_name,
                        "clusters_spec_version": self._clusters_spec_version,
                        "clusters_weights_digest": self._clusters_weights_digest,
                        "cluster_db_version": self._cluster_db_version,
                    },
                },
                "error": None,
            }

        if detected:
            features_flat[f"tp_semclust_source_{detected}"] = 1.0
        if detected and detected != self.primary_source:
            features_flat["tp_semclust_fallback_used"] = 1.0

        # Dim checks
        if int(vec.shape[0]) != int(self._orig_dim):
            features_flat["tp_semclust_dim_mismatch_flag"] = 1.0
            if self.require_embedding:
                raise RuntimeError(
                    f"semantic_cluster_extractor: embedding dim mismatch: embedding_dim={int(vec.shape[0])} "
                    f"model_orig_dim={int(self._orig_dim)} (clusters_spec_name={self.clusters_spec_name})"
                )
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            if self.emit_extra_metrics:
                features_flat["tp_semclust_embedding_dim"] = float(int(vec.shape[0]))
                features_flat["tp_semclust_model_orig_dim"] = float(int(self._orig_dim))
                features_flat["tp_semclust_model_reduced_dim"] = float(int(self._reduced_dim))
                features_flat["tp_semclust_n_clusters"] = float(int(self._n_clusters))
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
                    "semantic_cluster_meta": {
                        "clusters_spec_name": self.clusters_spec_name,
                        "clusters_spec_version": self._clusters_spec_version,
                        "clusters_weights_digest": self._clusters_weights_digest,
                        "cluster_db_version": self._cluster_db_version,
                    },
                },
                "error": None,
            }

        # Project via PCA and normalize
        t_compute0 = time.perf_counter()
        reduced = vec @ self._pca  # (reduced_dim,)
        reduced = _l2_normalize(reduced.reshape(1, -1), axis=1)

        # Nearest centroid by cosine similarity (inner product on L2-normalized vectors)
        want_k = 2 if self.emit_extra_metrics else 1
        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(reduced.astype("float32"), want_k)
            sim = float(scores[0, 0])
            cid = int(idx[0, 0])
            margin = float("nan")
            if self.emit_extra_metrics and scores.shape[1] >= 2:
                margin = float(scores[0, 0] - scores[0, 1])
        else:
            sims = (reduced @ self._centroids.T).reshape(-1)  # type: ignore[arg-type]
            if sims.size <= 0:
                # Should never happen (centroids validated in __init__), but keep safe behavior.
                cid = -1
                sim = float("nan")
            else:
                cid = int(np.argmax(sims))
                sim = float(sims[cid])
            margin = float("nan")
            if self.emit_extra_metrics and sims.size >= 2:
                # stable-ish second best
                # argsort for 32-256 clusters is fine; FAISS is preferred for large N
                s2 = np.sort(sims)[-2]
                margin = float(sim - float(s2))

        dist = 1.0 - sim
        features_flat["tp_semclust_present"] = 1.0
        features_flat["tp_semclust_id"] = float(int(cid))
        features_flat["tp_semclust_similarity"] = float(sim)
        features_flat["tp_semclust_distance"] = float(dist)

        t_compute = time.perf_counter() - t_compute0

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        if self.emit_extra_metrics:
            features_flat["tp_semclust_n_clusters"] = float(int(self._n_clusters))
            features_flat["tp_semclust_model_orig_dim"] = float(int(self._orig_dim))
            features_flat["tp_semclust_model_reduced_dim"] = float(int(self._reduced_dim))
            features_flat["tp_semclust_embedding_dim"] = float(int(vec.shape[0]))
            features_flat["tp_semclust_margin_top2"] = float(margin)
            features_flat["tp_semclust_compute_ms"] = float(round(t_compute * 1000.0, 3))

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
                "semantic_cluster_meta": {
                    "clusters_spec_name": self.clusters_spec_name,
                    "clusters_spec_version": self._clusters_spec_version,
                    "clusters_weights_digest": self._clusters_weights_digest,
                    "cluster_db_version": self._cluster_db_version,
                    "backend": "faiss_ip" if (self._faiss_index is not None) else "numpy_cosine",
                },
            },
            "error": None,
        }


