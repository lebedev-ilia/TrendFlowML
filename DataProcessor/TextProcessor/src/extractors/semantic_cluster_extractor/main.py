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


def _build_features_flat_keys() -> Tuple[str, ...]:
    return (
        "tp_semclust_present",
        "tp_semclust_require_primary_source_enabled",
        "tp_semclust_require_embedding_enabled",
        "tp_semclust_use_faiss_enabled",
        "tp_semclust_require_faiss_enabled",
        "tp_semclust_emit_extra_metrics_enabled",
        "tp_semclust_config_primary_title",
        "tp_semclust_config_primary_description",
        "tp_semclust_config_primary_hashtag",
        "tp_semclust_title_present",
        "tp_semclust_description_present",
        "tp_semclust_hashtag_present",
        "tp_semclust_source_title",
        "tp_semclust_source_description",
        "tp_semclust_source_hashtag",
        "tp_semclust_fallback_used",
        "tp_semclust_backend_faiss",
        "tp_semclust_dim_mismatch_flag",
        "tp_semclust_unsafe_relpath_flag",
        "tp_semclust_title_embed_missing_flag",
        "tp_semclust_description_embed_missing_flag",
        "tp_semclust_hashtag_embed_missing_flag",
        "tp_semclust_id",
        "tp_semclust_similarity",
        "tp_semclust_distance",
        "tp_semclust_n_clusters",
        "tp_semclust_model_orig_dim",
        "tp_semclust_model_reduced_dim",
        "tp_semclust_embedding_dim",
        "tp_semclust_margin_top2",
        "tp_semclust_compute_ms",
    )


_FEATURES_FLAT_KEYS = _build_features_flat_keys()
_SLOT_KEYS = ("title", "description", "hashtag")


class SemanticClusterExtractor(BaseExtractor):
    """
    Nearest-centroid semantic cluster classification over shared taxonomy (PCA + centroids via dp_models).
    Audit v3: fixed features_flat; per-slot load diagnostics; meta.backend on all branches.
    """

    VERSION = "1.3.0"

    def __init__(
        self,
        artifacts_dir: str | None = None,
        clusters_spec_name: str = "semantic_clusters_v1",
        primary_source: str = "title",
        allow_fallback_sources: Optional[List[str]] = None,
        require_primary_source: bool = False,
        require_embedding: bool = False,
        use_faiss: bool = True,
        require_faiss: bool = False,
        emit_extra_metrics: bool = False,
        cluster_model_path: str | None = None,
        pca_model_path: str | None = None,
        use_hdbscan: bool = False,
        source: str | None = None,
    ) -> None:
        init_sys_before = system_snapshot()
        init_mem_before = process_memory_bytes()

        if cluster_model_path is not None or pca_model_path is not None or use_hdbscan:
            raise RuntimeError(
                "semantic_cluster_extractor: legacy model paths / use_hdbscan are no longer supported under Audit v3. "
                "Use dp_models spec (clusters_spec_name=...) and nearest-centroid classifier only."
            )
        if source is not None:
            primary_source = str(source)

        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.clusters_spec_name = str(clusters_spec_name or "").strip()
        if not self.clusters_spec_name:
            raise RuntimeError("semantic_cluster_extractor: clusters_spec_name is required")

        self.primary_source = str(primary_source or "").strip().lower()
        if self.primary_source not in _SLOT_KEYS:
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

        init_sys_after = system_snapshot()
        init_mem_after = process_memory_bytes()
        self._init_metrics: Dict[str, Any] = {
            "pre_init": init_sys_before,
            "post_init": init_sys_after,
            "ram_peak_bytes": max(init_mem_before, init_mem_after),
        }

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
            if ss not in _SLOT_KEYS:
                raise RuntimeError("semantic_cluster_extractor: allow_fallback_sources must be subset of title|description|hashtag")
            if ss == self.primary_source:
                continue
            if ss not in out:
                out.append(ss)
        return out

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
            raise RuntimeError("semantic_cluster_extractor: embedding relpath escapes artifacts_dir")
        return cand

    @staticmethod
    def _pack_features_flat(values: Dict[str, Any]) -> Dict[str, Any]:
        nan = float("nan")
        out: Dict[str, Any] = {}
        for k in _FEATURES_FLAT_KEYS:
            if k not in values:
                raise KeyError(f"SemanticClusterExtractor: missing features_flat key {k!r}")
            v = values[k]
            if v is None:
                out[k] = nan
            elif isinstance(v, (bool, np.bool_)):
                out[k] = float(bool(v))
            else:
                out[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else nan
        return out

    def _mirrors_fragment(self) -> Dict[str, float]:
        return {
            "tp_semclust_require_primary_source_enabled": float(bool(self.require_primary_source)),
            "tp_semclust_require_embedding_enabled": float(bool(self.require_embedding)),
            "tp_semclust_use_faiss_enabled": float(bool(self.use_faiss)),
            "tp_semclust_require_faiss_enabled": float(bool(self.require_faiss)),
            "tp_semclust_emit_extra_metrics_enabled": float(bool(self.emit_extra_metrics)),
        }

    def _primary_config_onehot(self) -> Dict[str, float]:
        d = {f"tp_semclust_config_primary_{k}": 0.0 for k in _SLOT_KEYS}
        d[f"tp_semclust_config_primary_{self.primary_source}"] = 1.0
        return d

    def _semantic_meta(self) -> Dict[str, Any]:
        return {
            "clusters_spec_name": self.clusters_spec_name,
            "clusters_spec_version": self._clusters_spec_version,
            "clusters_weights_digest": self._clusters_weights_digest,
            "cluster_db_version": self._cluster_db_version,
            "backend": "faiss_ip" if (self._faiss_index is not None) else "numpy_cosine",
        }

    def _empty_features_template(self) -> Dict[str, Any]:
        nan = float("nan")
        ff: Dict[str, Any] = {
            "tp_semclust_present": 0.0,
            "tp_semclust_title_present": 0.0,
            "tp_semclust_description_present": 0.0,
            "tp_semclust_hashtag_present": 0.0,
            "tp_semclust_source_title": 0.0,
            "tp_semclust_source_description": 0.0,
            "tp_semclust_source_hashtag": 0.0,
            "tp_semclust_fallback_used": 0.0,
            "tp_semclust_dim_mismatch_flag": 0.0,
            "tp_semclust_unsafe_relpath_flag": 0.0,
            "tp_semclust_title_embed_missing_flag": 0.0,
            "tp_semclust_description_embed_missing_flag": 0.0,
            "tp_semclust_hashtag_embed_missing_flag": 0.0,
            "tp_semclust_id": nan,
            "tp_semclust_similarity": nan,
            "tp_semclust_distance": nan,
            "tp_semclust_n_clusters": nan,
            "tp_semclust_model_orig_dim": nan,
            "tp_semclust_model_reduced_dim": nan,
            "tp_semclust_embedding_dim": nan,
            "tp_semclust_margin_top2": nan,
            "tp_semclust_compute_ms": nan,
        }
        ff.update(self._mirrors_fragment())
        ff.update(self._primary_config_onehot())
        ff["tp_semclust_backend_faiss"] = 1.0 if (self._faiss_index is not None) else 0.0
        return ff

    def _apply_extra_block(
        self,
        ff: Dict[str, Any],
        *,
        success: bool,
        dim_mismatch: bool,
        vec: Optional[np.ndarray],
    ) -> None:
        nan = float("nan")
        if not self.emit_extra_metrics:
            ff["tp_semclust_n_clusters"] = nan
            ff["tp_semclust_model_orig_dim"] = nan
            ff["tp_semclust_model_reduced_dim"] = nan
            ff["tp_semclust_embedding_dim"] = nan
            ff["tp_semclust_margin_top2"] = nan
            ff["tp_semclust_compute_ms"] = nan
            return
        if success:
            return  # caller filled numerics
        if dim_mismatch and vec is not None:
            ff["tp_semclust_n_clusters"] = float(int(self._n_clusters))
            ff["tp_semclust_model_orig_dim"] = float(int(self._orig_dim))
            ff["tp_semclust_model_reduced_dim"] = float(int(self._reduced_dim))
            ff["tp_semclust_embedding_dim"] = float(int(vec.shape[0]))
            ff["tp_semclust_margin_top2"] = nan
            ff["tp_semclust_compute_ms"] = nan
            return
        ff["tp_semclust_n_clusters"] = nan
        ff["tp_semclust_model_orig_dim"] = nan
        ff["tp_semclust_model_reduced_dim"] = nan
        ff["tp_semclust_embedding_dim"] = nan
        ff["tp_semclust_margin_top2"] = nan
        ff["tp_semclust_compute_ms"] = nan

    def _scan_embedding_slots(
        self, doc: Any
    ) -> Tuple[Dict[str, Optional[np.ndarray]], bool, Dict[str, bool], Dict[str, float]]:
        vecs: Dict[str, Optional[np.ndarray]] = {k: None for k in _SLOT_KEYS}
        missing = {k: False for k in _SLOT_KEYS}
        loaded_flag = {k: 0.0 for k in _SLOT_KEYS}
        unsafe_any = False

        tp = getattr(doc, "tp_artifacts", None)
        emb = tp.get("embeddings") if isinstance(tp, dict) else None
        if not isinstance(emb, dict):
            return vecs, unsafe_any, missing, loaded_flag

        for key in _SLOT_KEYS:
            d = emb.get(key)
            rel = d.get("relpath") if isinstance(d, dict) else None
            if not isinstance(rel, str) or not rel:
                continue
            try:
                p = self._safe_join_artifacts_dir(self.artifacts_dir, rel)
            except Exception:
                unsafe_any = True
                continue
            if not p.exists():
                missing[key] = True
                continue
            try:
                v = np.load(p)
                v = np.asarray(v, dtype=np.float32).reshape(-1)
                if int(v.size) <= 0:
                    missing[key] = True
                    continue
                vecs[key] = v
                loaded_flag[key] = 1.0
            except Exception:
                missing[key] = True

        return vecs, unsafe_any, missing, loaded_flag

    def _select_embedding_vector(self, vecs: Dict[str, Optional[np.ndarray]]) -> Tuple[Optional[np.ndarray], str]:
        order: List[str] = [self.primary_source]
        if not self.require_primary_source:
            order.extend(self.allow_fallback_sources)
        seen: set[str] = set()
        for k in order:
            if k in seen:
                continue
            seen.add(k)
            v = vecs.get(k)
            if v is not None:
                return v, k
        return None, ""

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
            "result": {
                "features_flat": self._pack_features_flat(features_flat),
                "semantic_cluster_meta": self._semantic_meta(),
            },
            "error": None,
        }

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        mem_before = process_memory_bytes()

        if self._pca is None or self._centroids is None:  # type: ignore[truthy-bool]
            raise RuntimeError("semantic_cluster_extractor: models not loaded (this should have failed in __init__)")

        vecs, unsafe_any, missing_map, loaded_flag = self._scan_embedding_slots(doc)
        ff = self._empty_features_template()
        ff["tp_semclust_unsafe_relpath_flag"] = 1.0 if unsafe_any else 0.0
        for k in _SLOT_KEYS:
            ff[f"tp_semclust_{k}_present"] = float(loaded_flag[k])
            mf = 1.0 if missing_map[k] else 0.0
            ff[f"tp_semclust_{k}_embed_missing_flag"] = float(mf)

        vec, detected = self._select_embedding_vector(vecs)

        def _finish(ff_local: Dict[str, Any]) -> Dict[str, Any]:
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return self._build_return(
                features_flat=ff_local,
                sys_after=sys_after,
                mem_before=mem_before,
                mem_after=mem_after,
                total_s=total_s,
            )

        if vec is None:
            if self.require_embedding:
                raise RuntimeError(
                    "semantic_cluster_extractor: required embedding is missing. "
                    f"primary_source={self.primary_source!r} require_primary_source={self.require_primary_source}"
                )
            self._apply_extra_block(ff, success=False, dim_mismatch=False, vec=None)
            return _finish(ff)

        if detected:
            ff[f"tp_semclust_source_{detected}"] = 1.0
        if detected and detected != self.primary_source:
            ff["tp_semclust_fallback_used"] = 1.0

        if int(vec.shape[0]) != int(self._orig_dim):
            ff["tp_semclust_dim_mismatch_flag"] = 1.0
            if self.require_embedding:
                raise RuntimeError(
                    f"semantic_cluster_extractor: embedding dim mismatch: embedding_dim={int(vec.shape[0])} "
                    f"model_orig_dim={int(self._orig_dim)} (clusters_spec_name={self.clusters_spec_name})"
                )
            self._apply_extra_block(ff, success=False, dim_mismatch=True, vec=vec)
            return _finish(ff)

        t_compute0 = time.perf_counter()
        reduced = vec @ self._pca
        reduced = _l2_normalize(reduced.reshape(1, -1), axis=1)

        want_k = 2 if self.emit_extra_metrics else 1
        margin = float("nan")
        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(reduced.astype("float32"), want_k)
            sim = float(scores[0, 0])
            cid = int(idx[0, 0])
            if self.emit_extra_metrics and scores.shape[1] >= 2:
                margin = float(scores[0, 0] - scores[0, 1])
        else:
            sims = (reduced @ self._centroids.T).reshape(-1)  # type: ignore[arg-type]
            if sims.size <= 0:
                cid = -1
                sim = float("nan")
            else:
                cid = int(np.argmax(sims))
                sim = float(sims[cid])
            if self.emit_extra_metrics and sims.size >= 2:
                s2 = np.sort(sims)[-2]
                margin = float(sim - float(s2))

        dist = 1.0 - sim
        ff["tp_semclust_present"] = 1.0
        ff["tp_semclust_id"] = float(int(cid))
        ff["tp_semclust_similarity"] = float(sim)
        ff["tp_semclust_distance"] = float(dist)

        t_compute = time.perf_counter() - t_compute0

        if self.emit_extra_metrics:
            ff["tp_semclust_n_clusters"] = float(int(self._n_clusters))
            ff["tp_semclust_model_orig_dim"] = float(int(self._orig_dim))
            ff["tp_semclust_model_reduced_dim"] = float(int(self._reduced_dim))
            ff["tp_semclust_embedding_dim"] = float(int(vec.shape[0]))
            ff["tp_semclust_margin_top2"] = float(margin)
            ff["tp_semclust_compute_ms"] = float(round(t_compute * 1000.0, 3))
        else:
            self._apply_extra_block(ff, success=True, dim_mismatch=False, vec=vec)

        return _finish(ff)
