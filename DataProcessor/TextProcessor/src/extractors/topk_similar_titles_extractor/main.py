from __future__ import annotations

import json
import time
from collections import OrderedDict
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


def _l2n(x: np.ndarray, axis: int = 1, eps: float = 1e-10) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


_INDEX_CACHE: "OrderedDict[str, Tuple[float, Any, List[Any], int, int, str, str, str]]" = OrderedDict()
# key -> (created_at_s, index_or_corpus, ids, dim, corpus_size, corpus_weights_digest, corpus_version, id_kind)

_FEATURES_FLAT_KEYS: Tuple[str, ...] = (
    "tp_topktitles_present",
    "tp_topktitles_disabled_by_policy",
    "tp_topktitles_enabled",
    "tp_topktitles_require_title_embedding_enabled",
    "tp_topktitles_k",
    "tp_topktitles_corpus_size",
    "tp_topktitles_dim",
    "tp_topktitles_backend_faiss",
    "tp_topktitles_faiss_available",
    "tp_topktitles_require_faiss_enabled",
    "tp_topktitles_require_faiss_above_corpus_size",
    "tp_topktitles_allow_numpy_large_corpus_enabled",
    "tp_topktitles_max_corpus_for_numpy",
    "tp_topktitles_cache_enabled",
    "tp_topktitles_cache_ttl_s",
    "tp_topktitles_cache_max_entries",
    "tp_topktitles_export_topk_mode_ids_only",
    "tp_topktitles_export_topk_mode_ids_and_scores",
    "tp_topktitles_export_topk_mode_none",
    "tp_topktitles_max_export_k",
    "tp_topktitles_export_k_used",
    "tp_topktitles_export_k_truncated_flag",
    "tp_topktitles_unsafe_relpath_flag",
    "tp_topktitles_title_embed_missing_flag",
    "tp_topktitles_dim_mismatch_flag",
    "tp_topktitles_zero_norm_flag",
    "tp_topktitles_nan_inf_flag",
    "tp_topktitles_top1_score",
    "tp_topktitles_topk_mean_score",
)


class TopKSimilarCorpusTitlesExtractor(BaseExtractor):
    VERSION = "1.3.0"

    def __init__(
        self,
        corpus_spec_name: str = "similar_titles_corpus_v1",
        k: int = 5,
        export_topk_mode: str = "ids_and_scores",
        max_export_k: int = 50,
        enabled: bool = True,
        require_title_embedding: bool = False,
        require_faiss: bool = False,
        require_faiss_above_corpus_size: int = 200_000,
        allow_numpy_large_corpus: bool = False,
        max_corpus_for_numpy: int = 100_000,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 128,
        cache_enabled: bool = True,
        cache_ttl_s: float = 3600.0,
        cache_max_entries: int = 2,
        artifacts_dir: str | None = None,
    ) -> None:
        init_sys_before = system_snapshot()
        init_mem_before = process_memory_bytes()

        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.k = int(k)
        self.export_topk_mode = str(export_topk_mode or "ids_and_scores")
        self.max_export_k = int(max_export_k)
        self.enabled = bool(enabled)
        self.require_title_embedding = bool(require_title_embedding)
        self.require_faiss = bool(require_faiss)
        self.require_faiss_above_corpus_size = int(require_faiss_above_corpus_size)
        self.allow_numpy_large_corpus = bool(allow_numpy_large_corpus)
        self.max_corpus_for_numpy = int(max_corpus_for_numpy)
        self.hnsw_m = int(hnsw_m)
        self.hnsw_ef_construction = int(hnsw_ef_construction)
        self.hnsw_ef_search = int(hnsw_ef_search)
        self.cache_enabled = bool(cache_enabled)
        self.cache_ttl_s = float(cache_ttl_s)
        self.cache_max_entries = int(cache_max_entries)
        self._index: Any = None
        self._ids: List[Any] = []
        self._dim: Optional[int] = None
        self._corpus_size: int = 0
        self._corpus_weights_digest: str = "unknown"
        self._corpus_version: str = "unknown"
        self._corpus_spec_name: str = str(corpus_spec_name)
        self._id_kind: str = "unknown"

        self._load_corpus_from_dp_models(self._corpus_spec_name)

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
            raise RuntimeError("topk_similar_titles_extractor: relpath escapes artifacts_dir")
        return cand

    def _corpus_backend_label(self) -> str:
        if faiss is not None and self._index is not None and hasattr(self._index, "search") and not isinstance(self._index, np.ndarray):
            try:
                if isinstance(self._index, faiss.Index):  # type: ignore[arg-type]
                    return "faiss_hnsw_ip"
            except Exception:
                pass
        return "numpy_cosine"

    def _corpus_meta_dict(self) -> Dict[str, Any]:
        return {
            "corpus_spec_name": self._corpus_spec_name,
            "corpus_version": self._corpus_version,
            "corpus_weights_digest": self._corpus_weights_digest,
            "id_kind": self._id_kind,
            "corpus_size": int(self._corpus_size),
            "dim": int(self._dim or 0),
            "backend": self._corpus_backend_label(),
            "hnsw": {
                "m": int(self.hnsw_m),
                "ef_construction": int(self.hnsw_ef_construction),
                "ef_search": int(self.hnsw_ef_search),
            },
        }

    def _cache_key(self) -> str:
        backend = "faiss_hnsw_ip" if (faiss is not None) else "numpy_cosine"
        return "|".join(
            [
                f"spec={self._corpus_spec_name}",
                f"ver={self._corpus_version}",
                f"digest={self._corpus_weights_digest}",
                f"backend={backend}",
                f"hnsw_m={int(self.hnsw_m)}",
                f"hnsw_efc={int(self.hnsw_ef_construction)}",
                f"hnsw_efs={int(self.hnsw_ef_search)}",
            ]
        )

    def _cache_get(self, key: str) -> Optional[Tuple[Any, List[Any], int, int, str, str, str]]:
        if not self.cache_enabled:
            return None
        now = time.time()
        item = _INDEX_CACHE.get(key)
        if not item:
            return None
        created_at_s, index_or_corpus, ids, dim, corpus_size, corpus_weights_digest, corpus_version, id_kind = item
        if self.cache_ttl_s > 0 and (now - float(created_at_s)) > float(self.cache_ttl_s):
            try:
                del _INDEX_CACHE[key]
            except Exception:
                pass
            return None
        _INDEX_CACHE.move_to_end(key)
        return index_or_corpus, ids, dim, corpus_size, corpus_weights_digest, corpus_version, id_kind

    def _cache_put(self, key: str, index_or_corpus: Any, ids: List[Any], dim: int, corpus_size: int, weights_digest: str, version: str, id_kind: str) -> None:
        if not self.cache_enabled:
            return
        now = time.time()
        _INDEX_CACHE[key] = (now, index_or_corpus, ids, int(dim), int(corpus_size), str(weights_digest), str(version), str(id_kind))
        _INDEX_CACHE.move_to_end(key)
        while self.cache_max_entries > 0 and len(_INDEX_CACHE) > int(self.cache_max_entries):
            try:
                _INDEX_CACHE.popitem(last=False)
            except Exception:
                break

    def _load_corpus_from_dp_models(self, corpus_spec_name: str) -> None:
        if not corpus_spec_name:
            raise RuntimeError("topk_similar_titles_extractor: corpus_spec_name is required")

        mm = ModelManager()
        spec = mm.get_spec(model_name=corpus_spec_name)
        _, _, _, _, weights_digest, resolved = mm.resolve(spec)

        rp = spec.runtime_params if isinstance(spec.runtime_params, dict) else {}
        emb_rel = rp.get("embeddings_relpath")
        ids_rel = rp.get("ids_relpath")
        id_kind = rp.get("id_kind")
        if isinstance(id_kind, str) and id_kind:
            self._id_kind = id_kind
        if not isinstance(emb_rel, str) or not emb_rel:
            raise RuntimeError(f"topk_similar_titles_extractor: corpus spec missing runtime_params.embeddings_relpath: {corpus_spec_name}")
        if not isinstance(ids_rel, str) or not ids_rel:
            raise RuntimeError(f"topk_similar_titles_extractor: corpus spec missing runtime_params.ids_relpath: {corpus_spec_name}")

        emb_path = resolved.get(emb_rel) or resolved.get(str(emb_rel))
        ids_path = resolved.get(ids_rel) or resolved.get(str(ids_rel))
        if not emb_path:
            raise RuntimeError(f"topk_similar_titles_extractor: corpus embeddings artifact not resolved: {emb_rel}")
        if not ids_path:
            raise RuntimeError(f"topk_similar_titles_extractor: corpus ids artifact not resolved: {ids_rel}")

        emb = np.load(emb_path)
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 2:
            raise RuntimeError("topk_similar_titles_extractor: corpus embeddings must be 2D")
        if not np.isfinite(emb).all():
            raise RuntimeError("topk_similar_titles_extractor: corpus embeddings contain NaN/inf")
        emb = _l2n(emb, axis=1)
        self._dim = int(emb.shape[1])
        self._corpus_size = int(emb.shape[0])

        with open(ids_path, "r", encoding="utf-8") as f:
            ids = json.load(f)
        if not isinstance(ids, list) or len(ids) != emb.shape[0]:
            raise RuntimeError("topk_similar_titles_extractor: corpus ids must be a list with same length as embeddings")
        self._ids = ids
        self._corpus_weights_digest = str(weights_digest or "unknown")
        self._corpus_version = str(getattr(spec, "model_version", None) or "unknown")

        cache_key = self._cache_key()
        cached = self._cache_get(cache_key)
        if cached is not None:
            index_or_corpus, ids_cached, dim, corpus_size, wd, ver, id_kind_cached = cached
            self._index = index_or_corpus
            self._ids = ids_cached
            self._dim = int(dim)
            self._corpus_size = int(corpus_size)
            self._corpus_weights_digest = str(wd)
            self._corpus_version = str(ver)
            self._id_kind = str(id_kind_cached or self._id_kind)
            return

        if faiss is None:
            if self.require_faiss:
                raise RuntimeError("topk_similar_titles_extractor: faiss is required but not available")
            if int(self.require_faiss_above_corpus_size) > 0 and int(self._corpus_size) >= int(self.require_faiss_above_corpus_size):
                raise RuntimeError(
                    "topk_similar_titles_extractor: corpus too large for numpy backend; install faiss or lower require_faiss_above_corpus_size"
                )
            if int(self._corpus_size) > int(self.max_corpus_for_numpy) and not self.allow_numpy_large_corpus:
                raise RuntimeError(
                    "topk_similar_titles_extractor: numpy backend disabled for large corpus; set allow_numpy_large_corpus=true or install faiss"
                )
            self._index = emb
            self._cache_put(cache_key, self._index, self._ids, int(self._dim), int(self._corpus_size), self._corpus_weights_digest, self._corpus_version, self._id_kind)
            return

        index = faiss.IndexHNSWFlat(self._dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)  # type: ignore[attr-defined]
        index.hnsw.efConstruction = int(self.hnsw_ef_construction)
        index.hnsw.efSearch = int(self.hnsw_ef_search)
        index.add(emb.astype(np.float32))
        self._index = index
        self._cache_put(cache_key, self._index, self._ids, int(self._dim), int(self._corpus_size), self._corpus_weights_digest, self._corpus_version, self._id_kind)

    def _search_np(self, query: np.ndarray, corpus: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        sims = (_l2n(query, axis=1) @ corpus.T)
        k = min(k, sims.shape[1])
        idx = np.argsort(-sims, axis=1)[:, :k]
        scr = np.take_along_axis(sims, idx, axis=1)
        return idx, scr

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
                raise KeyError(f"TopKSimilarCorpusTitlesExtractor: missing features_flat key {k!r}")
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
            "tp_topktitles_present": 0.0,
            "tp_topktitles_disabled_by_policy": 0.0,
            "tp_topktitles_enabled": float(bool(self.enabled)),
            "tp_topktitles_require_title_embedding_enabled": float(bool(self.require_title_embedding)),
            "tp_topktitles_k": float(int(self.k)),
            "tp_topktitles_corpus_size": float(int(self._corpus_size)),
            "tp_topktitles_dim": float(int(self._dim or 0)),
            "tp_topktitles_backend_faiss": 1.0 if self._corpus_backend_label() == "faiss_hnsw_ip" else 0.0,
            "tp_topktitles_faiss_available": 1.0 if (faiss is not None) else 0.0,
            "tp_topktitles_require_faiss_enabled": float(bool(self.require_faiss)),
            "tp_topktitles_require_faiss_above_corpus_size": float(int(self.require_faiss_above_corpus_size)),
            "tp_topktitles_allow_numpy_large_corpus_enabled": float(bool(self.allow_numpy_large_corpus)),
            "tp_topktitles_max_corpus_for_numpy": float(int(self.max_corpus_for_numpy)),
            "tp_topktitles_cache_enabled": float(bool(self.cache_enabled)),
            "tp_topktitles_cache_ttl_s": float(self.cache_ttl_s),
            "tp_topktitles_cache_max_entries": float(int(self.cache_max_entries)),
            "tp_topktitles_export_topk_mode_ids_only": 1.0 if self.export_topk_mode == "ids_only" else 0.0,
            "tp_topktitles_export_topk_mode_ids_and_scores": 1.0 if self.export_topk_mode == "ids_and_scores" else 0.0,
            "tp_topktitles_export_topk_mode_none": 1.0 if self.export_topk_mode == "none" else 0.0,
            "tp_topktitles_max_export_k": float(int(self.max_export_k)),
            "tp_topktitles_export_k_used": nan,
            "tp_topktitles_export_k_truncated_flag": 0.0,
            "tp_topktitles_unsafe_relpath_flag": 0.0,
            "tp_topktitles_title_embed_missing_flag": 0.0,
            "tp_topktitles_dim_mismatch_flag": 0.0,
            "tp_topktitles_zero_norm_flag": 0.0,
            "tp_topktitles_nan_inf_flag": 0.0,
            "tp_topktitles_top1_score": nan,
            "tp_topktitles_topk_mean_score": nan,
        }

    def _build_return(
        self,
        *,
        features_flat: Dict[str, Any],
        topk_corpus_titles: Dict[str, Any],
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
            "result": {
                "features_flat": self._pack_features_flat(features_flat),
                "topk_similar_corpus_titles": topk_corpus_titles,
            },
            "error": error,
        }

    def extract(self, doc: Any) -> Dict[str, Any]:
        t0 = time.perf_counter()
        mem_before = process_memory_bytes()
        error: Optional[str] = None
        corpus_shell = {"corpus": self._corpus_meta_dict()}

        def _finish(ff: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return self._build_return(
                features_flat=ff,
                topk_corpus_titles=payload,
                sys_after=sys_after,
                mem_before=mem_before,
                mem_after=mem_after,
                total_s=total_s,
                error=error,
            )

        features_flat = self._base_features_flat()

        if not self.enabled:
            features_flat["tp_topktitles_disabled_by_policy"] = 1.0
            return _finish(features_flat, corpus_shell)

        if self._index is None or not self._ids or self._dim is None:
            raise RuntimeError("topk_similar_titles_extractor: corpus not loaded (this should have failed in __init__)")

        tp = getattr(doc, "tp_artifacts", None)
        emb = tp.get("embeddings") if isinstance(tp, dict) else None
        title_rel = emb.get("title", {}).get("relpath") if isinstance(emb, dict) and isinstance(emb.get("title"), dict) else None
        if not isinstance(title_rel, str) or not title_rel:
            if self.require_title_embedding:
                raise RuntimeError("topk_similar_titles_extractor: missing title embedding relpath in doc.tp_artifacts.embeddings.title")
            return _finish(features_flat, corpus_shell)

        try:
            p = self._safe_join_artifacts_dir(self.artifacts_dir, title_rel)
        except Exception:
            features_flat["tp_topktitles_unsafe_relpath_flag"] = 1.0
            if self.require_title_embedding:
                raise RuntimeError("topk_similar_titles_extractor: unsafe title embedding relpath")
            return _finish(features_flat, corpus_shell)

        if not p.exists():
            features_flat["tp_topktitles_title_embed_missing_flag"] = 1.0
            if self.require_title_embedding:
                raise RuntimeError("topk_similar_titles_extractor: title embedding file not found in per-run artifacts")
            return _finish(features_flat, corpus_shell)

        try:
            title = np.load(p).astype(np.float32).reshape(1, -1)
        except Exception:
            features_flat["tp_topktitles_title_embed_missing_flag"] = 1.0
            if self.require_title_embedding:
                raise RuntimeError("topk_similar_titles_extractor: failed to load title embedding from per-run artifacts")
            return _finish(features_flat, corpus_shell)

        if not np.isfinite(title).all():
            features_flat["tp_topktitles_nan_inf_flag"] = 1.0
            if self.require_title_embedding:
                raise RuntimeError("topk_similar_titles_extractor: title embedding contains NaN/inf")
            return _finish(features_flat, corpus_shell)
        if int(title.shape[1]) != int(self._dim):
            features_flat["tp_topktitles_dim_mismatch_flag"] = 1.0
            if self.require_title_embedding:
                raise RuntimeError(
                    f"topk_similar_titles_extractor: embedding dim mismatch: title_dim={int(title.shape[1])} corpus_dim={int(self._dim)}"
                )
            return _finish(features_flat, corpus_shell)

        if float(np.linalg.norm(title)) <= 0.0:
            features_flat["tp_topktitles_zero_norm_flag"] = 1.0
            if self.require_title_embedding:
                raise RuntimeError("topk_similar_titles_extractor: title embedding has zero norm")
            return _finish(features_flat, corpus_shell)

        title = _l2n(title, axis=1)

        top_ids: List[Any]
        top_scores: List[float]
        if faiss is not None and isinstance(self._index, faiss.Index):  # type: ignore[arg-type]
            scores, indices = self._index.search(title.astype(np.float32), min(self.k, len(self._ids)))
            top_ids = [self._ids[i] for i in indices[0].tolist()]
            top_scores = scores[0].astype(float).tolist()
        else:
            idx, scr = self._search_np(title, self._index, self.k)  # type: ignore[arg-type]
            top_ids = [self._ids[i] for i in idx[0].tolist()]
            top_scores = scr[0].astype(float).tolist()

        if top_scores:
            features_flat["tp_topktitles_top1_score"] = float(top_scores[0])
            features_flat["tp_topktitles_topk_mean_score"] = float(np.mean(np.asarray(top_scores, dtype=np.float32)))

        features_flat["tp_topktitles_present"] = 1.0

        corpus_meta = self._corpus_meta_dict()
        out_payload: Dict[str, Any] = {"corpus": corpus_meta}
        export_k = min(int(self.k), max(int(self.max_export_k), 0))
        features_flat["tp_topktitles_export_k_used"] = float(int(export_k))
        if int(self.k) > int(export_k):
            features_flat["tp_topktitles_export_k_truncated_flag"] = 1.0

        if self.export_topk_mode not in ("none", "ids_only", "ids_and_scores"):
            raise RuntimeError("topk_similar_titles_extractor: invalid export_topk_mode (expected none|ids_only|ids_and_scores)")

        if self.export_topk_mode == "ids_only":
            out_payload.update({"topk_similar_ids": top_ids[:export_k]})
        elif self.export_topk_mode == "ids_and_scores":
            out_payload.update({"topk_similar_ids": top_ids[:export_k], "topk_similar_scores": top_scores[:export_k]})

        return _finish(features_flat, out_payload)
