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


class TopKSimilarCorpusTitlesExtractor(BaseExtractor):
    VERSION = "1.2.0"

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

    @staticmethod
    def _safe_join_artifacts_dir(base_dir: Path, relpath: str) -> Path:
        base = base_dir.expanduser().resolve()
        cand = (base / relpath).resolve()
        if not (cand == base or base in cand.parents):
            raise RuntimeError("topk_similar_titles_extractor: relpath escapes artifacts_dir")
        return cand

    def _cache_key(self) -> str:
        # weights_digest is only known after dp_models resolve; set in _load_corpus_from_dp_models
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
        # refresh LRU
        _INDEX_CACHE.move_to_end(key)
        return index_or_corpus, ids, dim, corpus_size, corpus_weights_digest, corpus_version, id_kind

    def _cache_put(self, key: str, index_or_corpus: Any, ids: List[Any], dim: int, corpus_size: int, weights_digest: str, version: str, id_kind: str) -> None:
        if not self.cache_enabled:
            return
        now = time.time()
        _INDEX_CACHE[key] = (now, index_or_corpus, ids, int(dim), int(corpus_size), str(weights_digest), str(version), str(id_kind))
        _INDEX_CACHE.move_to_end(key)
        # evict
        while self.cache_max_entries > 0 and len(_INDEX_CACHE) > int(self.cache_max_entries):
            try:
                _INDEX_CACHE.popitem(last=False)
            except Exception:
                break

    def _load_corpus_from_dp_models(self, corpus_spec_name: str) -> None:
        """
        Strictly loads corpus assets via dp_models (offline + fail-fast).
        Corpus must be a dp_models spec with local_artifacts for:
        - embeddings.npy (float32, shape [N, D])
        - ids.json (list, len N)
        """
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

        # Index build
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
            # numpy backend: keep normalized corpus matrix in memory
            self._index = emb
            self._cache_put(cache_key, self._index, self._ids, int(self._dim), int(self._corpus_size), self._corpus_weights_digest, self._corpus_version, self._id_kind)
            return

        # cosine similarity via inner product on L2-normalized vectors
        index = faiss.IndexHNSWFlat(self._dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)  # type: ignore[attr-defined]
        index.hnsw.efConstruction = int(self.hnsw_ef_construction)
        index.hnsw.efSearch = int(self.hnsw_ef_search)
        index.add(emb.astype(np.float32))
        self._index = index
        self._cache_put(cache_key, self._index, self._ids, int(self._dim), int(self._corpus_size), self._corpus_weights_digest, self._corpus_version, self._id_kind)

    def _search_np(self, query: np.ndarray, corpus: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        # corpus is already L2-normalized at load time; do not renormalize per query.
        sims = (_l2n(query, axis=1) @ corpus.T)
        k = min(k, sims.shape[1])
        idx = np.argsort(-sims, axis=1)[:, :k]
        scr = np.take_along_axis(sims, idx, axis=1)
        return idx, scr

    def extract(self, doc: Any) -> Dict[str, Any]:
        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        def _stable_template() -> Dict[str, float]:
            return {
                "tp_topktitles_present": 0.0,
                "tp_topktitles_disabled_by_policy": 0.0,
                "tp_topktitles_enabled": float(bool(self.enabled)),
                "tp_topktitles_require_title_embedding_enabled": float(bool(self.require_title_embedding)),
                "tp_topktitles_k": float(int(self.k)),
                "tp_topktitles_corpus_size": float(int(self._corpus_size)),
                "tp_topktitles_dim": float(int(self._dim or 0)),
                "tp_topktitles_backend_faiss": 1.0 if (faiss is not None and hasattr(self._index, "search")) else 0.0,
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
                "tp_topktitles_export_k_used": float("nan"),
                "tp_topktitles_export_k_truncated_flag": 0.0,
                "tp_topktitles_unsafe_relpath_flag": 0.0,
                "tp_topktitles_dim_mismatch_flag": 0.0,
                "tp_topktitles_zero_norm_flag": 0.0,
                "tp_topktitles_nan_inf_flag": 0.0,
                "tp_topktitles_top1_score": float("nan"),
                "tp_topktitles_topk_mean_score": float("nan"),
            }

        features_flat = _stable_template()
        error: Optional[str] = None

        if not self.enabled:
            features_flat["tp_topktitles_disabled_by_policy"] = 1.0
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
                "result": {"features_flat": features_flat, "topk_similar_corpus_titles": {"corpus": {}}},
                "error": error,
            }

        if self._index is None or not self._ids or self._dim is None:
            raise RuntimeError("topk_similar_titles_extractor: corpus not loaded (this should have failed in __init__)")

        # deterministic: read title embedding from in-memory registry filled by TitleEmbedder
        tp = getattr(doc, "tp_artifacts", None)
        emb = tp.get("embeddings") if isinstance(tp, dict) else None
        title_rel = emb.get("title", {}).get("relpath") if isinstance(emb, dict) and isinstance(emb.get("title"), dict) else None
        if not isinstance(title_rel, str) or not title_rel:
            if self.require_title_embedding:
                raise RuntimeError("topk_similar_titles_extractor: missing title embedding relpath in doc.tp_artifacts.embeddings.title")
            # valid empty
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
                "result": {"features_flat": features_flat, "topk_similar_corpus_titles": {"corpus": {}}},
                "error": error,
            }

        try:
            p = self._safe_join_artifacts_dir(self.artifacts_dir, title_rel)
        except Exception:
            features_flat["tp_topktitles_unsafe_relpath_flag"] = 1.0
            if self.require_title_embedding:
                raise RuntimeError("topk_similar_titles_extractor: unsafe title embedding relpath")
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
                "result": {"features_flat": features_flat, "topk_similar_corpus_titles": {"corpus": {}}},
                "error": error,
            }

        if not p.exists():
            if self.require_title_embedding:
                raise RuntimeError("topk_similar_titles_extractor: title embedding file not found in per-run artifacts")
            # valid empty
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
                "result": {"features_flat": features_flat, "topk_similar_corpus_titles": {"corpus": {}}},
                "error": error,
            }

        title = np.load(p).astype(np.float32).reshape(1, -1)
        if not np.isfinite(title).all():
            features_flat["tp_topktitles_nan_inf_flag"] = 1.0
            if self.require_title_embedding:
                raise RuntimeError("topk_similar_titles_extractor: title embedding contains NaN/inf")
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
                "result": {"features_flat": features_flat, "topk_similar_corpus_titles": {"corpus": {}}},
                "error": error,
            }
        if int(title.shape[1]) != int(self._dim):
            features_flat["tp_topktitles_dim_mismatch_flag"] = 1.0
            if self.require_title_embedding:
                raise RuntimeError(
                    f"topk_similar_titles_extractor: embedding dim mismatch: title_dim={int(title.shape[1])} corpus_dim={int(self._dim)}"
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
                "result": {"features_flat": features_flat, "topk_similar_corpus_titles": {"corpus": {}}},
                "error": error,
            }

        if float(np.linalg.norm(title)) <= 0.0:
            features_flat["tp_topktitles_zero_norm_flag"] = 1.0
            if self.require_title_embedding:
                raise RuntimeError("topk_similar_titles_extractor: title embedding has zero norm")
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
                "result": {"features_flat": features_flat, "topk_similar_corpus_titles": {"corpus": {}}},
                "error": error,
            }

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

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        corpus_meta = {
            "corpus_spec_name": self._corpus_spec_name,
            "corpus_version": self._corpus_version,
            "corpus_weights_digest": self._corpus_weights_digest,
            "id_kind": self._id_kind,
            "corpus_size": int(self._corpus_size),
            "dim": int(self._dim or 0),
            "backend": "faiss_hnsw_ip" if (faiss is not None and isinstance(self._index, faiss.Index)) else "numpy_cosine",
            "hnsw": {
                "m": int(self.hnsw_m),
                "ef_construction": int(self.hnsw_ef_construction),
                "ef_search": int(self.hnsw_ef_search),
            },
        }

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
            "result": {"features_flat": features_flat, "topk_similar_corpus_titles": out_payload},
            "error": error,
        }


