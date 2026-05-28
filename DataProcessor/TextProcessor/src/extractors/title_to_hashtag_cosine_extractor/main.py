from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.path_utils import default_artifacts_dir


def _build_features_flat_keys() -> Tuple[str, ...]:
    return (
        "tp_titlehashcos_present",
        "tp_titlehashcos_cosine",
        "tp_titlehashcos_require_title_embedding_enabled",
        "tp_titlehashcos_require_hashtag_embedding_enabled",
        "tp_titlehashcos_title_present",
        "tp_titlehashcos_hashtag_present",
        "tp_titlehashcos_unsafe_relpath_flag",
        "tp_titlehashcos_title_embed_missing_flag",
        "tp_titlehashcos_hashtag_embed_missing_flag",
        "tp_titlehashcos_dim_mismatch_flag",
        "tp_titlehashcos_zero_norm_flag",
    )


_FEATURES_FLAT_KEYS = _build_features_flat_keys()


class TitleToHashtagCosineExtractor(BaseExtractor):
    """
    Cosine similarity between L2-normalized title and hashtag embeddings from tp_artifacts relpaths.
    Audit v3: fixed features_flat keys; unsafe vs missing-file flags; no in-extractor enabled gate.
    """

    VERSION = "1.2.0"

    def __init__(
        self,
        artifacts_dir: str | None = None,
        *,
        require_title_embedding: bool = False,
        require_hashtag_embedding: bool = False,
        **_: Any,
    ) -> None:
        init_sys_before = system_snapshot()
        init_mem_before = process_memory_bytes()

        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.require_title_embedding = bool(require_title_embedding)
        self.require_hashtag_embedding = bool(require_hashtag_embedding)

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
            raise RuntimeError("title_to_hashtag_cosine_extractor: relpath escapes artifacts_dir")
        return cand

    def _try_load_embedding(self, relpath: str) -> Tuple[Optional[np.ndarray], str]:
        """
        Returns (vector_or_none, status).
        status: ok | unsafe | missing | bad_file
        """
        try:
            p = self._safe_join_artifacts_dir(self.artifacts_dir, relpath)
        except Exception:
            return None, "unsafe"
        if not p.exists():
            return None, "missing"
        try:
            v = np.load(p)
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            if int(v.size) <= 0:
                return None, "bad_file"
            return v, "ok"
        except Exception:
            return None, "bad_file"

    @staticmethod
    def _l2n(v: np.ndarray) -> tuple[np.ndarray, float]:
        n = float(np.linalg.norm(v))
        if n > 0:
            return (v / n), n
        return v, n

    @staticmethod
    def _pack_features_flat(values: Dict[str, Any]) -> Dict[str, Any]:
        nan = float("nan")
        out: Dict[str, Any] = {}
        for k in _FEATURES_FLAT_KEYS:
            if k not in values:
                raise KeyError(f"TitleToHashtagCosineExtractor: missing features_flat key {k!r}")
            v = values[k]
            if v is None:
                out[k] = nan
            elif isinstance(v, (bool, np.bool_)):
                out[k] = float(bool(v))
            else:
                out[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else nan
        return out

    def _base_template(self) -> Dict[str, Any]:
        nan = float("nan")
        return {
            "tp_titlehashcos_present": 0.0,
            "tp_titlehashcos_cosine": nan,
            "tp_titlehashcos_require_title_embedding_enabled": float(bool(self.require_title_embedding)),
            "tp_titlehashcos_require_hashtag_embedding_enabled": float(bool(self.require_hashtag_embedding)),
            "tp_titlehashcos_title_present": 0.0,
            "tp_titlehashcos_hashtag_present": 0.0,
            "tp_titlehashcos_unsafe_relpath_flag": 0.0,
            "tp_titlehashcos_title_embed_missing_flag": 0.0,
            "tp_titlehashcos_hashtag_embed_missing_flag": 0.0,
            "tp_titlehashcos_dim_mismatch_flag": 0.0,
            "tp_titlehashcos_zero_norm_flag": 0.0,
        }

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

        tp = getattr(doc, "tp_artifacts", None)
        emb = tp.get("embeddings") if isinstance(tp, dict) else None
        title_rel = emb.get("title", {}).get("relpath") if isinstance(emb, dict) and isinstance(emb.get("title"), dict) else None
        hash_rel = emb.get("hashtag", {}).get("relpath") if isinstance(emb, dict) and isinstance(emb.get("hashtag"), dict) else None

        ff = self._base_template()

        tstat = "no_relpath"
        hstat = "no_relpath"
        title_vec: Optional[np.ndarray] = None
        hash_vec: Optional[np.ndarray] = None

        if isinstance(title_rel, str) and title_rel:
            title_vec, tstat = self._try_load_embedding(title_rel)
        if isinstance(hash_rel, str) and hash_rel:
            hash_vec, hstat = self._try_load_embedding(hash_rel)

        ff["tp_titlehashcos_title_present"] = float(title_vec is not None)
        ff["tp_titlehashcos_hashtag_present"] = float(hash_vec is not None)
        if tstat == "unsafe" or hstat == "unsafe":
            ff["tp_titlehashcos_unsafe_relpath_flag"] = 1.0
        if tstat in ("missing", "bad_file"):
            ff["tp_titlehashcos_title_embed_missing_flag"] = 1.0
        if hstat in ("missing", "bad_file"):
            ff["tp_titlehashcos_hashtag_embed_missing_flag"] = 1.0

        if title_vec is None and self.require_title_embedding:
            raise RuntimeError("TitleToHashtagCosineExtractor: required title embedding missing")
        if hash_vec is None and self.require_hashtag_embedding:
            raise RuntimeError("TitleToHashtagCosineExtractor: required hashtag embedding missing")

        def _finish(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            if extra:
                ff.update(extra)
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return self._build_return(
                features_flat=ff,
                sys_after=sys_after,
                mem_before=mem_before,
                mem_after=mem_after,
                total_s=total_s,
            )

        if title_vec is None or hash_vec is None:
            return _finish()

        if int(title_vec.size) != int(hash_vec.size):
            ff["tp_titlehashcos_dim_mismatch_flag"] = 1.0
            return _finish()

        a, an = self._l2n(title_vec)
        b, bn = self._l2n(hash_vec)
        if an <= 0.0 or bn <= 0.0:
            ff["tp_titlehashcos_zero_norm_flag"] = 1.0
            return _finish()

        sim = float(np.dot(a, b))
        ff["tp_titlehashcos_present"] = 1.0
        ff["tp_titlehashcos_cosine"] = float(sim)
        return _finish()
