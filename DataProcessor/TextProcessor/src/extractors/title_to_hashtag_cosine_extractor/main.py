from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.path_utils import default_artifacts_dir


class TitleToHashtagCosineExtractor(BaseExtractor):
    VERSION = "1.1.0"

    def __init__(
        self,
        artifacts_dir: str | None = None,
        *,
        enabled: bool = True,
        require_title_embedding: bool = False,
        require_hashtag_embedding: bool = False,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.enabled = bool(enabled)
        self.require_title_embedding = bool(require_title_embedding)
        self.require_hashtag_embedding = bool(require_hashtag_embedding)

    @staticmethod
    def _safe_join_artifacts_dir(base_dir: Path, relpath: str) -> Path:
        base = base_dir.expanduser().resolve()
        cand = (base / relpath).resolve()
        if not (cand == base or base in cand.parents):
            raise RuntimeError("title_to_hashtag_cosine_extractor: relpath escapes artifacts_dir")
        return cand

    @staticmethod
    def _l2n(v: np.ndarray) -> tuple[np.ndarray, float]:
        n = float(np.linalg.norm(v))
        if n > 0:
            return (v / n), n
        return v, n

    def _load_vector(self, relpath: str) -> Optional[np.ndarray]:
        try:
            p = self._safe_join_artifacts_dir(self.artifacts_dir, relpath)
        except Exception:
            return None
        if not p.exists():
            return None
        try:
            v = np.load(p)
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            return v
        except Exception:
            return None

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        def _stable_template() -> Dict[str, float]:
            return {
                # canonical prefix (new)
                "tp_titlehashcos_present": 0.0,
                "tp_titlehashcos_cosine": float("nan"),
                "tp_titlehashcos_disabled_by_policy": 0.0,
                "tp_titlehashcos_enabled": float(bool(self.enabled)),
                "tp_titlehashcos_require_title_embedding_enabled": float(bool(self.require_title_embedding)),
                "tp_titlehashcos_require_hashtag_embedding_enabled": float(bool(self.require_hashtag_embedding)),
                # inputs
                "tp_titlehashcos_title_present": 0.0,
                "tp_titlehashcos_hashtag_present": 0.0,
                # flags
                "tp_titlehashcos_unsafe_relpath_flag": 0.0,
                "tp_titlehashcos_dim_mismatch_flag": 0.0,
                "tp_titlehashcos_zero_norm_flag": 0.0,
            }

        features_flat = _stable_template()

        if not self.enabled:
            features_flat["tp_titlehashcos_disabled_by_policy"] = 1.0
            # legacy aliases (always present)
            features_flat.update(
                {
                    "tp_title_hashtag_cosine_present": 0.0,
                    "tp_title_hashtag_cosine": float("nan"),
                }
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

        tp = getattr(doc, "tp_artifacts", None)
        emb = tp.get("embeddings") if isinstance(tp, dict) else None
        title_rel = emb.get("title", {}).get("relpath") if isinstance(emb, dict) and isinstance(emb.get("title"), dict) else None
        hash_rel = emb.get("hashtag", {}).get("relpath") if isinstance(emb, dict) and isinstance(emb.get("hashtag"), dict) else None

        unsafe_relpath_flag = 0.0
        title_vec = None
        hash_vec = None

        if isinstance(title_rel, str) and title_rel:
            try:
                title_vec = self._load_vector(title_rel)
            except Exception:
                title_vec = None
        if isinstance(hash_rel, str) and hash_rel:
            try:
                hash_vec = self._load_vector(hash_rel)
            except Exception:
                hash_vec = None

        features_flat["tp_titlehashcos_title_present"] = float(title_vec is not None)
        features_flat["tp_titlehashcos_hashtag_present"] = float(hash_vec is not None)

        # If relpath was provided but vector couldn't be loaded, treat it as unsafe/invalid.
        # (We can't disambiguate IO errors vs traversal cleanly without deeper exception typing.)
        if (isinstance(title_rel, str) and title_rel and title_vec is None) or (isinstance(hash_rel, str) and hash_rel and hash_vec is None):
            unsafe_relpath_flag = 1.0
        features_flat["tp_titlehashcos_unsafe_relpath_flag"] = float(unsafe_relpath_flag)

        if title_vec is None and self.require_title_embedding:
            raise RuntimeError("TitleToHashtagCosineExtractor: required title embedding missing")
        if hash_vec is None and self.require_hashtag_embedding:
            raise RuntimeError("TitleToHashtagCosineExtractor: required hashtag embedding missing")

        if title_vec is None or hash_vec is None:
            # valid empty
            features_flat.update(
                {
                    "tp_title_hashtag_cosine_present": 0.0,
                    "tp_title_hashtag_cosine": float("nan"),
                }
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

        if int(title_vec.size) <= 0 or int(hash_vec.size) <= 0 or int(title_vec.size) != int(hash_vec.size):
            features_flat["tp_titlehashcos_dim_mismatch_flag"] = 1.0
            features_flat.update(
                {
                    "tp_title_hashtag_cosine_present": 0.0,
                    "tp_title_hashtag_cosine": float("nan"),
                }
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

        a, an = self._l2n(title_vec)
        b, bn = self._l2n(hash_vec)
        if an <= 0.0 or bn <= 0.0:
            features_flat["tp_titlehashcos_zero_norm_flag"] = 1.0
            features_flat.update(
                {
                    "tp_title_hashtag_cosine_present": 0.0,
                    "tp_title_hashtag_cosine": float("nan"),
                }
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

        sim = float(np.dot(a, b))
        features_flat["tp_titlehashcos_present"] = 1.0
        features_flat["tp_titlehashcos_cosine"] = float(sim)

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        # legacy aliases (always present)
        features_flat.update(
            {
                "tp_title_hashtag_cosine_present": 1.0,
                "tp_title_hashtag_cosine": float(sim),
            }
        )

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


