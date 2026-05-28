from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.path_utils import default_artifacts_dir

_FEATURES_FLAT_KEYS: Tuple[str, ...] = (
    "tp_embid_present",
    "tp_embid_strict_missing_primary_enabled",
    "tp_embid_policy_transcript_first",
    "tp_embid_policy_title_first",
    "tp_embid_policy_description_first",
    "tp_embid_policy_title_only",
    "tp_embid_policy_transcript_only",
    "tp_embid_primary_is_transcript",
    "tp_embid_primary_is_title",
    "tp_embid_primary_is_description",
    "tp_embid_unsafe_relpath_flag",
    "tp_embid_primary_embed_missing_flag",
    "tp_embid_nan_inf_flag",
)


class EmbeddingSourceIdExtractor(BaseExtractor):
    """
    A-policy: primary embedding source identifier.

    - Selects a primary embedding deterministically from doc.tp_artifacts (canonical + legacy).
    - Computes a portable vector_id from float32 values (no path dependency).
    - Privacy-safe: no absolute paths in result.
    - Output: fixed features_flat (scalars) + embedding_source_id (strings/meta).
    """

    VERSION = "1.3.0"

    def __init__(
        self,
        vector_store_uri: str = "faiss://semantic_titles_v1",
        model_version: str = "unknown",
        primary_source_policy: str = "transcript_first",
        strict_missing_primary: bool = True,
        artifacts_dir: str | None = None,
    ) -> None:
        init_sys_before = system_snapshot()
        init_mem_before = process_memory_bytes()

        self.vector_store_uri = vector_store_uri
        self.model_version = str(model_version or "unknown")
        self.primary_source_policy = str(primary_source_policy or "transcript_first").strip().lower()
        self.strict_missing_primary = bool(strict_missing_primary)
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        if self.primary_source_policy not in (
            "transcript_first",
            "title_first",
            "description_first",
            "title_only",
            "transcript_only",
        ):
            raise RuntimeError(
                "EmbeddingSourceIdExtractor: primary_source_policy must be one of: "
                "transcript_first|title_first|description_first|title_only|transcript_only"
            )

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
            raise RuntimeError("EmbeddingSourceIdExtractor: embedding_relpath escapes artifacts_dir")
        return cand

    @staticmethod
    def _vector_id_from_values(vec_f32: Any) -> str:
        """Portable deterministic ID: first 24 hex of sha256 over C-order little-endian float32 bytes."""
        import hashlib

        v = np.asarray(vec_f32, dtype=np.float32).reshape(-1)
        if v.dtype.byteorder not in ("<", "="):
            v = v.byteswap().newbyteorder("<")
        h = hashlib.sha256(v.tobytes(order="C")).hexdigest()
        return h[:24]

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
                raise KeyError(f"EmbeddingSourceIdExtractor: missing features_flat key {k!r}")
            v = values[k]
            if v is None:
                out[k] = nan
            elif isinstance(v, (bool, np.bool_)):
                out[k] = float(bool(v))
            else:
                out[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else nan
        return out

    def _policy_block(self) -> Dict[str, float]:
        pol = self.primary_source_policy
        return {
            "tp_embid_policy_transcript_first": 1.0 if pol == "transcript_first" else 0.0,
            "tp_embid_policy_title_first": 1.0 if pol == "title_first" else 0.0,
            "tp_embid_policy_description_first": 1.0 if pol == "description_first" else 0.0,
            "tp_embid_policy_title_only": 1.0 if pol == "title_only" else 0.0,
            "tp_embid_policy_transcript_only": 1.0 if pol == "transcript_only" else 0.0,
        }

    def _base_features_flat(self) -> Dict[str, Any]:
        ff: Dict[str, Any] = {
            "tp_embid_present": 0.0,
            "tp_embid_strict_missing_primary_enabled": float(bool(self.strict_missing_primary)),
            **self._policy_block(),
            "tp_embid_primary_is_transcript": 0.0,
            "tp_embid_primary_is_title": 0.0,
            "tp_embid_primary_is_description": 0.0,
            "tp_embid_unsafe_relpath_flag": 0.0,
            "tp_embid_primary_embed_missing_flag": 0.0,
            "tp_embid_nan_inf_flag": 0.0,
        }
        return ff

    def _set_primary_onehots(self, ff: Dict[str, Any], primary_source: Optional[str]) -> None:
        ps = primary_source or ""
        ff["tp_embid_primary_is_transcript"] = float(isinstance(ps, str) and ps.startswith("transcript_"))
        ff["tp_embid_primary_is_title"] = 1.0 if ps == "title" else 0.0
        ff["tp_embid_primary_is_description"] = 1.0 if ps == "description" else 0.0

    def _build_return(
        self,
        *,
        features_flat: Dict[str, Any],
        embedding_source_id: Dict[str, Any],
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
                "embedding_source_id": embedding_source_id,
            },
            "error": error,
        }

    def _pick_primary_relpath(self, doc: Any) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        tp = getattr(doc, "tp_artifacts", None)
        if not isinstance(tp, dict):
            return None, None, None
        emb = tp.get("embeddings") if isinstance(tp.get("embeddings"), dict) else {}
        ta = tp.get("transcript_aggregates") if isinstance(tp.get("transcript_aggregates"), dict) else {}
        tr = tp.get("transcripts") if isinstance(tp.get("transcripts"), dict) else {}

        def pick_transcript_mean() -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
            if isinstance(tr, dict):
                c = tr.get("combined")
                if isinstance(c, dict) and isinstance(c.get("agg_mean_relpath"), str) and c.get("agg_mean_relpath"):
                    return str(c.get("agg_mean_relpath")), "transcript_combined_mean", None
                w = tr.get("whisper")
                if isinstance(w, dict) and isinstance(w.get("agg_mean_relpath"), str) and w.get("agg_mean_relpath"):
                    return str(w.get("agg_mean_relpath")), "transcript_whisper_mean", None
                y = tr.get("youtube_auto")
                if isinstance(y, dict) and isinstance(y.get("agg_mean_relpath"), str) and y.get("agg_mean_relpath"):
                    return str(y.get("agg_mean_relpath")), "transcript_youtube_auto_mean", None
            if isinstance(ta, dict):
                c = ta.get("combined")
                if isinstance(c, dict) and isinstance(c.get("agg_mean_relpath"), str) and c.get("agg_mean_relpath"):
                    return str(c.get("agg_mean_relpath")), "transcript_combined_mean", None
                w = ta.get("whisper")
                if isinstance(w, dict) and isinstance(w.get("agg_mean_relpath"), str) and w.get("agg_mean_relpath"):
                    return str(w.get("agg_mean_relpath")), "transcript_whisper_mean", None
                y = ta.get("youtube_auto")
                if isinstance(y, dict) and isinstance(y.get("agg_mean_relpath"), str) and y.get("agg_mean_relpath"):
                    return str(y.get("agg_mean_relpath")), "transcript_youtube_auto_mean", None
            return None, None, None

        def pick_title() -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
            if not isinstance(emb, dict):
                return None, None, None
            t = emb.get("title")
            if isinstance(t, dict) and isinstance(t.get("relpath"), str) and t.get("relpath"):
                return str(t.get("relpath")), "title", t
            return None, None, None

        def pick_description() -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
            if not isinstance(emb, dict):
                return None, None, None
            d = emb.get("description")
            if isinstance(d, dict) and isinstance(d.get("relpath"), str) and d.get("relpath"):
                return str(d.get("relpath")), "description", d
            return None, None, None

        if self.primary_source_policy == "title_only":
            return pick_title()
        if self.primary_source_policy == "transcript_only":
            return pick_transcript_mean()
        if self.primary_source_policy == "description_first":
            for fn in (pick_description, pick_transcript_mean, pick_title):
                rel, src, meta = fn()
                if rel:
                    return rel, src, meta
            return None, None, None
        if self.primary_source_policy == "title_first":
            for fn in (pick_title, pick_transcript_mean, pick_description):
                rel, src, meta = fn()
                if rel:
                    return rel, src, meta
            return None, None, None

        for fn in (pick_transcript_mean, pick_title, pick_description):
            rel, src, meta = fn()
            if rel:
                return rel, src, meta
        return None, None, None

    def _meta_from_picked(self, picked_meta: Optional[Dict[str, Any]]) -> Tuple[Optional[str], str, str]:
        """model_name optional; model_version string; weights_digest string ('unknown' if missing)."""
        mv = str(self.model_version)
        mn: Optional[str] = None
        wd: Optional[str] = None
        if isinstance(picked_meta, dict):
            x = picked_meta.get("model_name")
            if isinstance(x, str) and x.strip():
                mn = x.strip()
            xv = picked_meta.get("model_version")
            if isinstance(xv, str) and xv.strip():
                mv = xv.strip()
            xw = picked_meta.get("weights_digest")
            if isinstance(xw, str) and xw.strip():
                wd = xw.strip()
        return mn, mv, (wd if wd else "unknown")

    def extract(self, doc: Any) -> Dict[str, Any]:
        t0 = time.perf_counter()
        mem_before = process_memory_bytes()
        error: Optional[str] = None

        def _finish(ff: Dict[str, Any], esid: Dict[str, Any]) -> Dict[str, Any]:
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return self._build_return(
                features_flat=ff,
                embedding_source_id=esid,
                sys_after=sys_after,
                mem_before=mem_before,
                mem_after=mem_after,
                total_s=total_s,
                error=error,
            )

        ff = self._base_features_flat()

        rel, primary_source, picked_meta = self._pick_primary_relpath(doc)
        if not rel:
            if self.strict_missing_primary:
                raise RuntimeError(
                    "EmbeddingSourceIdExtractor: primary embedding not found (missing upstream embeddings/transcript_aggregates)"
                )
            self._set_primary_onehots(ff, None)
            return _finish(ff, {"error": "no_embedding_found"})

        self._set_primary_onehots(ff, primary_source)

        try:
            p = self._safe_join_artifacts_dir(self.artifacts_dir, rel)
        except Exception:
            ff["tp_embid_unsafe_relpath_flag"] = 1.0
            if self.strict_missing_primary:
                raise RuntimeError("EmbeddingSourceIdExtractor: unsafe embedding relpath") from None
            return _finish(ff, {"error": "unsafe_relpath"})

        if not p.exists():
            ff["tp_embid_primary_embed_missing_flag"] = 1.0
            if self.strict_missing_primary:
                raise RuntimeError("EmbeddingSourceIdExtractor: embedding file not found in per-run artifacts")
            return _finish(ff, {"error": "embedding_file_missing"})

        try:
            vec = np.load(str(p))
        except Exception as e:
            ff["tp_embid_primary_embed_missing_flag"] = 1.0
            if self.strict_missing_primary:
                raise RuntimeError(f"EmbeddingSourceIdExtractor: failed to load embedding for id: {e}") from e
            return _finish(ff, {"error": "embedding_load_failed"})

        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if int(vec.size) <= 0:
            ff["tp_embid_primary_embed_missing_flag"] = 1.0
            if self.strict_missing_primary:
                raise RuntimeError("EmbeddingSourceIdExtractor: empty embedding vector")
            return _finish(ff, {"error": "embedding_empty"})

        if not np.isfinite(vec).all():
            ff["tp_embid_nan_inf_flag"] = 1.0
            if self.strict_missing_primary:
                raise RuntimeError("EmbeddingSourceIdExtractor: embedding contains NaN/inf")
            return _finish(ff, {"error": "embedding_non_finite"})

        vector_id = self._vector_id_from_values(vec)
        model_name, model_version, weights_digest = self._meta_from_picked(
            picked_meta if isinstance(picked_meta, dict) else None
        )

        ff["tp_embid_present"] = 1.0

        esid = {
            "vector_id": vector_id,
            "vector_store_uri": str(self.vector_store_uri),
            "model_name": model_name,
            "model_version": model_version,
            "weights_digest": weights_digest,
            "embedding_relpath": rel,
            "primary_source": primary_source,
        }
        return _finish(ff, esid)

