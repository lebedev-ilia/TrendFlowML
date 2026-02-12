from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.path_utils import default_artifacts_dir


class EmbeddingSourceIdExtractor(BaseExtractor):
    """
    A-policy: primary embedding source identifier.

    - Selects a primary embedding deterministically from doc.tp_artifacts (canonical + legacy).
    - Computes a portable vector_id from float32 values (no path dependency).
    - Privacy-safe: no absolute paths in result.
    - Output: features_flat (scalars) + embedding_source_id (strings/meta).
    """

    VERSION = "1.2.0"

    def __init__(
        self,
        vector_store_uri: str = "faiss://semantic_titles_v1",
        model_version: str = "unknown",
        primary_source_policy: str = "transcript_first",
        strict_missing_primary: bool = True,
        artifacts_dir: str | None = None,
    ) -> None:
        self.vector_store_uri = vector_store_uri
        self.model_version = model_version
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

    @staticmethod
    def _safe_join_artifacts_dir(base_dir: Path, relpath: str) -> Path:
        base = base_dir.expanduser().resolve()
        cand = (base / relpath).resolve()
        if not (cand == base or base in cand.parents):
            raise RuntimeError("EmbeddingSourceIdExtractor: embedding_relpath escapes artifacts_dir")
        return cand

    @staticmethod
    def _vector_id_from_values(vec_f32: "Any") -> str:
        """
        Portable deterministic ID computed from vector float32 values (not from filesystem path, not from .npy bytes).
        Output: 24 hex chars (first 12 bytes of sha256).
        """
        import numpy as np

        v = np.asarray(vec_f32, dtype=np.float32).reshape(-1)
        # ensure stable little-endian bytes across platforms
        if v.dtype.byteorder not in ("<", "="):
            v = v.byteswap().newbyteorder("<")
        import hashlib

        h = hashlib.sha256(v.tobytes(order="C")).hexdigest()
        return h[:24]

    def _pick_primary_relpath(self, doc: Any) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """
        Deterministic selection based on in-memory registry produced by upstream extractors.
        Default priority: transcript combined mean -> title -> description (transcript_first).
        """
        tp = getattr(doc, "tp_artifacts", None)
        if not isinstance(tp, dict):
            return None, None, None
        # title/description embedders write tp_artifacts.embeddings.{title,description};
        # transcript aggregator writes:
        # - canonical: tp_artifacts.transcripts.{combined,whisper,youtube_auto}.agg_mean_relpath
        # - legacy:    tp_artifacts.transcript_aggregates.{...}.agg_mean_relpath
        emb = tp.get("embeddings") if isinstance(tp.get("embeddings"), dict) else {}
        ta = tp.get("transcript_aggregates") if isinstance(tp.get("transcript_aggregates"), dict) else {}
        tr = tp.get("transcripts") if isinstance(tp.get("transcripts"), dict) else {}

        def pick_transcript_mean() -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
            # canonical first
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
            # legacy fallback
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

        # default: transcript_first
        for fn in (pick_transcript_mean, pick_title, pick_description):
            rel, src, meta = fn()
            if rel:
                return rel, src, meta
        return None, None, None

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        rel, primary_source, picked_meta = self._pick_primary_relpath(doc)
        if not rel:
            if self.strict_missing_primary:
                raise RuntimeError("EmbeddingSourceIdExtractor: primary embedding not found (missing upstream embeddings/transcript_aggregates)")
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
                    "features_flat": {
                        "tp_embid_present": 0.0,
                        "tp_embid_policy_transcript_first": 1.0 if (self.primary_source_policy == "transcript_first") else 0.0,
                        "tp_embid_policy_title_first": 1.0 if (self.primary_source_policy == "title_first") else 0.0,
                        "tp_embid_policy_description_first": 1.0 if (self.primary_source_policy == "description_first") else 0.0,
                        "tp_embid_policy_title_only": 1.0 if (self.primary_source_policy == "title_only") else 0.0,
                        "tp_embid_policy_transcript_only": 1.0 if (self.primary_source_policy == "transcript_only") else 0.0,
                    },
                    "embedding_source_id": {"error": "no_embedding_found"},
                },
                "error": None,
            }

        p = self._safe_join_artifacts_dir(self.artifacts_dir, rel)
        try:
            import numpy as np
            vec = np.load(str(p))
        except Exception as e:
            raise RuntimeError(f"EmbeddingSourceIdExtractor: failed to load embedding for id: {e}") from e

        vector_id = self._vector_id_from_values(vec)
        # model metadata: prefer per-embedding meta from tp_artifacts, else config
        model_name = None
        weights_digest = None
        if isinstance(picked_meta, dict):
            mn = picked_meta.get("model_name")
            if isinstance(mn, str) and mn:
                model_name = mn
            wd = picked_meta.get("weights_digest")
            if isinstance(wd, str) and wd:
                weights_digest = wd
        model_version = model_name or self.model_version

        features_flat = {
            "tp_embid_present": 1.0,
            "tp_embid_policy_transcript_first": 1.0 if (self.primary_source_policy == "transcript_first") else 0.0,
            "tp_embid_policy_title_first": 1.0 if (self.primary_source_policy == "title_first") else 0.0,
            "tp_embid_policy_description_first": 1.0 if (self.primary_source_policy == "description_first") else 0.0,
            "tp_embid_policy_title_only": 1.0 if (self.primary_source_policy == "title_only") else 0.0,
            "tp_embid_policy_transcript_only": 1.0 if (self.primary_source_policy == "transcript_only") else 0.0,
            "tp_embid_primary_is_transcript": 1.0 if (isinstance(primary_source, str) and primary_source.startswith("transcript_")) else 0.0,
            "tp_embid_primary_is_title": 1.0 if (primary_source == "title") else 0.0,
            "tp_embid_primary_is_description": 1.0 if (primary_source == "description") else 0.0,
        }

        out = {
            "features_flat": features_flat,
            "embedding_source_id": {
                "vector_id": vector_id,
                "vector_store_uri": self.vector_store_uri,
                "model_version": model_version,
                "weights_digest": str(weights_digest or "unknown"),
                "embedding_relpath": rel,
                "primary_source": primary_source,
            },
        }

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
            "result": out,
            "error": None,
        }


