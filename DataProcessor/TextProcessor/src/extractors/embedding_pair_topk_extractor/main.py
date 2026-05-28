from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.path_utils import default_artifacts_dir

# Machine schema (Audit v3): fixed export slots; config top_k_slots may be clamped.
SCHEMA_MAX_TOP_K_SLOTS = 8

# Stable key order for features_flat — must match embedding_pair_topk_extractor_output_v1.json
_FEATURES_FLAT_KEYS: Tuple[str, ...] = (
    "tp_embpair_present",
    "tp_embpair_enabled",
    "tp_embpair_disabled_by_policy",
    "tp_embpair_title_present",
    "tp_embpair_desc_present",
    "tp_embpair_transcript_chunks_present",
    "tp_embpair_used_legacy_key_flag",
    "tp_embpair_dim_mismatch_flag",
    "tp_embpair_unsafe_relpath_flag",
    "tp_embpair_nan_inf_flag",
    "tp_embpair_zero_norm_flag",
    "tp_embpair_top_k",
    "tp_embpair_top_k_slots",
    "tp_embpair_schema_slots_max",
    "tp_embpair_top_k_slots_requested",
    "tp_embpair_top_k_slots_clamped",
    "tp_embpair_title_desc_cosine",
    "tp_embpair_title_desc_present",
    "tp_embpair_title_transcript_topk_present",
    "tp_embpair_compute_title_desc_enabled",
    "tp_embpair_compute_title_transcript_topk_enabled",
    "tp_embpair_export_topk_slots_enabled",
    "tp_embpair_export_topk_indices_enabled",
    "tp_embpair_export_topk_summary_enabled",
    "tp_embpair_use_faiss_mode_auto",
    "tp_embpair_use_faiss_mode_never",
    "tp_embpair_use_faiss_mode_always",
    "tp_embpair_min_corpus_for_faiss",
    "tp_embpair_require_faiss_enabled",
    "tp_embpair_require_title_embedding_enabled",
    "tp_embpair_require_description_embedding_enabled",
    "tp_embpair_require_transcript_chunks_enabled",
    "tp_pairtopk_present",
    "tp_pairtopk_top_k",
    "tp_pairtopk_title_desc_cosine",
    "tp_embpair_title_transcript_top1",
    "tp_pairtopk_title_transcript_top1",
    "tp_embpair_title_transcript_top1_idx",
    "tp_embpair_title_transcript_top2",
    "tp_pairtopk_title_transcript_top2",
    "tp_embpair_title_transcript_top2_idx",
    "tp_embpair_title_transcript_top3",
    "tp_pairtopk_title_transcript_top3",
    "tp_embpair_title_transcript_top3_idx",
    "tp_embpair_title_transcript_top4",
    "tp_pairtopk_title_transcript_top4",
    "tp_embpair_title_transcript_top4_idx",
    "tp_embpair_title_transcript_top5",
    "tp_pairtopk_title_transcript_top5",
    "tp_embpair_title_transcript_top5_idx",
    "tp_embpair_title_transcript_top6",
    "tp_pairtopk_title_transcript_top6",
    "tp_embpair_title_transcript_top6_idx",
    "tp_embpair_title_transcript_top7",
    "tp_pairtopk_title_transcript_top7",
    "tp_embpair_title_transcript_top7_idx",
    "tp_embpair_title_transcript_top8",
    "tp_pairtopk_title_transcript_top8",
    "tp_embpair_title_transcript_top8_idx",
    "tp_embpair_title_transcript_topk_max",
    "tp_embpair_title_transcript_topk_mean",
    "tp_pairtopk_title_transcript_topk_max",
    "tp_pairtopk_title_transcript_topk_mean",
    "tp_embpair_n_chunks",
    "tp_embpair_transcript_source_whisper",
    "tp_embpair_transcript_source_youtube_auto",
    "tp_embpair_transcript_source_combined",
    "tp_embpair_use_faiss_mode",
    "tp_embpair_require_faiss",
)


class EmbeddingPairTopKExtractor(BaseExtractor):
    VERSION = "1.3.0"

    def __init__(
        self,
        artifacts_dir: str | None = None,
        *,
        enabled: bool = True,
        top_k: int = 10,
        top_k_slots: int = 5,
        transcript_source_priority: Sequence[str] | str = ("whisper", "youtube_auto"),
        compute_title_desc: bool = True,
        compute_title_transcript_topk: bool = True,
        export_topk_slots: bool = True,
        export_topk_indices: bool = True,
        export_topk_summary: bool = True,
        use_faiss_mode: str = "auto",  # auto|never|always
        min_corpus_for_faiss: int = 512,
        require_faiss: bool = False,
        use_cross_encoder: bool = False,
        temperature: float = 0.1,
        device: Optional[str] = "cpu",
        require_title_embedding: bool = False,
        require_description_embedding: bool = False,
        require_transcript_chunks: bool = False,
        emit_extra_metrics: bool = False,
    ) -> None:
        init_sys_before = system_snapshot()
        init_mem_before = process_memory_bytes()

        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.enabled = bool(enabled)
        self.top_k = int(max(1, top_k))
        req_slots = int(max(1, int(top_k_slots)))
        self.top_k_slots_requested = req_slots
        self.top_k_slots = min(req_slots, SCHEMA_MAX_TOP_K_SLOTS)
        self.top_k_slots_clamped = bool(req_slots > SCHEMA_MAX_TOP_K_SLOTS)
        self.temperature = float(temperature)
        self.device = str(device or "cpu")
        # Feature gating
        self.compute_title_desc = bool(compute_title_desc)
        self.compute_title_transcript_topk = bool(compute_title_transcript_topk)
        self.export_topk_slots = bool(export_topk_slots)
        self.export_topk_indices = bool(export_topk_indices)
        self.export_topk_summary = bool(export_topk_summary)
        self.use_faiss_mode = str(use_faiss_mode or "auto").strip().lower()
        self.min_corpus_for_faiss = int(max(0, min_corpus_for_faiss))
        self.require_faiss = bool(require_faiss)
        self.emit_extra_metrics = bool(emit_extra_metrics)
        self.require_title_embedding = bool(require_title_embedding)
        self.require_description_embedding = bool(require_description_embedding)
        self.require_transcript_chunks = bool(require_transcript_chunks)

        # Transcript source priority (deterministic policy). `combined` supported if present in tp_artifacts.
        if isinstance(transcript_source_priority, str):
            pr = [p.strip() for p in transcript_source_priority.split(",") if p.strip()]
        else:
            pr = [str(p).strip() for p in transcript_source_priority if str(p).strip()]
        self.transcript_source_priority = pr or ["whisper", "youtube_auto"]

        # Privacy policy: TranscriptChunkEmbedder does not store raw chunk texts.
        # Cross-encoder rerank requires raw texts → not supported in production by default.
        self.use_cross_encoder = bool(use_cross_encoder)
        if self.use_cross_encoder:
            raise RuntimeError(
                "EmbeddingPairTopKExtractor: cross-encoder rerank is disabled by policy "
                "(requires raw transcript chunk texts + dp_models spec)."
            )

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
            raise RuntimeError("embedding_pair_topk_extractor: relpath escapes artifacts_dir")
        return cand

    @staticmethod
    def _is_finite(x: np.ndarray) -> bool:
        return bool(np.isfinite(x).all())

    @staticmethod
    def _has_zero_norm_rows(x: np.ndarray, axis: int = 1) -> bool:
        n = np.linalg.norm(x, axis=axis)
        return bool(np.any(n <= 0.0))

    @staticmethod
    def _l2n(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(x, axis=axis, keepdims=True)
        n = np.maximum(n, eps)
        return x / n

    @staticmethod
    def _cosine_scalar(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 0.0 or nb <= 0.0:
            return float("nan")
        return float(np.dot(a, b) / (na * nb))

    def _use_faiss_backend(self, n_rows: int) -> bool:
        if self.use_faiss_mode not in ("auto", "never", "always"):
            raise RuntimeError("EmbeddingPairTopKExtractor: invalid use_faiss_mode (expected auto|never|always)")
        if self.use_faiss_mode == "never":
            return False
        if self.use_faiss_mode == "always":
            return True
        # auto
        return int(n_rows) >= int(self.min_corpus_for_faiss)

    def _retrieve_topk(self, query: np.ndarray, corpus: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        use_faiss = self._use_faiss_backend(int(corpus.shape[0]))
        if use_faiss:
            # Lazy import FAISS to avoid crashing on incompatible NumPy builds
            try:
                import faiss  # type: ignore

                dim = int(corpus.shape[1])
                index = faiss.IndexFlatIP(dim)
                # Avoid in-place mutation on caller buffers
                c = np.asarray(corpus, dtype=np.float32).copy()
                q = np.asarray(query, dtype=np.float32).copy()
                faiss.normalize_L2(c)
                faiss.normalize_L2(q)
                index.add(c)
                scores, indices = index.search(q, min(k, c.shape[0]))
                return indices, scores
            except Exception:
                if self.require_faiss:
                    raise RuntimeError("EmbeddingPairTopKExtractor: FAISS required but unavailable")
        qn = self._l2n(np.asarray(query, dtype=np.float32), axis=1)
        cn = self._l2n(np.asarray(corpus, dtype=np.float32), axis=1)
        sims = qn @ cn.T
        k = min(k, sims.shape[1])
        idx = np.argsort(-sims, axis=1)[:, :k]
        scr = np.take_along_axis(sims, idx, axis=1)
        return idx, scr

    def _load_rel_matrix(self, relpath: str) -> Tuple[Optional[np.ndarray], bool]:
        try:
            p = self._safe_join_artifacts_dir(self.artifacts_dir, relpath)
        except Exception:
            return None, True
        if not p.exists():
            return None, False
        try:
            m = np.asarray(np.load(p), dtype=np.float32)
            if m.ndim == 1:
                m = m.reshape(1, -1)
            return m, False
        except Exception:
            return None, False

    def _load_rel_vector(self, relpath: str) -> Tuple[Optional[np.ndarray], bool]:
        try:
            p = self._safe_join_artifacts_dir(self.artifacts_dir, relpath)
        except Exception:
            return None, True
        if not p.exists():
            return None, False
        try:
            v = np.asarray(np.load(p), dtype=np.float32).reshape(-1)
            return v, False
        except Exception:
            return None, False

    @staticmethod
    def _pack_features_flat(values: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in _FEATURES_FLAT_KEYS:
            if k not in values:
                raise KeyError(f"embedding_pair_topk_extractor: missing features_flat key {k!r}")
            out[k] = float(values[k])
        return out

    def _shell_features_disabled(self) -> Dict[str, float]:
        nan = float("nan")
        d: Dict[str, float] = {
            "tp_embpair_present": 0.0,
            "tp_embpair_enabled": float(bool(self.enabled)),
            "tp_embpair_disabled_by_policy": 1.0,
            "tp_embpair_title_present": 0.0,
            "tp_embpair_desc_present": 0.0,
            "tp_embpair_transcript_chunks_present": 0.0,
            "tp_embpair_used_legacy_key_flag": 0.0,
            "tp_embpair_dim_mismatch_flag": 0.0,
            "tp_embpair_unsafe_relpath_flag": 0.0,
            "tp_embpair_nan_inf_flag": 0.0,
            "tp_embpair_zero_norm_flag": 0.0,
            "tp_embpair_top_k": float(int(self.top_k)),
            "tp_embpair_top_k_slots": float(int(self.top_k_slots)),
            "tp_embpair_schema_slots_max": float(int(SCHEMA_MAX_TOP_K_SLOTS)),
            "tp_embpair_top_k_slots_requested": float(int(self.top_k_slots_requested)),
            "tp_embpair_top_k_slots_clamped": 1.0 if self.top_k_slots_clamped else 0.0,
            "tp_embpair_title_desc_cosine": nan,
            "tp_embpair_title_desc_present": 0.0,
            "tp_embpair_title_transcript_topk_present": 0.0,
            "tp_embpair_compute_title_desc_enabled": float(bool(self.compute_title_desc)),
            "tp_embpair_compute_title_transcript_topk_enabled": float(bool(self.compute_title_transcript_topk)),
            "tp_embpair_export_topk_slots_enabled": float(bool(self.export_topk_slots)),
            "tp_embpair_export_topk_indices_enabled": float(bool(self.export_topk_indices)),
            "tp_embpair_export_topk_summary_enabled": float(bool(self.export_topk_summary)),
            "tp_embpair_use_faiss_mode_auto": 1.0 if (self.use_faiss_mode == "auto") else 0.0,
            "tp_embpair_use_faiss_mode_never": 1.0 if (self.use_faiss_mode == "never") else 0.0,
            "tp_embpair_use_faiss_mode_always": 1.0 if (self.use_faiss_mode == "always") else 0.0,
            "tp_embpair_min_corpus_for_faiss": float(int(self.min_corpus_for_faiss)),
            "tp_embpair_require_faiss_enabled": float(bool(self.require_faiss)),
            "tp_embpair_require_title_embedding_enabled": float(bool(self.require_title_embedding)),
            "tp_embpair_require_description_embedding_enabled": float(bool(self.require_description_embedding)),
            "tp_embpair_require_transcript_chunks_enabled": float(bool(self.require_transcript_chunks)),
            "tp_pairtopk_present": 0.0,
            "tp_pairtopk_top_k": float(int(self.top_k)),
            "tp_pairtopk_title_desc_cosine": nan,
        }
        for i in range(1, SCHEMA_MAX_TOP_K_SLOTS + 1):
            d[f"tp_embpair_title_transcript_top{i}"] = nan
            d[f"tp_pairtopk_title_transcript_top{i}"] = nan
            d[f"tp_embpair_title_transcript_top{i}_idx"] = nan
        d["tp_embpair_title_transcript_topk_max"] = nan
        d["tp_embpair_title_transcript_topk_mean"] = nan
        d["tp_pairtopk_title_transcript_topk_max"] = nan
        d["tp_pairtopk_title_transcript_topk_mean"] = nan
        self._apply_extra_metrics_block(d, chunks=None, transcript_source_used=None)
        return d

    def _apply_extra_metrics_block(
        self,
        d: Dict[str, float],
        *,
        chunks: Optional[np.ndarray],
        transcript_source_used: Optional[str],
    ) -> None:
        nan = float("nan")
        if not self.emit_extra_metrics:
            d["tp_embpair_n_chunks"] = nan
            d["tp_embpair_transcript_source_whisper"] = nan
            d["tp_embpair_transcript_source_youtube_auto"] = nan
            d["tp_embpair_transcript_source_combined"] = nan
            d["tp_embpair_use_faiss_mode"] = nan
            d["tp_embpair_require_faiss"] = nan
            return
        n_chunks = float(int(chunks.shape[0])) if isinstance(chunks, np.ndarray) and chunks.ndim == 2 else nan
        src = transcript_source_used
        d["tp_embpair_n_chunks"] = float(n_chunks)
        d["tp_embpair_transcript_source_whisper"] = float(src == "whisper")
        d["tp_embpair_transcript_source_youtube_auto"] = float(src == "youtube_auto")
        d["tp_embpair_transcript_source_combined"] = float(src == "combined")
        d["tp_embpair_use_faiss_mode"] = float(0.0 if self.use_faiss_mode == "never" else (1.0 if self.use_faiss_mode == "always" else 0.5))
        d["tp_embpair_require_faiss"] = float(bool(self.require_faiss))

    def _build_extract_return(
        self,
        *,
        sys_after: Any,
        mem_before: int,
        mem_after: int,
        total_s: float,
        features_flat: Dict[str, float],
    ) -> Dict[str, Any]:
        gpu_peak_mb = self._gpu_peak_mb(sys_after)
        return {
            "device": self.device,
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
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        if not self.enabled:
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return self._build_extract_return(
                sys_after=sys_after,
                mem_before=mem_before,
                mem_after=mem_after,
                total_s=total_s,
                features_flat=self._shell_features_disabled(),
            )

        dim_mismatch_flag = 0.0
        unsafe_relpath_flag = 0.0
        nan_inf_flag = 0.0
        zero_norm_flag = 0.0
        used_legacy_key_flag = 0.0

        tp = getattr(doc, "tp_artifacts", None)
        emb = tp.get("embeddings") if isinstance(tp, dict) else None
        transcripts = tp.get("transcripts") if isinstance(tp, dict) else None
        tchunks = tp.get("transcript_chunks") if isinstance(tp, dict) else None

        title_rel = emb.get("title", {}).get("relpath") if isinstance(emb, dict) and isinstance(emb.get("title"), dict) else None
        desc_rel = emb.get("description", {}).get("relpath") if isinstance(emb, dict) and isinstance(emb.get("description"), dict) else None

        # pick transcript chunks deterministically via priority
        chunks_rel = None
        transcript_source_used = None
        seen = set()
        for src in list(self.transcript_source_priority):
            if src in seen:
                continue
            seen.add(src)
            # canonical
            if isinstance(transcripts, dict):
                d2 = transcripts.get(src)
                rel2 = d2.get("chunk_embeddings_relpath") if isinstance(d2, dict) else None
                if isinstance(rel2, str) and rel2:
                    chunks_rel = rel2
                    transcript_source_used = str(src)
                    used_legacy_key_flag = 0.0
                    break
            # legacy fallback
            if isinstance(tchunks, dict):
                d = tchunks.get(src)
                rel = d.get("embeddings_relpath") if isinstance(d, dict) else None
                if isinstance(rel, str) and rel:
                    chunks_rel = rel
                    transcript_source_used = str(src)
                    used_legacy_key_flag = 1.0
                    break

        title = None
        desc = None
        chunks = None
        if isinstance(title_rel, str) and title_rel:
            title, unsafe = self._load_rel_vector(title_rel)
            if unsafe:
                unsafe_relpath_flag = 1.0
        if isinstance(desc_rel, str) and desc_rel:
            desc, unsafe = self._load_rel_vector(desc_rel)
            if unsafe:
                unsafe_relpath_flag = 1.0
        if isinstance(chunks_rel, str) and chunks_rel:
            chunks, unsafe = self._load_rel_matrix(chunks_rel)
            if unsafe:
                unsafe_relpath_flag = 1.0

        title_present = float(title is not None and isinstance(title, np.ndarray) and title.size > 0)
        desc_present = float(desc is not None and isinstance(desc, np.ndarray) and desc.size > 0)
        chunks_present = float(chunks is not None and isinstance(chunks, np.ndarray) and chunks.size > 0)

        if self.require_title_embedding and not bool(title_present):
            raise RuntimeError("EmbeddingPairTopKExtractor: required title embedding missing")
        if self.require_description_embedding and self.compute_title_desc and not bool(desc_present):
            raise RuntimeError("EmbeddingPairTopKExtractor: required description embedding missing for title-desc cosine")
        if self.require_transcript_chunks and self.compute_title_transcript_topk and not bool(chunks_present):
            raise RuntimeError("EmbeddingPairTopKExtractor: required transcript chunks missing for top-k")

        # Validate finiteness / norms for vectors
        if isinstance(title, np.ndarray) and title.size > 0:
            if not self._is_finite(title):
                nan_inf_flag = 1.0
            if float(np.linalg.norm(title)) <= 0.0:
                zero_norm_flag = 1.0
        if isinstance(desc, np.ndarray) and desc.size > 0:
            if not self._is_finite(desc):
                nan_inf_flag = 1.0
            if float(np.linalg.norm(desc)) <= 0.0:
                zero_norm_flag = 1.0
        if isinstance(chunks, np.ndarray) and chunks.size > 0:
            if chunks.ndim != 2:
                dim_mismatch_flag = 1.0
            if not self._is_finite(chunks):
                nan_inf_flag = 1.0
            if chunks.ndim == 2 and self._has_zero_norm_rows(chunks, axis=1):
                zero_norm_flag = 1.0

        title_desc_cos = float("nan")
        title_desc_present = 0.0
        if self.compute_title_desc and title is not None and desc is not None:
            if nan_inf_flag or zero_norm_flag:
                title_desc_cos = float("nan")
                title_desc_present = 0.0
            else:
                if int(title.size) != int(desc.size):
                    dim_mismatch_flag = 1.0
                    title_desc_cos = float("nan")
                    title_desc_present = 0.0
                else:
                    title_desc_cos = float(self._cosine_scalar(title, desc))
                    title_desc_present = 1.0 if (title_desc_cos == title_desc_cos) else 0.0

        # TopK over transcript chunks (valid empty if transcript is missing)
        topk_scores: np.ndarray | None = None
        topk_indices: np.ndarray | None = None
        title_transcript_present = 0.0
        if self.compute_title_transcript_topk and title is not None and chunks is not None and chunks.size > 0 and chunks.ndim == 2:
            if nan_inf_flag or zero_norm_flag:
                topk_scores = None
                topk_indices = None
                title_transcript_present = 0.0
            else:
                try:
                    q = title.reshape(1, -1)
                    if int(q.shape[1]) != int(chunks.shape[1]):
                        dim_mismatch_flag = 1.0
                        topk_scores = None
                        topk_indices = None
                        title_transcript_present = 0.0
                    else:
                        idx, scr = self._retrieve_topk(q, chunks, self.top_k)
                        topk_indices = np.asarray(idx, dtype=np.int32).reshape(-1)
                        topk_scores = np.asarray(scr, dtype=np.float32).reshape(-1)
                        title_transcript_present = 1.0 if (topk_scores.size > 0 and np.isfinite(topk_scores).any()) else 0.0
                except Exception:
                    dim_mismatch_flag = 1.0
                    topk_scores = None
                    topk_indices = None
                    title_transcript_present = 0.0

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        # features_flat: only numeric scalars (top-k list summarized)
        present = 1.0 if (bool(title_desc_present) or bool(title_transcript_present)) else 0.0
        nan = float("nan")
        features_flat: Dict[str, float] = {
            "tp_embpair_present": float(present),
            "tp_embpair_enabled": float(bool(self.enabled)),
            "tp_embpair_disabled_by_policy": 0.0,
            "tp_embpair_title_present": float(title_present),
            "tp_embpair_desc_present": float(desc_present),
            "tp_embpair_transcript_chunks_present": float(chunks_present),
            "tp_embpair_used_legacy_key_flag": float(used_legacy_key_flag),
            "tp_embpair_dim_mismatch_flag": float(dim_mismatch_flag),
            "tp_embpair_unsafe_relpath_flag": float(unsafe_relpath_flag),
            "tp_embpair_nan_inf_flag": float(nan_inf_flag),
            "tp_embpair_zero_norm_flag": float(zero_norm_flag),
            "tp_embpair_top_k": float(int(self.top_k)),
            "tp_embpair_top_k_slots": float(int(self.top_k_slots)),
            "tp_embpair_schema_slots_max": float(int(SCHEMA_MAX_TOP_K_SLOTS)),
            "tp_embpair_top_k_slots_requested": float(int(self.top_k_slots_requested)),
            "tp_embpair_top_k_slots_clamped": 1.0 if self.top_k_slots_clamped else 0.0,
            "tp_embpair_title_desc_cosine": float(title_desc_cos),
            "tp_embpair_title_desc_present": float(title_desc_present),
            "tp_embpair_title_transcript_topk_present": float(title_transcript_present),
            "tp_embpair_compute_title_desc_enabled": float(bool(self.compute_title_desc)),
            "tp_embpair_compute_title_transcript_topk_enabled": float(bool(self.compute_title_transcript_topk)),
            "tp_embpair_export_topk_slots_enabled": float(bool(self.export_topk_slots)),
            "tp_embpair_export_topk_indices_enabled": float(bool(self.export_topk_indices)),
            "tp_embpair_export_topk_summary_enabled": float(bool(self.export_topk_summary)),
            "tp_embpair_use_faiss_mode_auto": 1.0 if (self.use_faiss_mode == "auto") else 0.0,
            "tp_embpair_use_faiss_mode_never": 1.0 if (self.use_faiss_mode == "never") else 0.0,
            "tp_embpair_use_faiss_mode_always": 1.0 if (self.use_faiss_mode == "always") else 0.0,
            "tp_embpair_min_corpus_for_faiss": float(int(self.min_corpus_for_faiss)),
            "tp_embpair_require_faiss_enabled": float(bool(self.require_faiss)),
            "tp_embpair_require_title_embedding_enabled": float(bool(self.require_title_embedding)),
            "tp_embpair_require_description_embedding_enabled": float(bool(self.require_description_embedding)),
            "tp_embpair_require_transcript_chunks_enabled": float(bool(self.require_transcript_chunks)),
            # Back-compat aliases (old names) — tp_pairtopk_present stays title-transcript-only (Q4-A)
            "tp_pairtopk_present": float(title_transcript_present),
            "tp_pairtopk_top_k": float(int(self.top_k)),
            "tp_pairtopk_title_desc_cosine": float(title_desc_cos),
        }

        # Export slots top1..top8 (fixed schema)
        topk_vals: list[float] = []
        topk_idx_vals: list[float] = []
        if topk_scores is not None and topk_scores.size > 0:
            topk_vals = [float(x) for x in topk_scores[: self.top_k_slots]]
        if topk_indices is not None and topk_indices.size > 0:
            topk_idx_vals = [float(int(x)) for x in topk_indices[: self.top_k_slots]]
        while len(topk_vals) < SCHEMA_MAX_TOP_K_SLOTS:
            topk_vals.append(nan)
        while len(topk_idx_vals) < SCHEMA_MAX_TOP_K_SLOTS:
            topk_idx_vals.append(nan)

        for i in range(1, SCHEMA_MAX_TOP_K_SLOTS + 1):
            features_flat[f"tp_embpair_title_transcript_top{i}"] = nan
            features_flat[f"tp_pairtopk_title_transcript_top{i}"] = nan
            features_flat[f"tp_embpair_title_transcript_top{i}_idx"] = nan

        if self.export_topk_slots:
            for i in range(1, self.top_k_slots + 1):
                idx = i - 1
                features_flat[f"tp_embpair_title_transcript_top{i}"] = float(topk_vals[idx])
                features_flat[f"tp_pairtopk_title_transcript_top{i}"] = float(topk_vals[idx])

        if self.export_topk_indices:
            for i in range(1, self.top_k_slots + 1):
                idx = i - 1
                features_flat[f"tp_embpair_title_transcript_top{i}_idx"] = float(topk_idx_vals[idx])

        features_flat["tp_embpair_title_transcript_topk_max"] = nan
        features_flat["tp_embpair_title_transcript_topk_mean"] = nan
        features_flat["tp_pairtopk_title_transcript_topk_max"] = nan
        features_flat["tp_pairtopk_title_transcript_topk_mean"] = nan
        if self.export_topk_summary:
            finite = [x for x in (topk_scores.tolist() if isinstance(topk_scores, np.ndarray) else []) if np.isfinite(x)]
            features_flat["tp_embpair_title_transcript_topk_max"] = float(np.max(finite)) if finite else nan
            features_flat["tp_embpair_title_transcript_topk_mean"] = float(np.mean(finite)) if finite else nan
            features_flat["tp_pairtopk_title_transcript_topk_max"] = float(features_flat["tp_embpair_title_transcript_topk_max"])
            features_flat["tp_pairtopk_title_transcript_topk_mean"] = float(features_flat["tp_embpair_title_transcript_topk_mean"])

        self._apply_extra_metrics_block(features_flat, chunks=chunks, transcript_source_used=transcript_source_used)

        return self._build_extract_return(
            sys_after=sys_after,
            mem_before=mem_before,
            mem_after=mem_after,
            total_s=total_s,
            features_flat=features_flat,
        )
