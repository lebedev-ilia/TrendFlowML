"""
CommentsEmbedder — извлекает L2-нормализованные эмбеддинги для комментариев.

- Использует общий ModelRegistry (SentenceTransformer переиспользуется)
- Батчинг encode, inference_mode
- Сохраняет артефакт одним массивом (N, D) в .artifacts
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.model_registry import get_model_with_meta
from src.core.path_utils import default_artifacts_dir
from src.core.text_utils import normalize_whitespace
from src.schemas.models import VideoDocument


class CommentsEmbedder(BaseExtractor):
    VERSION = "1.2.0"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        cache_enabled: bool = False,
        cache_ttl_days: Optional[float] = 7.0,
        cache_max_items: Optional[int] = 50_000,
        cache_max_bytes: Optional[int] = 5_000_000_000,
        cache_cleanup_on_init: bool = True,
        cache_cleanup_max_seconds: float = 0.2,
        artifacts_dir: Optional[str] = None,
        device: Optional[str] = "cpu",
        fp16: bool = True,
        batch_size: int = 64,
        max_comments: int = 200,
        max_total_chars: int = 20000,
        max_chars_per_comment: int = 400,
        min_chars_per_comment: int = 3,
        dedup_comments: bool = True,
        selection_policy: str = "by_likes_then_recency",
        compute_embeddings: bool = True,
        write_artifact: bool = True,
        # Back-compat alias: if False → write_artifact=False
        write_embedding_artifact: bool = True,
        emit_extra_metrics: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cache_enabled = bool(cache_enabled)
        self.cache_ttl_days = float(cache_ttl_days) if cache_ttl_days is not None else None
        self.cache_max_items = int(cache_max_items) if cache_max_items is not None else None
        self.cache_max_bytes = int(cache_max_bytes) if cache_max_bytes is not None else None
        self.cache_cleanup_on_init = bool(cache_cleanup_on_init)
        self.cache_cleanup_max_seconds = float(cache_cleanup_max_seconds)
        try:
            from src.core.path_utils import default_cache_dir  # type: ignore

            base_cache = default_cache_dir() / "embed_cache"
        except Exception:
            base_cache = default_artifacts_dir().parent / "_cache" / "embed_cache"
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else base_cache
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.device = str(device or "cpu")
        self.fp16 = fp16 and ("cuda" in self.device)
        self.batch_size = batch_size
        self.max_comments = int(max(1, max_comments))
        self.max_total_chars = int(max(0, max_total_chars))
        self.max_chars_per_comment = int(max(1, max_chars_per_comment))
        self.min_chars_per_comment = int(max(0, min_chars_per_comment))
        self.dedup_comments = bool(dedup_comments)
        self.selection_policy = str(selection_policy or "by_likes_then_recency").strip().lower()
        if self.selection_policy not in ("by_likes_then_recency", "by_likes", "by_recency", "first_k"):
            raise RuntimeError("CommentsEmbedder: selection_policy must be one of: by_likes_then_recency|by_likes|by_recency|first_k")
        self.compute_embeddings = bool(compute_embeddings)
        self.write_artifact = bool(write_artifact) and bool(write_embedding_artifact)
        self.emit_extra_metrics = bool(emit_extra_metrics)

        # metrics: init snapshots
        init_sys_before = system_snapshot()
        init_mem_before = process_memory_bytes()

        # Resolve & load model strictly via dp_models (offline, fail-fast).
        self.model, self.weights_digest, self.model_version = get_model_with_meta(
            model_name=self.model_name, device=self.device, fp16=self.fp16
        )
        self._model_digest_u24 = int(self.weights_digest[:6], 16) if len(self.weights_digest) >= 6 else 0

        if self.cache_cleanup_on_init and self.cache_enabled:
            self._cleanup_cache_best_effort()

        init_sys_after = system_snapshot()
        init_mem_after = process_memory_bytes()
        self._init_metrics: Dict[str, Any] = {
            "pre_init": init_sys_before,
            "post_init": init_sys_after,
            "ram_peak_bytes": max(init_mem_before, init_mem_after),
        }

    @staticmethod
    def _hash_list(texts: List[str], model_name: str) -> str:
        payload = (model_name + "||" + "\n".join(texts)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _is_cache_entry_expired(self, path: Path) -> bool:
        if self.cache_ttl_days is None:
            return False
        try:
            age_s = max(0.0, time.time() - float(path.stat().st_mtime))
            return age_s > float(self.cache_ttl_days) * 86400.0
        except Exception:
            return False

    def _cleanup_cache_best_effort(self) -> None:
        t0 = time.perf_counter()
        try:
            entries = []
            for fn in os.listdir(self.cache_dir):
                if not fn.endswith(".npy"):
                    continue
                p = self.cache_dir / fn
                try:
                    st = p.stat()
                except Exception:
                    continue
                if self._is_cache_entry_expired(p):
                    try:
                        p.unlink()
                    except Exception:
                        pass
                    continue
                entries.append((float(st.st_mtime), int(st.st_size), p))
                if (time.perf_counter() - t0) > self.cache_cleanup_max_seconds:
                    break
            if (time.perf_counter() - t0) > self.cache_cleanup_max_seconds:
                return
            entries.sort(key=lambda t: t[0])  # oldest first
            total_bytes = sum(s for _, s, _ in entries)
            while entries:
                too_many = (self.cache_max_items is not None) and (len(entries) > int(self.cache_max_items))
                too_big = (self.cache_max_bytes is not None) and (total_bytes > int(self.cache_max_bytes))
                if not (too_many or too_big):
                    break
                _mtime, size, p = entries.pop(0)
                try:
                    p.unlink()
                    total_bytes = max(0, total_bytes - int(size))
                except Exception:
                    pass
                if (time.perf_counter() - t0) > self.cache_cleanup_max_seconds:
                    break
        except Exception:
            return

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out_batches: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            with torch.no_grad():
                raw = self.model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
            raw = np.asarray(raw, dtype=np.float32)
            # l2 normalize
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normed = raw / norms
            out_batches.append(normed)
        return np.vstack(out_batches)

    def _select_comments(self, doc: VideoDocument, texts: List[str]) -> tuple[List[str], Dict[str, float]]:
        """
        Deterministic comment selection + truncation.
        Returns (selected_texts, stats).
        """
        n_input = len(texts)
        likes = getattr(doc, "comments_likes", None)
        recency = getattr(doc, "comments_recency", None)

        # default ordering: stable input order
        order = list(range(n_input))

        def _valid_weights(arr: Any) -> bool:
            return isinstance(arr, list) and len(arr) == n_input and all(isinstance(x, (int, float)) for x in arr)

        if self.selection_policy == "by_likes_then_recency":
            if _valid_weights(likes):
                if _valid_weights(recency):
                    order.sort(key=lambda i: (float(likes[i]), float(recency[i])), reverse=True)
                else:
                    order.sort(key=lambda i: float(likes[i]), reverse=True)
            elif _valid_weights(recency):
                # Fallback to recency only if likes not available
                order.sort(key=lambda i: float(recency[i]), reverse=True)
            # else: keep default order (first_k behavior)
        elif self.selection_policy == "by_likes":
            if _valid_weights(likes):
                order.sort(key=lambda i: float(likes[i]), reverse=True)
            # else: keep default order (first_k behavior)
        elif self.selection_policy == "by_recency":
            if _valid_weights(recency):
                order.sort(key=lambda i: float(recency[i]), reverse=True)
            # else: keep default order (first_k behavior)
        elif self.selection_policy == "first_k":
            pass
        else:
            # Should be unreachable due to __init__ validation; keep defensive.
            raise RuntimeError(f"CommentsEmbedder: unknown selection_policy={self.selection_policy!r}")

        picked: List[str] = []
        total_chars = 0
        truncated_by_total = 0.0
        for i in order[: self.max_comments]:
            t = texts[i]
            if self.max_total_chars > 0 and (total_chars + len(t)) > self.max_total_chars:
                truncated_by_total = 1.0
                break
            picked.append(t)
            total_chars += len(t)
        return picked, {
            "n_input": float(n_input),
            "n_selected": float(len(picked)),
            "total_chars_used": float(total_chars),
            "truncated_by_total_chars_flag": float(truncated_by_total),
        }

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()
        error: Optional[str] = None

        # gather comments
        raw_comments = (doc.comments or []) if hasattr(doc, "comments") else []
        texts_norm: List[str] = []
        texts_src_idx: List[int] = []
        for src_i, c in enumerate(raw_comments):
            try:
                t = normalize_whitespace(getattr(c, "text", ""))
            except Exception:
                continue
            if not t:
                continue
            if self.max_chars_per_comment and len(t) > self.max_chars_per_comment:
                t = t[: self.max_chars_per_comment]
            if len(t) < self.min_chars_per_comment:
                continue
            texts_norm.append(t)
            texts_src_idx.append(int(src_i))

        n_before_dedup = len(texts_norm)
        if self.dedup_comments and texts_norm:
            seen = set()
            uniq_txt: List[str] = []
            uniq_idx: List[int] = []
            for t, src_i in zip(texts_norm, texts_src_idx):
                if t in seen:
                    continue
                seen.add(t)
                uniq_txt.append(t)
                uniq_idx.append(int(src_i))
            texts_norm = uniq_txt
            texts_src_idx = uniq_idx
        n_after_dedup = len(texts_norm)

        def _stable_features_template() -> Dict[str, float]:
            return {
                "tp_commentsemb_present": 0.0,  # embeddings computed (not "artifact exists")
                "tp_commentsemb_count": float("nan"),
                "tp_commentsemb_dim": float("nan"),
                "tp_commentsemb_n_input": float(n_before_dedup),
                "tp_commentsemb_n_deduped": float(n_after_dedup),
                "tp_commentsemb_n_selected": float("nan"),
                "tp_commentsemb_total_chars_used": float("nan"),
                "tp_commentsemb_truncated_by_total_chars_flag": 0.0,
                "tp_commentsemb_cache_enabled": float(bool(self.cache_enabled)),
                "tp_commentsemb_cache_hit": float("nan"),
                "tp_commentsemb_fp16": float(bool(self.fp16)),
                "tp_commentsemb_device_cuda": float("cuda" in str(self.device).lower()),
                "tp_commentsemb_model_digest_u24": float(int(self._model_digest_u24)),
                "tp_commentsemb_compute_enabled": float(bool(self.compute_embeddings)),
                "tp_commentsemb_write_artifact_enabled": float(bool(self.write_artifact)),
                "tp_commentsemb_artifact_written": 0.0,
                "tp_commentsemb_select_ms": float("nan"),
                "tp_commentsemb_encode_ms": float("nan"),
            }

        if not texts_norm:
            # valid empty (comments may be missing)
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            features_flat = _stable_features_template()
            features_flat["tp_commentsemb_count"] = 0.0
            features_flat["tp_commentsemb_cache_hit"] = 0.0
            features_flat["tp_commentsemb_n_selected"] = 0.0
            return {
                "device": self.device,
                "version": self.VERSION,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "weights_digest": self.weights_digest,
                "system": {
                    "pre_init": self._init_metrics.get("pre_init"),
                    "post_init": self._init_metrics.get("post_init"),
                    "post_process": sys_after,
                    "peaks": {
                        "ram_peak_mb": int(max(self._init_metrics.get("ram_peak_bytes", 0), mem_before, mem_after) / 1024 / 1024),
                        "gpu_peak_mb": 0,
                    },
                },
                "timings_s": {"total": round(total_s, 3)},
                "result": {"features_flat": features_flat},
                "error": None,
            }

        # selection
        t_sel0 = time.perf_counter()
        selected, sel_stats = self._select_comments(doc, texts_norm)
        sel_s = time.perf_counter() - t_sel0

        # Derive selected source indices by matching selected texts back to the deduped list.
        # This is privacy-safe (indices only) and enables weight alignment in CommentsAggregationExtractor.
        selected_src_indices: List[int] = []
        if selected and texts_src_idx and len(texts_src_idx) == len(texts_norm):
            pos_by_text: Dict[str, int] = {}
            for j, t in enumerate(texts_norm):
                if t not in pos_by_text:
                    pos_by_text[t] = int(j)
            for t in selected:
                j = pos_by_text.get(t)
                if j is None:
                    continue
                selected_src_indices.append(int(texts_src_idx[int(j)]))

        if not self.compute_embeddings:
            # feature-gating: skip compute + write
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            features_flat = _stable_features_template()
            features_flat["tp_commentsemb_cache_hit"] = 0.0
            features_flat["tp_commentsemb_n_selected"] = float(sel_stats.get("n_selected", float("nan")))
            features_flat["tp_commentsemb_total_chars_used"] = float(sel_stats.get("total_chars_used", float("nan")))
            features_flat["tp_commentsemb_truncated_by_total_chars_flag"] = float(sel_stats.get("truncated_by_total_chars_flag", 0.0))
            features_flat["tp_commentsemb_select_ms"] = float(round(sel_s * 1000.0, 3))
            return {
                "device": self.device,
                "version": self.VERSION,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "weights_digest": self.weights_digest,
                "system": {
                    "pre_init": self._init_metrics.get("pre_init"),
                    "post_init": self._init_metrics.get("post_init"),
                    "post_process": sys_after,
                    "peaks": {
                        "ram_peak_mb": int(max(self._init_metrics.get("ram_peak_bytes", 0), mem_before, mem_after) / 1024 / 1024),
                        "gpu_peak_mb": 0,
                    },
                },
                "timings_s": {"select": round(sel_s, 3), "total": round(total_s, 3)},
                "result": {"features_flat": features_flat},
                "error": None,
            }

        # Optional cache (content-addressed) - not source-of-truth.
        cache_hit = False
        embs = None
        cache_key = None
        if self.cache_enabled:
            # NOTE: cache key uses SHA256 of normalized selected comments; not persisted in NPZ.
            cache_key = hashlib.sha256(
                (
                    f"{self.model_name}|{self.weights_digest}|sel={self.selection_policy}|mc={self.max_comments}|mtc={self.max_total_chars}|mch={self.max_chars_per_comment}|dedup={int(self.dedup_comments)}"
                    + "||"
                    + "\n".join(selected)
                ).encode("utf-8")
            ).hexdigest()
            cache_path = (self.cache_dir / f"comments_embeddings_{cache_key}.npy").resolve()
            try:
                if cache_path.exists() and (not self._is_cache_entry_expired(cache_path)):
                    embs = np.asarray(np.load(cache_path), dtype=np.float32)
                    if isinstance(embs, np.ndarray) and embs.ndim == 2 and embs.shape[0] > 0:
                        cache_hit = True
            except Exception:
                try:
                    cache_path.unlink(missing_ok=True)
                except Exception:
                    pass
                embs = None

        # encode
        encode_s = float("nan")
        if embs is None:
            t_enc = time.perf_counter()
            embs = self._encode_texts(selected)
            encode_s = time.perf_counter() - t_enc
            if self.cache_enabled and cache_key:
                try:
                    cache_path = (self.cache_dir / f"comments_embeddings_{cache_key}.npy").resolve()
                    tmpc = cache_path.with_suffix(".tmp.npy")
                    np.save(tmpc, np.asarray(embs, dtype=np.float32))
                    tmpc.replace(cache_path)
                except Exception:
                    pass

        # save artifact
        emb_path = self.artifacts_dir / "comments_embeddings.npy"
        artifact_written = False
        if self.write_artifact:
            try:
                tmp = emb_path.with_suffix(".tmp.npy")
                np.save(tmp, embs.astype(np.float32))
                tmp.replace(emb_path)
                artifact_written = True
            except Exception as e:
                raise RuntimeError(f"CommentsEmbedder: artifact_save_error: {e}") from e

        # In-memory registry for downstream (no absolute paths in result/NPZ).
        try:
            tp = getattr(doc, "tp_artifacts", None)
            if not isinstance(tp, dict):
                tp = {}
                setattr(doc, "tp_artifacts", tp)
            tp.setdefault("embeddings", {})
            tp.setdefault("comments", {})
            if artifact_written:
                tp["embeddings"]["comments"] = {
                    "relpath": emb_path.name,
                    "kind": "matrix",
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "weights_digest": self.weights_digest,
                    "rows": int(embs.shape[0]),
                    "dim": int(embs.shape[1]) if embs.ndim == 2 else 0,
                }
                # Persist selected indices as a per-run sub-artifact to enable deterministic weighting downstream.
                if selected_src_indices:
                    idx_path = self.artifacts_dir / "comments_selected_indices.npy"
                    tmpi = idx_path.with_suffix(".tmp.npy")
                    np.save(tmpi, np.asarray(selected_src_indices, dtype=np.int32))
                    tmpi.replace(idx_path)
                    tp["comments"]["selected_indices_relpath"] = idx_path.name
                    tp["comments"]["selected_indices_count"] = int(len(selected_src_indices))
        except Exception:
            pass

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        features_flat = _stable_features_template()
        features_flat.update(
            {
                "tp_commentsemb_present": 1.0,
                "tp_commentsemb_count": float(int(embs.shape[0])),
                "tp_commentsemb_dim": float(int(embs.shape[1]) if embs.ndim == 2 else 0),
                "tp_commentsemb_cache_hit": float(bool(cache_hit)) if self.cache_enabled else 0.0,
                "tp_commentsemb_n_selected": float(sel_stats.get("n_selected", float("nan"))),
                "tp_commentsemb_total_chars_used": float(sel_stats.get("total_chars_used", float("nan"))),
                "tp_commentsemb_truncated_by_total_chars_flag": float(sel_stats.get("truncated_by_total_chars_flag", 0.0)),
                "tp_commentsemb_select_ms": float(round(sel_s * 1000.0, 3)),
                "tp_commentsemb_encode_ms": float(round(float(encode_s) * 1000.0, 3)) if encode_s == encode_s else float("nan"),
                "tp_commentsemb_artifact_written": float(bool(artifact_written)),
            }
        )

        result: Dict[str, Any] = {
            "device": self.device,
            "version": self.VERSION,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "weights_digest": self.weights_digest,
            "system": {
                "pre_init": self._init_metrics.get("pre_init"),
                "post_init": self._init_metrics.get("post_init"),
                "post_process": sys_after,
                "peaks": {
                    "ram_peak_mb": int(max(self._init_metrics.get("ram_peak_bytes", 0), mem_before, mem_after) / 1024 / 1024),
                    "gpu_peak_mb": 0,
                },
            },
            "timings_s": {"select": round(sel_s, 3), "encode": round(float(encode_s), 3) if encode_s == encode_s else float("nan"), "total": round(total_s, 3)},
            "result": {"features_flat": features_flat},
            "error": error,
        }

    @property
    def supports_batch(self) -> bool:
        """CommentsEmbedder supports batch processing."""
        return True

    def extract_batch(self, docs: List[VideoDocument]) -> List[Dict[str, Any]]:
        """
        Batch processing: collect all selected comments from all documents,
        encode in batches, then distribute back per-document.
        """
        import time
        from collections import defaultdict

        started = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        n_docs = len(docs)
        if n_docs == 0:
            return []

        # Step 1: Process each document: normalize, dedup, select comments
        # Structure: doc_idx -> (selected_texts, selected_src_indices, stats, n_before_dedup, n_after_dedup)
        doc_comments_data: Dict[int, Tuple[List[str], List[int], Dict[str, float], int, int]] = {}
        all_selected_comments_flat: List[str] = []
        comment_to_doc: List[int] = []  # (doc_idx)

        for doc_idx, doc in enumerate(docs):
            doc_t0 = time.perf_counter()
            
            # Gather and normalize comments (same logic as extract())
            raw_comments = (doc.comments or []) if hasattr(doc, "comments") else []
            texts_norm: List[str] = []
            texts_src_idx: List[int] = []
            for src_i, c in enumerate(raw_comments):
                try:
                    t = normalize_whitespace(getattr(c, "text", ""))
                except Exception:
                    continue
                if not t:
                    continue
                if self.max_chars_per_comment and len(t) > self.max_chars_per_comment:
                    t = t[: self.max_chars_per_comment]
                if len(t) < self.min_chars_per_comment:
                    continue
                texts_norm.append(t)
                texts_src_idx.append(int(src_i))

            n_before_dedup = len(texts_norm)
            if self.dedup_comments and texts_norm:
                seen = set()
                uniq_txt: List[str] = []
                uniq_idx: List[int] = []
                for t, src_i in zip(texts_norm, texts_src_idx):
                    if t in seen:
                        continue
                    seen.add(t)
                    uniq_txt.append(t)
                    uniq_idx.append(int(src_i))
                texts_norm = uniq_txt
                texts_src_idx = uniq_idx
            n_after_dedup = len(texts_norm)

            if not texts_norm:
                # Empty case - will be handled in result building
                doc_comments_data[doc_idx] = ([], [], {}, n_before_dedup, n_after_dedup)
                continue

            # Selection (per-doc, depends on doc.comments_likes/recency)
            t_sel0 = time.perf_counter()
            selected, sel_stats = self._select_comments(doc, texts_norm)
            sel_s = time.perf_counter() - t_sel0
            sel_stats["select_ms"] = sel_s  # Add timing to stats

            # Derive selected source indices
            selected_src_indices: List[int] = []
            if selected and texts_src_idx and len(texts_src_idx) == len(texts_norm):
                pos_by_text: Dict[str, int] = {}
                for j, t in enumerate(texts_norm):
                    if t not in pos_by_text:
                        pos_by_text[t] = int(j)
                for t in selected:
                    j = pos_by_text.get(t)
                    if j is None:
                        continue
                    selected_src_indices.append(int(texts_src_idx[int(j)]))

            # Store and collect for batch encoding
            doc_comments_data[doc_idx] = (selected, selected_src_indices, sel_stats, n_before_dedup, n_after_dedup)
            for comment in selected:
                all_selected_comments_flat.append(comment)
                comment_to_doc.append(doc_idx)

        # Step 2: Batch encode all selected comments
        t_enc0 = time.perf_counter()
        if all_selected_comments_flat:
            all_embeddings = self._encode_texts(all_selected_comments_flat)  # (N_total_comments, D)
        else:
            all_embeddings = np.zeros((0, 0), dtype=np.float32)
        t_enc_s = time.perf_counter() - t_enc0

        # Step 3: Process each document: distribute embeddings, save artifacts, build results
        results: List[Dict[str, Any]] = []
        global_comment_idx = 0

        for doc_idx, doc in enumerate(docs):
            doc_t0 = time.perf_counter()
            doc_sys_before = system_snapshot()
            doc_mem_before = process_memory_bytes()

            selected, selected_src_indices, sel_stats, n_before_dedup, n_after_dedup = doc_comments_data.get(doc_idx, ([], [], {}, 0, 0))

            def _stable_features_template() -> Dict[str, float]:
                return {
                    "tp_commentsemb_present": 0.0,
                    "tp_commentsemb_count": float("nan"),
                    "tp_commentsemb_dim": float("nan"),
                    "tp_commentsemb_n_input": float(n_before_dedup),
                    "tp_commentsemb_n_deduped": float(n_after_dedup),
                    "tp_commentsemb_n_selected": float("nan"),
                    "tp_commentsemb_total_chars_used": float("nan"),
                    "tp_commentsemb_truncated_by_total_chars_flag": 0.0,
                    "tp_commentsemb_cache_enabled": float(bool(self.cache_enabled)),
                    "tp_commentsemb_cache_hit": float("nan"),
                    "tp_commentsemb_fp16": float(bool(self.fp16)),
                    "tp_commentsemb_device_cuda": float("cuda" in str(self.device).lower()),
                    "tp_commentsemb_model_digest_u24": float(int(self._model_digest_u24)),
                    "tp_commentsemb_compute_enabled": float(bool(self.compute_embeddings)),
                    "tp_commentsemb_write_artifact_enabled": float(bool(self.write_artifact)),
                    "tp_commentsemb_artifact_written": 0.0,
                    "tp_commentsemb_select_ms": float("nan"),
                    "tp_commentsemb_encode_ms": float("nan"),
                }

            error: Optional[str] = None

            if not selected:
                # Empty case
                sys_after = system_snapshot()
                mem_after = process_memory_bytes()
                total_s = time.perf_counter() - doc_t0
                features_flat = _stable_features_template()
                features_flat["tp_commentsemb_count"] = 0.0
                features_flat["tp_commentsemb_cache_hit"] = 0.0
                features_flat["tp_commentsemb_n_selected"] = 0.0
                results.append({
                    "device": self.device,
                    "version": self.VERSION,
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "weights_digest": self.weights_digest,
                    "system": {
                        "pre_init": self._init_metrics.get("pre_init"),
                        "post_init": self._init_metrics.get("post_init"),
                        "post_process": sys_after,
                        "peaks": {
                            "ram_peak_mb": int(max(self._init_metrics.get("ram_peak_bytes", 0), doc_mem_before, mem_after) / 1024 / 1024),
                            "gpu_peak_mb": 0,
                        },
                    },
                    "timings_s": {"total": round(total_s, 3)},
                    "result": {"features_flat": features_flat},
                    "error": None,
                })
                continue

            if not self.compute_embeddings:
                # Feature-gating: skip compute + write
                sys_after = system_snapshot()
                mem_after = process_memory_bytes()
                total_s = time.perf_counter() - doc_t0
                features_flat = _stable_features_template()
                features_flat["tp_commentsemb_cache_hit"] = 0.0
                features_flat["tp_commentsemb_n_selected"] = float(sel_stats.get("n_selected", float("nan")))
                features_flat["tp_commentsemb_total_chars_used"] = float(sel_stats.get("total_chars_used", float("nan")))
                features_flat["tp_commentsemb_truncated_by_total_chars_flag"] = float(sel_stats.get("truncated_by_total_chars_flag", 0.0))
                features_flat["tp_commentsemb_select_ms"] = float(round(sel_stats.get("select_ms", 0.0) * 1000.0, 3))
                results.append({
                    "device": self.device,
                    "version": self.VERSION,
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "weights_digest": self.weights_digest,
                    "system": {
                        "pre_init": self._init_metrics.get("pre_init"),
                        "post_init": self._init_metrics.get("post_init"),
                        "post_process": sys_after,
                        "peaks": {
                            "ram_peak_mb": int(max(self._init_metrics.get("ram_peak_bytes", 0), doc_mem_before, mem_after) / 1024 / 1024),
                            "gpu_peak_mb": 0,
                        },
                    },
                    "timings_s": {"select": round(sel_stats.get("select_ms", 0.0), 3), "total": round(total_s, 3)},
                    "result": {"features_flat": features_flat},
                    "error": None,
                })
                continue

            # Extract embeddings for this document's comments
            n_selected = len(selected)
            doc_embeddings = all_embeddings[global_comment_idx:global_comment_idx + n_selected]
            global_comment_idx += n_selected

            # Get per-doc artifacts directory
            doc_artifacts_dir = getattr(doc, "_tp_artifacts_dir", None)
            if doc_artifacts_dir:
                emb_path = Path(doc_artifacts_dir) / "comments_embeddings.npy"
                idx_path = Path(doc_artifacts_dir) / "comments_selected_indices.npy"
            else:
                emb_path = self.artifacts_dir / "comments_embeddings.npy"
                idx_path = self.artifacts_dir / "comments_selected_indices.npy"

            # Save artifacts
            artifact_written = False
            if self.write_artifact:
                try:
                    emb_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = emb_path.with_suffix(".tmp.npy")
                    np.save(tmp, doc_embeddings.astype(np.float32))
                    tmp.replace(emb_path)
                    artifact_written = True

                    # Save selected indices
                    if selected_src_indices:
                        tmpi = idx_path.with_suffix(".tmp.npy")
                        np.save(tmpi, np.asarray(selected_src_indices, dtype=np.int32))
                        tmpi.replace(idx_path)
                except Exception as e:
                    error = f"artifact_save_error: {e}"

            # Update tp_artifacts
            if artifact_written:
                try:
                    tp = getattr(doc, "tp_artifacts", None)
                    if not isinstance(tp, dict):
                        tp = {}
                        setattr(doc, "tp_artifacts", tp)
                    tp.setdefault("embeddings", {})
                    tp.setdefault("comments", {})
                    tp["embeddings"]["comments"] = {
                        "relpath": emb_path.name,
                        "kind": "matrix",
                        "model_name": self.model_name,
                        "model_version": self.model_version,
                        "weights_digest": self.weights_digest,
                        "rows": int(doc_embeddings.shape[0]),
                        "dim": int(doc_embeddings.shape[1]) if doc_embeddings.ndim == 2 else 0,
                    }
                    if selected_src_indices:
                        tp["comments"]["selected_indices_relpath"] = idx_path.name
                        tp["comments"]["selected_indices_count"] = int(len(selected_src_indices))
                except Exception:
                    pass

            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - doc_t0

            features_flat = _stable_features_template()
            features_flat.update({
                "tp_commentsemb_present": 1.0,
                "tp_commentsemb_count": float(int(doc_embeddings.shape[0])),
                "tp_commentsemb_dim": float(int(doc_embeddings.shape[1]) if doc_embeddings.ndim == 2 else 0),
                "tp_commentsemb_cache_hit": 0.0,  # Cache not used in batch mode (could be added later)
                "tp_commentsemb_n_selected": float(sel_stats.get("n_selected", float("nan"))),
                "tp_commentsemb_total_chars_used": float(sel_stats.get("total_chars_used", float("nan"))),
                "tp_commentsemb_truncated_by_total_chars_flag": float(sel_stats.get("truncated_by_total_chars_flag", 0.0)),
                "tp_commentsemb_select_ms": float(round(sel_s * 1000.0, 3)),
                "tp_commentsemb_encode_ms": float(round(t_enc_s * 1000.0 / n_docs, 3)) if t_enc_s == t_enc_s else float("nan"),  # Per-doc share
                "tp_commentsemb_artifact_written": float(bool(artifact_written)),
            })

            results.append({
                "device": self.device,
                "version": self.VERSION,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "weights_digest": self.weights_digest,
                "system": {
                    "pre_init": self._init_metrics.get("pre_init"),
                    "post_init": self._init_metrics.get("post_init"),
                    "post_process": sys_after,
                    "peaks": {
                        "ram_peak_mb": int(max(self._init_metrics.get("ram_peak_bytes", 0), doc_mem_before, mem_after) / 1024 / 1024),
                        "gpu_peak_mb": 0,
                    },
                },
                "timings_s": {
                    "select": round(sel_s, 3),
                    "encode": round(t_enc_s / n_docs, 3),  # Per-doc share
                    "total": round(total_s, 3),
                },
                "result": {"features_flat": features_flat},
                "error": error,
            })

        return results

        return result


