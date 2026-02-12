from __future__ import annotations

import hashlib
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.model_registry import get_model_with_meta
from src.core.path_utils import default_artifacts_dir
from src.schemas.models import VideoDocument


class HashtagEmbedder(BaseExtractor):
    VERSION = "1.2.0"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        cache_enabled: bool = False,
        cache_ttl_days: Optional[float] = 30.0,
        cache_max_items: Optional[int] = 200_000,
        cache_max_bytes: Optional[int] = 2_000_000_000,
        cache_cleanup_on_init: bool = True,
        cache_cleanup_max_seconds: float = 0.2,
        artifacts_dir: Optional[str] = None,
        device: Optional[str] = "cpu",
        fp16: bool = True,
        batch_size: int = 128,
        require_hashtags: bool = False,
        # Back-compat alias (deprecated): if True → require_hashtags=True
        strict_missing_hashtags: bool = True,
        max_tags: int = 50,
        max_tag_len: int = 64,
        normalize_casefold: bool = True,
        strip_hash_prefix: bool = True,
        use_frequencies: bool = False,
        aggregation: str = "mean",
        compute_embedding: bool = True,
        write_artifact: bool = True,
        # Back-compat alias (deprecated): if False → write_artifact=False
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
        base_cache = default_artifacts_dir().parent / "_cache" / "embed_cache"
        # Prefer TextProcessor cache root if available; fallback to a subdir near artifacts_dir.
        try:
            from src.core.path_utils import default_cache_dir  # type: ignore
            base_cache = default_cache_dir() / "embed_cache"
        except Exception:
            pass
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else base_cache
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.device = str(device or "cpu")
        self.fp16 = fp16 and ("cuda" in self.device)
        self.batch_size = batch_size
        self.require_hashtags = bool(require_hashtags) or bool(strict_missing_hashtags)
        self.max_tags = int(max(1, max_tags))
        self.max_tag_len = int(max(1, max_tag_len))
        self.normalize_casefold = bool(normalize_casefold)
        self.strip_hash_prefix = bool(strip_hash_prefix)
        self.use_frequencies = bool(use_frequencies)
        self.aggregation = str(aggregation or "mean").strip().lower()
        self.compute_embedding = bool(compute_embedding)
        self.write_artifact = bool(write_artifact) and bool(write_embedding_artifact)
        self.emit_extra_metrics = bool(emit_extra_metrics)

        # Resolve & load model strictly via dp_models (offline, fail-fast).
        self._model, self.weights_digest, self.model_version = get_model_with_meta(
            model_name=self.model_name, device=self.device, fp16=self.fp16
        )
        self._model_digest_u24 = int(self.weights_digest[:6], 16) if len(self.weights_digest) >= 6 else 0

        # Cache is optional; default off. Create dirs only when enabled.
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.cache_cleanup_on_init:
                self._cleanup_cache_best_effort()

    def _is_cache_entry_expired(self, path: Path) -> bool:
        if self.cache_ttl_days is None:
            return False
        try:
            age_s = max(0.0, time.time() - float(path.stat().st_mtime))
            return age_s > float(self.cache_ttl_days) * 86400.0
        except Exception:
            return False

    def _cache_path_vector(self, h: str) -> Path:
        return self.cache_dir / f"{h}.npy"

    def _cache_path_aux(self, h: str) -> Path:
        return self.cache_dir / f"{h}.aux.npy"

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

    def _canonicalize_tags(self, tags: List[str]) -> tuple[List[str], Dict[str, int], int]:
        """
        Returns (canonical_tags_sorted, freq_by_tag, n_truncated).
        """
        cleaned: List[str] = []
        for t in tags:
            try:
                s = str(t or "").strip()
            except Exception:
                continue
            if not s:
                continue
            if self.strip_hash_prefix and s.startswith("#"):
                s = s[1:]
            s = s.strip()
            if not s:
                continue
            if self.normalize_casefold:
                s = s.casefold()
            if len(s) > self.max_tag_len:
                s = s[: self.max_tag_len]
            cleaned.append(s)

        freq = Counter(cleaned)
        uniq_sorted = sorted(freq.keys())
        n_truncated = 0
        if len(uniq_sorted) > self.max_tags:
            n_truncated = len(uniq_sorted) - self.max_tags
            uniq_sorted = uniq_sorted[: self.max_tags]
            freq = {k: int(freq[k]) for k in uniq_sorted}
        else:
            freq = {k: int(v) for k, v in freq.items()}
        return uniq_sorted, freq, int(n_truncated)

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            with torch.no_grad():
                raw = self._model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
            raw = np.asarray(raw, dtype=np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out.append(raw / norms)
        return np.vstack(out)

    @staticmethod
    def _agg(embs: np.ndarray, *, weights: Optional[np.ndarray], agg: str) -> np.ndarray:
        """
        Aggregate tag embeddings into a single vector.
        embs: (N, D), assumed float32.
        weights: optional (N,) non-negative.
        """
        if embs.ndim != 2 or embs.shape[0] == 0:
            # Should be unreachable in normal flow (empty handled earlier); keep as defensive.
            return np.zeros((0,), dtype=np.float32)
        a = str(agg or "mean").strip().lower()
        m = np.asarray(embs, dtype=np.float32)
        if weights is None:
            if a == "mean":
                v = m.mean(axis=0)
            elif a == "max":
                v = m.max(axis=0)
            elif a == "logsumexp":
                mx = m.max(axis=0)
                v = mx + np.log(np.exp(m - mx).sum(axis=0) + 1e-9)
                v = v - np.log(float(m.shape[0]) + 1e-9)
            else:
                raise ValueError(f"Unknown aggregation: {agg}")
        else:
            w = np.asarray(weights, dtype=np.float32).reshape(-1)
            w = np.maximum(w, 0.0)
            if w.sum() <= 0:
                w = np.ones((m.shape[0],), dtype=np.float32)
            w = w / (w.sum() + 1e-9)
            if a == "mean":
                v = (m * w.reshape(-1, 1)).sum(axis=0)
            elif a == "max":
                v = m.max(axis=0)
            elif a == "logsumexp":
                mx = m.max(axis=0)
                v = mx + np.log((np.exp(m - mx) * w.reshape(-1, 1)).sum(axis=0) + 1e-9)
            else:
                raise ValueError(f"Unknown aggregation: {agg}")
        nrm = float(np.linalg.norm(v))
        if nrm > 0:
            v = v / nrm
        return np.asarray(v, dtype=np.float32).reshape(-1)

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        tags_raw = getattr(doc, "hashtags", None)
        if tags_raw is None or not isinstance(tags_raw, list):
            if self.require_hashtags:
                raise RuntimeError(
                    "HashtagEmbedder requires doc.hashtags list (missing). "
                    "Ensure TagsExtractor ran with mutate_doc_hashtags=true and enable_extract_hashtags=true."
                )
            tags_raw = []

        canonical, freq, n_truncated = self._canonicalize_tags(list(tags_raw))
        n_tags_in = len(tags_raw) if isinstance(tags_raw, list) else 0

        # Optional privacy-safe hint: upstream TagsExtractor had hashtags extraction disabled by policy.
        disabled_by_policy_hint = 0.0
        try:
            tp = getattr(doc, "tp_artifacts", None)
            if isinstance(tp, dict) and isinstance(tp.get("tags"), dict):
                v = tp["tags"].get("hashtags_disabled_by_policy")
                if isinstance(v, (int, float)):
                    disabled_by_policy_hint = float(v)
        except Exception:
            disabled_by_policy_hint = 0.0

        def _stable_features_template() -> Dict[str, float]:
            # Stable schema, even when gated/empty.
            base: Dict[str, float] = {
                "tp_hashemb_present": 0.0,  # embedding computed (not "artifact exists")
                "tp_hashemb_dim": float("nan"),
                "tp_hashemb_tag_count": float("nan"),
                "tp_hashemb_l2_norm": float("nan"),
                # policy/inputs
                "tp_hashemb_require_hashtags_enabled": float(bool(self.require_hashtags)),
                "tp_hashemb_disabled_by_policy_hint": float(disabled_by_policy_hint),
                "tp_hashemb_n_input_tags": float(int(n_tags_in)),
                "tp_hashemb_n_unique_tags": float(int(len(canonical))),
                "tp_hashemb_n_tags_truncated": float(int(n_truncated)),
                # gating
                "tp_hashemb_compute_enabled": float(bool(self.compute_embedding)),
                "tp_hashemb_write_artifact_enabled": float(bool(self.write_artifact)),
                "tp_hashemb_artifact_written": 0.0,
                # cache
                "tp_hashemb_cache_enabled": float(bool(self.cache_enabled)),
                "tp_hashemb_cache_hit": float("nan"),
                # model/device
                "tp_hashemb_model_digest_u24": float(int(self._model_digest_u24)),
                "tp_hashemb_fp16": float(bool(self.fp16)),
                "tp_hashemb_device_cuda": float("cuda" in str(self.device).lower()),
                # timings
                "tp_hashemb_encode_ms": float("nan"),
                "tp_hashemb_agg_ms": float("nan"),
                # params flags
                "tp_hashemb_use_frequencies": float(bool(self.use_frequencies)),
                "tp_hashemb_agg_mean": float(self.aggregation == "mean"),
                "tp_hashemb_agg_max": float(self.aggregation == "max"),
                "tp_hashemb_agg_logsumexp": float(self.aggregation == "logsumexp"),
            }
            return base

        if not canonical:
            # valid empty
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            features_flat = _stable_features_template()
            features_flat["tp_hashemb_n_unique_tags"] = 0.0
            features_flat["tp_hashemb_tag_count"] = 0.0
            features_flat["tp_hashemb_cache_hit"] = 0.0
            return {
                "device": self.device,
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

        if not self.compute_embedding:
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            features_flat = _stable_features_template()
            features_flat["tp_hashemb_cache_hit"] = 0.0
            features_flat["tp_hashemb_tag_count"] = float(int(len(canonical)))
            return {
                "device": self.device,
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

        # Deterministic signature: includes model digest and canonicalized tags (sorted) + aggregation params.
        sig = f"{self.model_name}|{self.weights_digest}|agg={self.aggregation}|freq={int(self.use_frequencies)}|max_tags={int(self.max_tags)}|max_len={int(self.max_tag_len)}"
        if self.use_frequencies:
            payload = " ".join([f"{t}:{int(freq.get(t,1))}" for t in canonical])
        else:
            payload = " ".join(canonical)
        h = hashlib.sha256((sig + "||" + payload).encode("utf-8")).hexdigest()

        cache_hit = False
        vec_cached = None
        if self.cache_enabled:
            p = self._cache_path_vector(h)
            try:
                if p.exists() and (not self._is_cache_entry_expired(p)):
                    vec_cached = np.asarray(np.load(p), dtype=np.float32).reshape(-1)
                    cache_hit = vec_cached.size > 0
            except Exception:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

        t_enc_s = float("nan")
        t_agg_s = float("nan")
        if vec_cached is not None and vec_cached.size > 0:
            mean_vec = vec_cached
        else:
            t_enc0 = time.perf_counter()
            embs = self._encode_texts(canonical)
            t_enc_s = time.perf_counter() - t_enc0

            w = None
            if self.use_frequencies:
                w = np.asarray([float(freq.get(t, 1)) for t in canonical], dtype=np.float32)
            t_agg0 = time.perf_counter()
            mean_vec = self._agg(embs, weights=w, agg=self.aggregation)
            t_agg_s = time.perf_counter() - t_agg0

            if self.cache_enabled:
                try:
                    tmpc = self._cache_path_vector(h).with_suffix(".tmp.npy")
                    np.save(tmpc, mean_vec.astype(np.float32))
                    tmpc.replace(self._cache_path_vector(h))
                except Exception:
                    pass

        # Embedding computed in-memory; write artifact is optional.
        artifact_written = False
        out_path = self.artifacts_dir / "hashtag_embedding.npy"
        if self.write_artifact:
            tmp = out_path.with_suffix(".tmp.npy")
            np.save(tmp, mean_vec.astype(np.float32))
            tmp.replace(out_path)
            artifact_written = True

        # In-memory registry for downstream (no absolute paths in result/NPZ).
        if artifact_written:
            try:
                tp = getattr(doc, "tp_artifacts", None)
                if not isinstance(tp, dict):
                    tp = {}
                    setattr(doc, "tp_artifacts", tp)
                tp.setdefault("embeddings", {})
                tp["embeddings"]["hashtag"] = {
                    "relpath": out_path.name,
                    "kind": "vector",
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "weights_digest": self.weights_digest,
                    "dim": int(mean_vec.size),
                }
            except Exception:
                pass

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        features_flat = _stable_features_template()
        features_flat.update(
            {
                "tp_hashemb_present": 1.0,
                "tp_hashemb_dim": float(int(mean_vec.size)),
                "tp_hashemb_tag_count": float(int(len(canonical))),
                "tp_hashemb_l2_norm": float(np.linalg.norm(mean_vec)),
                "tp_hashemb_artifact_written": float(bool(artifact_written)),
                "tp_hashemb_cache_hit": float(bool(cache_hit)) if self.cache_enabled else 0.0,
                "tp_hashemb_encode_ms": float(round(float(t_enc_s) * 1000.0, 3)) if t_enc_s == t_enc_s else float("nan"),
                "tp_hashemb_agg_ms": float(round(float(t_agg_s) * 1000.0, 3)) if t_agg_s == t_agg_s else float("nan"),
            }
        )

        return {
            "device": self.device,
            "version": self.VERSION,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "weights_digest": self.weights_digest,
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

    @property
    def supports_batch(self) -> bool:
        """HashtagEmbedder supports batch processing."""
        return True

    def extract_batch(self, docs: List[VideoDocument]) -> List[Dict[str, Any]]:
        """
        Batch processing: encode all hashtags in one go, then aggregate per-document.
        """
        import time
        from collections import defaultdict

        started = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        n_docs = len(docs)
        if n_docs == 0:
            return []

        # Step 1: Canonicalize tags per document (preserve per-doc logic)
        doc_canonical: List[List[str]] = []
        doc_freq: List[Dict[str, int]] = []
        doc_n_truncated: List[int] = []
        doc_n_input: List[int] = []
        doc_disabled_hint: List[float] = []
        doc_has_tags: List[bool] = []

        for doc in docs:
            tags_raw = getattr(doc, "hashtags", None)
            if tags_raw is None or not isinstance(tags_raw, list):
                if self.require_hashtags:
                    # Will be handled in per-doc result
                    tags_raw = []
                else:
                    tags_raw = []

            canonical, freq, n_truncated = self._canonicalize_tags(list(tags_raw))
            n_tags_in = len(tags_raw) if isinstance(tags_raw, list) else 0

            disabled_by_policy_hint = 0.0
            try:
                tp = getattr(doc, "tp_artifacts", None)
                if isinstance(tp, dict) and isinstance(tp.get("tags"), dict):
                    v = tp["tags"].get("hashtags_disabled_by_policy")
                    if isinstance(v, (int, float)):
                        disabled_by_policy_hint = float(v)
            except Exception:
                pass

            doc_canonical.append(canonical)
            doc_freq.append(freq)
            doc_n_truncated.append(n_truncated)
            doc_n_input.append(n_tags_in)
            doc_disabled_hint.append(disabled_by_policy_hint)
            doc_has_tags.append(len(canonical) > 0)

        # Step 2: Collect all unique tags across all documents for batch encoding
        all_tags_set = set()
        tag_to_doc_indices: Dict[str, List[int]] = defaultdict(list)
        for doc_idx, canonical in enumerate(doc_canonical):
            for tag in canonical:
                all_tags_set.add(tag)
                tag_to_doc_indices[tag].append(doc_idx)

        all_tags_list = sorted(all_tags_set)  # Deterministic order

        # Step 3: Batch encode all unique tags
        t_enc0 = time.perf_counter()
        if all_tags_list:
            all_embeddings = self._encode_texts(all_tags_list)  # (N_unique, D)
        else:
            all_embeddings = np.zeros((0, 0), dtype=np.float32)
        t_enc_s = time.perf_counter() - t_enc0

        # Step 4: Build tag -> embedding map
        tag_to_emb: Dict[str, np.ndarray] = {}
        for i, tag in enumerate(all_tags_list):
            tag_to_emb[tag] = all_embeddings[i]

        # Step 5: Process each document: aggregate its tags, write artifact, build result
        results: List[Dict[str, Any]] = []

        for doc_idx, doc in enumerate(docs):
            doc_t0 = time.perf_counter()
            canonical = doc_canonical[doc_idx]
            freq = doc_freq[doc_idx]
            n_truncated = doc_n_truncated[doc_idx]
            n_tags_in = doc_n_input[doc_idx]
            disabled_by_policy_hint = doc_disabled_hint[doc_idx]
            has_tags = doc_has_tags[doc_idx]

            def _stable_features_template() -> Dict[str, float]:
                base: Dict[str, float] = {
                    "tp_hashemb_present": 0.0,
                    "tp_hashemb_dim": float("nan"),
                    "tp_hashemb_tag_count": float("nan"),
                    "tp_hashemb_l2_norm": float("nan"),
                    "tp_hashemb_require_hashtags_enabled": float(bool(self.require_hashtags)),
                    "tp_hashemb_disabled_by_policy_hint": float(disabled_by_policy_hint),
                    "tp_hashemb_n_input_tags": float(int(n_tags_in)),
                    "tp_hashemb_n_unique_tags": float(int(len(canonical))),
                    "tp_hashemb_n_tags_truncated": float(int(n_truncated)),
                    "tp_hashemb_compute_enabled": float(bool(self.compute_embedding)),
                    "tp_hashemb_write_artifact_enabled": float(bool(self.write_artifact)),
                    "tp_hashemb_artifact_written": 0.0,
                    "tp_hashemb_cache_enabled": float(bool(self.cache_enabled)),
                    "tp_hashemb_cache_hit": float("nan"),
                    "tp_hashemb_model_digest_u24": float(int(self._model_digest_u24)),
                    "tp_hashemb_fp16": float(bool(self.fp16)),
                    "tp_hashemb_device_cuda": float("cuda" in str(self.device).lower()),
                    "tp_hashemb_encode_ms": float("nan"),
                    "tp_hashemb_agg_ms": float("nan"),
                    "tp_hashemb_use_frequencies": float(bool(self.use_frequencies)),
                    "tp_hashemb_agg_mean": float(self.aggregation == "mean"),
                    "tp_hashemb_agg_max": float(self.aggregation == "max"),
                    "tp_hashemb_agg_logsumexp": float(self.aggregation == "logsumexp"),
                }
                return base

            error: Optional[str] = None

            if not has_tags:
                # Empty case
                sys_after = system_snapshot()
                mem_after = process_memory_bytes()
                total_s = time.perf_counter() - doc_t0
                features_flat = _stable_features_template()
                features_flat["tp_hashemb_n_unique_tags"] = 0.0
                features_flat["tp_hashemb_tag_count"] = 0.0
                features_flat["tp_hashemb_cache_hit"] = 0.0
                results.append({
                    "device": self.device,
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
                })
                continue

            if not self.compute_embedding:
                sys_after = system_snapshot()
                mem_after = process_memory_bytes()
                total_s = time.perf_counter() - doc_t0
                features_flat = _stable_features_template()
                features_flat["tp_hashemb_cache_hit"] = 0.0
                features_flat["tp_hashemb_tag_count"] = float(int(len(canonical)))
                results.append({
                    "device": self.device,
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
                })
                continue

            # Aggregate embeddings for this document's tags
            doc_embs_list: List[np.ndarray] = []
            for tag in canonical:
                if tag in tag_to_emb:
                    doc_embs_list.append(tag_to_emb[tag])

            if not doc_embs_list:
                # Should not happen if canonical is non-empty, but defensive
                mean_vec = np.zeros((0,), dtype=np.float32)
                t_agg_s = 0.0
            else:
                doc_embs = np.vstack(doc_embs_list)  # (n_tags, D)
                w = None
                if self.use_frequencies:
                    w = np.asarray([float(freq.get(t, 1)) for t in canonical], dtype=np.float32)
                t_agg0 = time.perf_counter()
                mean_vec = self._agg(doc_embs, weights=w, agg=self.aggregation)
                t_agg_s = time.perf_counter() - t_agg0

            # Write artifact to per-doc directory
            artifact_written = False
            doc_artifacts_dir = getattr(doc, "_tp_artifacts_dir", None)
            if doc_artifacts_dir:
                out_path = Path(doc_artifacts_dir) / "hashtag_embedding.npy"
            else:
                out_path = self.artifacts_dir / "hashtag_embedding.npy"

            if self.write_artifact and mean_vec.size > 0:
                try:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out_path.with_suffix(".tmp.npy")
                    np.save(tmp, mean_vec.astype(np.float32))
                    tmp.replace(out_path)
                    artifact_written = True
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
                    tp["embeddings"]["hashtag"] = {
                        "relpath": out_path.name,
                        "kind": "vector",
                        "model_name": self.model_name,
                        "model_version": self.model_version,
                        "weights_digest": self.weights_digest,
                        "dim": int(mean_vec.size),
                    }
                except Exception:
                    pass

            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - doc_t0

            features_flat = _stable_features_template()
            if mean_vec.size > 0:
                features_flat.update({
                    "tp_hashemb_present": 1.0,
                    "tp_hashemb_dim": float(int(mean_vec.size)),
                    "tp_hashemb_tag_count": float(int(len(canonical))),
                    "tp_hashemb_l2_norm": float(np.linalg.norm(mean_vec)),
                    "tp_hashemb_artifact_written": float(bool(artifact_written)),
                    "tp_hashemb_cache_hit": 0.0,  # Cache not used in batch mode (could be added later)
                    "tp_hashemb_encode_ms": float(round(t_enc_s * 1000.0 / n_docs, 3)) if t_enc_s == t_enc_s else float("nan"),  # Per-doc share
                    "tp_hashemb_agg_ms": float(round(t_agg_s * 1000.0, 3)) if t_agg_s == t_agg_s else float("nan"),
                })
            else:
                features_flat["tp_hashemb_cache_hit"] = 0.0

            results.append({
                "device": self.device,
                "version": self.VERSION,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "weights_digest": self.weights_digest,
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
                "error": error,
            })

        return results


