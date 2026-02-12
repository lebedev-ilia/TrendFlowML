"""
TitleEmbedder - извлекает L2-нормализованные эмбеддинги для заголовков (titles)
и одновременно предоставляет L2-нормы необработанных векторов (title_embedding_norm).

Особенности:
- Батчинг
- Локальный кеш по SHA256(content + model_name) — сохраняются и векторы, и нормы
- GPU (cuda) поддержка, опционально fp16
- Сохранение/загрузка кеша на диск (atomic save)
- Возвращает numpy массивы:
    - embeddings: shape (N, D) — L2-нормированные векторы
    - norms: shape (N,) — L2-нормы raw (unnormalized) векторов
"""

import os
import hashlib
import time
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import torch

from src.core.base_extractor import BaseExtractor  # noqa: E402
from src.core.text_utils import normalize_whitespace  # noqa: E402
from src.schemas.models import VideoDocument  # noqa: E402
from src.core.metrics import system_snapshot, process_memory_bytes  # noqa: E402
from src.core.model_registry import get_model_with_meta  # noqa: E402
from src.core.path_utils import default_artifacts_dir, default_cache_dir  # noqa: E402


class TitleEmbedder(BaseExtractor):
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
        device: Optional[str] = "cpu",
        fp16: bool = True,
        batch_size: int = 128,
        artifacts_dir: Optional[str] = None,
        require_title: bool = False,
        compute_embedding: bool = True,
        write_artifact: bool = True,
        # Back-compat alias: if False → write_artifact=False
        write_embedding_artifact: bool = True,
        compute_raw_norm: bool = True,
        emit_extra_metrics: bool = False,
    ):
        self.model_name = model_name
        self.cache_enabled = bool(cache_enabled)
        self.cache_ttl_days = float(cache_ttl_days) if cache_ttl_days is not None else None
        self.cache_max_items = int(cache_max_items) if cache_max_items is not None else None
        self.cache_max_bytes = int(cache_max_bytes) if cache_max_bytes is not None else None
        self.cache_cleanup_on_init = bool(cache_cleanup_on_init)
        self.cache_cleanup_max_seconds = float(cache_cleanup_max_seconds)
        base_cache = default_cache_dir() / "embed_cache"
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else base_cache
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.device = str(device or "cpu")
        self.require_title = bool(require_title)
        self.compute_embedding = bool(compute_embedding)
        self.write_artifact = bool(write_artifact) and bool(write_embedding_artifact)
        self.compute_raw_norm = bool(compute_raw_norm)
        self.emit_extra_metrics = bool(emit_extra_metrics)

        # metrics: init snapshots
        init_sys_before = system_snapshot()
        init_mem_before = process_memory_bytes()

        self.fp16 = fp16 and ("cuda" in self.device)

        # Resolve & load model strictly via dp_models (offline, fail-fast).
        self.model, self.weights_digest, self.model_version = get_model_with_meta(
            model_name=self.model_name, device=self.device, fp16=(fp16 and ("cuda" in self.device))
        )
        self.fp16 = fp16 and ("cuda" in self.device)
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

    # release_resources removed: models are shared via registry and persist

    @staticmethod
    def _hash_text(text: str, model_key: str) -> str:
        normalized = " ".join(text.strip().split())
        payload = (model_key + "||" + normalized).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _cache_path_vector(self, h: str) -> Path:
        return self.cache_dir / f"{h}.npy"

    def _cache_path_norm(self, h: str) -> Path:
        return self.cache_dir / f"{h}.norm.npy"

    def _is_cache_entry_expired(self, path: Path) -> bool:
        if self.cache_ttl_days is None:
            return False
        try:
            age_s = max(0.0, time.time() - float(path.stat().st_mtime))
            return age_s > float(self.cache_ttl_days) * 86400.0
        except Exception:
            return False

    def _load_vector_from_cache(self, h: str) -> Optional[np.ndarray]:
        if not self.cache_enabled:
            return None
        p = self._cache_path_vector(h)
        if p.exists():
            try:
                if self._is_cache_entry_expired(p):
                    try:
                        p.unlink()
                    except Exception:
                        pass
                    return None
                arr = np.load(p)
                # ensure float32
                return arr.astype(np.float32)
            except Exception:
                try:
                    p.unlink()
                except Exception:
                    pass
        return None

    def _load_norm_from_cache(self, h: str) -> Optional[float]:
        if not self.cache_enabled:
            return None
        p = self._cache_path_norm(h)
        if p.exists():
            try:
                if self._is_cache_entry_expired(p):
                    try:
                        p.unlink()
                    except Exception:
                        pass
                    return None
                arr = np.load(p)
                return float(arr.item())
            except Exception:
                try:
                    p.unlink()
                except Exception:
                    pass
        return None

    def _save_vector_to_cache(self, h: str, vector: np.ndarray):
        if not self.cache_enabled:
            return
        p = self._cache_path_vector(h)
        tmp = p.with_suffix(".tmp.npy")
        # ensure dtype float32
        to_save = np.asarray(vector, dtype=np.float32)
        np.save(tmp, to_save)
        tmp.replace(p)

    def _save_norm_to_cache(self, h: str, val: float):
        if not self.cache_enabled:
            return
        p = self._cache_path_norm(h)
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, np.array(val, dtype=np.float32))
        tmp.replace(p)

    def _cleanup_cache_best_effort(self) -> None:
        """
        Best-effort cache pruning:
        - TTL eviction (mtime based)
        - max_items/max_bytes eviction (LRU by mtime)
        Runs with a small time budget to avoid impacting latency.
        """
        t0 = time.perf_counter()
        try:
            entries = []
            for fn in os.listdir(self.cache_dir):
                if not (fn.endswith(".npy") or fn.endswith(".norm.npy")):
                    continue
                p = self.cache_dir / fn
                try:
                    st = p.stat()
                except Exception:
                    continue
                # TTL removal first
                if self._is_cache_entry_expired(p):
                    try:
                        p.unlink()
                    except Exception:
                        pass
                    continue
                entries.append((float(st.st_mtime), int(st.st_size), p))
                if (time.perf_counter() - t0) > self.cache_cleanup_max_seconds:
                    break

            # If we ran out of budget, do not attempt size-based pruning now.
            if (time.perf_counter() - t0) > self.cache_cleanup_max_seconds:
                return

            # Enforce max_items/max_bytes by removing oldest first.
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

    def embed_titles(
        self,
        titles: List[str],
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Возвращает L2-нормализованные эмбеддинги для списка заголовков.
        (Совместимая версия — без явного возвращения норм).
        """
        embeddings, _ = self.embed_titles_with_norms(titles, use_cache=use_cache, return_norms=False)
        return embeddings

    def embed_titles_with_norms(
        self,
        titles: List[str],
        use_cache: bool = True,
        return_norms: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Возвращает (embeddings, norms)
        embeddings: np.ndarray shape (N, D) — L2-normalized embeddings
        norms: np.ndarray shape (N,) — L2 norms of raw/unnormalized embeddings (if return_norms=True)
        """
        if not isinstance(titles, (list, tuple)):
            raise ValueError("titles must be a list of strings")

        n_total = len(titles)
        embeddings: List[Optional[np.ndarray]] = [None] * n_total
        norms: List[Optional[float]] = [None] * n_total
        to_compute_indices = []
        hashes = []
        model_key = f"{self.model_name}|{self.weights_digest}"

        # 1) try load both vector and norm from cache (preferred)
        for i, t in enumerate(titles):
            h = self._hash_text(t, model_key)
            hashes.append(h)
            if use_cache and self.cache_enabled:
                vec = self._load_vector_from_cache(h)
                nrm = self._load_norm_from_cache(h) if self.compute_raw_norm else None
                if vec is not None and nrm is not None:
                    embeddings[i] = vec
                    norms[i] = float(nrm)
                else:
                    # if vector exists but norm missing, re-compute (so treat as missing)
                    to_compute_indices.append(i)
            else:
                to_compute_indices.append(i)

        # 2) compute missing ones in batches: we will request raw vectors (normalize_embeddings=False)
        if len(to_compute_indices) > 0:
            texts_to_compute = [titles[i] for i in to_compute_indices]
            computed_raw_batches = []
            m = len(texts_to_compute)
            for start in range(0, m, self.batch_size):
                end = min(m, start + self.batch_size)
                batch = texts_to_compute[start:end]
                with torch.no_grad():
                    # IMPORTANT: request raw vectors so we can compute raw norms
                    raw = self.model.encode(
                        batch,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=False,
                    )
                raw = np.asarray(raw, dtype=np.float32)
                computed_raw_batches.append(raw)

            raw_all = np.vstack(computed_raw_batches)
            # compute raw norms
            raw_norms = np.linalg.norm(raw_all, axis=1) if self.compute_raw_norm else np.ones((raw_all.shape[0],), dtype=np.float32)
            # avoid zero norms
            raw_norms_safe = raw_norms.copy()
            raw_norms_safe[raw_norms_safe == 0] = 1.0
            # normalized vectors
            normalized_vectors = raw_all / raw_norms_safe.reshape(-1, 1)

            # assign back and cache both normalized vector and raw norm
            j = 0
            for idx in to_compute_indices:
                vec = normalized_vectors[j]
                nrm = float(raw_norms[j]) if self.compute_raw_norm else float("nan")
                embeddings[idx] = vec
                norms[idx] = nrm
                if use_cache and self.cache_enabled:
                    try:
                        self._save_vector_to_cache(hashes[idx], vec)
                        if self.compute_raw_norm:
                            self._save_norm_to_cache(hashes[idx], nrm)
                    except Exception:
                        # swallow caching errors
                        pass
                j += 1

        # 3) all filled — stack and final safety normalization for embeddings
        emb_stack = np.vstack([e for e in embeddings]).astype(np.float32)
        # safety L2 normalize (in case cached vectors were not normalized)
        emb_norms = np.linalg.norm(emb_stack, axis=1, keepdims=True)
        emb_norms[emb_norms == 0] = 1.0
        emb_stack = emb_stack / emb_norms

        if return_norms:
            norms_arr = np.array([float(x) if x is not None else float("nan") for x in norms], dtype=np.float32)
            return emb_stack, norms_arr
        else:
            return emb_stack, None


    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        """
        Реализация интерфейса BaseExtractor с метриками и сохранением артефактов.
        Возвращает словарь с полями device, version, timings, system, result, error.
        """
        import time

        started = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()
        error: Optional[str] = None

        # Preconditions: title must be a non-empty string (production-grade, no-fallback).
        t0 = time.perf_counter()
        title = normalize_whitespace(getattr(doc, "title", None))
        title_present = float(isinstance(title, str) and bool(title.strip()))
        if self.require_title and title_present == 0.0:
            raise RuntimeError("TitleEmbedder requires non-empty VideoDocument.title (require_title=true)")

        # Stable features schema (always present).
        def _stable_features_template() -> Dict[str, float]:
            return {
                "tp_titleemb_present": 0.0,  # embedding computed (not "artifact exists")
                "tp_titleemb_dim": float("nan"),
                "tp_titleemb_norm_raw": float("nan"),
                "tp_titleemb_l2_norm": float("nan"),
                "tp_titleemb_title_present": float(title_present),
                "tp_titleemb_require_title_enabled": float(bool(self.require_title)),
                "tp_titleemb_compute_enabled": float(bool(self.compute_embedding)),
                "tp_titleemb_write_artifact_enabled": float(bool(self.write_artifact)),
                "tp_titleemb_artifact_written": 0.0,
                "tp_titleemb_cache_enabled": float(bool(self.cache_enabled)),
                "tp_titleemb_cache_hit": float("nan"),
                "tp_titleemb_fp16": float(bool(self.fp16)),
                "tp_titleemb_device_cuda": float("cuda" in str(self.device).lower()),
                "tp_titleemb_model_digest_u24": float(int(self._model_digest_u24)),
                "tp_titleemb_encode_ms": float("nan"),
                "tp_titleemb_compute_raw_norm": float(bool(self.compute_raw_norm)),
            }

        if title_present == 0.0:
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = (time.perf_counter() - started)
            features_flat = _stable_features_template()
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
                "timings_s": {"encode": 0.0, "total": round(total_s, 3)},
                "result": {"features_flat": features_flat},
                "error": None,
            }
            return result

        if not self.compute_embedding:
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = (time.perf_counter() - started)
            features_flat = _stable_features_template()
            features_flat["tp_titleemb_cache_hit"] = 0.0
            result = {
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
                "timings_s": {"encode": 0.0, "total": round(total_s, 3)},
                "result": {"features_flat": features_flat},
                "error": None,
            }
            return result

        # processing block: encode
        cache_hit = False
        embeddings, norms = self.embed_titles_with_norms([title], use_cache=True, return_norms=True)
        encode_s = (time.perf_counter() - t0)

        vec = embeddings[0]
        norm_val = float(norms[0]) if (norms is not None and self.compute_raw_norm) else float("nan")
        # Cache hit heuristic: if encode is very small and cache enabled, likely hit. Better: check cache directly.
        try:
            model_key = f"{self.model_name}|{self.weights_digest}"
            h0 = self._hash_text(title, model_key)
            cache_hit = bool(self.cache_enabled) and (self._cache_path_vector(h0).exists())
        except Exception:
            cache_hit = False

        # Write per-run artifact (fixed name); optional.
        emb_path = self.artifacts_dir / "title_embedding.npy"
        artifact_written = False
        try:
            if self.write_artifact:
                tmp = emb_path.with_suffix(".tmp.npy")
                np.save(tmp, vec.astype(np.float32))
                tmp.replace(emb_path)
                artifact_written = True
        except Exception as e:
            error = f"artifact_save_error: {e}"

        # In-memory registry for downstream (no absolute paths in result/NPZ).
        try:
            tp = getattr(doc, "tp_artifacts", None)
            if not isinstance(tp, dict):
                tp = {}
                setattr(doc, "tp_artifacts", tp)
            tp.setdefault("embeddings", {})
            if artifact_written:
                tp["embeddings"]["title"] = {
                    "relpath": emb_path.name,
                    "kind": "vector",
                    "model_name": self.model_name,
                    "weights_digest": self.weights_digest,
                    "model_version": self.model_version,
                    "dim": int(vec.size),
                }
        except Exception:
            pass

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = (time.perf_counter() - started)

        def _gpu_used_mb(snap: Any) -> int:
            try:
                g = (snap or {}).get("gpu") or {}
                arr = g.get("gpus") or []
                return max([int(x.get("memory_used_mb", 0)) for x in arr] or [0])
            except Exception:
                return 0

        gpu_peak_mb = max(
            _gpu_used_mb(self._init_metrics.get("pre_init")),
            _gpu_used_mb(self._init_metrics.get("post_init")),
            _gpu_used_mb(sys_after),
        )

        features_flat = _stable_features_template()
        features_flat.update(
            {
                "tp_titleemb_present": 1.0,
                "tp_titleemb_dim": float(int(vec.size)),
                "tp_titleemb_norm_raw": float(norm_val),
                "tp_titleemb_l2_norm": float(np.linalg.norm(vec)),
                "tp_titleemb_cache_hit": float(bool(cache_hit)) if self.cache_enabled else 0.0,
                "tp_titleemb_encode_ms": float(round(encode_s * 1000.0, 3)),
                "tp_titleemb_artifact_written": float(bool(artifact_written)),
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
                    "gpu_peak_mb": int(gpu_peak_mb),
                },
            },
            "timings_s": {
                "encode": round(encode_s, 3),
                "total": round(total_s, 3),
            },
            "result": {"features_flat": features_flat},
            "error": error,
        }

        return result

    @property
    def supports_batch(self) -> bool:
        return True

    def extract_batch(self, docs: List[VideoDocument]) -> List[Dict[str, Any]]:
        """
        Batch-optimized title embedding extraction.

        - Encodes all present titles in one (batched) pass via `embed_titles_with_norms`.
        - Preserves per-document artifact layout by writing `title_embedding.npy` into:
          - `doc._tp_artifacts_dir` if provided by orchestrator (preferred), else
          - `self.artifacts_dir`.
        - Updates `doc.tp_artifacts["embeddings"]["title"]` exactly like `extract()`.

        NOTE: This method is safe to call directly, but orchestrator-level batching
        (shared model instance across docs) will be introduced in later stages.
        """
        started = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        docs_local = list(docs or [])
        if not docs_local:
            return []

        # Pre-normalize titles (deterministic).
        titles: List[str] = []
        title_present_flags: List[float] = []
        for d in docs_local:
            t = normalize_whitespace(getattr(d, "title", None))
            present = float(isinstance(t, str) and bool(t.strip()))
            title_present_flags.append(present)
            titles.append(t if isinstance(t, str) else "")

        def _stable_features_template(title_present: float) -> Dict[str, float]:
            return {
                "tp_titleemb_present": 0.0,
                "tp_titleemb_dim": float("nan"),
                "tp_titleemb_norm_raw": float("nan"),
                "tp_titleemb_l2_norm": float("nan"),
                "tp_titleemb_title_present": float(title_present),
                "tp_titleemb_require_title_enabled": float(bool(self.require_title)),
                "tp_titleemb_compute_enabled": float(bool(self.compute_embedding)),
                "tp_titleemb_write_artifact_enabled": float(bool(self.write_artifact)),
                "tp_titleemb_artifact_written": 0.0,
                "tp_titleemb_cache_enabled": float(bool(self.cache_enabled)),
                "tp_titleemb_cache_hit": float("nan"),
                "tp_titleemb_fp16": float(bool(self.fp16)),
                "tp_titleemb_device_cuda": float("cuda" in str(self.device).lower()),
                "tp_titleemb_model_digest_u24": float(int(self._model_digest_u24)),
                "tp_titleemb_encode_ms": float("nan"),
                "tp_titleemb_compute_raw_norm": float(bool(self.compute_raw_norm)),
            }

        # If compute is disabled, return stable empty for all.
        if not self.compute_embedding:
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - started
            out: List[Dict[str, Any]] = []
            for present in title_present_flags:
                ff = _stable_features_template(present)
                ff["tp_titleemb_cache_hit"] = 0.0
                out.append(
                    {
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
                        "timings_s": {"encode": 0.0, "total": round(total_s, 3)},
                        "result": {"features_flat": ff},
                        "error": None,
                    }
                )
            return out

        # If require_title, enforce per-doc (fail-fast for the doc, not whole batch).
        # Here we preserve overall batch processing: docs missing title get an error payload.
        missing_required = [i for i, p in enumerate(title_present_flags) if (self.require_title and p == 0.0)]

        # Collect indices to encode (only those with present titles and not missing-required).
        idx_to_encode = [i for i, p in enumerate(title_present_flags) if p == 1.0 and i not in set(missing_required)]
        titles_to_encode = [titles[i] for i in idx_to_encode]

        encode_s_total = 0.0
        embeddings = None
        norms = None
        if titles_to_encode:
            t_enc0 = time.perf_counter()
            embeddings, norms = self.embed_titles_with_norms(titles_to_encode, use_cache=True, return_norms=True)
            encode_s_total = time.perf_counter() - t_enc0

        # Build outputs per doc
        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - started

        # GPU peak (best-effort, shared across docs)
        def _gpu_used_mb(snap: Any) -> int:
            try:
                g = (snap or {}).get("gpu") or {}
                arr = g.get("gpus") or []
                return max([int(x.get("memory_used_mb", 0)) for x in arr] or [0])
            except Exception:
                return 0

        gpu_peak_mb = max(
            _gpu_used_mb(self._init_metrics.get("pre_init")),
            _gpu_used_mb(self._init_metrics.get("post_init")),
            _gpu_used_mb(sys_after),
        )

        # Map encoded vectors back
        enc_pos = {doc_idx: j for j, doc_idx in enumerate(idx_to_encode)}
        approx_encode_s_per_doc = (encode_s_total / max(1, len(idx_to_encode))) if idx_to_encode else 0.0

        out: List[Dict[str, Any]] = []
        for i, d in enumerate(docs_local):
            error: Optional[str] = None
            present = float(title_present_flags[i])
            if self.require_title and present == 0.0:
                # Required but missing -> per-doc fail-fast semantics (in batch)
                error = "TitleEmbedder requires non-empty VideoDocument.title (require_title=true)"
                ff = _stable_features_template(present)
                out.append(
                    {
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
                                "gpu_peak_mb": int(gpu_peak_mb),
                            },
                        },
                        "timings_s": {"encode": 0.0, "total": round(total_s, 3)},
                        "result": {"features_flat": ff},
                        "error": error,
                        "status": "error",
                    }
                )
                continue

            if present == 0.0:
                ff = _stable_features_template(present)
                out.append(
                    {
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
                                "gpu_peak_mb": int(gpu_peak_mb),
                            },
                        },
                        "timings_s": {"encode": 0.0, "total": round(total_s, 3)},
                        "result": {"features_flat": ff},
                        "error": None,
                    }
                )
                continue

            j = enc_pos.get(i)
            if j is None or embeddings is None:
                # Should not happen; fallback to single-doc extract for safety
                out.append(self.extract(d))
                continue

            vec = np.asarray(embeddings[j], dtype=np.float32).reshape(-1)
            norm_val = float(norms[j]) if (norms is not None and self.compute_raw_norm) else float("nan")

            # Cache-hit heuristic: check cache file existence
            cache_hit = False
            try:
                model_key = f"{self.model_name}|{self.weights_digest}"
                h0 = self._hash_text(titles[i], model_key)
                cache_hit = bool(self.cache_enabled) and (self._cache_path_vector(h0).exists())
            except Exception:
                cache_hit = False

            # Resolve per-doc artifacts dir
            artifacts_dir = None
            try:
                artifacts_dir = getattr(d, "_tp_artifacts_dir", None)
            except Exception:
                artifacts_dir = None
            art_dir_path = Path(str(artifacts_dir)).expanduser().resolve() if artifacts_dir else self.artifacts_dir
            try:
                art_dir_path.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            # Write per-run artifact (fixed name); optional.
            emb_path = art_dir_path / "title_embedding.npy"
            artifact_written = False
            try:
                if self.write_artifact:
                    tmp = emb_path.with_suffix(".tmp.npy")
                    np.save(tmp, vec.astype(np.float32))
                    tmp.replace(emb_path)
                    artifact_written = True
            except Exception as e:
                error = f"artifact_save_error: {e}"

            # In-memory registry for downstream (no absolute paths in result/NPZ).
            try:
                tp = getattr(d, "tp_artifacts", None)
                if not isinstance(tp, dict):
                    tp = {}
                    setattr(d, "tp_artifacts", tp)
                tp.setdefault("embeddings", {})
                if artifact_written:
                    tp["embeddings"]["title"] = {
                        "relpath": emb_path.name,
                        "kind": "vector",
                        "model_name": self.model_name,
                        "weights_digest": self.weights_digest,
                        "model_version": self.model_version,
                        "dim": int(vec.size),
                    }
            except Exception:
                pass

            ff = _stable_features_template(present)
            ff.update(
                {
                    "tp_titleemb_present": 1.0,
                    "tp_titleemb_dim": float(int(vec.size)),
                    "tp_titleemb_norm_raw": float(norm_val),
                    "tp_titleemb_l2_norm": float(np.linalg.norm(vec)),
                    "tp_titleemb_cache_hit": float(bool(cache_hit)) if self.cache_enabled else 0.0,
                    "tp_titleemb_encode_ms": float(round(approx_encode_s_per_doc * 1000.0, 3)),
                    "tp_titleemb_artifact_written": float(bool(artifact_written)),
                }
            )

            out.append(
                {
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
                            "gpu_peak_mb": int(gpu_peak_mb),
                        },
                    },
                    "timings_s": {
                        "encode": round(approx_encode_s_per_doc, 3),
                        "total": round(total_s, 3),
                    },
                    "result": {"features_flat": ff},
                    "error": error,
                }
            )

        return out


# Example usage
# if __name__ == "__main__":
#    titles = [
#        "Что такое искусственный интеллект и как он работает?",
#        "5 простых трюков для ускорения Python кода",
#    ]
#    embedder = TitleEmbedder(
#        model_name="sentence-transformers/all-mpnet-base-v2",
#        cache_dir="./embed_cache",
#        fp16=False,           # True → попробуйте на современной NVidia GPU
#        batch_size=64,
#    )
#    embs, norms = embedder.embed_titles_with_norms(titles, use_cache=True, return_norms=True)
#    print("embeddings shape:", embs.shape)     # (2, dim)
#    print("first embedding (norm):", np.linalg.norm(embs[0]))
#    print("raw norms:", norms)                 # norms of raw vectors (before normalization)
