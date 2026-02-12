"""
DescriptionEmbedder — извлекает L2-нормализованные эмбеддинги для описаний (description)
и одновременно вычисляет L2-нормы необработанных векторов (description_embedding_norm).

Особенности:
- Поддержка длинных описаний (chunk-and-aggregate)
- Attention-weighted pooling по длине чанка
- Батчинг
- GPU (cuda) поддержка, fp16 опционально
- Кеш по SHA256(content + model_name)
- Сохранение артефактов и метрик
"""

import os
import hashlib
import time
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import torch
from src.core.model_registry import get_model_with_meta
from src.core.path_utils import default_artifacts_dir, default_cache_dir

from src.core.base_extractor import BaseExtractor  # noqa
from src.core.text_utils import normalize_whitespace  # noqa
from src.schemas.models import VideoDocument  # noqa
from src.core.metrics import system_snapshot, process_memory_bytes  # noqa


class DescriptionEmbedder(BaseExtractor):
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
        batch_size: int = 32,
        artifacts_dir: Optional[str] = None,
        tokenizer_spec_name: str = "shared_tokenizer_v1",
        max_chunk_tokens_model: int = 512,
        pooling_strategy: str = "length_weighted_mean",
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
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.tokenizer_spec_name = str(tokenizer_spec_name or "shared_tokenizer_v1")
        self.max_chunk_tokens_model = int(max_chunk_tokens_model)
        self.pooling_strategy = str(pooling_strategy or "length_weighted_mean").strip().lower()

        self.device = str(device or "cpu")

        self.fp16 = fp16 and ("cuda" in self.device)
        self.compute_embedding = bool(compute_embedding)
        self.write_artifact = bool(write_artifact) and bool(write_embedding_artifact)
        self.compute_raw_norm = bool(compute_raw_norm)
        self.emit_extra_metrics = bool(emit_extra_metrics)

        # --- инициализация модели ---
        init_sys_before = system_snapshot()
        init_mem_before = process_memory_bytes()

        # Resolve model + tokenizer via dp_models (offline, fail-fast).
        try:
            from dp_models import get_global_model_manager  # type: ignore
            from tokenizers import Tokenizer  # type: ignore

            mm = get_global_model_manager()
            # Tokenizer (strict, offline)

            tok_spec = mm.get_spec(model_name=str(self.tokenizer_spec_name))
            _td, _tp, _trt, _teng, tok_digest, tok_arts = mm.resolve(tok_spec)
            tok_path = list((tok_arts or {}).values())[0] if tok_arts else None
            if not tok_path:
                raise RuntimeError(f"{self.tokenizer_spec_name} artifacts are empty")
            self.tokenizer_weights_digest = str(tok_digest or "unknown")
            self._tokenizer = Tokenizer.from_file(tok_path)
        except Exception as e:
            raise RuntimeError(f"DescriptionEmbedder | dp_models resolve failed: {e}") from e

        # Model (strict, offline) via dp_models.
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

    # release_resources removed: models are shared via registry and persist

    # -------------------- КЕШ --------------------
    @staticmethod
    def _hash_text(text: str, key: str) -> str:
        normalized = " ".join(text.strip().split())
        payload = (key + "||" + normalized).encode("utf-8")
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
                    p.unlink(missing_ok=True)
                    return None
                arr = np.load(p)
                return arr.astype(np.float32)
            except Exception:
                p.unlink(missing_ok=True)
        return None

    def _load_norm_from_cache(self, h: str) -> Optional[float]:
        if not self.cache_enabled:
            return None
        p = self._cache_path_norm(h)
        if p.exists():
            try:
                if self._is_cache_entry_expired(p):
                    p.unlink(missing_ok=True)
                    return None
                arr = np.load(p)
                return float(arr.item())
            except Exception:
                p.unlink(missing_ok=True)
        return None

    def _save_vector_to_cache(self, h: str, vec: np.ndarray):
        if not self.cache_enabled:
            return
        p = self._cache_path_vector(h)
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, np.asarray(vec, dtype=np.float32))
        tmp.replace(p)

    def _save_norm_to_cache(self, h: str, val: float):
        if not self.cache_enabled:
            return
        p = self._cache_path_norm(h)
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, np.array(val, dtype=np.float32))
        tmp.replace(p)

    def _cleanup_cache_best_effort(self) -> None:
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

    # -------------------- ЛОГИКА --------------------
    def _chunk_text_token_aware(self, text: str) -> tuple[List[str], List[int]]:
        """
        Token-aware chunking using shared tokenizer.
        Returns (chunk_texts, chunk_token_counts).
        """
        enc = None
        try:
            enc = self._tokenizer.encode(text, add_special_tokens=False)  # type: ignore[arg-type]
        except Exception:
            enc = self._tokenizer.encode(text)  # type: ignore[no-untyped-call]
        ids = list(getattr(enc, "ids", []) or [])
        if len(ids) <= max(1, self.max_chunk_tokens_model):
            return [text], [len(ids)]
        out_txt: List[str] = []
        out_tok: List[int] = []
        step = max(1, self.max_chunk_tokens_model)
        for i in range(0, len(ids), step):
            chunk_ids = ids[i : i + step]
            if not chunk_ids:
                continue
            try:
                chunk_text = self._tokenizer.decode([int(x) for x in chunk_ids], skip_special_tokens=True)
            except Exception:
                raise RuntimeError("DescriptionEmbedder: tokenizer decode failed (strict token-aware chunking)") from None
            if isinstance(chunk_text, str) and chunk_text.strip():
                out_txt.append(chunk_text.strip())
                out_tok.append(int(len(chunk_ids)))
        if not out_txt:
            return [text], [len(ids)]
        return out_txt, out_tok

    def _weighted_pooling(self, embeddings: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weighted_sum = (embeddings * weights.unsqueeze(-1)).sum(dim=0)
        denom = weights.sum() + 1e-9
        return weighted_sum / denom

    @staticmethod
    def _pool(embeds: torch.Tensor, weights: torch.Tensor, strategy: str) -> torch.Tensor:
        s = str(strategy or "length_weighted_mean").strip().lower()
        if embeds.ndim != 2:
            raise ValueError("embeds must be 2D (N, D)")
        if s == "mean":
            return embeds.mean(dim=0)
        if s == "length_weighted_mean":
            return (embeds * weights.unsqueeze(-1)).sum(dim=0) / (weights.sum() + 1e-9)
        if s == "max":
            return torch.max(embeds, dim=0).values
        if s == "logsumexp":
            # stable logsumexp pooling; scale back by N to keep magnitude comparable-ish
            return torch.logsumexp(embeds, dim=0) - torch.log(torch.tensor(float(embeds.shape[0]), device=embeds.device) + 1e-9)
        raise ValueError(f"Unknown pooling_strategy: {strategy}")

    # -------------------- ОСНОВНОЙ МЕТОД --------------------
    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        start = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()
        error: Optional[str] = None

        text = normalize_whitespace(doc.description or "")

        # Stable features schema (always present).
        def _stable_features_template() -> Dict[str, float]:
            return {
                "tp_descemb_present": 0.0,  # embedding computed (not "artifact exists")
                "tp_descemb_dim": float("nan"),
                "tp_descemb_norm_raw": float("nan"),
                "tp_descemb_l2_norm": float("nan"),
                "tp_descemb_description_present": float(bool(text)),
                "tp_descemb_compute_enabled": float(bool(self.compute_embedding)),
                "tp_descemb_write_artifact_enabled": float(bool(self.write_artifact)),
                "tp_descemb_artifact_written": 0.0,
                "tp_descemb_cache_enabled": float(bool(self.cache_enabled)),
                "tp_descemb_cache_hit": float("nan"),
                "tp_descemb_fp16": float(bool(self.fp16)),
                "tp_descemb_device_cuda": float("cuda" in str(self.device).lower()),
                "tp_descemb_model_digest_u24": float(int(self._model_digest_u24)),
                "tp_descemb_pooling_length_weighted": float(self.pooling_strategy == "length_weighted_mean"),
                "tp_descemb_n_chunks": float("nan"),
                "tp_descemb_avg_chunk_tokens": float("nan"),
                "tp_descemb_chunk_ms": float("nan"),
                "tp_descemb_encode_ms": float("nan"),
                "tp_descemb_pool_ms": float("nan"),
            }

        if not text:
            # Valid empty: do NOT create fake vectors, do NOT register tp_artifacts.
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = (time.perf_counter() - start)

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
                        "gpu_peak_mb": int(gpu_peak_mb),
                    },
                },
                "timings_s": {"total": round(total_s, 3)},
                "result": {"features_flat": _stable_features_template()},
                "error": None,
            }

        # Feature gating: allow disabling vector computation entirely (then no relpath for downstream).
        if not self.compute_embedding:
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = (time.perf_counter() - start)
            features_flat = _stable_features_template()
            features_flat["tp_descemb_cache_hit"] = 0.0
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

        config_sig = f"{self.model_name}|{self.weights_digest}|tok={self.tokenizer_spec_name}|tokd={self.tokenizer_weights_digest}|mxtok={int(self.max_chunk_tokens_model)}|pool={self.pooling_strategy}"
        h = self._hash_text(text, config_sig)

        cache_hit = False
        # extra-metrics placeholders (filled only on compute path)
        chunk_s = float("nan")
        encode_s = float("nan")
        pool_s = float("nan")
        n_chunks = float("nan")
        avg_chunk_tokens = float("nan")
        vec_cached = self._load_vector_from_cache(h) if self.cache_enabled else None
        norm_cached = self._load_norm_from_cache(h) if (self.cache_enabled and self.compute_raw_norm) else None
        if vec_cached is not None and (norm_cached is not None or (not self.compute_raw_norm)):
            cache_hit = True
            pooled_np = np.asarray(vec_cached, dtype=np.float32).reshape(-1)
            norm_val = float(norm_cached) if (norm_cached is not None and self.compute_raw_norm) else float("nan")
        else:
            # --- chunk and embed (token-aware) ---
            t_chunk0 = time.perf_counter()
            chunks, chunk_tok_counts = self._chunk_text_token_aware(text)
            chunk_s = time.perf_counter() - t_chunk0
            try:
                n_chunks = float(int(len(chunks)))
                avg_chunk_tokens = float(np.mean(np.asarray(chunk_tok_counts, dtype=np.float32))) if chunk_tok_counts else float("nan")
            except Exception:
                n_chunks = float("nan")
                avg_chunk_tokens = float("nan")

            t_enc0 = time.perf_counter()
            with torch.no_grad():
                embeds = self.model.encode(
                    chunks,
                    batch_size=self.batch_size,
                    convert_to_tensor=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
            encode_s = time.perf_counter() - t_enc0

            # --- pooling ---
            t_pool0 = time.perf_counter()
            weights = torch.tensor(chunk_tok_counts, dtype=torch.float32, device=embeds.device)
            weights = weights / (weights.sum() + 1e-9)
            pooled = self._pool(embeds, weights, self.pooling_strategy)
            pooled_norm = torch.nn.functional.normalize(pooled, p=2, dim=0)
            pooled_np = pooled_norm.detach().cpu().numpy().astype(np.float32).reshape(-1)
            norm_val = float(torch.linalg.norm(pooled).item()) if self.compute_raw_norm else float("nan")
            pool_s = time.perf_counter() - t_pool0

            # cache (best-effort)
            if self.cache_enabled:
                try:
                    self._save_vector_to_cache(h, pooled_np)
                    if self.compute_raw_norm:
                        self._save_norm_to_cache(h, float(norm_val))
                except Exception:
                    pass
        # --- save per-run artifact (fixed name; optional) ---
        artifact_written = False
        emb_path = self.artifacts_dir / "description_embedding.npy"
        if self.write_artifact:
            tmp = emb_path.with_suffix(".tmp.npy")
            try:
                np.save(tmp, pooled_np.astype(np.float32))
                tmp.replace(emb_path)
                artifact_written = True
            except Exception as e:
                raise RuntimeError(f"DescriptionEmbedder: artifact_save_error: {e}") from e

        # In-memory registry for downstream (no absolute paths in result/NPZ).
        if artifact_written:
            try:
                tp = getattr(doc, "tp_artifacts", None)
                if not isinstance(tp, dict):
                    tp = {}
                    setattr(doc, "tp_artifacts", tp)
                tp.setdefault("embeddings", {})
                tp["embeddings"]["description"] = {
                    "relpath": emb_path.name,
                    "kind": "vector",
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "weights_digest": self.weights_digest,
                    "tokenizer_spec_name": self.tokenizer_spec_name,
                    "tokenizer_weights_digest": self.tokenizer_weights_digest,
                    "max_chunk_tokens_model": int(self.max_chunk_tokens_model),
                    "pooling_strategy": self.pooling_strategy,
                    "dim": int(pooled_np.size),
                }
            except Exception:
                pass

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = (time.perf_counter() - start)

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
                "tp_descemb_present": 1.0,
                "tp_descemb_dim": float(int(pooled_np.size)),
                "tp_descemb_norm_raw": float(norm_val),
                "tp_descemb_l2_norm": float(np.linalg.norm(pooled_np)),
                "tp_descemb_cache_hit": float(bool(cache_hit)) if self.cache_enabled else 0.0,
                "tp_descemb_artifact_written": float(bool(artifact_written)),
                "tp_descemb_n_chunks": float(n_chunks),
                "tp_descemb_avg_chunk_tokens": float(avg_chunk_tokens),
                "tp_descemb_chunk_ms": float(round(float(chunk_s) * 1000.0, 3)) if chunk_s == chunk_s else float("nan"),
                "tp_descemb_encode_ms": float(round(float(encode_s) * 1000.0, 3)) if encode_s == encode_s else float("nan"),
                "tp_descemb_pool_ms": float(round(float(pool_s) * 1000.0, 3)) if pool_s == pool_s else float("nan"),
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
            "timings_s": {"total": round(total_s, 3)},
            "result": {"features_flat": features_flat},
            "error": error,
        }
        return result
