"""
CLAP (Contrastive Language-Audio Pre-training) экстрактор для семантических аудио эмбеддингов.
"""
import warnings
import os
from contextlib import redirect_stdout, redirect_stderr, contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
import threading
import numpy as np
import torch
import torchaudio
from typing import Dict, Any, Optional, List, Tuple

# CRITICAL: Set HuggingFace offline mode BEFORE any imports that might use transformers
# This prevents network requests for tokenizers and models
# Use direct assignment (not setdefault) to ensure values are set even if already present
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Подавляем warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", message="torchaudio._backend.set_audio_backend has been deprecated")

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

from .utils.resource_profile import (
    prefix_snapshot,
    resource_profile_enabled,
    snapshot_process_resources,
)

logger = logging.getLogger(__name__)

# Try to import CLAP, fallback to stub if not available
try:
    import laion_clap
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False
    logger.warning("LAION CLAP not available. Using stub implementation.")

# Try to silence transformers/huggingface logs if available
try:  # pragma: no cover
    from transformers.utils import logging as hf_logging  # type: ignore

    hf_logging.set_verbosity_error()
except Exception:
    pass

# Reduce TensorFlow verbosity if present in deps
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all,1=info,2=warning,3=error


class CLAPExtractor(BaseExtractor):
    """CLAP экстрактор для семантических аудио эмбеддингов."""
    
    name: str = "clap_extractor"
    version: str = "1.1.1"
    description: str = "CLAP семантические аудио эмбеддинги"
    category: str = "advanced"
    dependencies: list = ["laion_clap", "torch", "torchaudio"]
    estimated_duration: float = 3.0
    device: str = "cpu"

    def __init__(self, device: Optional[str] = None, sample_rate: int = 48000):
        super().__init__(device)
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.embedding_dim = 512
        self.model = None
        self._model_loaded = False
        self._model_lock = threading.Lock()
        self._max_retries = 3
        self._retry_delay = 1.0
        
        # CLAP параметры
        self.max_audio_length = 10.0  # секунд
        self.batch_size = 1
        # Robust aggregation knobs (Audit v3)
        self._trim_norm_low_q = 0.05
        self._trim_norm_high_q = 0.95
        
        # Скрываем шумные инфо-логи инициализации экстрактора
        # # self.logger.debug(f"CLAP Extractor initialized on {self.device} | target_sr={self.sample_rate}")
        # Производительные настройки
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        if not CLAP_AVAILABLE:
            self.logger.error("CLAP Python package is not available; real extraction is required")
        else:
            # self.logger.debug("CLAP package detected; model will be loaded lazily on first use")
            pass

    def _load_model(self):
        """Загрузка CLAP модели."""
        if self._model_loaded:
            # self.logger.debug(f"CLAP model already initialized | model_is_none={self.model is None}")
            return True
            
        try:
            if not CLAP_AVAILABLE:
                raise RuntimeError("CLAP package not available")

            # Enforce offline/no-network policy (ModelManager sets env globally too).
            # CRITICAL: Ensure offline flags are set (they should already be set at module level)
            # Force set again here to be absolutely sure
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            try:
                from dp_models import get_global_model_manager  # type: ignore

                _ = get_global_model_manager()
            except Exception:
                # ModelManager not available, but offline flags are already set above
                pass

            # Инициализация CLAP модели (подавляем stdout/stderr спам весов и прогресс-бары)
            # Отключаем tqdm прогресс-бары через переменную окружения
            original_tqdm_disable = os.environ.get("TQDM_DISABLE")
            os.environ["TQDM_DISABLE"] = "1"
            # PyTorch 2.6+ default weights_only=True несовместим с laion_clap checkpoint (numpy pickle).
            # Monkey-patch torch.load чтобы laion_clap мог загрузить checkpoint c weights_only=False.
            # Checkpoint из dp_models — доверенный локальный источник (ModelManager-verified).
            import torch as _torch_patch
            _orig_torch_load = _torch_patch.load
            def _patched_torch_load(*_a, **_kw):
                _kw.setdefault("weights_only", False)
                return _orig_torch_load(*_a, **_kw)
            _torch_patch.load = _patched_torch_load
            try:
                with open(os.devnull, "w") as devnull:
                    with redirect_stdout(devnull), redirect_stderr(devnull):
                        try:
                            self.model = laion_clap.CLAP_Module(enable_fusion=False)
                        except Exception as init_error:
                            error_msg = str(init_error)
                            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                                raise TimeoutError(
                                    f"CLAP | model initialization timed out. "
                                    f"Original error: {error_msg}"
                                ) from init_error
                            raise RuntimeError(
                                f"CLAP | failed to initialize CLAP_Module: {error_msg}"
                            ) from init_error
                        # Load weights strictly from local artifacts via ModelManager.
                        ckpt_path: Optional[str] = None
                        weights_digest = "unknown"
                        try:
                            from dp_models import get_global_model_manager  # type: ignore
                            from dp_models.errors import ModelManagerError  # type: ignore

                            mm = get_global_model_manager()
                            spec = mm.get_spec(model_name="laion_clap")
                            _, _, _, _, weights_digest, artifacts = mm.resolve(spec)
                            # First declared artifact is the checkpoint.
                            if artifacts:
                                ckpt_path = list(artifacts.values())[0]
                        except Exception as e:
                            # Fail-fast: do not allow implicit downloads.
                            raise RuntimeError(f"CLAP | failed to resolve local checkpoint via ModelManager: {e}") from e

                        if not ckpt_path:
                            raise RuntimeError("CLAP | checkpoint path is empty (ModelManager spec invalid)")

                        # Проверяем доступность файла перед загрузкой
                        if not os.path.exists(ckpt_path):
                            raise FileNotFoundError(f"CLAP | checkpoint file not found: {ckpt_path}")
                        
                        # Проверяем размер файла (должен быть > 0)
                        try:
                            file_size = os.path.getsize(ckpt_path)
                            if file_size == 0:
                                raise ValueError(f"CLAP | checkpoint file is empty: {ckpt_path}")
                            self.logger.debug(f"CLAP | loading checkpoint from {ckpt_path} (size: {file_size / (1024*1024):.2f} MB)")
                        except OSError as e:
                            raise OSError(f"CLAP | cannot access checkpoint file {ckpt_path}: {e}") from e

                        # Try to pass explicit ckpt path (API differs across laion_clap versions).
                        try:
                            import inspect

                            sig = inspect.signature(self.model.load_ckpt)  # type: ignore[attr-defined]
                            if "ckpt_path" in sig.parameters:
                                self.model.load_ckpt(ckpt_path=ckpt_path)  # type: ignore[misc]
                            elif "ckpt" in sig.parameters:
                                self.model.load_ckpt(ckpt=ckpt_path)  # type: ignore[misc]
                            elif "path" in sig.parameters:
                                self.model.load_ckpt(path=ckpt_path)  # type: ignore[misc]
                            else:
                                self.model.load_ckpt(ckpt_path)  # type: ignore[misc]
                        except TypeError as e:
                            # Audit v3: no implicit / no-network weight loading. We require an explicit local path.
                            raise RuntimeError(
                                "CLAP | load_ckpt signature mismatch; audited mode requires explicit local ckpt_path via ModelManager"
                            ) from e
                        except Exception as load_error:
                            # Обертываем ошибки загрузки с более детальной информацией
                            error_msg = str(load_error)
                            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                                raise TimeoutError(
                                    f"CLAP | checkpoint loading timed out for {ckpt_path} "
                                    f"(size: {file_size / (1024*1024):.2f} MB). "
                                    f"Original error: {error_msg}"
                                ) from load_error
                            raise RuntimeError(
                                f"CLAP | failed to load checkpoint from {ckpt_path}: {error_msg}"
                            ) from load_error
            finally:
                # Восстанавливаем оригинальное значение TQDM_DISABLE
                if original_tqdm_disable is not None:
                    os.environ["TQDM_DISABLE"] = original_tqdm_disable
                elif "TQDM_DISABLE" in os.environ:
                    del os.environ["TQDM_DISABLE"]
            self.model.eval()

            # Перенос на устройство
            try:
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model.to(self.device)
                else:
                    self.model.to("cpu")
            except Exception:
                self.model.to("cpu")

            self._model_loaded = True
            # # self.logger.debug("CLAP model loaded successfully")
            
            # Проверяем устройство модели
            model_device = getattr(next(self.model.parameters()), 'device', self.device)
            # self.logger.debug(f"CLAP model device: {model_device}")
            
            # Дополнительная проверка для GPU
            if self.device == "cuda" and torch.cuda.is_available():
                if model_device.type != "cuda":
                    self.logger.warning(f"CLAP model not on GPU! Expected cuda, got {model_device}")
                else:
                    # # self.logger.debug(f"CLAP model successfully loaded on GPU: {model_device}")
                    pass
            
            return True

        except Exception as e:
            self._model_loaded = False
            self.model = None
            raise

    def _robust_aggregate(self, seq: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Robust aggregation of per-segment embeddings into a single vector.
        Strategy: compute per-segment L2 norms and trim extreme segments by quantiles,
        then take mean of the remaining valid embeddings.
        """
        if seq.ndim != 2 or seq.size == 0:
            return np.zeros((self.embedding_dim,), dtype=np.float32)
        if mask.ndim != 1 or mask.shape[0] != seq.shape[0]:
            valid = np.ones((seq.shape[0],), dtype=bool)
        else:
            valid = mask.astype(bool)
        if not np.any(valid):
            return np.zeros((seq.shape[1],), dtype=np.float32)

        valid_seq = seq[valid]
        # Remove rows containing NaN/Inf defensively (should already be excluded by mask).
        finite_rows = np.isfinite(valid_seq).all(axis=1)
        valid_seq = valid_seq[finite_rows]
        if valid_seq.size == 0:
            return np.zeros((seq.shape[1],), dtype=np.float32)

        norms = np.linalg.norm(valid_seq, axis=1)
        norms = np.asarray(norms, dtype=np.float32)
        if norms.size >= 4:
            lo = float(np.quantile(norms, self._trim_norm_low_q))
            hi = float(np.quantile(norms, self._trim_norm_high_q))
            keep = (norms >= lo) & (norms <= hi)
            if np.any(keep):
                valid_seq = valid_seq[keep]

        emb = np.mean(valid_seq, axis=0).astype(np.float32)
        # Ensure finite
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return emb.reshape(-1)

    def run_segments(
        self,
        input_uri: str,
        tmp_path: str,
        segments: List[Dict[str, Any]],
        *,
        segment_parallelism: int = 1,
        max_inflight: int = 1,
        model_batch_size: int = 1,
    ) -> ExtractorResult:
        """
        Compute CLAP embeddings over Segmenter-provided audio windows and aggregate.
        Produces:
          - embedding (mean over segments) shape [D]
          - embedding_sequence shape [N, D]
          - segment_center_sec shape [N] (+ start/end + segment_mask)
        """
        start_time = time.time()
        t_total0 = time.perf_counter()
        stage_timings_ms: Dict[str, float] = {}
        clap_resource_profile: Optional[Dict[str, Any]] = None
        try:
            if not CLAP_AVAILABLE:
                raise RuntimeError("CLAP Python package is not available (required)")
            if not self._validate_input(input_uri):
                return self._create_result(
                    success=False,
                    error="Некорректный входной файл",
                    processing_time=time.time() - start_time,
                )
            if not isinstance(segments, list) or not segments:
                raise ValueError("segments is empty (no-fallback)")

            if resource_profile_enabled():
                try:
                    clap_resource_profile = {
                        **prefix_snapshot("at_start", snapshot_process_resources()),
                    }
                except Exception:
                    clap_resource_profile = None

            # Scheduler-controlled: only batching is supported here (segment_parallelism is unsafe for CLAP inference).
            # However, segment_parallelism can be used for parallel loading/preprocessing of segments.
            self.batch_size = max(1, int(model_batch_size or 1))

            # Time axis (strict alignment with Segmenter windows)
            starts: list[float] = [float(seg.get("start_sec", 0.0) or 0.0) for seg in segments]
            ends: list[float] = [float(seg.get("end_sec", 0.0) or 0.0) for seg in segments]
            centers: list[float] = [float(seg.get("center_sec", 0.0) or 0.0) for seg in segments]
            total_segments = len(segments)
            
            # Начальное сообщение о начале обработки
            if self.progress_callback and total_segments > 0:
                self.progress_callback("clap", 0, total_segments, f"Starting preprocessing: 0/{total_segments} segments (0%)")

            # Parallel load + preprocess per segment (CPU/GPU) and then run inference in micro-batches.
            # Use segment_parallelism for preprocessing workers (default to 4 if not specified).
            preprocess_workers = max(1, int(segment_parallelism or 4))
            # Limit workers to avoid excessive parallelism for small batches
            preprocess_workers = min(preprocess_workers, len(segments), 8)
            
            processed_list: list[Optional[torch.Tensor]] = []
            trimmed_flags: list[bool] = []
            last_reported_pct = -1

            t_pre0 = time.perf_counter()
            
            if len(segments) == 1 or preprocess_workers == 1:
                # Sequential processing for single segment or when parallelism disabled
                for seg_idx, seg in enumerate(segments):
                    try:
                        processed, was_trimmed = self._load_and_preprocess_segment(seg, input_uri)
                        processed_list.append(processed)
                        trimmed_flags.append(bool(was_trimmed))
                    except Exception:
                        # Mask failed segment, keep strict alignment.
                        processed_list.append(None)
                        trimmed_flags.append(False)
                    
                    # Progress reporting: каждые 10% сегментов или на первом/последнем
                    if self.progress_callback and total_segments > 0:
                        current = seg_idx + 1
                        pct = int(current * 100 / total_segments)
                        if current == 1 or current == total_segments or (pct % 10 == 0 and pct != last_reported_pct):
                            self.progress_callback("clap", current, total_segments, f"Preprocessing segments: {current}/{total_segments} ({pct}%)")
                            last_reported_pct = pct
            else:
                # Parallel processing for multiple segments
                # Create list of (index, segment) tuples to preserve order
                indexed_segments = [(idx, seg) for idx, seg in enumerate(segments)]
                completed_count = 0
                
                with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
                    # Submit all tasks
                    future_to_index = {
                        executor.submit(self._load_and_preprocess_segment, seg, input_uri): idx
                        for idx, seg in indexed_segments
                    }
                    
                    # Collect results in order
                    results: list[Optional[Tuple[torch.Tensor, bool]]] = [None] * len(segments)
                    for future in as_completed(future_to_index):
                        idx = future_to_index[future]
                        try:
                            results[idx] = future.result()
                            completed_count += 1
                            
                            # Progress reporting: каждые 10% сегментов или на первом/последнем
                            if self.progress_callback and total_segments > 0:
                                pct = int(completed_count * 100 / total_segments)
                                if completed_count == 1 or completed_count == total_segments or (pct % 10 == 0 and pct != last_reported_pct):
                                    self.progress_callback("clap", completed_count, total_segments, f"Preprocessing segments: {completed_count}/{total_segments} ({pct}%)")
                                    last_reported_pct = pct
                        except Exception as e:
                            self.logger.error(f"Error loading/preprocessing segment {idx}: {e}")
                            results[idx] = None
                    
                    for r in results:
                        if r is None:
                            processed_list.append(None)
                            trimmed_flags.append(False)
                        else:
                            t, was_trimmed = r
                            processed_list.append(t)
                            trimmed_flags.append(bool(was_trimmed))

            stage_timings_ms["preprocess_segments_ms"] = (time.perf_counter() - t_pre0) * 1000.0

            def _infer_batch(wavs_1d: list[torch.Tensor]) -> list[np.ndarray]:
                # Try batched inference. If CLAP impl doesn't support batching, fall back to per-item.
                try:
                    self._ensure_model_loaded()
                    if not self._model_loaded or self.model is None:
                        raise RuntimeError("CLAP model is not loaded")

                    # Pad to max length for stacking.
                    lens = [int(w.shape[-1]) for w in wavs_1d]
                    max_len = int(max(lens)) if lens else 0
                    if max_len <= 0:
                        return []
                    batch = []
                    for w in wavs_1d:
                        w = w.contiguous().float()
                        if int(w.shape[-1]) < max_len:
                            pad = torch.zeros((max_len - int(w.shape[-1])), device=w.device, dtype=w.dtype)
                            w = torch.cat([w, pad], dim=-1)
                        batch.append(w)
                    audio_tensor = torch.stack(batch, dim=0)  # [B, T]
                    if self.device == "cuda" and torch.cuda.is_available():
                        audio_tensor = audio_tensor.to(self.device, non_blocking=True)
                    use_cuda = self.device == "cuda" and torch.cuda.is_available()
                    if use_cuda:
                        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float32)
                    else:
                        autocast_ctx = nullcontext()
                    with torch.inference_mode():
                        with autocast_ctx:
                            with open(os.devnull, "w") as devnull:
                                with redirect_stdout(devnull):
                                    emb_t = self.model.get_audio_embedding_from_data(  # type: ignore[call-arg]
                                        audio_tensor,
                                        use_tensor=True,
                                    )
                    emb = emb_t.detach().cpu().float().numpy()
                    emb = np.asarray(emb)
                    if emb.ndim == 1:
                        return [emb.astype(np.float32).reshape(-1)]
                    return [np.asarray(emb[i], dtype=np.float32).reshape(-1) for i in range(int(emb.shape[0]))]
                except Exception:
                    out: list[np.ndarray] = []
                    for w in wavs_1d:
                        e = self._extract_clap_embeddings(w)
                        out.append(np.asarray(e, dtype=np.float32).reshape(-1))
                    return out

            emb_list: list[Optional[np.ndarray]] = []
            bs = int(self.batch_size)
            # Inference only for valid (non-masked) segments, but preserve alignment.
            valid_indices = [i for i, t in enumerate(processed_list) if isinstance(t, torch.Tensor)]
            total_batches = (len(valid_indices) + bs - 1) // bs if valid_indices else 0
            last_infer_pct = -1
            
            # Pre-fill with None to preserve strict alignment
            emb_list = [None] * len(processed_list)
            t_inf0 = time.perf_counter()
            for batch_idx, i0 in enumerate(range(0, len(valid_indices), bs)):
                idxs = valid_indices[i0 : i0 + bs]
                chunk = [processed_list[j] for j in idxs]  # type: ignore[list-item]
                chunk_t = [t for t in chunk if isinstance(t, torch.Tensor)]
                out = _infer_batch(chunk_t)
                for j, e in zip(idxs, out):
                    emb_list[j] = e
                
                # Progress reporting: каждые 10% батчей или на первом/последнем
                if self.progress_callback and total_batches > 0:
                    current = batch_idx + 1
                    pct = int(current * 100 / total_batches)
                    if current == 1 or current == total_batches or (pct % 10 == 0 and pct != last_infer_pct):
                        self.progress_callback("clap", current, total_batches, f"Running inference: {current}/{total_batches} batches ({pct}%)")
                        last_infer_pct = pct

            stage_timings_ms["inference_ms"] = (time.perf_counter() - t_inf0) * 1000.0

            # Build strict-aligned arrays
            t_build0 = time.perf_counter()
            N = len(processed_list)
            D = int(self.embedding_dim)
            seq = np.full((N, D), np.nan, dtype=np.float32)
            seg_mask = np.zeros((N,), dtype=np.bool_)
            seg_norm = np.full((N,), np.nan, dtype=np.float32)
            for i, e in enumerate(emb_list):
                if e is None:
                    continue
                v = np.asarray(e, dtype=np.float32).reshape(-1)
                if v.size != D or (not np.isfinite(v).all()):
                    continue
                seq[i, :] = v
                seg_mask[i] = True
                seg_norm[i] = float(np.linalg.norm(v))

            if not np.any(seg_mask):
                raise RuntimeError("CLAP | all segments failed (segment_mask all false)")

            stage_timings_ms["build_outputs_ms"] = (time.perf_counter() - t_build0) * 1000.0

            t_agg0 = time.perf_counter()
            emb_mean = self._robust_aggregate(seq, seg_mask)
            stage_timings_ms["robust_aggregate_ms"] = (time.perf_counter() - t_agg0) * 1000.0

            # extra stats
            emb_norm = float(np.linalg.norm(emb_mean))
            emb_mag_mean = float(np.mean(np.abs(emb_mean)))
            emb_mag_std = float(np.std(np.abs(emb_mean)))
            non_zero_count = int(np.count_nonzero(emb_mean))
            trimmed_segments_count = int(sum(1 for x in trimmed_flags if x))
            trimmed_ratio = float(trimmed_segments_count / float(len(trimmed_flags) or 1))

            stage_timings_ms["total_ms"] = (time.perf_counter() - t_total0) * 1000.0

            if clap_resource_profile is not None:
                try:
                    clap_resource_profile = {
                        **clap_resource_profile,
                        **prefix_snapshot("at_end", snapshot_process_resources()),
                    }
                except Exception:
                    pass

            payload: Dict[str, Any] = {
                "embedding": emb_mean,
                "embedding_sequence": seq,
                "segment_start_sec": np.asarray(starts, dtype=np.float32),
                "segment_end_sec": np.asarray(ends, dtype=np.float32),
                "segment_center_sec": np.asarray(centers, dtype=np.float32),
                "segment_mask": seg_mask,
                "segment_embedding_norm": seg_norm,
                "stage_timings_ms": stage_timings_ms,
                "clap_resource_profile": clap_resource_profile,
                "segments_count": int(np.sum(seg_mask)),
                "embedding_dim": int(emb_mean.shape[0]),
                # Report model SR (after preprocess) rather than input SR.
                "sample_rate": int(self.sample_rate),
                "max_audio_length_sec": float(self.max_audio_length),
                "trimmed_segments_count": trimmed_segments_count,
                "trimmed_ratio": trimmed_ratio,
                "clap_norm": emb_norm,
                "clap_magnitude_mean": emb_mag_mean,
                "clap_magnitude_std": emb_mag_std,
                "clap_non_zero_count": non_zero_count,
                "device_used": self.device,
                "scheduler_knobs": {
                    "segment_parallelism": int(preprocess_workers),
                    "max_inflight": int(1),
                    "model_batch_size": int(self.batch_size),
                },
            }
            return self._create_result(True, payload=payload, processing_time=time.time() - start_time)
        except Exception as e:
            return self._create_result(False, error=str(e), processing_time=time.time() - start_time)

    def _initialize_model_with_retry(self) -> None:
        """Инициализация модели с повторными попытками и экспоненциальной паузой."""
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                self._load_model()
                return
            except (RuntimeError, TimeoutError, OSError, FileNotFoundError, ValueError) as e:
                last_error = e
                if attempt == self._max_retries - 1:
                    # На последней попытке пробрасываем ошибку с полным контекстом
                    error_type = type(e).__name__
                    raise RuntimeError(
                        f"CLAP | failed to load model after {self._max_retries} attempts. "
                        f"Last error ({error_type}): {e}"
                    ) from e
                retry_delay = self._retry_delay * (2 ** attempt)
                try:
                    self.logger.warning(
                        f"CLAP | model loading attempt {attempt + 1}/{self._max_retries} failed: {e}. "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                except (ValueError, OSError, AttributeError, RuntimeError, TypeError):
                    # Logging handler may be closed, ignore
                    pass
                time.sleep(retry_delay)
        if last_error is not None:
            raise RuntimeError(
                f"CLAP | failed to load model after {self._max_retries} attempts. "
                f"Last error: {last_error}"
            ) from last_error

    def _ensure_model_loaded(self) -> None:
        """Ленивая и потокобезопасная загрузка модели."""
        if not self._model_loaded or self.model is None:
            with self._model_lock:
                if not self._model_loaded or self.model is None:
                    self._initialize_model_with_retry()

    def _load_and_preprocess_segment(
        self,
        seg: Dict[str, Any],
        input_uri: str,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Загрузка и предобработка одного сегмента аудио.
        Этот метод может быть вызван параллельно для разных сегментов.
        
        Args:
            seg: Словарь с метаданными сегмента (start_sample, end_sample, etc.)
            input_uri: URI к входному аудио файлу
        
        Returns:
            (Предобработанный тензор аудио [T], was_trimmed)
        """
        ss = int(seg.get("start_sample"))
        es = int(seg.get("end_sample"))
        wav_t, sr = self.audio_utils.load_audio_segment(input_uri, start_sample=ss, end_sample=es, target_sr=None)
        if self.device == "cuda" and torch.cuda.is_available():
            wav_t = wav_t.to(self.device, non_blocking=True)
        processed, was_trimmed = self._preprocess_audio(wav_t, sr)
        # normalize dims to 1D [T]
        if isinstance(processed, torch.Tensor) and processed.dim() == 2 and processed.size(0) == 1:
            processed = processed[0]
        return processed, bool(was_trimmed)

    def _preprocess_audio(self, waveform: torch.Tensor, sr: int) -> Tuple[torch.Tensor, bool]:
        """Предобработка аудио для CLAP."""
        # self.logger.debug(f"Preprocess audio: input_shape={tuple(waveform.shape)}, input_sr={sr}")
        # Ресемплирование до 48kHz если необходимо
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(self.device)
            waveform = resampler(waveform)
            # self.logger.debug(f"Resampled audio to {self.sample_rate} Hz | shape={tuple(waveform.shape)}")
        
        # Обрезка до максимальной длины
        max_samples = int(self.max_audio_length * self.sample_rate)
        was_trimmed = False
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]
            was_trimmed = True
            # self.logger.debug(f"Trimmed audio to max {self.max_audio_length}s | samples={max_samples}")
        
        # Нормализация
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        # self.logger.debug("Normalized audio waveform")
        
        return waveform, was_trimmed

    def _extract_clap_embeddings(self, waveform: torch.Tensor) -> np.ndarray:
        """Извлечение CLAP эмбеддингов."""
        self._ensure_model_loaded()
        if not self._model_loaded or self.model is None:
            raise RuntimeError("CLAP model is not loaded")
        
        try:
            # Готовим вход: моно [T] float32, батч [1, T]
            audio_tensor = waveform
            if audio_tensor.dim() == 2 and audio_tensor.size(0) > 1:
                audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
            elif audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            audio_tensor = audio_tensor.contiguous().float()

            # Перенос на устройство
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    audio_tensor = audio_tensor.pin_memory()
                except Exception:
                    pass
                audio_tensor = audio_tensor.to(self.device, non_blocking=True)

            # Используем autocast на CUDA для снижения нагрузки на память/ускорения
            use_cuda = self.device == "cuda" and torch.cuda.is_available()
            # Используем современный API autocast с float32 для стабильности CLAP
            if use_cuda:
                autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float32)
            else:
                autocast_ctx = nullcontext()
            with torch.inference_mode():
                with autocast_ctx:
                    with open(os.devnull, "w") as devnull:
                        with redirect_stdout(devnull):
                            emb_t = self.model.get_audio_embedding_from_data(
                                audio_tensor,
                                use_tensor=True,
                            )

            # Приводим к np.ndarray [D]
            emb = emb_t.detach().cpu().float().numpy()
            emb = np.asarray(emb)
            if emb.ndim == 2 and emb.shape[0] == 1:
                emb = emb[0]

            # Очистка ссылок; не вызываем empty_cache на каждом шаге
            if torch.cuda.is_available():
                try:
                    del emb_t
                    del audio_tensor
                except Exception:
                    pass

            # Дополнительная отладочная информация
            emb_norm = float(np.linalg.norm(emb))
            emb_mean = float(np.mean(emb))
            emb_std = float(np.std(emb))
            non_zero_count = int(np.count_nonzero(emb))
            
            # self.logger.debug(f"Extracted CLAP embedding | shape={emb.shape}, norm={emb_norm:.4f}, mean={emb_mean:.4f}, std={emb_std:.4f}, non_zero={non_zero_count}")
            
            # Проверяем на нулевые эмбеддинги
            if emb_norm < 1e-6:
                self.logger.warning(f"CLAP embedding is nearly zero! norm={emb_norm:.6f}, mean={emb_mean:.6f}, std={emb_std:.6f}")
            
            return emb
                
        except Exception:
            raise

    @property
    def supports_batch(self) -> bool:
        """CLAP extractor поддерживает batch processing для сегментов."""
        return True
    
    def extract_batch_segments(
        self,
        audio_files_with_segments: List[Dict[str, Any]],
        *,
        max_workers: Optional[int] = None,
        max_segments_per_batch: Optional[int] = None,
    ) -> List[ExtractorResult]:
        """
        Батчевая обработка сегментов из нескольких видео с гибридным подходом.
        
        Гибридный подход (вариант C):
        - Собирает сегменты из всех видео
        - Группирует в батчи по max_segments_per_batch (если задан)
        - Обрабатывает батчи последовательно
        - Распределяет результаты обратно по видео
        
        Args:
            audio_files_with_segments: Список словарей с ключами:
                - 'input_uri': URI к входному аудио/видео файлу
                - 'tmp_path': Путь к временной директории для обработки
                - 'segments': Список сегментов для обработки
                - 'file_id': Идентификатор файла (для распределения результатов)
            max_workers: Не используется для GPU extractors (оставлено для совместимости)
            max_segments_per_batch: Максимальное количество сегментов в одном батче (None = без лимита)
        
        Returns:
            Список ExtractorResult для каждого файла
        """
        start_time = time.time()
        
        if not CLAP_AVAILABLE:
            # Возвращаем ошибки для всех файлов
            return [
                self._create_result(
                    success=False,
                    error="CLAP Python package is not available (required)",
                    processing_time=time.time() - start_time,
                )
                for _ in audio_files_with_segments
            ]
        
        if not audio_files_with_segments:
            return []
        
        try:
            # Audit v3 correctness-first:
            # - Preserve per-file strict alignment + mask semantics by reusing run_segments().
            # - Cross-video batching is intentionally not performed here.
            results: List[ExtractorResult] = []
            for file_info in audio_files_with_segments:
                input_uri = file_info.get("input_uri")
                tmp_path = file_info.get("tmp_path")
                segments = file_info.get("segments", [])
                if not input_uri or not tmp_path:
                    results.append(
                        self._create_result(
                            success=False,
                            error="Missing input_uri/tmp_path",
                            processing_time=time.time() - start_time,
                        )
                    )
                    continue
                try:
                    r = self.run_segments(
                        input_uri,
                        tmp_path,
                        segments,
                        segment_parallelism=1,
                        max_inflight=1,
                        model_batch_size=1,
                    )
                    results.append(r)
                except Exception as e:
                    results.append(
                        self._create_result(
                            success=False,
                            error=str(e),
                            processing_time=time.time() - start_time,
                        )
                    )
            return results
            
        except Exception as e:
            self.logger.error(f"Error in extract_batch_segments: {e}")
            # Возвращаем ошибки для всех файлов
            return [
                self._create_result(
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time,
                )
                for _ in audio_files_with_segments
            ]
    
    @contextmanager
    def model_context(self):
        """Контекстный менеджер для безопасной работы с моделью."""
        try:
            self._ensure_model_loaded()
            yield self.model
        except Exception as e:
            self.logger.error(f"Model operation failed: {e}")
            raise
        finally:
            pass

    def _create_stub_embeddings(self) -> np.ndarray:
        """Заглушки не поддерживаются."""
        raise RuntimeError("Stub embeddings are disabled for CLAP extractor")

    def warm_up(self) -> None:
        """Предварительная инициализация модели (ускоряет первый вызов)."""
        try:
            # self.logger.debug("Warm-up: loading model and running dummy pass")
            self._load_model()
            # Прогоняем небольшой тихий тензор, чтобы прогреть граф/контекст
            dummy = torch.zeros(1, int(self.sample_rate * 0.1), device="cpu", dtype=torch.float32)
            _ = self._extract_clap_embeddings(dummy)
            # self.logger.debug("Warm-up completed")
        except Exception:
            # Тихо игнорируем – будет fallback на stub во время реального вызова
            pass

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """Legacy full-audio mode is disabled in audited contract (Audit v3)."""
        start_time = time.time()
        return self._create_result(
            success=False,
            error="CLAPExtractor.run() is disabled in audited mode. Use run_segments() with Segmenter windows (families.clap).",
            processing_time=time.time() - start_time,
        )

    def _compute_clap_statistics(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Вычисление статистик CLAP эмбеддингов."""
        try:
            # Clean embeddings of NaN and inf values
            if embeddings.ndim > 1:
                embeddings_clean = embeddings[~np.isnan(embeddings).any(axis=1)]
                if embeddings_clean.size == 0:
                    embeddings_clean = np.zeros((1, embeddings.shape[1]))
            else:
                embeddings_clean = embeddings[~np.isnan(embeddings)]
                if embeddings_clean.size == 0:
                    embeddings_clean = np.zeros(embeddings.shape)
            
            # Ensure we have valid data
            if embeddings_clean.size == 0:
                embeddings_clean = np.zeros(self.embedding_dim)
            
            # Основные статистики
            mean_embedding = np.mean(embeddings_clean, axis=0) if embeddings_clean.ndim > 1 else embeddings_clean
            if embeddings_clean.ndim > 1:
                std_embedding = np.std(embeddings_clean, axis=0)
            else:
                # For 1D arrays, compute std across the entire array and replicate it
                std_value = np.std(embeddings_clean)
                std_embedding = np.full_like(embeddings_clean, std_value)
            
            # Replace any remaining NaN values with 0
            mean_embedding = np.nan_to_num(mean_embedding, nan=0.0, posinf=0.0, neginf=0.0)
            std_embedding = np.nan_to_num(std_embedding, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Норма эмбеддинга
            norm = np.linalg.norm(mean_embedding)
            if np.isnan(norm) or np.isinf(norm):
                norm = 0.0
            
            # Минимальные и максимальные значения
            min_vals = np.min(embeddings_clean, axis=0) if embeddings_clean.ndim > 1 else embeddings_clean
            max_vals = np.max(embeddings_clean, axis=0) if embeddings_clean.ndim > 1 else embeddings_clean
            
            # Replace NaN values in min/max
            min_vals = np.nan_to_num(min_vals, nan=0.0, posinf=0.0, neginf=0.0)
            max_vals = np.nan_to_num(max_vals, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Количество ненулевых компонент
            non_zero_count = np.count_nonzero(mean_embedding)
            
            # Calculate magnitude statistics safely
            abs_mean = np.abs(mean_embedding)
            magnitude_mean = float(np.mean(abs_mean)) if abs_mean.size > 0 else 0.0
            magnitude_std = float(np.std(abs_mean)) if abs_mean.size > 0 else 0.0
            
            # Ensure no NaN values in final results
            magnitude_mean = 0.0 if np.isnan(magnitude_mean) or np.isinf(magnitude_mean) else magnitude_mean
            magnitude_std = 0.0 if np.isnan(magnitude_std) or np.isinf(magnitude_std) else magnitude_std
            
            return {
                "clap_mean": mean_embedding.tolist(),
                "clap_std": std_embedding.tolist(),
                "clap_min": min_vals.tolist(),
                "clap_max": max_vals.tolist(),
                "clap_norm": float(norm),
                "clap_non_zero_count": int(non_zero_count),
                "clap_magnitude_mean": magnitude_mean,
                "clap_magnitude_std": magnitude_std,
                "total_features": len(mean_embedding)
            }
            
        except Exception as e:
            self.logger.error(f"Error computing CLAP statistics: {e}")
            return {
                "clap_mean": [0.0] * self.embedding_dim,
                "clap_std": [0.0] * self.embedding_dim,
                "clap_min": [0.0] * self.embedding_dim,
                "clap_max": [0.0] * self.embedding_dim,
                "clap_norm": 0.0,
                "clap_non_zero_count": 0,
                "clap_magnitude_mean": 0.0,
                "clap_magnitude_std": 0.0,
                "total_features": self.embedding_dim
            }

    def _log_extraction_metrics(self, audio_shape, processing_time: float, embedding_quality: float):
        try:
            gpu_mem = 0.0
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
            # self.logger.info(
            #     f"Extraction metrics - Audio: {audio_shape}, Time: {processing_time:.2f}s, "
            #     f"Quality: {embedding_quality:.3f}, GPU Memory: {gpu_mem:.2f}GB"
            # )
        except Exception:
            pass



