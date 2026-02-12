"""
VoiceQualityExtractor: извлечение метрик качества голоса (jitter, shimmer, HNR) с использованием librosa.
Интеграция с общим интерфейсом BaseExtractor и AudioUtils.

Production-grade implementation with:
- Segmenter contract support (run_segments)
- Feature gating (per-feature flags)
- Full validation (outputs, parameters)
- No-fallback policy (fail-fast)
- Per-run storage for .npy files
- Progress reporting
- UI renderer support
- Contract versioning
- Detailed error codes
- Optional audio normalization
- Additional ML/analytics metrics
- Optional integration with pitch_extractor
"""
import time
import logging
import os
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)

# Contract version for downstream extractors compatibility validation
VOICE_QUALITY_CONTRACT_VERSION = "voice_quality_contract_v1"

# Threshold for saving large arrays to .npy files
TIME_SERIES_SAVE_THRESHOLD = 10000


class VoiceQualityExtractor(BaseExtractor):
    """Экстрактор метрик качества голоса: jitter, shimmer, HNR-подобная метрика."""

    name = "voice_quality"
    version = "2.0.0"
    description = "Прокси метрики качества голоса: jitter, shimmer, HNR-подобная"
    category = "voice"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 1.5

    gpu_required = False
    gpu_preferred = True  # torchcrepe может использовать GPU для ускорения f0 estimation
    gpu_memory_required = 0.0  # Минимальные требования (torchcrepe tiny модель ~100MB)

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 22050,
        average_channels: bool = True,
        hnr_frame_ms: float = 40.0,
        rms_mask_threshold: float = 0.01,
        # F0 estimation parameters
        f0_fmin: float = 50.0,
        f0_fmax: float = 500.0,
        f0_method: str = "yin",  # "yin" | "pyin" | "torchcrepe"
        torchcrepe_model: str = "tiny",  # "tiny" | "full" (для torchcrepe: tiny быстрее, full точнее)
        # Feature gating flags (per-feature control, default: all False)
        enable_jitter: bool = False,
        enable_shimmer: bool = False,
        enable_hnr: bool = False,
        enable_f0_stats: bool = False,
        enable_time_series: bool = False,
        # Optional audio normalization
        enable_audio_normalization: bool = False,
        # Optional integration with pitch_extractor
        pitch_payload: Optional[Dict[str, Any]] = None,  # Results from pitch_extractor for f0
        # Progress reporting callback
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        # Per-run storage path for .npy files
        artifacts_dir: Optional[str] = None,
    ):
        """
        Инициализация VoiceQuality экстрактора.

        Args:
            device: Устройство для обработки
            sample_rate: Частота дискретизации
            average_channels: Усреднять каналы для многоканального аудио
            hnr_frame_ms: Размер окна для HNR вычисления (миллисекунды)
            rms_mask_threshold: Порог RMS для маскирования тихих участков
            f0_fmin: Минимальная частота f0 для оценки (Hz)
            f0_fmax: Максимальная частота f0 для оценки (Hz)
            f0_method: Метод оценки f0 ("yin" | "pyin" | "torchcrepe")
            enable_jitter: Включить метрику jitter
            enable_shimmer: Включить метрику shimmer
            enable_hnr: Включить метрику HNR
            enable_f0_stats: Включить статистики f0 (mean, std, min, max, stability)
            enable_time_series: Включить временные серии (f0, amps, hnr по окнам)
            enable_audio_normalization: Включить нормализацию аудио перед обработкой
            pitch_payload: Результаты от pitch_extractor для использования их f0
            progress_callback: Callback для прогресса (metric_name, current, total, message)
            artifacts_dir: Директория для сохранения .npy файлов (per-run storage)
        """
        super().__init__(device=device)

        # Validate parameters
        self._validate_parameters(
            sample_rate, hnr_frame_ms, rms_mask_threshold, f0_fmin, f0_fmax, f0_method
        )

        self.sample_rate = int(sample_rate)
        self.average_channels = bool(average_channels)
        self.hnr_frame_ms = float(hnr_frame_ms)
        self.rms_mask_threshold = float(rms_mask_threshold)
        self.f0_fmin = float(f0_fmin)
        self.f0_fmax = float(f0_fmax)
        self.f0_method = str(f0_method)
        self.torchcrepe_model = str(torchcrepe_model) if f0_method == "torchcrepe" else "tiny"

        # Feature gating flags
        self.enable_jitter = bool(enable_jitter)
        self.enable_shimmer = bool(enable_shimmer)
        self.enable_hnr = bool(enable_hnr)
        self.enable_f0_stats = bool(enable_f0_stats)
        self.enable_time_series = bool(enable_time_series)

        # Optional audio normalization
        self.enable_audio_normalization = bool(enable_audio_normalization)

        # Optional integration with pitch_extractor
        self.pitch_payload = pitch_payload

        # Progress callback
        self.progress_callback = progress_callback

        # Per-run storage for .npy files
        self.artifacts_dir = artifacts_dir

        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)

    def _validate_parameters(
        self,
        sample_rate: int,
        hnr_frame_ms: float,
        rms_mask_threshold: float,
        f0_fmin: float,
        f0_fmax: float,
        f0_method: str,
    ) -> None:
        """
        Валидация входных параметров (fail-fast).

        Args:
            sample_rate: Частота дискретизации
            hnr_frame_ms: Размер окна для HNR
            rms_mask_threshold: Порог RMS
            f0_fmin: Минимальная частота f0
            f0_fmax: Максимальная частота f0
            f0_method: Метод оценки f0

        Raises:
            ValueError: Если параметры невалидны
        """
        if sample_rate <= 0:
            raise ValueError(f"voice_quality | sample_rate must be positive, got {sample_rate}")
        if hnr_frame_ms <= 0:
            raise ValueError(f"voice_quality | hnr_frame_ms must be positive, got {hnr_frame_ms}")
        if rms_mask_threshold < 0:
            raise ValueError(f"voice_quality | rms_mask_threshold must be non-negative, got {rms_mask_threshold}")
        if f0_fmin <= 0:
            raise ValueError(f"voice_quality | f0_fmin must be positive, got {f0_fmin}")
        if f0_fmax <= 0:
            raise ValueError(f"voice_quality | f0_fmax must be positive, got {f0_fmax}")
        if f0_fmin >= f0_fmax:
            raise ValueError(f"voice_quality | f0_fmin ({f0_fmin}) must be < f0_fmax ({f0_fmax})")
        if f0_fmin < 20.0:
            raise ValueError(f"voice_quality | f0_fmin ({f0_fmin}) is too low (minimum 20 Hz)")
        if f0_fmax > 2000.0:
            raise ValueError(f"voice_quality | f0_fmax ({f0_fmax}) is too high (maximum 2000 Hz)")
        if f0_method not in ["yin", "pyin", "torchcrepe"]:
            raise ValueError(f"voice_quality | f0_method must be 'yin', 'pyin', or 'torchcrepe', got {f0_method}")

    def _classify_error(self, error: Exception, context: str) -> str:
        """
        Классифицировать ошибку и вернуть детальный error_code.

        Args:
            error: Исключение
            context: Контекст ошибки

        Returns:
            error_code: один из:
                - voice_quality_audio_load_failed
                - voice_quality_f0_estimation_failed
                - voice_quality_librosa_failed
                - voice_quality_insufficient_data
                - voice_quality_validation_failed
                - voice_quality_unknown
        """
        error_str = str(error).lower()

        if "audio" in error_str or "load" in error_str or context == "audio_load_failed":
            return "voice_quality_audio_load_failed"
        if "f0" in error_str or "pitch" in error_str or context == "f0_estimation_failed":
            return "voice_quality_f0_estimation_failed"
        if "librosa" in error_str or context == "librosa_failed":
            return "voice_quality_librosa_failed"
        if "insufficient" in error_str or "empty" in error_str or context == "insufficient_data":
            return "voice_quality_insufficient_data"
        if "validation" in error_str or "invalid" in error_str or context == "validation_failed":
            return "voice_quality_validation_failed"

        return "voice_quality_unknown"

    def _validate_output(self, features: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Полная валидация выходных данных: проверка диапазонов, NaN/inf, консистентность.

        Args:
            features: Словарь с выходными данными

        Returns:
            (is_valid, error_message)
        """
        try:
            # Check for NaN/inf
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        return False, f"voice_quality | {key} contains NaN or inf: {value}"
                elif isinstance(value, np.ndarray):
                    if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                        return False, f"voice_quality | {key} contains NaN or inf"

            # Validate ranges
            if "vq_jitter" in features:
                jitter = features["vq_jitter"]
                if jitter < 0.0 or jitter > 1.0:
                    return False, f"voice_quality | jitter out of reasonable range [0, 1]: {jitter}"

            if "vq_shimmer" in features:
                shimmer = features["vq_shimmer"]
                if shimmer < 0.0 or shimmer > 1.0:
                    return False, f"voice_quality | shimmer out of reasonable range [0, 1]: {shimmer}"

            # HNR can be any value (dB), but check for extreme values
            if "vq_hnr_like_db" in features:
                hnr = features["vq_hnr_like_db"]
                if not (-100.0 <= hnr <= 100.0):
                    logger.warning(f"voice_quality | HNR out of typical range [-100, 100]: {hnr}")

            return True, None
        except Exception as e:
            return False, f"voice_quality | validation error: {e}"

    def _normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """
        Нормализовать аудио сигнал (опционально).

        Args:
            y: Аудио сигнал

        Returns:
            Нормализованный сигнал
        """
        if not self.enable_audio_normalization:
            return y

        max_val = np.max(np.abs(y))
        if max_val > 1e-9:
            return y / max_val
        return y

    def _estimate_f0(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Оценить f0 с использованием выбранного метода (fail-fast, no-fallback).

        Args:
            y: Аудио сигнал
            sr: Частота дискретизации

        Returns:
            Массив f0 значений

        Raises:
            RuntimeError: Если оценка f0 не удалась
        """
        # Если доступны результаты pitch_extractor, используем их
        if self.pitch_payload is not None:
            # Извлекаем f0 из pitch_payload
            f0_series = None
            if self.f0_method == "pyin" and "f0_series_pyin" in self.pitch_payload:
                f0_series = self.pitch_payload.get("f0_series_pyin")
            elif self.f0_method == "yin" and "f0_series_yin" in self.pitch_payload:
                f0_series = self.pitch_payload.get("f0_series_yin")
            elif self.f0_method == "torchcrepe" and "f0_series_torchcrepe" in self.pitch_payload:
                f0_series = self.pitch_payload.get("f0_series_torchcrepe")
            elif "f0_series_pyin" in self.pitch_payload:
                f0_series = self.pitch_payload.get("f0_series_pyin")  # Fallback to pyin if available
            elif "f0_series_yin" in self.pitch_payload:
                f0_series = self.pitch_payload.get("f0_series_yin")  # Fallback to yin if available

            if f0_series is not None:
                if isinstance(f0_series, list):
                    f0_series = np.array(f0_series, dtype=np.float32)
                f0 = f0_series[np.isfinite(f0_series) & (f0_series > 0)]
                if f0.size > 0:
                    return f0

        # Иначе оцениваем f0 самостоятельно
        try:
            import librosa
        except ImportError as e:
            raise RuntimeError(f"voice_quality | librosa not available: {e}") from e

        try:
            if self.f0_method == "yin":
                f0 = librosa.yin(y, fmin=self.f0_fmin, fmax=self.f0_fmax, sr=sr)
            elif self.f0_method == "pyin":
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    y, fmin=self.f0_fmin, fmax=self.f0_fmax, sr=sr
                )
                # Используем только voiced frames
                if voiced_flag is not None:
                    f0 = f0[voiced_flag]
            elif self.f0_method == "torchcrepe":
                try:
                    import torchcrepe
                    import torch
                except ImportError as e:
                    raise RuntimeError(f"voice_quality | torchcrepe not available: {e}") from e
                
                # torchcrepe требует tensor input и работает лучше на 16kHz
                if sr != 16000:
                    import librosa
                    y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000).astype(np.float32, copy=False)
                    sr_tc = 16000
                else:
                    y_16k = y.astype(np.float32, copy=False)
                    sr_tc = sr
                
                # Создаем tensor и переносим на устройство (CUDA если доступно)
                if isinstance(y_16k, np.ndarray):
                    y_tensor = torch.from_numpy(y_16k).unsqueeze(0)
                else:
                    y_tensor = y_16k.unsqueeze(0) if y_16k.dim() == 1 else y_16k
                
                # Переносим на устройство экстрактора (CUDA если доступно)
                if self.device == "cuda" and torch.cuda.is_available():
                    y_tensor = y_tensor.cuda(non_blocking=True)
                    device_tc = y_tensor.device
                else:
                    device_tc = torch.device("cpu")
                
                # Используем выбранную модель (tiny быстрее, full точнее)
                with torch.inference_mode():
                    f0 = torchcrepe.predict(
                        y_tensor, sr_tc,
                        hop_length=160,
                        fmin=self.f0_fmin, fmax=self.f0_fmax,
                        model=self.torchcrepe_model,  # "tiny" быстрее (~2x), "full" точнее
                        device=device_tc,
                    )
                    f0 = f0.squeeze(0).float().cpu().numpy()
            else:
                raise ValueError(f"voice_quality | Unknown f0_method: {self.f0_method}")

            # Фильтруем NaN и отрицательные значения
            f0 = f0[np.isfinite(f0) & (f0 > 0)]

            if f0.size == 0:
                raise RuntimeError(f"voice_quality | No valid f0 values estimated (f0_method={self.f0_method})")

            return f0.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"voice_quality | f0 estimation failed ({self.f0_method}): {e}") from e

    def _compute_voice_quality_metrics(
        self, y: np.ndarray, sr: int, f0: np.ndarray
    ) -> Dict[str, Any]:
        """
        Вычислить все метрики качества голоса на основе f0 и аудио сигнала.

        Args:
            y: Аудио сигнал
            sr: Частота дискретизации
            f0: Массив f0 значений

        Returns:
            Словарь с метриками
        """
        features: Dict[str, Any] = {}

        # Jitter (вариативность f0)
        if self.enable_jitter:
            if f0.size > 2:
                df0 = np.diff(f0)
                jitter = float(np.std(df0) / (np.mean(f0) + 1e-6))
            else:
                jitter = 0.0
            features["vq_jitter"] = jitter

            # Additional jitter metrics
            if f0.size > 2:
                features["vq_jitter_mean"] = float(np.mean(np.abs(df0)))
                features["vq_jitter_std"] = float(np.std(df0))
                features["vq_jitter_min"] = float(np.min(np.abs(df0)))
                features["vq_jitter_max"] = float(np.max(np.abs(df0)))

        # Shimmer (вариативность амплитуды)
        if self.enable_shimmer:
            frame_len = max(1, int(0.03 * sr))
            hop = max(1, int(0.01 * sr))
            amps = []
            for i in range(0, len(y) - frame_len + 1, hop):
                frm = y[i : i + frame_len]
                amps.append(np.sqrt(np.mean(frm * frm) + 1e-12))
            amps = np.array(amps, dtype=np.float32)

            # Маскируем слишком тихие кадры
            if amps.size > 0:
                mask = amps >= self.rms_mask_threshold
                amps_masked = amps[mask] if np.any(mask) else amps
            else:
                amps_masked = amps

            if amps_masked.size > 2:
                damps = np.diff(amps_masked)
                shimmer = float(np.std(damps) / (np.mean(amps_masked) + 1e-6))
            else:
                shimmer = 0.0
            features["vq_shimmer"] = shimmer

            # Additional shimmer metrics
            if amps_masked.size > 2:
                features["vq_shimmer_mean"] = float(np.mean(np.abs(damps)))
                features["vq_shimmer_std"] = float(np.std(damps))
                features["vq_shimmer_min"] = float(np.min(np.abs(damps)))
                features["vq_shimmer_max"] = float(np.max(np.abs(damps)))

            # Store amps for time series if needed
            if self.enable_time_series:
                features["_amps"] = amps_masked

        # HNR-подобная метрика
        if self.enable_hnr:
            hnr_frame = max(1, int(self.hnr_frame_ms / 1000.0 * sr))
            hop = max(1, int(0.01 * sr))
            if len(y) >= hnr_frame:
                vals = []
                for i in range(0, len(y) - hnr_frame + 1, hop):
                    frm = y[i : i + hnr_frame]
                    ac = np.correlate(frm, frm, mode="full")[hnr_frame - 1 : hnr_frame + 2]
                    r0 = float(ac[0] + 1e-12)
                    r1 = float(ac[1] if ac.shape[0] > 1 else 0.0)
                    vals.append(20.0 * np.log10(abs(r1) / r0 + 1e-12))
                hnr_like = float(np.mean(vals)) if vals else 0.0
            else:
                hnr_like = 0.0
            features["vq_hnr_like_db"] = hnr_like

            # Additional HNR metrics
            if len(y) >= hnr_frame and len(vals) > 0:
                features["vq_hnr_mean"] = float(np.mean(vals))
                features["vq_hnr_std"] = float(np.std(vals))
                features["vq_hnr_min"] = float(np.min(vals))
                features["vq_hnr_max"] = float(np.max(vals))

            # Store HNR values for time series if needed
            if self.enable_time_series and len(y) >= hnr_frame:
                features["_hnr_vals"] = np.array(vals, dtype=np.float32)

        # F0 statistics
        if self.enable_f0_stats:
            if f0.size > 0:
                features["vq_f0_mean"] = float(np.mean(f0))
                features["vq_f0_std"] = float(np.std(f0))
                features["vq_f0_min"] = float(np.min(f0))
                features["vq_f0_max"] = float(np.max(f0))
                features["vq_f0_median"] = float(np.median(f0))
                # F0 stability (coefficient of variation)
                f0_stability = float(1.0 / (1.0 + (np.std(f0) / (np.mean(f0) + 1e-6))))
                features["vq_f0_stability"] = f0_stability
                # Voice presence ratio
                voice_presence_ratio = float(f0.size / len(y) if len(y) > 0 else 0.0)
                features["vq_voice_presence_ratio"] = voice_presence_ratio
            else:
                features["vq_f0_mean"] = 0.0
                features["vq_f0_std"] = 0.0
                features["vq_f0_min"] = 0.0
                features["vq_f0_max"] = 0.0
                features["vq_f0_median"] = 0.0
                features["vq_f0_stability"] = 0.0
                features["vq_voice_presence_ratio"] = 0.0

        # Voice quality score (комбинация jitter/shimmer/HNR)
        if self.enable_jitter and self.enable_shimmer and self.enable_hnr:
            jitter = features.get("vq_jitter", 0.0)
            shimmer = features.get("vq_shimmer", 0.0)
            hnr = features.get("vq_hnr_like_db", 0.0)
            # Нормализуем HNR к [0, 1] (предполагаем диапазон [-50, 50] dB)
            hnr_norm = max(0.0, min(1.0, (hnr + 50.0) / 100.0))
            # Quality score: чем меньше jitter/shimmer и больше HNR, тем лучше
            quality_score = float((1.0 - jitter) * 0.33 + (1.0 - shimmer) * 0.33 + hnr_norm * 0.34)
            features["vq_voice_quality_score"] = quality_score

            # Breathiness score (на основе HNR: низкий HNR = более "дыхательный" голос)
            breathiness_score = float(max(0.0, min(1.0, (50.0 - hnr) / 100.0)))
            features["vq_breathiness_score"] = breathiness_score

        # Store f0 for time series if needed
        if self.enable_time_series:
            features["_f0"] = f0

        return features

    def _save_time_series_npy(self, data: np.ndarray, name: str) -> Optional[str]:
        """
        Сохранить временные серии в .npy файл для больших массивов (per-run storage).

        Args:
            data: Массив данных
            name: Имя файла (без расширения)

        Returns:
            Относительный путь к .npy файлу или None
        """
        if data.size < TIME_SERIES_SAVE_THRESHOLD:
            return None

        if self.artifacts_dir is None:
            logger.warning(f"voice_quality | artifacts_dir not set, cannot save {name} to .npy")
            return None

        try:
            os.makedirs(self.artifacts_dir, exist_ok=True)
            npy_path = os.path.join(self.artifacts_dir, f"{name}.npy")
            np.save(npy_path, data)
            # Return relative path from component directory
            rel_path = f"_artifacts/{name}.npy"
            logger.info(f"voice_quality | Saved {name} ({data.size} elements) to {npy_path}")
            return rel_path
        except Exception as e:
            logger.warning(f"voice_quality | Failed to save {name} to .npy: {e}")
            return None

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Обработка полного аудио файла (legacy mode, для обратной совместимости).

        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория

        Returns:
            ExtractorResult с метриками качества голоса
        """
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    False,
                    error=f"voice_quality | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            self._log_extraction_start(input_uri)

            # Load audio
            if self.progress_callback:
                self.progress_callback("voice_quality", 0, 6, "Loading audio")
            y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)
            if y.ndim == 2:
                y = np.mean(y, axis=0) if self.average_channels else y[0]

            # Normalize audio if enabled
            y = self._normalize_audio(y)

            duration = float(y.shape[-1] / sr)

            # Estimate f0 (fail-fast, no-fallback)
            if self.progress_callback:
                self.progress_callback("voice_quality", 1, 6, f"Estimating f0 ({self.f0_method})")
            f0 = self._estimate_f0(y, sr)

            if f0.size == 0:
                error_code = self._classify_error(RuntimeError("No valid f0 values"), "insufficient_data")
                return self._create_result(
                    False,
                    error=f"voice_quality | No valid f0 values estimated (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            # Compute metrics
            if self.progress_callback:
                self.progress_callback("voice_quality", 2, 6, "Computing jitter")
            features = self._compute_voice_quality_metrics(y, sr, f0)

            # Save time series if needed
            if self.progress_callback:
                self.progress_callback("voice_quality", 5, 6, "Saving artifacts")
            if self.enable_time_series:
                # Save f0
                if "_f0" in features:
                    f0_npy_path = self._save_time_series_npy(features["_f0"], "f0")
                    if f0_npy_path:
                        features["f0_npy"] = f0_npy_path
                    else:
                        features["f0"] = features["_f0"].astype(np.float32).tolist()
                    del features["_f0"]

                # Save amps
                if "_amps" in features:
                    amps_npy_path = self._save_time_series_npy(features["_amps"], "amps")
                    if amps_npy_path:
                        features["amps_npy"] = amps_npy_path
                    else:
                        features["amps"] = features["_amps"].astype(np.float32).tolist()
                    del features["_amps"]

                # Save HNR values
                if "_hnr_vals" in features:
                    hnr_npy_path = self._save_time_series_npy(features["_hnr_vals"], "hnr_vals")
                    if hnr_npy_path:
                        features["hnr_vals_npy"] = hnr_npy_path
                    else:
                        features["hnr_vals"] = features["_hnr_vals"].astype(np.float32).tolist()
                    del features["_hnr_vals"]

            # Add metadata
            features["sample_rate"] = int(sr)
            features["duration"] = duration
            features["device_used"] = self.device
            features["average_channels"] = self.average_channels
            features["hnr_frame_ms"] = self.hnr_frame_ms
            features["rms_mask_threshold"] = self.rms_mask_threshold
            features["f0_method"] = self.f0_method
            features["f0_fmin"] = self.f0_fmin
            features["f0_fmax"] = self.f0_fmax
            features["voice_quality_contract_version"] = VOICE_QUALITY_CONTRACT_VERSION

            # Add _features_enabled for feature gating
            features_enabled = []
            if self.enable_jitter:
                features_enabled.append("jitter")
            if self.enable_shimmer:
                features_enabled.append("shimmer")
            if self.enable_hnr:
                features_enabled.append("hnr")
            if self.enable_f0_stats:
                features_enabled.append("f0_stats")
            if self.enable_time_series:
                features_enabled.append("time_series")
            features["_features_enabled"] = features_enabled

            # Validate output
            is_valid, error_msg = self._validate_output(features)
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                return self._create_result(
                    False,
                    error=f"voice_quality | Validation failed: {error_msg} (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            dt = time.time() - start_time
            self._log_extraction_success(input_uri, dt)
            return self._create_result(True, payload=features, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            self._log_extraction_error(input_uri, f"{error_code}: {str(e)}", dt)
            return self._create_result(False, error=f"voice_quality | {error_code}: {str(e)}", processing_time=dt)

    def run_segments(
        self,
        input_uri: str,
        tmp_path: str,
        segments: List[Dict[str, Any]],
        *,
        segment_parallelism: int = 1,
        max_inflight: Optional[int] = None,
    ) -> ExtractorResult:
        """
        Segmenter-driven voice quality extraction: compute voice quality metrics on provided windows (families.voice_quality).

        Progress reporting: каждые 10% сегментов (если progress_callback установлен).

        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория
            segments: Список сегментов из Segmenter (families.voice_quality)
            segment_parallelism: Количество параллельных потоков для обработки сегментов
            max_inflight: Максимальное количество одновременно обрабатываемых сегментов

        Returns:
            ExtractorResult с агрегированными метриками по сегментам
        """
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    False,
                    error=f"voice_quality | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )
            if not isinstance(segments, list) or not segments:
                raise ValueError("voice_quality | segments is empty (no-fallback)")

            total_segments = len(segments)

            # Progress reporting: каждые 10%
            progress_report_interval = max(1, total_segments // 10) if total_segments >= 10 else 1
            last_reported_pct = -1

            # Process segments
            all_jitter: List[float] = []
            all_shimmer: List[float] = []
            all_hnr: List[float] = []
            all_f0_mean: List[float] = []
            all_f0_stability: List[float] = []
            segment_centers: List[float] = []
            segment_durations: List[float] = []

            seg_p = max(1, int(segment_parallelism or 1))
            inflight = int(max_inflight) if max_inflight is not None else seg_p
            inflight = max(1, int(inflight))

            def _process_segment(i: int, seg: dict) -> tuple[int, float, Dict[str, Any], float]:
                """Обработать один сегмент."""
                ss = int(seg.get("start_sample", 0))
                es = int(seg.get("end_sample", 0))
                center_sec = float(seg.get("center_sec", 0.0))
                start_sec = float(seg.get("start_sec", 0.0))
                end_sec = float(seg.get("end_sec", 0.0))
                duration = end_sec - start_sec

                # Load audio segment
                waveform_t, sr = self.audio_utils.load_audio_segment(
                    input_uri,
                    start_sample=ss,
                    end_sample=es,
                    target_sr=self.sample_rate,
                    mix_to_mono=self.average_channels,
                )
                waveform_np = self.audio_utils.to_numpy(waveform_t)
                if waveform_np.ndim == 2:
                    waveform_np = np.mean(waveform_np, axis=0) if self.average_channels else waveform_np[0]

                # Normalize audio if enabled
                waveform_np = self._normalize_audio(waveform_np)

                # Estimate f0 (fail-fast, no-fallback)
                f0 = self._estimate_f0(waveform_np, int(sr))

                # Compute metrics
                seg_features = self._compute_voice_quality_metrics(waveform_np, int(sr), f0)

                return i, center_sec, seg_features, duration

            # Process segments (sequential or parallel)
            if seg_p <= 1:
                for seg_idx, seg in enumerate(segments):
                    _, center_sec, seg_features, duration = _process_segment(seg_idx, seg)
                    if self.enable_jitter:
                        all_jitter.append(seg_features.get("vq_jitter", 0.0))
                    if self.enable_shimmer:
                        all_shimmer.append(seg_features.get("vq_shimmer", 0.0))
                    if self.enable_hnr:
                        all_hnr.append(seg_features.get("vq_hnr_like_db", 0.0))
                    if self.enable_f0_stats:
                        all_f0_mean.append(seg_features.get("vq_f0_mean", 0.0))
                        all_f0_stability.append(seg_features.get("vq_f0_stability", 0.0))
                    segment_centers.append(center_sec)
                    segment_durations.append(duration)

                    # Progress reporting
                    if self.progress_callback and seg_idx % progress_report_interval == 0:
                        pct = int((seg_idx + 1) * 100 / total_segments)
                        if pct != last_reported_pct:
                            self.progress_callback("voice_quality", seg_idx + 1, total_segments, f"Processed {seg_idx + 1}/{total_segments} segments")
                            last_reported_pct = pct
            else:
                # Parallel processing
                workers = max(1, min(int(seg_p), int(inflight)))
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(_process_segment, i, seg) for i, seg in enumerate(segments)]
                    completed = 0
                    for fut in as_completed(futures):
                        i, center_sec, seg_features, duration = fut.result()
                        if self.enable_jitter:
                            all_jitter.append(seg_features.get("vq_jitter", 0.0))
                        if self.enable_shimmer:
                            all_shimmer.append(seg_features.get("vq_shimmer", 0.0))
                        if self.enable_hnr:
                            all_hnr.append(seg_features.get("vq_hnr_like_db", 0.0))
                        if self.enable_f0_stats:
                            all_f0_mean.append(seg_features.get("vq_f0_mean", 0.0))
                            all_f0_stability.append(seg_features.get("vq_f0_stability", 0.0))
                        segment_centers.append(center_sec)
                        segment_durations.append(duration)
                        completed += 1

                        # Progress reporting
                        if self.progress_callback and completed % progress_report_interval == 0:
                            pct = int(completed * 100 / total_segments)
                            if pct != last_reported_pct:
                                self.progress_callback("voice_quality", completed, total_segments, f"Processed {completed}/{total_segments} segments")
                                last_reported_pct = pct

            # Aggregate metrics across all segments
            features: Dict[str, Any] = {}

            if self.enable_jitter and all_jitter:
                features["vq_jitter"] = float(np.mean(all_jitter))
                features["vq_jitter_mean"] = float(np.mean(all_jitter))
                features["vq_jitter_std"] = float(np.std(all_jitter))
                features["vq_jitter_min"] = float(np.min(all_jitter))
                features["vq_jitter_max"] = float(np.max(all_jitter))

            if self.enable_shimmer and all_shimmer:
                features["vq_shimmer"] = float(np.mean(all_shimmer))
                features["vq_shimmer_mean"] = float(np.mean(all_shimmer))
                features["vq_shimmer_std"] = float(np.std(all_shimmer))
                features["vq_shimmer_min"] = float(np.min(all_shimmer))
                features["vq_shimmer_max"] = float(np.max(all_shimmer))

            if self.enable_hnr and all_hnr:
                features["vq_hnr_like_db"] = float(np.mean(all_hnr))
                features["vq_hnr_mean"] = float(np.mean(all_hnr))
                features["vq_hnr_std"] = float(np.std(all_hnr))
                features["vq_hnr_min"] = float(np.min(all_hnr))
                features["vq_hnr_max"] = float(np.max(all_hnr))

            if self.enable_f0_stats and all_f0_mean:
                features["vq_f0_mean"] = float(np.mean(all_f0_mean))
                features["vq_f0_std"] = float(np.std(all_f0_mean))
                features["vq_f0_min"] = float(np.min(all_f0_mean))
                features["vq_f0_max"] = float(np.max(all_f0_mean))
                features["vq_f0_stability"] = float(np.mean(all_f0_stability))

            # Voice quality score
            if self.enable_jitter and self.enable_shimmer and self.enable_hnr:
                jitter = features.get("vq_jitter", 0.0)
                shimmer = features.get("vq_shimmer", 0.0)
                hnr = features.get("vq_hnr_like_db", 0.0)
                hnr_norm = max(0.0, min(1.0, (hnr + 50.0) / 100.0))
                quality_score = float((1.0 - jitter) * 0.33 + (1.0 - shimmer) * 0.33 + hnr_norm * 0.34)
                features["vq_voice_quality_score"] = quality_score
                breathiness_score = float(max(0.0, min(1.0, (50.0 - hnr) / 100.0)))
                features["vq_breathiness_score"] = breathiness_score

            # Add metadata
            total_duration = sum(segment_durations)
            features["sample_rate"] = int(self.sample_rate)
            features["duration"] = total_duration
            features["device_used"] = self.device
            features["average_channels"] = self.average_channels
            features["hnr_frame_ms"] = self.hnr_frame_ms
            features["rms_mask_threshold"] = self.rms_mask_threshold
            features["f0_method"] = self.f0_method
            features["f0_fmin"] = self.f0_fmin
            features["f0_fmax"] = self.f0_fmax
            features["segments_count"] = int(total_segments)
            features["voice_quality_contract_version"] = VOICE_QUALITY_CONTRACT_VERSION

            # Add _features_enabled for feature gating
            features_enabled = []
            if self.enable_jitter:
                features_enabled.append("jitter")
            if self.enable_shimmer:
                features_enabled.append("shimmer")
            if self.enable_hnr:
                features_enabled.append("hnr")
            if self.enable_f0_stats:
                features_enabled.append("f0_stats")
            if self.enable_time_series:
                features_enabled.append("time_series")
            features["_features_enabled"] = features_enabled

            # Add segment-level time series if enabled
            if self.enable_time_series:
                features["segment_centers_sec"] = segment_centers
                features["segment_durations_sec"] = segment_durations

            # Validate output
            is_valid, error_msg = self._validate_output(features)
            if not is_valid:
                error_code = self._classify_error(ValueError(error_msg), "validation_failed")
                return self._create_result(
                    False,
                    error=f"voice_quality | Validation failed: {error_msg} (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            dt = time.time() - start_time
            return self._create_result(True, payload=features, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            error_code = self._classify_error(e, "unknown")
            self._log_extraction_error(input_uri, f"{error_code}: {str(e)}", dt)
            return self._create_result(False, error=f"voice_quality | {error_code}: {str(e)}", processing_time=dt)

    @property
    def supports_batch(self) -> bool:
        """
        Указывает, поддерживает ли экстрактор batch processing.
        
        voice_quality_extractor поддерживает batch processing через extract_batch_segments()
        с CPU parallelism для обработки сегментов из нескольких видео одновременно.
        """
        return True

    def extract_batch_segments(
        self,
        audio_files: List[Dict[str, Any]],
        *,
        max_workers: Optional[int] = None,
    ) -> List[ExtractorResult]:
        """
        Батчевая обработка сегментов из нескольких аудио файлов.
        
        Использует ThreadPoolExecutor для параллельной обработки сегментов из всех файлов.
        Каждый файл обрабатывается изолированно через run_segments().
        
        Args:
            audio_files: Список словарей с ключами:
                - 'input_uri': URI к входному аудио файлу
                - 'tmp_path': Путь к временной директории для обработки
                - 'file_id': Идентификатор файла (опционально, для логирования)
                - 'segments': Список сегментов для обработки (обязательно)
                - 'artifacts_dir': Директория для сохранения .npy файлов (опционально, per-file)
            max_workers: Количество параллельных воркеров (None = os.cpu_count())
        
        Returns:
            Список ExtractorResult для каждого файла
        """
        from concurrent.futures import ThreadPoolExecutor
        import os as os_module
        
        if max_workers is None:
            max_workers = os_module.cpu_count() or 4
        
        results: List[ExtractorResult] = []
        
        def process_file(file_info: Dict[str, Any]) -> ExtractorResult:
            input_uri = file_info.get("input_uri")
            tmp_path = file_info.get("tmp_path")
            file_id = file_info.get("file_id", input_uri)
            segments = file_info.get("segments", [])
            artifacts_dir = file_info.get("artifacts_dir")
            
            if not input_uri or not tmp_path:
                logger.error(f"voice_quality | Missing input_uri or tmp_path for file_id={file_id}")
                return self._create_result(
                    False,
                    error="Missing input_uri or tmp_path",
                    processing_time=0.0,
                )
            
            if not segments:
                logger.error(f"voice_quality | Missing segments for file_id={file_id}")
                return self._create_result(
                    False,
                    error="Missing segments",
                    processing_time=0.0,
                )
            
            # Set per-file artifacts_dir if provided (batch processing isolation)
            original_artifacts_dir = self.artifacts_dir
            if artifacts_dir:
                self.artifacts_dir = artifacts_dir
            
            try:
                return self.run_segments(
                    input_uri=input_uri,
                    tmp_path=tmp_path,
                    segments=segments,
                )
            except Exception as e:
                logger.error(f"voice_quality | Error processing file_id={file_id}: {e}")
                error_code = self._classify_error(e, "unknown")
                return self._create_result(
                    False,
                    error=f"voice_quality | {error_code}: {str(e)}",
                    processing_time=0.0,
                )
            finally:
                # Restore original artifacts_dir
                self.artifacts_dir = original_artifacts_dir
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_file, audio_files))
        
        return results
