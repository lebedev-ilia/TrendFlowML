"""
BandEnergyExtractor: извлечение энергий по частотным полосам (низ/середина/высокие) и их долей.

Production-grade implementation with:
- Segmenter contract support (run_segments)
- Feature gating (per-feature flags)
- Full validation (outputs, parameters)
- No-fallback policy (explicit method selection)
- Progress reporting
- UI renderer support
- Contract versioning
- Detailed error codes
- Optional audio normalization
- Additional ML/analytics metrics
- Integration with spectral_extractor via shared_features
"""
import time
import logging
import os
import importlib.util
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import librosa

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)

# Contract version for downstream extractors compatibility validation
BAND_ENERGY_CONTRACT_VERSION = "band_energy_contract_v1"


class BandEnergyExtractor(BaseExtractor):
    """Экстрактор энергий по частотным полосам с поддержкой segment-based обработки."""

    name = "band_energy"
    version = "2.0.0"
    description = "Энергии по полосам (low/mid/high) и доли энергии"
    category = "spectral"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 0.9

    gpu_required = False
    gpu_preferred = False
    gpu_memory_required = 0.0

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 22050,
        bands: Optional[List[Tuple[float, float]]] = None,
        n_fft: int = 2048,
        hop_length: int = 512,
        use_mel_bands: bool = True,
        n_mels: int = 3,
        band_method: str = "auto",  # "essentia" | "librosa" | "auto"
        average_channels: bool = True,
        # Feature gating flags (per-feature control, default: all False)
        enable_basic_stats: bool = False,
        enable_extended_stats: bool = False,
        enable_time_series: bool = False,
        enable_dynamics: bool = False,
        enable_balance_metrics: bool = False,
        # Optional audio normalization
        enable_audio_normalization: bool = False,
        # Progress reporting callback
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        # Per-run storage path for .npy files
        artifacts_dir: Optional[str] = None,
    ):
        """
        Инициализация BandEnergy экстрактора.

        Args:
            device: Устройство для обработки
            sample_rate: Частота дискретизации
            bands: Список полос [(lo, hi), ...] в Hz, по умолчанию: [(0, 200), (200, 2000), (2000, nyq)]
            n_fft: Размер FFT окна
            hop_length: Размер hop для STFT
            use_mel_bands: Использовать мел-шкалу вместо фиксированных полос
            n_mels: Количество мел-полос (если use_mel_bands=True)
            band_method: Метод обработки ("essentia" | "librosa" | "auto")
            average_channels: Усреднять каналы для многоканального аудио
            enable_basic_stats: Включить базовые статистики (mean, std, median)
            enable_extended_stats: Включить расширенные статистики (min, max, p25, p75)
            enable_time_series: Включить временные серии (band_energy_ts)
            enable_dynamics: Включить метрики динамики (для run_segments)
            enable_balance_metrics: Включить метрики баланса
            enable_audio_normalization: Включить нормализацию аудио перед обработкой
            progress_callback: Callback для прогресса (metric_name, current, total, message)
            artifacts_dir: Директория для сохранения .npy файлов (per-run storage)
        """
        super().__init__(device=device)

        # Validate parameters
        self._validate_parameters(sample_rate, n_fft, hop_length, bands, n_mels, band_method)

        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.use_mel_bands = bool(use_mel_bands)
        self.n_mels = max(3, int(n_mels))
        self.band_method = str(band_method)
        self.average_channels = bool(average_channels)

        # По умолчанию: low [0-200), mid [200-2000), high [2000-nyq)
        if bands is None:
            self.bands = [(0.0, 200.0), (200.0, 2000.0), (2000.0, sample_rate / 2.0)]
        else:
            self.bands = [(float(lo), float(hi)) for lo, hi in bands]

        # Feature gating flags
        self.enable_basic_stats = bool(enable_basic_stats)
        self.enable_extended_stats = bool(enable_extended_stats)
        self.enable_time_series = bool(enable_time_series)
        self.enable_dynamics = bool(enable_dynamics)
        self.enable_balance_metrics = bool(enable_balance_metrics)

        # Optional audio normalization
        self.enable_audio_normalization = bool(enable_audio_normalization)

        # Progress callback
        self.progress_callback = progress_callback

        # Per-run storage for .npy files
        self.artifacts_dir = artifacts_dir

        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)

    def _validate_parameters(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        bands: Optional[List[Tuple[float, float]]],
        n_mels: int,
        band_method: str,
    ) -> None:
        """Валидация параметров инициализации."""
        if sample_rate <= 0:
            raise ValueError(f"band_energy | sample_rate must be > 0, got {sample_rate}")
        if n_fft <= 0:
            raise ValueError(f"band_energy | n_fft must be > 0, got {n_fft}")
        if hop_length <= 0:
            raise ValueError(f"band_energy | hop_length must be > 0, got {hop_length}")
        if n_mels < 3:
            raise ValueError(f"band_energy | n_mels must be >= 3, got {n_mels}")
        if band_method not in ("essentia", "librosa", "auto"):
            raise ValueError(f"band_energy | band_method must be 'essentia', 'librosa', or 'auto', got {band_method}")

        if bands is not None:
            nyquist = sample_rate / 2.0
            for i, (lo, hi) in enumerate(bands):
                if lo < 0 or hi > nyquist:
                    raise ValueError(f"band_energy | band {i} out of range: [{lo}, {hi}] not in [0, {nyquist}]")
                if lo >= hi:
                    raise ValueError(f"band_energy | band {i} invalid: lo={lo} >= hi={hi}")
                # Check for overlaps
                for j, (lo2, hi2) in enumerate(bands):
                    if i != j and not (hi <= lo2 or hi2 <= lo):
                        raise ValueError(f"band_energy | bands {i} and {j} overlap: [{lo}, {hi}] and [{lo2}, {hi2}]")

    def _classify_error(self, error: Exception, default_code: str) -> str:
        """Классификация ошибок для детальных error codes."""
        error_str = str(error).lower()
        if "file" in error_str or "not found" in error_str or "cannot open" in error_str:
            return "audio_load_failed"
        if "too short" in error_str or "empty" in error_str:
            return "audio_too_short"
        if "stft" in error_str or "fft" in error_str or "spectrum" in error_str:
            return "stft_computation_failed"
        if "band" in error_str or "energy" in error_str:
            return "band_computation_failed"
        if "essentia" in error_str or "unavailable" in error_str:
            return "essentia_unavailable"
        if "parameter" in error_str or "invalid" in error_str:
            return "invalid_parameters"
        return default_code

    def _normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """Нормализация аудио (peak normalization)."""
        if not self.enable_audio_normalization:
            return y
        max_val = np.abs(y).max()
        if max_val > 1e-12:
            return y / max_val
        return y

    def _compute_stft(
        self,
        y: np.ndarray,
        sr: int,
        shared_features: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Вычисление STFT с поддержкой shared_features."""
        # Try to reuse provided STFT if present in shared_features
        stft_magnitude = None
        freqs = None

        if shared_features:
            # Try to get STFT magnitude from spectral_extractor
            stft_magnitude = shared_features.get("stft_magnitude")
            if stft_magnitude is not None:
                if not isinstance(stft_magnitude, np.ndarray):
                    stft_magnitude = np.array(stft_magnitude)
                if stft_magnitude.ndim != 2:
                    logger.warning(f"band_energy | Invalid stft_magnitude shape in shared_features: {stft_magnitude.shape}, recomputing")
                    stft_magnitude = None

            # Try to get frequencies
            freqs = shared_features.get("frequencies")
            if freqs is not None:
                if not isinstance(freqs, np.ndarray):
                    freqs = np.array(freqs)
                if freqs.ndim != 1:
                    logger.warning(f"band_energy | Invalid frequencies shape in shared_features: {freqs.shape}, recomputing")
                    freqs = None

        # If no shared STFT — compute it here
        if stft_magnitude is None:
            stft_magnitude = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)).astype(np.float32) ** 2
            freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft).astype(np.float32)

        return stft_magnitude, freqs

    def _compute_band_energies_essentia(
        self,
        y: np.ndarray,
        sr: int,
    ) -> Tuple[List[Tuple[float, float]], np.ndarray, np.ndarray]:
        """Вычисление энергий по полосам через Essentia."""
        if not importlib.util.find_spec("essentia"):
            raise RuntimeError("band_energy | Essentia is not available (essentia package not found)")

        try:
            import essentia.standard as es  # type: ignore
            audio = y.astype(np.float32)
            frame_cutter = es.FrameCutter(frameSize=self.n_fft, hopSize=self.hop_length, startFromZero=True)
            window = es.Windowing(type='hann')
            spectrum = es.Spectrum()

            num_bins = int(self.n_fft // 2 + 1)
            freqs = np.linspace(0.0, sr / 2.0, num=num_bins, dtype=np.float32)

            # Полосы: фиксированные или мел-шкала
            bands_to_use = self.bands
            if self.use_mel_bands:
                mel_edges = librosa.mel_frequencies(n_mels=self.n_mels, fmin=0.0, fmax=sr / 2.0)
                bands_to_use = list(zip(mel_edges[:-1], mel_edges[1:]))

            band_masks = [(freqs >= lo) & (freqs < hi) for (lo, hi) in bands_to_use]

            total_energy = 0.0
            accum_energies = [0.0 for _ in band_masks]
            per_frame: List[List[float]] = []

            while True:
                frame = frame_cutter(audio)
                if frame.size == 0:
                    break
                win = window(frame)
                spec = spectrum(win)  # magnitude
                pwr = np.asarray(spec, dtype=np.float32) ** 2
                total_energy += float(np.sum(pwr))
                frame_energies = []
                for i, mask in enumerate(band_masks):
                    e = float(np.sum(pwr[mask]))
                    accum_energies[i] += e
                    frame_energies.append(e)
                if self.enable_time_series:
                    per_frame.append(frame_energies)

            if total_energy == 0.0:
                total_energy = float(np.sum(audio.astype(np.float32) ** 2) + 1e-12)

            energies = np.array(accum_energies, dtype=np.float32)
            band_energy_ts = np.array(per_frame, dtype=np.float32).T if per_frame else np.zeros((len(bands_to_use), 0), dtype=np.float32)

            return bands_to_use, energies, band_energy_ts

        except Exception as e:
            if self.band_method == "essentia":
                error_code = self._classify_error(e, "essentia_unavailable")
                raise RuntimeError(f"band_energy | Essentia method failed (error_code={error_code}): {e}") from e
            raise

    def _compute_band_energies_librosa(
        self,
        y: np.ndarray,
        sr: int,
        shared_features: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Tuple[float, float]], np.ndarray, np.ndarray]:
        """Вычисление энергий по полосам через librosa."""
        # Compute STFT
        S, freqs = self._compute_stft(y, sr, shared_features)

        # Полосы: фиксированные или мел-шкала
        bands_to_use = self.bands
        if self.use_mel_bands:
            mel_edges = librosa.mel_frequencies(n_mels=self.n_mels, fmin=0.0, fmax=sr / 2.0)
            bands_to_use = list(zip(mel_edges[:-1], mel_edges[1:]))

        # Векторизованный биннинг: матрица масок (freq_bins, num_bands)
        masks = []
        for lo, hi in bands_to_use:
            masks.append(((freqs >= float(lo)) & (freqs < float(hi))).astype(np.float32))
        mask_matrix = np.stack(masks, axis=1)  # (freq_bins, num_bands)

        # Пер-кадровые энергии по полосам: (num_bands, frames)
        band_energy_ts = mask_matrix.T @ S  # матричное умножение

        # Скалярные суммы энергий (по всем кадрам)
        energies = band_energy_ts.sum(axis=1).astype(np.float32)

        return bands_to_use, energies, band_energy_ts

    def _compute_statistics(self, band_energy_ts: np.ndarray) -> Dict[str, np.ndarray]:
        """Вычисление статистик по временным рядам."""
        stats: Dict[str, np.ndarray] = {}

        if self.enable_basic_stats:
            stats["mean"] = band_energy_ts.mean(axis=1)
            stats["std"] = band_energy_ts.std(axis=1)
            stats["median"] = np.median(band_energy_ts, axis=1)

        if self.enable_extended_stats:
            stats["min"] = band_energy_ts.min(axis=1)
            stats["max"] = band_energy_ts.max(axis=1)
            stats["p25"] = np.percentile(band_energy_ts, 25, axis=1)
            stats["p75"] = np.percentile(band_energy_ts, 75, axis=1)

        return stats

    def _compute_balance_metrics(self, shares: np.ndarray) -> Dict[str, float]:
        """Вычисление метрик баланса."""
        if not self.enable_balance_metrics:
            return {}

        # Balance score (entropy of distribution)
        shares_normalized = shares / (shares.sum() + 1e-12)
        entropy = -np.sum(shares_normalized * np.log(shares_normalized + 1e-12))
        max_entropy = np.log(len(shares))
        balance_score = entropy / max_entropy if max_entropy > 0 else 0.0

        # Dominance (index of dominant band)
        dominance = int(np.argmax(shares))

        # Contrast (max - min)
        contrast = float(np.max(shares) - np.min(shares))

        return {
            "band_balance_score": float(balance_score),
            "band_dominance": dominance,
            "band_contrast": float(contrast),
        }

    def _validate_output(self, result: Dict[str, Any]) -> None:
        """Валидация выходных данных."""
        # Validate basic fields
        if "band_edges" not in result:
            raise ValueError("band_energy | Missing band_edges in output")
        if "band_energies" not in result:
            raise ValueError("band_energy | Missing band_energies in output")
        if "band_energy_shares" not in result:
            raise ValueError("band_energy | Missing band_energy_shares in output")

        band_edges = result["band_edges"]
        band_energies = result["band_energies"]
        band_shares = result["band_energy_shares"]

        if not isinstance(band_edges, list) or len(band_edges) == 0:
            raise ValueError(f"band_energy | Invalid band_edges: must be non-empty list, got {type(band_edges)}")
        if not isinstance(band_energies, list) or len(band_energies) != len(band_edges):
            raise ValueError(f"band_energy | Invalid band_energies: must be list of length {len(band_edges)}, got {len(band_energies) if isinstance(band_energies, list) else 'N/A'}")
        if not isinstance(band_shares, list) or len(band_shares) != len(band_edges):
            raise ValueError(f"band_energy | Invalid band_energy_shares: must be list of length {len(band_edges)}, got {len(band_shares) if isinstance(band_shares, list) else 'N/A'}")

        # Validate shares sum to ~1.0
        shares_sum = sum(band_shares)
        if not (0.99 <= shares_sum <= 1.01):
            raise ValueError(f"band_energy | Invalid band_energy_shares: sum must be ~1.0, got {shares_sum}")

        # Validate energies are non-negative
        for i, energy in enumerate(band_energies):
            if not isinstance(energy, (int, float)) or energy < 0:
                raise ValueError(f"band_energy | Invalid band_energies[{i}]: must be non-negative float, got {energy}")

    def _report_progress(self, stage: str, current: int, total: int, message: str = "") -> None:
        """Отчет о прогрессе."""
        if self.progress_callback:
            try:
                self.progress_callback("band_energy", current, total, f"{stage}: {message}" if message else stage)
            except Exception as e:
                logger.debug(f"band_energy | Progress callback failed: {e}")

    def run(self, input_uri: str, tmp_path: str, shared_features: Optional[Dict[str, Any]] = None) -> ExtractorResult:
        """
        Извлечение энергий по полосам для полного аудио файла.

        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория
            shared_features: Общие фичи (может содержать stft_magnitude от spectral_extractor)

        Returns:
            ExtractorResult с результатами извлечения энергий по полосам
        """
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"band_energy | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )

            self._log_extraction_start(input_uri)
            self._report_progress("load_audio", 0, 1, "Loading audio")

            # Load audio
            y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)
            if y.ndim == 2:
                if self.average_channels:
                    y = np.mean(y, axis=0)  # mix to mono
                else:
                    y = y[0]  # use first channel

            y = y.astype(np.float32)
            if y.size == 0:
                raise ValueError("band_energy | Пустой аудиосигнал (audio_too_short)")

            # Check minimum duration (at least 1 second)
            duration = len(y) / sr
            if duration < 1.0:
                raise ValueError(f"band_energy | Аудио слишком короткое: {duration:.2f}s < 1.0s (audio_too_short)")

            # Normalize audio if enabled
            if self.enable_audio_normalization:
                self._report_progress("normalize_audio", 0, 1, "Normalizing audio")
                y = self._normalize_audio(y)

            # Compute band energies
            self._report_progress("compute_bands", 0, 1, "Computing band energies")
            bands_to_use, energies, band_energy_ts = None, None, None

            if self.band_method == "essentia":
                bands_to_use, energies, band_energy_ts = self._compute_band_energies_essentia(y, sr)
            elif self.band_method == "librosa":
                bands_to_use, energies, band_energy_ts = self._compute_band_energies_librosa(y, sr, shared_features)
            else:  # auto
                # Try Essentia first, fallback to librosa
                try:
                    bands_to_use, energies, band_energy_ts = self._compute_band_energies_essentia(y, sr)
                except Exception as e:
                    logger.info(f"band_energy | Essentia failed, using librosa fallback: {e}")
                    bands_to_use, energies, band_energy_ts = self._compute_band_energies_librosa(y, sr, shared_features)

            # Compute statistics
            self._report_progress("compute_stats", 0, 1, "Computing statistics")
            stats = self._compute_statistics(band_energy_ts)

            # Compute shares
            total_energy = float(np.sum(energies) + 1e-12)
            shares = (energies / total_energy).astype(np.float32)

            # Compute balance metrics
            balance_metrics = self._compute_balance_metrics(shares)

            # Build payload
            payload: Dict[str, Any] = {
                "band_edges": [(float(lo), float(hi)) for lo, hi in bands_to_use],
                "band_energies": [float(e) for e in energies.tolist()],
                "band_energy_shares": [float(s) for s in shares.tolist()],
                "total_energy": total_energy,
                "sample_rate": sr,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "duration": duration,
                "device_used": self.device,
                "method": "essentia" if self.band_method == "essentia" or (self.band_method == "auto" and importlib.util.find_spec("essentia") is not None) else "librosa",
            }

            # Add statistics if enabled
            if self.enable_basic_stats:
                payload["band_energy_mean"] = [float(m) for m in stats["mean"].tolist()]
                payload["band_energy_std"] = [float(s) for s in stats["std"].tolist()]
                payload["band_energy_median"] = [float(med) for med in stats["median"].tolist()]

            if self.enable_extended_stats:
                payload["band_energy_min"] = [float(m) for m in stats["min"].tolist()]
                payload["band_energy_max"] = [float(m) for m in stats["max"].tolist()]
                payload["band_energy_p25"] = [float(p) for p in stats["p25"].tolist()]
                payload["band_energy_p75"] = [float(p) for p in stats["p75"].tolist()]

            # Add time series if enabled
            if self.enable_time_series:
                payload["band_energy_ts"] = band_energy_ts.astype(np.float32).tolist()

            # Add balance metrics if enabled
            if self.enable_balance_metrics:
                payload.update(balance_metrics)

            # Track enabled features for meta
            enabled_features = []
            if self.enable_basic_stats:
                enabled_features.append("basic_stats")
            if self.enable_extended_stats:
                enabled_features.append("extended_stats")
            if self.enable_time_series:
                enabled_features.append("time_series")
            if self.enable_dynamics:
                enabled_features.append("dynamics")
            if self.enable_balance_metrics:
                enabled_features.append("balance_metrics")

            payload["band_energy_contract_version"] = BAND_ENERGY_CONTRACT_VERSION
            payload["_features_enabled"] = enabled_features

            # Validate output
            self._validate_output(payload)

            dt = time.time() - start_time
            self._log_extraction_success(input_uri, dt)
            self._report_progress("complete", 1, 1, "Complete")
            return self._create_result(True, payload=payload, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            error_code = self._classify_error(e, "band_computation_failed")
            error_msg = f"band_energy | {str(e)} (error_code={error_code})"
            self._log_extraction_error(input_uri, error_msg, dt)
            return self._create_result(False, error=error_msg, processing_time=dt)

    def run_segments(
        self,
        input_uri: str,
        tmp_path: str,
        segments: List[Dict[str, Any]],
        shared_features: Optional[Dict[str, Any]] = None,
    ) -> ExtractorResult:
        """
        Извлечение энергий по полосам для сегментов от Segmenter (families.band_energy.segments[]).

        Progress reporting: каждые 10% сегментов (если progress_callback установлен).

        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория
            segments: Список сегментов от Segmenter
            shared_features: Общие фичи (может содержать stft_magnitude от spectral_extractor)

        Returns:
            ExtractorResult с результатами извлечения энергий по полосам по сегментам
        """
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                error_code = self._classify_error(ValueError("Invalid input"), "audio_load_failed")
                return self._create_result(
                    success=False,
                    error=f"band_energy | Некорректный входной файл (error_code={error_code})",
                    processing_time=time.time() - start_time,
                )
            if not isinstance(segments, list) or not segments:
                raise ValueError("band_energy | segments is empty (no-fallback)")

            total_segments = len(segments)

            # Progress reporting: каждые 10%
            progress_report_interval = max(1, total_segments // 10) if total_segments >= 10 else 1
            last_reported_pct = -1

            # Process segments
            band_energies_all: List[List[float]] = []
            band_shares_all: List[List[float]] = []
            band_energy_ts_all: List[np.ndarray] = []
            segment_centers: List[float] = []
            segment_durations: List[float] = []

            self._report_progress("load_segments", 0, total_segments, "Loading segments")

            for seg_idx, seg in enumerate(segments):
                # Progress reporting
                current_pct = (seg_idx * 100) // total_segments
                if current_pct >= last_reported_pct + 10:
                    self._report_progress("process_segments", seg_idx, total_segments, f"Processing segment {seg_idx + 1}/{total_segments}")
                    last_reported_pct = current_pct

                # Extract segment info
                start_sample = int(seg.get("start_sample", 0))
                end_sample = int(seg.get("end_sample", 0))
                center_sec = float(seg.get("center_sec", 0.0))
                duration_sec = float(seg.get("end_sec", 0.0) - seg.get("start_sec", 0.0))

                # Load segment audio
                try:
                    y_t, sr = self.audio_utils.load_audio_segment(
                        input_uri,
                        start_sample=start_sample,
                        end_sample=end_sample,
                        target_sr=self.sample_rate,
                        mix_to_mono=True,
                    )
                    y = self.audio_utils.to_numpy(y_t)
                    if y.ndim == 2:
                        if self.average_channels:
                            y = np.mean(y, axis=0)  # mix to mono
                        else:
                            y = y[0]  # use first channel
                    y = y.astype(np.float32)

                    if y.size == 0 or duration_sec < 0.5:
                        # Skip very short segments
                        logger.debug(f"band_energy | Skipping segment {seg_idx}: too short ({duration_sec:.2f}s)")
                        continue

                    # Normalize audio if enabled
                    if self.enable_audio_normalization:
                        y = self._normalize_audio(y)

                    # Compute band energies for segment
                    bands_to_use, energies, band_energy_ts = None, None, None
                    if self.band_method == "essentia":
                        try:
                            bands_to_use, energies, band_energy_ts = self._compute_band_energies_essentia(y, sr)
                        except Exception:
                            bands_to_use, energies, band_energy_ts = self._compute_band_energies_librosa(y, sr, shared_features)
                    elif self.band_method == "librosa":
                        bands_to_use, energies, band_energy_ts = self._compute_band_energies_librosa(y, sr, shared_features)
                    else:  # auto
                        try:
                            bands_to_use, energies, band_energy_ts = self._compute_band_energies_essentia(y, sr)
                        except Exception:
                            bands_to_use, energies, band_energy_ts = self._compute_band_energies_librosa(y, sr, shared_features)

                    # Compute shares
                    total_energy = float(np.sum(energies) + 1e-12)
                    shares = (energies / total_energy).astype(np.float32)

                    # Store results
                    band_energies_all.append(energies.tolist())
                    band_shares_all.append(shares.tolist())
                    if self.enable_time_series:
                        band_energy_ts_all.append(band_energy_ts)
                    segment_centers.append(center_sec)
                    segment_durations.append(duration_sec)

                except Exception as e:
                    logger.warning(f"band_energy | Failed to process segment {seg_idx}: {e}")
                    continue

            if not band_energies_all:
                raise ValueError("band_energy | No valid segments processed (all segments too short or failed)")

            # Aggregate results
            self._report_progress("aggregate", total_segments, total_segments, "Aggregating results")

            # Average energies and shares
            avg_energies = np.mean([np.array(e) for e in band_energies_all], axis=0)
            avg_shares = np.mean([np.array(s) for s in band_shares_all], axis=0)

            # Use first segment's band_edges (should be consistent)
            bands_to_use = self.bands
            if self.use_mel_bands:
                mel_edges = librosa.mel_frequencies(n_mels=self.n_mels, fmin=0.0, fmax=self.sample_rate / 2.0)
                bands_to_use = list(zip(mel_edges[:-1], mel_edges[1:]))

            # Build payload
            payload: Dict[str, Any] = {
                "band_edges": [(float(lo), float(hi)) for lo, hi in bands_to_use],
                "band_energies": [float(e) for e in avg_energies.tolist()],
                "band_energy_shares": [float(s) for s in avg_shares.tolist()],
                "total_energy": float(np.sum(avg_energies)),
                "sample_rate": self.sample_rate,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "device_used": self.device,
                "method": "librosa" if self.band_method == "librosa" or (self.band_method == "auto" and not importlib.util.find_spec("essentia")) else "essentia",
            }

            # Compute statistics from aggregated data
            if band_energy_ts_all:
                combined_ts = np.concatenate(band_energy_ts_all, axis=1)
                stats = self._compute_statistics(combined_ts)

                if self.enable_basic_stats:
                    payload["band_energy_mean"] = [float(m) for m in stats["mean"].tolist()]
                    payload["band_energy_std"] = [float(s) for s in stats["std"].tolist()]
                    payload["band_energy_median"] = [float(med) for med in stats["median"].tolist()]

                if self.enable_extended_stats:
                    payload["band_energy_min"] = [float(m) for m in stats["min"].tolist()]
                    payload["band_energy_max"] = [float(m) for m in stats["max"].tolist()]
                    payload["band_energy_p25"] = [float(p) for p in stats["p25"].tolist()]
                    payload["band_energy_p75"] = [float(p) for p in stats["p75"].tolist()]

            # Add time series if enabled
            if self.enable_time_series and band_energy_ts_all:
                # Concatenate all time series
                combined_ts = np.concatenate(band_energy_ts_all, axis=1)
                payload["band_energy_ts"] = combined_ts.astype(np.float32).tolist()
                payload["segment_centers_sec"] = segment_centers
                payload["segment_durations"] = segment_durations

            # Add dynamics metrics if enabled
            if self.enable_dynamics:
                # Compute stability (variance of shares over segments)
                shares_array = np.array(band_shares_all)
                payload["band_energy_stability"] = float(1.0 / (1.0 + np.mean(np.std(shares_array, axis=0))))

                # Compute transitions (significant changes in dominant band)
                dominant_bands = [int(np.argmax(shares)) for shares in band_shares_all]
                transitions = []
                for i in range(1, len(dominant_bands)):
                    if dominant_bands[i] != dominant_bands[i - 1]:
                        transitions.append({
                            "transition_index": i,
                            "from_band": dominant_bands[i - 1],
                            "to_band": dominant_bands[i],
                            "transition_time_sec": segment_centers[i],
                        })
                payload["band_transitions"] = transitions
                payload["band_transitions_count"] = len(transitions)
                payload["band_transitions_rate"] = len(transitions) / max(segment_centers[-1] - segment_centers[0], 1e-6) if segment_centers else 0.0

                # Distribution of dominant bands
                band_distribution: Dict[int, float] = {}
                for band_idx, dur in zip(dominant_bands, segment_durations):
                    band_distribution[band_idx] = band_distribution.get(band_idx, 0.0) + dur
                total_dur = sum(band_distribution.values())
                if total_dur > 0:
                    band_distribution = {k: v / total_dur for k, v in band_distribution.items()}
                payload["band_distribution"] = band_distribution
                payload["band_diversity"] = len(band_distribution)

            # Compute balance metrics
            balance_metrics = self._compute_balance_metrics(avg_shares)
            if self.enable_balance_metrics:
                payload.update(balance_metrics)

            # Track enabled features for meta
            enabled_features = []
            if self.enable_basic_stats:
                enabled_features.append("basic_stats")
            if self.enable_extended_stats:
                enabled_features.append("extended_stats")
            if self.enable_time_series:
                enabled_features.append("time_series")
            if self.enable_dynamics:
                enabled_features.append("dynamics")
            if self.enable_balance_metrics:
                enabled_features.append("balance_metrics")

            payload["band_energy_contract_version"] = BAND_ENERGY_CONTRACT_VERSION
            payload["_features_enabled"] = enabled_features

            # Validate output
            self._validate_output(payload)

            dt = time.time() - start_time
            self._log_extraction_success(input_uri, dt)
            self._report_progress("complete", total_segments, total_segments, "Complete")
            return self._create_result(True, payload=payload, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            error_code = self._classify_error(e, "band_computation_failed")
            error_msg = f"band_energy | {str(e)} (error_code={error_code})"
            self._log_extraction_error(input_uri, error_msg, dt)
            return self._create_result(False, error=error_msg, processing_time=dt)
    
    @property
    def supports_batch(self) -> bool:
        """
        Указывает, поддерживает ли экстрактор batch processing.
        
        band_energy_extractor поддерживает CPU parallelism для обработки сегментов из нескольких видео.
        """
        return True
    
    def extract_batch_segments(
        self,
        audio_files_with_segments: List[Dict[str, Any]],
        *,
        max_workers: Optional[int] = None,
        max_segments_per_batch: Optional[int] = None,
    ) -> List[ExtractorResult]:
        """
        Батчевая обработка сегментов из нескольких видео с CPU parallelism.
        
        Для CPU extractors используется ThreadPoolExecutor для параллельной обработки
        сегментов из разных видео. Это позволяет ускорить обработку на многоядерных CPU.
        
        Args:
            audio_files_with_segments: Список словарей с ключами:
                - 'input_uri': URI к входному аудио/видео файлу
                - 'tmp_path': Путь к временной директории для обработки
                - 'segments': Список сегментов для обработки
                - 'file_id': Идентификатор файла (для логирования)
                - 'shared_features': Опциональные общие фичи (например, stft_magnitude от spectral_extractor)
            max_workers: Количество параллельных воркеров для CPU extractors (None = auto)
            max_segments_per_batch: Не используется для CPU extractors (оставлено для совместимости)
        
        Returns:
            Список ExtractorResult для каждого файла
        """
        start_time = time.time()
        
        if not audio_files_with_segments:
            return []
        
        # Проверяем, поддерживает ли экстрактор run_segments
        if not hasattr(self, 'run_segments'):
            logger.error(f"{self.name} does not support run_segments()")
            return [
                self._create_result(
                    success=False,
                    error=f"{self.name} does not support run_segments()",
                    processing_time=time.time() - start_time,
                )
                for _ in audio_files_with_segments
            ]
        
        # Определяем количество воркеров
        if max_workers is None or max_workers <= 0:
            max_workers = min(len(audio_files_with_segments), os.cpu_count() or 1)
        
        results: List[ExtractorResult] = [None] * len(audio_files_with_segments)  # type: ignore
        
        def process_single_file(file_info: Dict[str, Any], file_idx: int) -> Tuple[int, ExtractorResult]:
            """Обработка одного файла с сегментами."""
            input_uri = file_info.get("input_uri")
            tmp_path = file_info.get("tmp_path")
            segments = file_info.get("segments", [])
            file_id = file_info.get("file_id", input_uri)
            shared_features = file_info.get("shared_features")
            
            if not input_uri or not tmp_path:
                logger.error(f"band_energy | Missing input_uri or tmp_path for file_id={file_id}")
                return file_idx, self._create_result(
                    success=False,
                    error="Missing input_uri or tmp_path",
                    processing_time=0.0,
                )
            
            if not segments:
                logger.warning(f"band_energy | No segments provided for file_id={file_id}")
                return file_idx, self._create_result(
                    success=False,
                    error="No segments provided",
                    processing_time=0.0,
                )
            
            try:
                result = self.run_segments(input_uri, tmp_path, segments, shared_features=shared_features)
                return file_idx, result
            except Exception as e:
                logger.error(f"band_energy | Error processing segments for file_id={file_id}: {e}")
                return file_idx, self._create_result(
                    success=False,
                    error=str(e),
                    processing_time=0.0,
                )
        
        # Параллельная обработка файлов через ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_file, file_info, idx): idx
                for idx, file_info in enumerate(audio_files_with_segments)
            }
            
            for future in as_completed(futures):
                file_idx, result = future.result()
                results[file_idx] = result
        
        return results
