"""
BandEnergyExtractor: извлечение энергий по частотным полосам (низ/середина/высокие) и их долей.

Production-grade implementation with:
- Segmenter contract support (run_segments)
- Feature gating (per-feature flags)
- Full validation (outputs, parameters)
- Audit v3 no-fallback policy (librosa-only; no Essentia/auto fallback)
- Progress reporting
- UI renderer support
- Contract versioning
- Detailed error codes
- Audio normalization enabled by default (Audit v3)
- Additional ML/analytics metrics
- Integration with spectral_extractor via shared_features
"""
import time
import logging
import os
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import librosa

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils
from src.extractors.band_energy_extractor.utils.resource_profile import (
    prefix_snapshot,
    resource_profile_enabled,
    snapshot_process_resources,
)

logger = logging.getLogger(__name__)

# Contract version for downstream extractors compatibility validation
BAND_ENERGY_CONTRACT_VERSION = "band_energy_contract_v1"


class BandEnergyExtractor(BaseExtractor):
    """Экстрактор энергий по частотным полосам с поддержкой segment-based обработки."""

    name = "band_energy"
    version = "2.1.1"
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
        # Audit v3: canonical output uses fixed 3 bands (low/mid/high).
        use_mel_bands: bool = False,
        n_mels: int = 3,
        # Audit v3: librosa-only; fail-fast for other values.
        band_method: str = "librosa",  # "librosa" (audit v3)
        average_channels: bool = True,
        # Feature gating flags (per-feature control, default: all False)
        enable_basic_stats: bool = False,
        enable_extended_stats: bool = False,
        enable_time_series: bool = False,
        enable_dynamics: bool = False,
        enable_balance_metrics: bool = False,
        # Optional audio normalization
        enable_audio_normalization: bool = True,
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
        self._validate_parameters(sample_rate, n_fft, hop_length, bands, use_mel_bands, n_mels, band_method)

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

        # Audit v3 (band_energy): we keep only minimal + optional segment-aligned sequences.
        # Stats/dynamics are removed from the audited contract (fail-fast if enabled).
        if self.enable_basic_stats or self.enable_extended_stats or self.enable_dynamics:
            raise RuntimeError(
                "band_energy | Audit v3: basic/extended stats and dynamics are not supported in audited contract. "
                "Disable enable_basic_stats/enable_extended_stats/enable_dynamics."
            )

        # Optional audio normalization
        self.enable_audio_normalization = bool(enable_audio_normalization)

        # Progress callback
        self.progress_callback = progress_callback

        # Per-run storage for .npy files
        self.artifacts_dir = artifacts_dir

        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        # Cache band masks for per-segment speed (avoid rebuilding per window).
        self._mask_cache_key: Optional[Tuple[int, int]] = None  # (sr, n_fft)
        self._mask_cache_matrix: Optional[np.ndarray] = None

    def _mask_matrix_for_freqs(self, freqs: np.ndarray, sr: int) -> np.ndarray:
        """
        Cached band mask matrix for (sr, n_fft).

        Audit v3 for this extractor uses fixed 3 bands; caching is safe and speeds up run_segments().
        """
        key = (int(sr), int(self.n_fft))
        if self._mask_cache_key == key and self._mask_cache_matrix is not None:
            if int(self._mask_cache_matrix.shape[0]) == int(freqs.shape[0]):
                return self._mask_cache_matrix

        masks = [((freqs >= float(lo)) & (freqs < float(hi))).astype(np.float32) for lo, hi in self.bands]
        mask_matrix = np.stack(masks, axis=1)  # (freq_bins, 3)
        self._mask_cache_key = key
        self._mask_cache_matrix = mask_matrix
        return mask_matrix

    def _validate_parameters(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        bands: Optional[List[Tuple[float, float]]],
        use_mel_bands: bool,
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
        # Audit v3: librosa-only
        if band_method != "librosa":
            raise RuntimeError(f"band_energy | Audit v3: band_method must be 'librosa', got {band_method!r}")
        # Audit v3: fixed bands only
        if use_mel_bands:
            raise RuntimeError("band_energy | Audit v3: mel bands are not supported (use fixed 3 bands)")
        if n_mels < 3:
            raise ValueError(f"band_energy | n_mels must be >= 3, got {n_mels}")

        if bands is not None:
            if len(bands) != 3:
                raise RuntimeError(f"band_energy | Audit v3: bands must have length 3 (low/mid/high), got {len(bands)}")
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
                    stft_magnitude = np.asarray(stft_magnitude)
                if stft_magnitude.ndim != 2:
                    logger.warning(f"band_energy | Invalid stft_magnitude shape in shared_features: {stft_magnitude.shape}, recomputing")
                    stft_magnitude = None

            # Try to get frequencies
            freqs = shared_features.get("frequencies")
            if freqs is not None:
                if not isinstance(freqs, np.ndarray):
                    freqs = np.asarray(freqs)
                if freqs.ndim != 1:
                    logger.warning(f"band_energy | Invalid frequencies shape in shared_features: {freqs.shape}, recomputing")
                    freqs = None

            # Best-effort guard: ignore shared STFT if it looks unrelated to current window.
            if stft_magnitude is not None and freqs is not None:
                try:
                    exp_bins = int(self.n_fft // 2 + 1)
                    if int(stft_magnitude.shape[0]) != exp_bins or int(freqs.shape[0]) != exp_bins:
                        stft_magnitude = None
                        freqs = None
                    else:
                        approx_frames = max(1, int(np.ceil(float(len(y)) / float(self.hop_length))))
                        shared_frames = int(stft_magnitude.shape[1])
                        if shared_frames > int(approx_frames * 3.0) or shared_frames < int(max(1, approx_frames // 3)):
                            stft_magnitude = None
                            freqs = None
                except Exception:
                    stft_magnitude = None
                    freqs = None

        # If no shared STFT — compute it here
        if stft_magnitude is None:
            stft_c = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
            mag = np.abs(stft_c).astype(np.float32)
            mag *= mag  # power; avoids **2 temp
            stft_magnitude = mag
            freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft).astype(np.float32)

        return stft_magnitude, freqs

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

        # Векторизованный биннинг: матрица масок (freq_bins, 3) — кэшируется по (sr, n_fft)
        mask_matrix = self._mask_matrix_for_freqs(freqs, sr)

        # Пер-кадровые энергии по полосам: (num_bands, frames)
        band_energy_ts = mask_matrix.T @ S  # матричное умножение

        # Скалярные суммы энергий (по всем кадрам)
        energies = band_energy_ts.sum(axis=1).astype(np.float32)

        return bands_to_use, energies, band_energy_ts

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
        if "band_energy_shares" not in result:
            raise ValueError("band_energy | Missing band_energy_shares in output")

        band_edges = result["band_edges"]
        band_shares = result["band_energy_shares"]

        if not isinstance(band_edges, list) or len(band_edges) != 3:
            raise ValueError(f"band_energy | Invalid band_edges: must be list of length 3, got {type(band_edges)} len={len(band_edges) if isinstance(band_edges, list) else 'N/A'}")
        if not isinstance(band_shares, list) or len(band_shares) != len(band_edges):
            raise ValueError(f"band_energy | Invalid band_energy_shares: must be list of length {len(band_edges)}, got {len(band_shares) if isinstance(band_shares, list) else 'N/A'}")

        # Validate shares sum to ~1.0
        shares_sum = sum(band_shares)
        if not (0.99 <= shares_sum <= 1.01):
            raise ValueError(f"band_energy | Invalid band_energy_shares: sum must be ~1.0, got {shares_sum}")
        # Validate shares are finite non-negative
        for i, s in enumerate(band_shares):
            if not isinstance(s, (int, float)) or (float(s) < 0) or (not np.isfinite(float(s))):
                raise ValueError(f"band_energy | Invalid band_energy_shares[{i}]: must be finite non-negative float, got {s}")

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
        t0 = time.perf_counter()
        stage_ms: Dict[str, float] = {}
        res_prof: Optional[Dict[str, Any]] = None
        if resource_profile_enabled():
            res_prof = {"at_start": snapshot_process_resources()}
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
            t_load0 = time.perf_counter()
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
            stage_ms["load_audio_ms"] = float((time.perf_counter() - t_load0) * 1000.0)

            # Check minimum duration (at least 1 second)
            duration = len(y) / sr
            if duration < 1.0:
                raise ValueError(f"band_energy | Аудио слишком короткое: {duration:.2f}s < 1.0s (audio_too_short)")

            # Normalize audio if enabled
            if self.enable_audio_normalization:
                self._report_progress("normalize_audio", 0, 1, "Normalizing audio")
                t_norm0 = time.perf_counter()
                y = self._normalize_audio(y)
                stage_ms["normalize_audio_ms"] = float((time.perf_counter() - t_norm0) * 1000.0)

            # Compute band energies
            self._report_progress("compute_bands", 0, 1, "Computing band energies")
            t_bands0 = time.perf_counter()
            bands_to_use, energies, _band_energy_ts = self._compute_band_energies_librosa(y, sr, shared_features)
            stage_ms["compute_bands_ms"] = float((time.perf_counter() - t_bands0) * 1000.0)

            # Compute shares
            t_sh0 = time.perf_counter()
            total_energy = float(np.sum(energies) + 1e-12)
            shares = (energies / total_energy).astype(np.float32)
            stage_ms["compute_shares_ms"] = float((time.perf_counter() - t_sh0) * 1000.0)

            # Compute balance metrics
            t_bal0 = time.perf_counter()
            balance_metrics = self._compute_balance_metrics(shares)
            stage_ms["balance_metrics_ms"] = float((time.perf_counter() - t_bal0) * 1000.0)

            # Build payload
            payload: Dict[str, Any] = {
                "band_edges": [(float(lo), float(hi)) for lo, hi in bands_to_use],
                "band_energy_shares": [float(s) for s in shares.tolist()],
                "sample_rate": sr,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "duration": duration,
                "device_used": self.device,
                "method": "librosa",
            }

            # Add balance metrics if enabled
            if self.enable_balance_metrics:
                payload.update(balance_metrics)

            # Track enabled features for meta
            enabled_features = []
            if self.enable_time_series:
                enabled_features.append("time_series")
            if self.enable_balance_metrics:
                enabled_features.append("balance_metrics")

            payload["band_energy_contract_version"] = BAND_ENERGY_CONTRACT_VERSION
            payload["_features_enabled"] = enabled_features
            payload["stage_timings_ms"] = {**stage_ms, "total_ms": float((time.perf_counter() - t0) * 1000.0)}
            if res_prof is not None:
                res_prof["at_end"] = snapshot_process_resources()
                payload["band_energy_resource_profile"] = {
                    **prefix_snapshot("at_start", res_prof.get("at_start", {})),
                    **prefix_snapshot("at_end", res_prof.get("at_end", {})),
                }

            # Validate output
            t_val0 = time.perf_counter()
            self._validate_output(payload)
            stage_ms["validate_output_ms"] = float((time.perf_counter() - t_val0) * 1000.0)
            payload["stage_timings_ms"]["validate_output_ms"] = stage_ms["validate_output_ms"]
            payload["stage_timings_ms"]["total_ms"] = float((time.perf_counter() - t0) * 1000.0)

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
        t0 = time.perf_counter()
        stage_ms: Dict[str, float] = {}
        res_prof: Optional[Dict[str, Any]] = None
        if resource_profile_enabled():
            res_prof = {"at_start": snapshot_process_resources()}
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

            t_segmeta0 = time.perf_counter()
            # Pre-collect segment timing for strict alignment.
            segment_centers = [float(seg.get("center_sec", 0.0)) for seg in segments]
            segment_durations = [float(seg.get("end_sec", 0.0) - seg.get("start_sec", 0.0)) for seg in segments]
            stage_ms["load_segments_ms"] = float((time.perf_counter() - t_segmeta0) * 1000.0)
            # Always compute per-segment shares for aggregation; optionally expose it as a sequence.
            band_shares_by_segment = np.full((total_segments, 3), np.nan, dtype=np.float32)
            segment_mask = np.zeros((total_segments,), dtype=bool)

            self._report_progress("load_segments", 0, total_segments, "Loading segments")

            t_proc0 = time.perf_counter()
            n_masked_short = 0
            for seg_idx, seg in enumerate(segments):
                # Progress reporting
                current_pct = (seg_idx * 100) // total_segments
                if current_pct >= last_reported_pct + 10:
                    self._report_progress("process_segments", seg_idx, total_segments, f"Processing segment {seg_idx + 1}/{total_segments}")
                    last_reported_pct = current_pct

                # Extract segment info
                start_sample = int(seg.get("start_sample", 0))
                end_sample = int(seg.get("end_sample", 0))
                duration_sec = segment_durations[seg_idx]

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
                        # Audit v3: do not drop indices. Keep alignment; mark as invalid in mask.
                        logger.debug(f"band_energy | Segment {seg_idx} too short ({duration_sec:.2f}s): masked out")
                        n_masked_short += 1
                        continue

                    # Normalize audio if enabled
                    if self.enable_audio_normalization:
                        y = self._normalize_audio(y)

                    # Compute band energies for segment (audit v3: librosa-only)
                    bands_to_use, energies, _band_energy_ts = self._compute_band_energies_librosa(y, sr, shared_features)

                    # Compute shares
                    total_energy = float(np.sum(energies) + 1e-12)
                    shares = (energies / total_energy).astype(np.float32)

                    # Store results (aligned)
                    band_shares_by_segment[seg_idx, :] = shares.astype(np.float32)
                    segment_mask[seg_idx] = True

                except Exception as e:
                    logger.warning(f"band_energy | Failed to process segment {seg_idx}: {e}")
                    continue
            stage_ms["process_segments_ms"] = float((time.perf_counter() - t_proc0) * 1000.0)

            if not bool(np.any(segment_mask)):
                raise ValueError("band_energy | No valid segments processed (all segments too short or failed)")

            # Aggregate results
            self._report_progress("aggregate", total_segments, total_segments, "Aggregating results")

            # Aggregate shares (ignore masked rows)
            t_ag0 = time.perf_counter()
            avg_shares = np.nanmean(band_shares_by_segment, axis=0).astype(np.float32)
            stage_ms["aggregate_results_ms"] = float((time.perf_counter() - t_ag0) * 1000.0)

            # Use first segment's band_edges (should be consistent)
            bands_to_use = self.bands
            if self.use_mel_bands:
                mel_edges = librosa.mel_frequencies(n_mels=self.n_mels, fmin=0.0, fmax=self.sample_rate / 2.0)
                bands_to_use = list(zip(mel_edges[:-1], mel_edges[1:]))

            # Build payload
            span_sec: Optional[float] = None
            try:
                starts = [float(seg.get("start_sec", 0.0)) for seg in segments]
                ends = [float(seg.get("end_sec", 0.0)) for seg in segments]
                if starts and ends:
                    span_sec = float(max(ends) - min(starts))
            except Exception:
                span_sec = None
            payload: Dict[str, Any] = {
                "band_edges": [(float(lo), float(hi)) for lo, hi in bands_to_use],
                "band_energy_shares": [float(s) for s in avg_shares.tolist()],
                "sample_rate": self.sample_rate,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "duration": span_sec,
                "device_used": self.device,
                "method": "librosa",
            }

            # Optional segment-aligned sequence (Audit v3): expose shares_by_segment + mask.
            if self.enable_time_series:
                payload["segment_centers_sec"] = segment_centers
                payload["segment_durations"] = segment_durations
                payload["segment_mask"] = segment_mask.astype(bool).tolist()
                payload["band_shares_by_segment"] = band_shares_by_segment.astype(np.float32).tolist()

            # Compute balance metrics
            balance_metrics = self._compute_balance_metrics(avg_shares)
            if self.enable_balance_metrics:
                payload.update(balance_metrics)

            # Track enabled features for meta
            enabled_features = []
            if self.enable_time_series:
                enabled_features.append("time_series")
            if self.enable_balance_metrics:
                enabled_features.append("balance_metrics")

            payload["band_energy_contract_version"] = BAND_ENERGY_CONTRACT_VERSION
            payload["_features_enabled"] = enabled_features
            payload["stage_timings_ms"] = {
                **stage_ms,
                "segments_count": int(total_segments),
                "segments_valid": int(np.sum(segment_mask)),
                "segments_masked_short": int(n_masked_short),
                "total_ms": float((time.perf_counter() - t0) * 1000.0),
            }
            if res_prof is not None:
                res_prof["at_end"] = snapshot_process_resources()
                payload["band_energy_resource_profile"] = {
                    **prefix_snapshot("at_start", res_prof.get("at_start", {})),
                    **prefix_snapshot("at_end", res_prof.get("at_end", {})),
                }

            # Validate output
            t_val0 = time.perf_counter()
            self._validate_output(payload)
            stage_ms["validate_output_ms"] = float((time.perf_counter() - t_val0) * 1000.0)
            payload["stage_timings_ms"]["validate_output_ms"] = stage_ms["validate_output_ms"]
            payload["stage_timings_ms"]["total_ms"] = float((time.perf_counter() - t0) * 1000.0)

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
