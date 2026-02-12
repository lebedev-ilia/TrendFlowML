"""
Source separation extractor (inprocess PyTorch model) + Segmenter windows.

We intentionally do NOT output stems (too large). We output:
- per-window energy shares for [vocals, drums, bass, other]
- aggregated mean shares (and basic dispersion)

Policy:
- NO fallback (model missing => ERROR)
- NO runtime downloads (ModelManager enforced)
- uses Segmenter `audio/segments.json` family: `source_separation`
- `<5s` audio => ERROR
- truly silent audio => EMPTY (status="empty", empty_reason="audio_silent")
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Tuple

import numpy as np
import torch

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)

# Contract version for downstream extractors compatibility validation
SOURCE_SEPARATION_CONTRACT_VERSION = "source_separation_contract_v1"


class SourceSeparationExtractor(BaseExtractor):
    name = "source_separation_extractor"
    version = "3.0.0"
    description = "Source separation shares via inprocess PyTorch model (log-mel input)"
    category = "source_separation"
    dependencies = ["numpy", "dp_models", "torchaudio", "torch"]
    estimated_duration = 12.0

    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 2.0  # PyTorch model requires GPU memory

    def __init__(
        self,
        device: str = "auto",
        model_size: str = "large",
        batch_size: int = 8,
        # Feature gating flags (per-feature control, default: all False)
        enable_share_sequence: bool = False,
        enable_energy_sequence: bool = False,
        enable_share_mean: bool = False,
        enable_share_std: bool = False,
        enable_quality_metrics: bool = False,
        # Silence detection
        silence_peak_threshold: float = 1e-3,
        silence_rms_threshold: float = 1e-4,
        enable_silence_detection: bool = True,
        # Progress reporting callback
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        """
        Инициализация экстрактора разделения источников.
        
        Args:
            device: Устройство для обработки
            model_size: large (inprocess model selection via ModelManager)
            batch_size: Размер батча для обработки окон
            enable_share_sequence: Включить share_sequence (per-segment shares)
            enable_energy_sequence: Включить energy_sequence (per-segment energies)
            enable_share_mean: Включить share_mean (mean shares)
            enable_share_std: Включить share_std (std shares)
            enable_quality_metrics: Включить quality_metrics (метрики качества)
            silence_peak_threshold: Порог peak для детекции тишины
            silence_rms_threshold: Порог RMS для детекции тишины
            enable_silence_detection: Включить проверку на тишину
            progress_callback: Callback для прогресса (batch_index, total_batches, message)
        """
        super().__init__(device=device)
        self.model_size = str(model_size or "large").strip().lower()
        if self.model_size not in ("large",):
            raise ValueError(f"source_separation | unsupported model_size={self.model_size}. Expected: large")
        self.batch_size = max(1, int(batch_size))
        
        # Feature gating flags
        self.enable_share_sequence = bool(enable_share_sequence)
        self.enable_energy_sequence = bool(enable_energy_sequence)
        self.enable_share_mean = bool(enable_share_mean)
        self.enable_share_std = bool(enable_share_std)
        self.enable_quality_metrics = bool(enable_quality_metrics)
        
        # Silence detection
        self.silence_peak_threshold = float(silence_peak_threshold)
        self.silence_rms_threshold = float(silence_rms_threshold)
        self.enable_silence_detection = bool(enable_silence_detection)
        
        # Progress callback
        self.progress_callback = progress_callback

        # ModelManager: resolve in-process model (no-network).
        try:
            from dp_models import get_global_model_manager  # type: ignore

            self._mm = get_global_model_manager()
        except Exception as e:
            raise RuntimeError(f"source_separation | ModelManager is required but failed to init: {e}") from e

        spec_name = f"source_separation_{self.model_size}_inprocess"
        try:
            self.model_spec = self._mm.get_spec(model_name=spec_name)
            _dev, _prec, rt, _eng, wd, _arts = self._mm.resolve(self.model_spec)
            if str(rt) != "inprocess":
                raise RuntimeError(f"source_separation | expected runtime=inprocess in spec {spec_name}, got {rt}")
            self.model_name = str(self.model_spec.model_name)
            self.weights_digest = str(wd)
            
            # Load model via ModelManager
            resolved_model = self._mm.get(model_name=spec_name)
            self.model = resolved_model.handle
            self.models_used_entry = resolved_model.models_used_entry
            
            # Get preprocessing params from runtime_params
            rp = self.model_spec.runtime_params or {}
            self.sample_rate = int(rp.get("sample_rate") or 44100)
            self.n_fft = int(rp.get("n_fft") or 2048)
            self.hop_length = int(rp.get("hop_length") or 512)
            self.n_mels = int(rp.get("n_mels") or 64)
            
            # Get source order from runtime_params
            source_order_raw = rp.get("source_order")
            if isinstance(source_order_raw, list) and source_order_raw:
                self._source_names = [str(x) for x in source_order_raw]
            else:
                self._source_names = ["vocals", "drums", "bass", "other"]
            
            # Model is already loaded via TorchStateDictProvider
            # Move model to appropriate device (BaseExtractor sets self.device)
            if hasattr(self.model, 'to'):
                try:
                    # Import torch here to ensure it's available
                    import torch as torch_module
                    # Use the device determined by BaseExtractor
                    target_device = self.device
                    if str(target_device).lower().startswith("cuda") and not torch_module.cuda.is_available():
                        target_device = "cpu"
                        self.device = "cpu"
                    
                    self.model = self.model.to(target_device)
                except Exception as e:
                    logger.warning(f"source_separation | failed to move model to device {self.device}: {e}, using CPU")
                    self.model = self.model.to("cpu") if hasattr(self.model, 'to') else self.model
                    self.device = "cpu"
            
            # Set model to eval mode
            if hasattr(self.model, 'eval'):
                self.model.eval()
        except Exception as e:
            raise RuntimeError(f"source_separation | failed to resolve/load model via ModelManager: {e}") from e

        # Validate preprocessing parameters (informative)
        self._validate_preprocessing_params()
        
        # Validate source_order
        self._validate_source_order()

        self.audio_utils = AudioUtils(device=device, sample_rate=self.sample_rate)

        # Build mel transform lazily (import torch/torchaudio here).
        try:
            import torch
            import torchaudio

            self._torch = torch
            self._mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                power=2.0,
            )
            self._amptodb = torchaudio.transforms.AmplitudeToDB(stype="power")
        except Exception as e:
            raise RuntimeError(f"source_separation | torchaudio/torch is required for mel preprocessing: {e}") from e


    def _validate_preprocessing_params(self) -> None:
        """
        Информативная валидация параметров предобработки (логирование предупреждений, не ошибок).
        """
        warnings = []
        
        # Validate sample_rate
        if self.sample_rate < 8000 or self.sample_rate > 48000:
            warnings.append(f"source_separation | sample_rate={self.sample_rate} is outside typical range [8000, 48000]")
        
        # Validate n_fft
        if self.n_fft < 512 or self.n_fft > 4096:
            warnings.append(f"source_separation | n_fft={self.n_fft} is outside typical range [512, 4096]")
        
        # Validate hop_length
        if self.hop_length < 128 or self.hop_length > 2048:
            warnings.append(f"source_separation | hop_length={self.hop_length} is outside typical range [128, 2048]")
        
        # Validate n_mels
        if self.n_mels < 32 or self.n_mels > 128:
            warnings.append(f"source_separation | n_mels={self.n_mels} is outside typical range [32, 128]")
        
        # Validate hop_length <= n_fft
        if self.hop_length > self.n_fft:
            warnings.append(f"source_separation | hop_length={self.hop_length} > n_fft={self.n_fft} (may cause issues)")
        
        # Log warnings if any
        for warning in warnings:
            logger.warning(warning)

    def _validate_source_order(self) -> None:
        """
        Полная валидация source_order: проверка длины, отсутствие дубликатов, валидность типов.
        """
        if not isinstance(self._source_names, list):
            raise ValueError(f"source_separation | source_order must be a list, got {type(self._source_names)}")
        
        if len(self._source_names) != 4:
            raise ValueError(f"source_separation | source_order length ({len(self._source_names)}) != 4 (expected)")
        
        # Check for duplicates
        if len(self._source_names) != len(set(self._source_names)):
            raise ValueError("source_separation | source_order contain duplicates")
        
        # Check types (all should be strings)
        for i, name in enumerate(self._source_names):
            if not isinstance(name, str):
                raise ValueError(f"source_separation | source_order[{i}] must be str, got {type(name)}")


    def _validate_shares_and_energies(
        self, 
        shares: np.ndarray, 
        energies: np.ndarray, 
        num_segments: int
    ) -> tuple[bool, Optional[str]]:
        """
        Полная валидация shares и energies: проверка NaN/inf, диапазонов [0,1] для shares, 
        неотрицательности для energies, нормализации shares (сумма ≈ 1.0), согласованность размеров.
        
        Args:
            shares: массив долей (float32, shape [N, 4])
            energies: массив энергий (float32, shape [N, 4])
            num_segments: ожидаемое количество сегментов
        
        Returns:
            (is_valid, error_message)
        """
        # Validate shares
        if shares.size == 0:
            return True, None  # Empty is valid
        
        # Check dtype
        if shares.dtype != np.float32:
            return False, f"source_separation | shares dtype must be float32, got {shares.dtype}"
        
        # Check shape
        if shares.ndim != 2:
            return False, f"source_separation | shares must be 2D [N, 4], got shape {shares.shape}"
        
        if shares.shape[0] != num_segments:
            return False, f"source_separation | shares shape[0] ({shares.shape[0]}) != num_segments ({num_segments})"
        
        if shares.shape[1] != 4:
            return False, f"source_separation | shares shape[1] ({shares.shape[1]}) != 4 (expected)"
        
        # Check for NaN/inf
        if np.any(np.isnan(shares)):
            return False, "source_separation | shares contain NaN values"
        if np.any(np.isinf(shares)):
            return False, "source_separation | shares contain inf values"
        
        # Check ranges [0, 1]
        if np.any(shares < 0.0) or np.any(shares > 1.0):
            return False, f"source_separation | shares out of range [0, 1]: min={np.min(shares)}, max={np.max(shares)}"
        
        # Check normalization (sum should be close to 1.0 per row)
        row_sums = np.sum(shares, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            return False, f"source_separation | shares not normalized (sum per row should be 1.0): min_sum={np.min(row_sums)}, max_sum={np.max(row_sums)}"
        
        # Validate energies
        if energies.size == 0:
            return True, None  # Empty is valid
        
        # Check dtype
        if energies.dtype != np.float32:
            return False, f"source_separation | energies dtype must be float32, got {energies.dtype}"
        
        # Check shape
        if energies.ndim != 2:
            return False, f"source_separation | energies must be 2D [N, 4], got shape {energies.shape}"
        
        if energies.shape[0] != num_segments:
            return False, f"source_separation | energies shape[0] ({energies.shape[0]}) != num_segments ({num_segments})"
        
        if energies.shape[1] != 4:
            return False, f"source_separation | energies shape[1] ({energies.shape[1]}) != 4 (expected)"
        
        # Check for NaN/inf
        if np.any(np.isnan(energies)):
            return False, "source_separation | energies contain NaN values"
        if np.any(np.isinf(energies)):
            return False, "source_separation | energies contain inf values"
        
        # Check non-negativity
        if np.any(energies < 0.0):
            return False, f"source_separation | energies contain negative values: min={np.min(energies)}"
        
        return True, None

    @staticmethod
    def _rms_and_peak(x: np.ndarray) -> tuple[float, float]:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return 0.0, 0.0
        rms = float(np.sqrt(float(np.mean(x * x)) + 1e-12))
        peak = float(np.max(np.abs(x)) + 1e-12)
        return rms, peak

    def _mel_log(self, wav_1d: np.ndarray) -> np.ndarray:
        """
        wav_1d: float32 [-1..1], shape [T]
        returns log-mel [n_mels, frames] float32
        """
        t = self._torch.from_numpy(np.asarray(wav_1d, dtype=np.float32).reshape(1, -1))  # [1, T]
        # Always on CPU for stable preprocessing.
        t = t.cpu()
        with self._torch.no_grad():
            mel = self._mel(t)  # [1, n_mels, frames]
            mel_db = self._amptodb(mel)  # [1, n_mels, frames]
        out = mel_db.squeeze(0).contiguous().numpy().astype(np.float32)
        return out

    def _compute_advanced_features(self, shares: np.ndarray) -> Dict[str, Any]:
        """
        Вычисляет расширенные фичи на основе временных рядов shares.
        
        Args:
            shares: массив долей источников [T, 4] float32, где T - количество сегментов
        
        Returns:
            Словарь с расширенными фичами:
            - Transition features (delta): mean_delta_*, max_delta_*, std_delta_* для каждого источника
            - Stability features: *_stability для каждого источника
            - Distribution features: *_dominance_ratio, *_mean_share для каждого источника
            - Energy balance: source_entropy_mean, source_entropy_std, energy_balance_mean
            - Musical heuristics: vocals_presence_ratio, drums_flux, bass_floor_p20
        """
        features = {}
        
        if shares.shape[0] < 2:
            # Недостаточно данных для временных фич
            return features
        
        T, num_sources = shares.shape
        source_names = self._source_names  # ["vocals", "drums", "bass", "other"]
        eps = 1e-8
        
        # 1. Transition features (delta): |P_t - P_{t-1}|
        dP = np.abs(np.diff(shares, axis=0))  # [T-1, 4]
        
        for i, name in enumerate(source_names):
            dP_i = dP[:, i]
            features[f"{name}_delta_mean"] = float(dP_i.mean())
            # std() может вернуть NaN для одного значения, используем ddof=0 для консистентности
            features[f"{name}_delta_std"] = float(dP_i.std(ddof=0) if len(dP_i) > 0 else 0.0)
            features[f"{name}_delta_max"] = float(dP_i.max() if len(dP_i) > 0 else 0.0)
        
        # 2. Stability features: 1 - std(P_i)
        for i, name in enumerate(source_names):
            stability = 1.0 - shares[:, i].std()
            features[f"{name}_stability"] = float(max(0.0, min(1.0, stability)))  # Clamp to [0, 1]
        
        # 3. Distribution features
        # 3.1 Mean share (уже есть в share_mean, но добавляем для полноты)
        for i, name in enumerate(source_names):
            features[f"{name}_mean_share"] = float(shares[:, i].mean())
        
        # 3.2 Dominance ratio: mean(P_i == max(P))
        dominant = np.argmax(shares, axis=1)  # [T]
        for i, name in enumerate(source_names):
            dominance_ratio = float((dominant == i).mean())
            features[f"{name}_dominance_ratio"] = dominance_ratio
        
        # 4. Energy balance / entropy
        # 4.1 Source entropy per segment: -sum(P * log(P))
        entropy_per_segment = -np.sum(shares * np.log(shares + eps), axis=1)  # [T]
        features["source_entropy_mean"] = float(entropy_per_segment.mean())
        features["source_entropy_std"] = float(entropy_per_segment.std())
        
        # 4.2 Energy balance: 1 - std(P over sources)
        balance_per_segment = 1.0 - shares.std(axis=1)  # [T]
        features["energy_balance_mean"] = float(balance_per_segment.mean())
        
        # 5. Musical heuristics
        # 5.1 Vocals presence ratio: mean(P_vocals > τ), τ = 0.35
        vocals_idx = source_names.index("vocals") if "vocals" in source_names else 0
        vocals_threshold = 0.35
        features["vocals_presence_ratio"] = float((shares[:, vocals_idx] > vocals_threshold).mean())
        
        # 5.2 Drums flux: mean(|P_drums(t) - P_drums(t-1)|)
        drums_idx = source_names.index("drums") if "drums" in source_names else 1
        drums_flux = np.abs(np.diff(shares[:, drums_idx])).mean()
        features["drums_flux"] = float(drums_flux)
        
        # 5.3 Bass floor: percentile(P_bass, 20)
        bass_idx = source_names.index("bass") if "bass" in source_names else 2
        bass_floor = np.percentile(shares[:, bass_idx], 20)
        features["bass_floor_p20"] = float(bass_floor)
        
        return features

    def _infer_energies_batch(self, batch: np.ndarray) -> np.ndarray:
        """
        Выполнить batch inference для получения энергий источников через inprocess PyTorch модель.
        
        Args:
            batch: паддингнутые mel features, shape [B, n_mels, T] float32
        
        Returns:
            энергии источников, shape [B, 4] float32
        
        Raises:
            RuntimeError при ошибках inference
        """
        try:
            # Convert numpy array to torch tensor
            batch_tensor = torch.from_numpy(batch).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                
                # Handle different output formats
                if isinstance(outputs, torch.Tensor):
                    energies_tensor = outputs
                elif isinstance(outputs, (list, tuple)):
                    # If model returns multiple outputs, take the first one (energies)
                    energies_tensor = outputs[0]
                elif isinstance(outputs, dict):
                    # If model returns dict, try common keys
                    energies_tensor = outputs.get("energies") or outputs.get("energy") or outputs.get("output")
                    if energies_tensor is None:
                        # Take first value from dict
                        energies_tensor = list(outputs.values())[0]
                else:
                    raise RuntimeError(f"source_separation | unexpected model output type: {type(outputs)}")
                
                # Convert to numpy
                energies = energies_tensor.cpu().numpy().astype(np.float32)
            
            # Validate output shape
            if energies.ndim != 2 or energies.shape[1] != 4:
                raise RuntimeError(f"source_separation | unexpected energy output shape: {energies.shape} (expected [{batch.shape[0]}, 4])")
            
            # Ensure non-negative energies and handle NaN/inf
            energies = np.nan_to_num(energies, nan=0.0, posinf=0.0, neginf=0.0)
            energies = np.maximum(energies, 0.0)
            
            return energies
        except Exception as e:
            raise RuntimeError(f"source_separation | inference failed: {e}") from e

    def run_segments(
        self, 
        input_uri: str, 
        tmp_path: str, 
        segments: List[Dict[str, Any]]
    ) -> ExtractorResult:
        """
        Segmenter-driven source separation: compute energy shares on provided windows.
        
        Progress reporting: каждые 10% батчей (если progress_callback установлен).
        """
        start_time = time.time()
        timings = {}  # Детальное профилирование этапов
        
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)
            if not isinstance(segments, list) or not segments:
                raise ValueError("segments is empty (no-fallback)")

            dur_sec = float(max((float(s.get("end_sec", 0.0)) for s in segments), default=0.0))
            if dur_sec < 5.0:
                raise RuntimeError(f"source_separation | audio too short (<5s): duration_sec={dur_sec:.3f}")

            # Этап 1: Загрузка аудио и вычисление mel features
            t_load_start = time.time()
            # Load audio windows and compute mel.
            mels: list[np.ndarray] = []
            starts: list[float] = []
            ends: list[float] = []
            centers: list[float] = []
            peaks: list[float] = []
            rmss: list[float] = []

            for seg in segments:
                ss = int(seg.get("start_sample"))
                es = int(seg.get("end_sample"))
                st = float(seg.get("start_sec"))
                en = float(seg.get("end_sec"))
                c = float(seg.get("center_sec"))
                wav_t, sr = self.audio_utils.load_audio_segment(input_uri, start_sample=ss, end_sample=es, target_sr=self.sample_rate)
                wav = self.audio_utils.to_numpy(wav_t)
                wav = wav[0] if wav.ndim == 2 else wav.reshape(-1)
                wav = np.asarray(wav, dtype=np.float32).reshape(-1)
                if int(sr) != int(self.sample_rate):
                    raise RuntimeError(f"source_separation | segment SR mismatch: got {sr} expected {self.sample_rate}")

                rms, peak = self._rms_and_peak(wav)
                rmss.append(float(rms))
                peaks.append(float(peak))
                mels.append(self._mel_log(wav))
                starts.append(st)
                ends.append(en)
                centers.append(c)
            
            t_load_end = time.time()
            timings["load_audio_sec"] = t_load_end - t_load_start
            logger.info(f"source_separation | loaded {len(segments)} segments in {timings['load_audio_sec']:.3f}s")
            
            # Progress reporting: загрузка завершена
            if self.progress_callback:
                self.progress_callback(0, 1, f"Loaded {len(segments)} segments ({timings['load_audio_sec']:.1f}s)")

            # Этап 2: Silence detection (if enabled)
            t_silence_start = time.time()
            if self.enable_silence_detection:
                # Global silence decision: only if ALL windows are silent.
                if (max(peaks) if peaks else 0.0) < self.silence_peak_threshold and (max(rmss) if rmss else 0.0) < self.silence_rms_threshold:
                    payload: Dict[str, Any] = {
                        "status": "empty",
                        "empty_reason": "audio_silent",
                        "segments_count": int(len(segments)),
                        "sample_rate": int(self.sample_rate),
                        "model_name": self.model_name,
                        "device_used": self.device,
                        "source_separation_contract_version": SOURCE_SEPARATION_CONTRACT_VERSION,
                    }
                    timings["silence_detection_sec"] = time.time() - t_silence_start
                    return self._create_result(True, payload=payload, processing_time=time.time() - start_time)
            
            t_silence_end = time.time()
            timings["silence_detection_sec"] = t_silence_end - t_silence_start
            
            # Progress reporting: silence detection завершена
            if self.progress_callback:
                self.progress_callback(0, 1, f"Silence detection completed ({timings['silence_detection_sec']:.1f}s)")

            # Этап 3: Padding для батчинга
            t_pad_start = time.time()
            # Pad mel time dimension for batching
            t_max = int(max(m.shape[1] for m in mels)) if mels else 0
            if t_max <= 0:
                raise RuntimeError("source_separation | empty mel features")
            batch_in = np.zeros((len(mels), self.n_mels, t_max), dtype=np.float32)
            for i, m in enumerate(mels):
                batch_in[i, :, : m.shape[1]] = m
            
            t_pad_end = time.time()
            timings["padding_sec"] = t_pad_end - t_pad_start

            # Этап 4: Batch inference
            t_inference_start = time.time()
            # Determine batch size (auto-split if >100 segments)
            effective_batch_size = self.batch_size
            if len(mels) > 100:
                effective_batch_size = min(100, self.batch_size)  # Auto-split large batches

            # Process in batches
            energies = []
            total_batches = (batch_in.shape[0] + effective_batch_size - 1) // effective_batch_size
            progress_report_interval = max(1, total_batches // 10) if total_batches >= 10 else 1
            last_reported_pct = -1
            
            # Progress reporting: начало inference
            if self.progress_callback:
                self.progress_callback(0, total_batches, f"Starting inference: {total_batches} batches")

            for batch_idx, start in enumerate(range(0, batch_in.shape[0], effective_batch_size)):
                b = batch_in[start : start + effective_batch_size]
                batch_energies = self._infer_energies_batch(b)
                energies.append(batch_energies)
                
                # Progress reporting
                if self.progress_callback and batch_idx % progress_report_interval == 0:
                    pct = int((batch_idx + 1) * 100 / total_batches)
                    if pct != last_reported_pct:
                        batch_elapsed = time.time() - t_inference_start
                        self.progress_callback(batch_idx + 1, total_batches, f"Inference: {batch_idx + 1}/{total_batches} batches ({pct}%, {batch_elapsed:.1f}s)")
                        last_reported_pct = pct
            
            t_inference_end = time.time()
            timings["inference_sec"] = t_inference_end - t_inference_start
            logger.info(f"source_separation | inference completed: {timings['inference_sec']:.3f}s for {total_batches} batches")

            energy = np.concatenate(energies, axis=0) if energies else np.zeros((0, 4), dtype=np.float32)
            if energy.shape[0] != len(segments):
                raise RuntimeError(f"source_separation | energy count mismatch: {energy.shape[0]} vs {len(segments)}")

            # Этап 5: Нормализация энергий
            t_postprocess_start = time.time()
            # Ensure energies are non-negative and finite (handle NaN/inf)
            energy = np.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)
            energy = np.maximum(energy, 0.0)  # Ensure non-negative
            
            # Compute shares with safe normalization
            total = np.sum(energy, axis=1, keepdims=True)
            # Handle zero total case (all energies are zero) - set equal shares
            zero_mask = (total.flatten() < 1e-9)
            total = np.where(zero_mask.reshape(-1, 1), 1.0, total + 1e-9)
            shares = energy / total  # [N,4]
            
            # For zero-energy segments, set equal shares (1/4 each)
            if np.any(zero_mask):
                shares[zero_mask] = 0.25  # Equal shares for silent segments
            
            # Ensure shares are finite and in valid range [0, 1]
            shares = np.nan_to_num(shares, nan=0.25, posinf=1.0, neginf=0.0)
            shares = np.clip(shares, 0.0, 1.0)
            
            # Renormalize to ensure sum = 1.0 (defensive)
            row_sums = np.sum(shares, axis=1, keepdims=True)
            shares = shares / (row_sums + 1e-9)
            
            # Progress reporting: postprocessing завершен
            if self.progress_callback:
                self.progress_callback(1, 1, f"Postprocessing completed ({time.time() - t_postprocess_start:.1f}s)")

            # Validate shares and energies
            is_valid, error_msg = self._validate_shares_and_energies(shares, energy, len(segments))
            if not is_valid:
                raise ValueError(f"source_separation | validation failed: {error_msg}")

            # Этап 6: Вычисление агрегатов
            t_aggregates_start = time.time()
            # Compute basic aggregates (always needed for downstream)
            share_mean = np.mean(shares, axis=0).astype(np.float32) if shares.size else np.zeros((4,), dtype=np.float32)
            share_std = np.std(shares, axis=0).astype(np.float32) if shares.size else np.zeros((4,), dtype=np.float32)

            payload: Dict[str, Any] = {
                "segments_count": int(len(segments)),
                "sample_rate": int(self.sample_rate),
                "device_used": self.device,
                "model_name": self.model_name,
                "source_order": self._source_names,
                "source_separation_contract_version": SOURCE_SEPARATION_CONTRACT_VERSION,
            }
            
            # Feature gating: share_sequence
            if self.enable_share_sequence:
                payload["share_sequence"] = shares.astype(np.float32)
            
            # Feature gating: energy_sequence
            if self.enable_energy_sequence:
                payload["energy_sequence"] = energy.astype(np.float32)
            
            # Feature gating: share_mean
            if self.enable_share_mean:
                payload["share_mean"] = share_mean
            
            # Feature gating: share_std
            if self.enable_share_std:
                payload["share_std"] = share_std
            
            # Always include segment timestamps (needed for downstream)
            payload["segment_start_sec"] = starts
            payload["segment_end_sec"] = ends
            payload["segment_center_sec"] = centers
            
            # Additional aggregates (if any feature is enabled)
            if self.enable_share_mean or self.enable_share_sequence:
                # Dominant source
                dominant_source_id = int(np.argmax(share_mean)) if share_mean.size else -1
                dominant_source_share = float(np.max(share_mean)) if share_mean.size else 0.0
                payload["dominant_source_id"] = dominant_source_id
                payload["dominant_source_share"] = dominant_source_share
                
                # Source balance score (entropy-based, normalized)
                if share_mean.size > 1:
                    # Entropy of shares (higher = more balanced)
                    entropy = float(-np.sum(share_mean * np.log(share_mean + 1e-9)))
                    max_entropy = float(np.log(4.0))  # log(num_sources)
                    balance_score = float(entropy / max_entropy) if max_entropy > 0 else 0.0
                    payload["source_balance_score"] = balance_score
                
                # Source transitions and distribution (if share_sequence is enabled)
                if self.enable_share_sequence:
                    # Dominant source per segment
                    dominant_sources = np.argmax(shares, axis=1).astype(np.int32)
                    
                    # Transitions count
                    if len(dominant_sources) > 1:
                        transitions = sum(1 for i in range(len(dominant_sources) - 1) if dominant_sources[i] != dominant_sources[i + 1])
                        payload["source_transitions_count"] = int(transitions)
                    
                    # Source distribution (time ratios)
                    source_duration = {}
                    source_segments_count = {}
                    total_duration = float(max(ends) if ends else 0.0)
                    for i, src_id in enumerate(dominant_sources):
                        src_id_int = int(src_id)
                        if src_id_int not in source_duration:
                            source_duration[src_id_int] = 0.0
                            source_segments_count[src_id_int] = 0
                        seg_duration = float(ends[i] - starts[i])
                        source_duration[src_id_int] += seg_duration
                        source_segments_count[src_id_int] += 1
                    
                    source_distribution = {}
                    if total_duration > 0:
                        for src_id, duration in source_duration.items():
                            source_distribution[int(src_id)] = float(duration / total_duration)
                    payload["source_distribution"] = source_distribution
                    payload["source_segments_per_source"] = {int(k): int(v) for k, v in source_segments_count.items()}
                    payload["source_duration_per_source"] = {int(k): float(v) for k, v in source_duration.items()}
                    
                    # Source stability score (inverse of transitions frequency)
                    if total_duration > 0:
                        transitions_freq = float(transitions) / total_duration if transitions > 0 else 0.0
                        stability_score = float(1.0 / (1.0 + transitions_freq))  # 0 = unstable, 1 = stable
                        payload["source_stability_score"] = stability_score
                    
                    # Advanced features (temporal, stability, distribution, musical heuristics)
                    advanced_features = self._compute_advanced_features(shares)
                    payload.update(advanced_features)
            
            # Feature gating: quality metrics
            if self.enable_quality_metrics:
                quality_metrics = {}
                if self.enable_share_mean:
                    quality_metrics["share_mean_min"] = float(np.min(share_mean))
                    quality_metrics["share_mean_max"] = float(np.max(share_mean))
                    quality_metrics["share_mean_std"] = float(np.std(share_mean))
                if self.enable_share_std:
                    quality_metrics["share_std_mean"] = float(np.mean(share_std))
                    quality_metrics["share_std_max"] = float(np.max(share_std))
                if self.enable_share_sequence:
                    quality_metrics["share_sequence_min"] = float(np.min(shares))
                    quality_metrics["share_sequence_max"] = float(np.max(shares))
                    quality_metrics["share_sequence_mean"] = float(np.mean(shares))
                if self.enable_energy_sequence:
                    quality_metrics["energy_sequence_min"] = float(np.min(energy))
                    quality_metrics["energy_sequence_max"] = float(np.max(energy))
                    quality_metrics["energy_sequence_mean"] = float(np.mean(energy))
                payload["source_quality_metrics"] = quality_metrics
            
            # Track enabled features for meta
            enabled_features = []
            if self.enable_share_sequence:
                enabled_features.append("share_sequence")
            if self.enable_energy_sequence:
                enabled_features.append("energy_sequence")
            if self.enable_share_mean:
                enabled_features.append("share_mean")
            if self.enable_share_std:
                enabled_features.append("share_std")
            if self.enable_quality_metrics:
                enabled_features.append("quality_metrics")
            
            payload["_features_enabled"] = enabled_features
            
            t_aggregates_end = time.time()
            timings["aggregates_sec"] = t_aggregates_end - t_aggregates_start
            t_postprocess_end = time.time()
            timings["postprocess_sec"] = t_postprocess_end - t_postprocess_start
            total_time = time.time() - start_time
            
            # Log detailed profiling
            logger.info(f"source_separation | run_segments completed: segments={len(segments)}, enabled_features={enabled_features}")
            logger.info(f"source_separation | profiling: load={timings.get('load_audio_sec', 0):.3f}s, silence={timings.get('silence_detection_sec', 0):.3f}s, pad={timings.get('padding_sec', 0):.3f}s, inference={timings.get('inference_sec', 0):.3f}s, aggregates={timings.get('aggregates_sec', 0):.3f}s, postprocess={timings.get('postprocess_sec', 0):.3f}s, total={total_time:.3f}s")

            return self._create_result(True, payload=payload, processing_time=total_time)

        except Exception as e:
            return self._create_result(False, error=str(e), processing_time=time.time() - start_time)

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        return self._create_result(
            success=False,
            error="source_separation_extractor | run() is not supported in production. Use run_segments() with Segmenter families.source_separation windows.",
            processing_time=0.0,
        )

    def _validate_input(self, input_uri: str) -> bool:
        if not super()._validate_input(input_uri):
            return False
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".mp4", ".avi", ".mov"}
        if not any(input_uri.lower().endswith(ext) for ext in audio_extensions):
            self.logger.error(f"Файл не является поддерживаемым аудио/видео форматом: {input_uri}")
            return False
        return True

    @property
    def supports_batch(self) -> bool:
        """Source separation extractor поддерживает batch processing для сегментов через inprocess PyTorch модель."""
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
        - Вычисляет mel features для каждого сегмента
        - Группирует в батчи по max_segments_per_batch (если задан) или использует batch_size
        - Обрабатывает батчи через inprocess PyTorch модель для получения энергий источников
        - Распределяет результаты обратно по видео
        
        Args:
            audio_files_with_segments: Список словарей с ключами:
                - 'input_uri': URI к входному аудио/видео файлу
                - 'tmp_path': Путь к временной директории для обработки
                - 'segments': Список сегментов для обработки
                - 'file_id': Идентификатор файла (для распределения результатов)
            max_workers: Не используется для GPU extractors (оставлено для совместимости)
            max_segments_per_batch: Максимальное количество сегментов в одном батче (None = использует batch_size)
        
        Returns:
            Список ExtractorResult для каждого файла
        """
        start_time = time.time()
        
        if not audio_files_with_segments:
            return []
        
        try:
            # Этап 1: Сбор всех сегментов с привязкой к файлам
            all_segments_with_metadata: List[Dict[str, Any]] = []
            file_segment_ranges: Dict[str, Tuple[int, int]] = {}
            
            for file_info in audio_files_with_segments:
                file_id = file_info.get("file_id", "unknown")
                segments = file_info.get("segments", [])
                input_uri = file_info.get("input_uri")
                tmp_path = file_info.get("tmp_path")
                
                if not input_uri or not tmp_path or not segments:
                    continue
                
                # Проверка длительности для каждого файла
                dur_sec = float(max((float(s.get("end_sec", 0.0)) for s in segments), default=0.0))
                if dur_sec < 5.0:
                    self.logger.warning(f"source_separation | file_id={file_id} audio too short (<5s): duration_sec={dur_sec:.3f}")
                    continue
                
                start_idx = len(all_segments_with_metadata)
                for seg in segments:
                    all_segments_with_metadata.append({
                        "segment": seg,
                        "file_id": file_id,
                        "input_uri": input_uri,
                        "tmp_path": tmp_path,
                    })
                end_idx = len(all_segments_with_metadata)
                file_segment_ranges[file_id] = (start_idx, end_idx)
            
            if not all_segments_with_metadata:
                return [
                    self._create_result(
                        success=False,
                        error="No segments provided or all files too short",
                        processing_time=time.time() - start_time,
                    )
                    for _ in audio_files_with_segments
                ]
            
            # Этап 2: Загрузка аудио, вычисление mel features
            mels: List[Dict[str, Any]] = []
            starts: List[float] = []
            ends: List[float] = []
            centers: List[float] = []
            peaks: List[float] = []
            rmss: List[float] = []
            
            for seg_meta in all_segments_with_metadata:
                seg = seg_meta["segment"]
                input_uri = seg_meta["input_uri"]
                
                try:
                    ss = int(seg.get("start_sample"))
                    es = int(seg.get("end_sample"))
                    st = float(seg.get("start_sec", 0.0))
                    en = float(seg.get("end_sec", 0.0))
                    c = float(seg.get("center_sec", 0.0))
                    
                    wav_t, sr = self.audio_utils.load_audio_segment(
                        input_uri, start_sample=ss, end_sample=es, target_sr=self.sample_rate
                    )
                    wav = self.audio_utils.to_numpy(wav_t)
                    wav = wav[0] if wav.ndim == 2 else wav.reshape(-1)
                    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
                    
                    if int(sr) != int(self.sample_rate):
                        raise RuntimeError(f"source_separation | segment SR mismatch: got {sr} expected {self.sample_rate}")
                    
                    rms, peak = self._rms_and_peak(wav)
                    mel = self._mel_log(wav)
                    
                    mels.append({
                        "mel": mel,
                        "file_id": seg_meta["file_id"],
                    })
                    starts.append(st)
                    ends.append(en)
                    centers.append(c)
                    peaks.append(float(peak))
                    rmss.append(float(rms))
                except Exception as e:
                    self.logger.error(f"Error preprocessing segment for file_id={seg_meta['file_id']}: {e}")
                    continue
            
            if not mels:
                return [
                    self._create_result(
                        success=False,
                        error="Failed to preprocess any segments",
                        processing_time=time.time() - start_time,
                    )
                    for _ in audio_files_with_segments
                ]
            
            # Этап 3: Паддинг mel features для батчинга
            t_max = int(max(m["mel"].shape[1] for m in mels)) if mels else 0
            if t_max <= 0:
                return [
                    self._create_result(
                        success=False,
                        error="Empty mel features",
                        processing_time=time.time() - start_time,
                    )
                    for _ in audio_files_with_segments
                ]
            
            batch_in = np.zeros((len(mels), self.n_mels, t_max), dtype=np.float32)
            for i, m in enumerate(mels):
                batch_in[i, :, : m["mel"].shape[1]] = m["mel"]
            
            # Этап 4: Определение размера батча
            effective_batch_size = max_segments_per_batch
            if effective_batch_size is None:
                effective_batch_size = self.batch_size
                if len(mels) > 100:
                    effective_batch_size = min(100, self.batch_size)  # Auto-split large batches
            
            # Этап 5: Обработка батчей через inprocess PyTorch модель
            energies_chunks: List[np.ndarray] = []
            
            for batch_start in range(0, batch_in.shape[0], effective_batch_size):
                batch_end = min(batch_start + effective_batch_size, batch_in.shape[0])
                b = batch_in[batch_start:batch_end]
                
                try:
                    batch_energies = self._infer_energies_batch(b)
                    energies_chunks.append(batch_energies)
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_start // effective_batch_size}: {e}")
                    # Добавляем нулевые энергии для неудачных сегментов
                    batch_size_actual = batch_end - batch_start
                    energies_chunks.append(np.zeros((batch_size_actual, 4), dtype=np.float32))
            
            energy_all = np.concatenate(energies_chunks, axis=0) if energies_chunks else np.zeros((0, 4), dtype=np.float32)
            
            # Вычисляем shares для всех сегментов с безопасной обработкой NaN
            # Ensure energies are non-negative and finite (handle NaN/inf)
            energy_all = np.nan_to_num(energy_all, nan=0.0, posinf=0.0, neginf=0.0)
            energy_all = np.maximum(energy_all, 0.0)  # Ensure non-negative
            
            # Compute shares with safe normalization
            total = np.sum(energy_all, axis=1, keepdims=True)
            # Handle zero total case (all energies are zero) - set equal shares
            zero_mask = (total.flatten() < 1e-9)
            total = np.where(zero_mask.reshape(-1, 1), 1.0, total + 1e-9)
            shares_all = energy_all / total  # [N, 4]
            
            # For zero-energy segments, set equal shares (1/4 each)
            if np.any(zero_mask):
                shares_all[zero_mask] = 0.25  # Equal shares for silent segments
            
            # Ensure shares are finite and in valid range [0, 1]
            shares_all = np.nan_to_num(shares_all, nan=0.25, posinf=1.0, neginf=0.0)
            shares_all = np.clip(shares_all, 0.0, 1.0)
            
            # Renormalize to ensure sum = 1.0 (defensive)
            row_sums = np.sum(shares_all, axis=1, keepdims=True)
            shares_all = shares_all / (row_sums + 1e-9)
            
            # Этап 6: Распределение результатов обратно по файлам
            results: List[ExtractorResult] = []
            
            for file_info in audio_files_with_segments:
                file_id = file_info.get("file_id", "unknown")
                file_start, file_end = file_segment_ranges.get(file_id, (0, 0))
                
                # Извлекаем результаты для этого файла
                file_energies: List[np.ndarray] = []
                file_shares: List[np.ndarray] = []
                file_starts: List[float] = []
                file_ends: List[float] = []
                file_centers: List[float] = []
                file_peaks: List[float] = []
                file_rmss: List[float] = []
                
                for idx in range(len(mels)):
                    if mels[idx]["file_id"] == file_id:
                        if idx < energy_all.shape[0]:
                            file_energies.append(energy_all[idx])
                            file_shares.append(shares_all[idx])
                        file_starts.append(starts[idx])
                        file_ends.append(ends[idx])
                        file_centers.append(centers[idx])
                        file_peaks.append(peaks[idx])
                        file_rmss.append(rmss[idx])
                
                if not file_energies:
                    results.append(self._create_result(
                        success=False,
                        error="No energies generated for this file",
                        processing_time=time.time() - start_time,
                    ))
                    continue
                
                # Формируем массивы для файла
                energy = np.stack(file_energies, axis=0).astype(np.float32)
                shares = np.stack(file_shares, axis=0).astype(np.float32)
                
                # Silence detection (если включено)
                if self.enable_silence_detection:
                    if (max(file_peaks) if file_peaks else 0.0) < self.silence_peak_threshold and (max(file_rmss) if file_rmss else 0.0) < self.silence_rms_threshold:
                        payload: Dict[str, Any] = {
                            "status": "empty",
                            "empty_reason": "audio_silent",
                            "segments_count": int(len(file_energies)),
                            "sample_rate": int(self.sample_rate),
                            "model_name": self.model_name,
                            "device_used": self.device,
                            "source_separation_contract_version": SOURCE_SEPARATION_CONTRACT_VERSION,
                        }
                        results.append(self._create_result(True, payload=payload, processing_time=time.time() - start_time))
                        continue
                
                # Validate shares and energies
                is_valid, error_msg = self._validate_shares_and_energies(shares, energy, len(file_energies))
                if not is_valid:
                    results.append(self._create_result(
                        success=False,
                        error=f"Validation failed: {error_msg}",
                        processing_time=time.time() - start_time,
                    ))
                    continue
                
                # Compute aggregates (аналогично run_segments)
                share_mean = np.mean(shares, axis=0).astype(np.float32) if shares.size else np.zeros((4,), dtype=np.float32)
                share_std = np.std(shares, axis=0).astype(np.float32) if shares.size else np.zeros((4,), dtype=np.float32)
                
                payload: Dict[str, Any] = {
                    "segments_count": int(len(file_energies)),
                    "sample_rate": int(self.sample_rate),
                    "device_used": self.device,
                    "model_name": self.model_name,
                    "source_order": self._source_names,
                    "source_separation_contract_version": SOURCE_SEPARATION_CONTRACT_VERSION,
                }
                
                # Feature gating (аналогично run_segments)
                if self.enable_share_sequence:
                    payload["share_sequence"] = shares.astype(np.float32)
                if self.enable_energy_sequence:
                    payload["energy_sequence"] = energy.astype(np.float32)
                if self.enable_share_mean:
                    payload["share_mean"] = share_mean
                if self.enable_share_std:
                    payload["share_std"] = share_std
                
                payload["segment_start_sec"] = file_starts
                payload["segment_end_sec"] = file_ends
                payload["segment_center_sec"] = file_centers
                
                # Additional aggregates (если нужно)
                if self.enable_share_mean or self.enable_share_sequence:
                    dominant_source_id = int(np.argmax(share_mean)) if share_mean.size else -1
                    dominant_source_share = float(np.max(share_mean)) if share_mean.size else 0.0
                    payload["dominant_source_id"] = dominant_source_id
                    payload["dominant_source_share"] = dominant_source_share
                    
                    if share_mean.size > 1:
                        entropy = float(-np.sum(share_mean * np.log(share_mean + 1e-9)))
                        max_entropy = float(np.log(4.0))
                        balance_score = float(entropy / max_entropy) if max_entropy > 0 else 0.0
                        payload["source_balance_score"] = balance_score
                    
                    if self.enable_share_sequence:
                        dominant_sources = np.argmax(shares, axis=1).astype(np.int32)
                        
                        if len(dominant_sources) > 1:
                            transitions = sum(1 for i in range(len(dominant_sources) - 1) if dominant_sources[i] != dominant_sources[i + 1])
                            payload["source_transitions_count"] = int(transitions)
                        
                        source_duration = {}
                        source_segments_count = {}
                        total_duration = float(max(file_ends) if file_ends else 0.0)
                        for i, src_id in enumerate(dominant_sources):
                            src_id_int = int(src_id)
                            if src_id_int not in source_duration:
                                source_duration[src_id_int] = 0.0
                                source_segments_count[src_id_int] = 0
                            seg_duration = float(file_ends[i] - file_starts[i])
                            source_duration[src_id_int] += seg_duration
                            source_segments_count[src_id_int] += 1
                        
                        source_distribution = {}
                        if total_duration > 0:
                            for src_id, duration in source_duration.items():
                                source_distribution[int(src_id)] = float(duration / total_duration)
                        payload["source_distribution"] = source_distribution
                        payload["source_segments_per_source"] = {int(k): int(v) for k, v in source_segments_count.items()}
                        payload["source_duration_per_source"] = {int(k): float(v) for k, v in source_duration.items()}
                        
                        if total_duration > 0:
                            transitions_freq = float(transitions) / total_duration if transitions > 0 else 0.0
                            stability_score = float(1.0 / (1.0 + transitions_freq))
                            payload["source_stability_score"] = stability_score
                        
                        # Advanced features (temporal, stability, distribution, musical heuristics)
                        advanced_features = self._compute_advanced_features(shares)
                        payload.update(advanced_features)
                
                if self.enable_quality_metrics:
                    quality_metrics = {}
                    if self.enable_share_mean:
                        quality_metrics["share_mean_min"] = float(np.min(share_mean))
                        quality_metrics["share_mean_max"] = float(np.max(share_mean))
                        quality_metrics["share_mean_std"] = float(np.std(share_mean))
                    if self.enable_share_std:
                        quality_metrics["share_std_mean"] = float(np.mean(share_std))
                        quality_metrics["share_std_max"] = float(np.max(share_std))
                    if self.enable_share_sequence:
                        quality_metrics["share_sequence_min"] = float(np.min(shares))
                        quality_metrics["share_sequence_max"] = float(np.max(shares))
                        quality_metrics["share_sequence_mean"] = float(np.mean(shares))
                    if self.enable_energy_sequence:
                        quality_metrics["energy_sequence_min"] = float(np.min(energy))
                        quality_metrics["energy_sequence_max"] = float(np.max(energy))
                        quality_metrics["energy_sequence_mean"] = float(np.mean(energy))
                    payload["source_quality_metrics"] = quality_metrics
                
                # Track enabled features
                enabled_features = []
                if self.enable_share_sequence:
                    enabled_features.append("share_sequence")
                if self.enable_energy_sequence:
                    enabled_features.append("energy_sequence")
                if self.enable_share_mean:
                    enabled_features.append("share_mean")
                if self.enable_share_std:
                    enabled_features.append("share_std")
                if self.enable_quality_metrics:
                    enabled_features.append("quality_metrics")
                
                payload["_features_enabled"] = enabled_features
                
                results.append(self._create_result(
                    success=True,
                    payload=payload,
                    processing_time=time.time() - start_time,
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in extract_batch_segments: {e}")
            return [
                self._create_result(
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time,
                )
                for _ in audio_files_with_segments
            ]
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_size": self.model_size,
            "batch_size": self.batch_size,
            "device": self.device,
            "model_name": getattr(self, "model_name", None),
            "weights_digest": getattr(self, "weights_digest", None),
            "models_used_entry": getattr(self, "models_used_entry", None),
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "source_order": self._source_names,
        }
