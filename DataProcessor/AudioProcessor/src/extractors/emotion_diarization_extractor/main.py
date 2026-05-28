"""
Emotion diarization extractor (SpeechBrain Speech_Emotion_Diarization) + Segmenter time windows.

Policy:
- NO runtime downloads (ModelManager enforced).
- Uses Segmenter `audio/segments.json` family: `emotion`.
- `<5s` audio -> ERROR.
- Truly silent audio -> EMPTY (payload.status="empty", empty_reason="audio_silent").
- Uses local SpeechBrain from component directory (speechbrain/).
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple

import numpy as np
import torch

# Enforce offline mode (Audit v3). ModelManager should set these too, but we set defensively.
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Import base classes first (before modifying sys.path)
from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

from .utils.resource_profile import (
    prefix_snapshot,
    resource_profile_enabled,
    snapshot_process_resources,
)

# Add local speechbrain to path (after base imports to avoid breaking module imports)
_extractor_dir = Path(__file__).resolve().parent
_speechbrain_path = _extractor_dir / "speechbrain"
if _speechbrain_path.exists() and str(_speechbrain_path) not in sys.path:
    sys.path.insert(0, str(_speechbrain_path))

logger = logging.getLogger(__name__)

# Contract version for downstream extractors compatibility validation
EMOTION_CONTRACT_VERSION = "emotion_contract_v1"


class EmotionDiarizationExtractor(BaseExtractor):
    name = "emotion_diarization_extractor"
    version = "3.1.2"
    description = "Emotion diarization via SpeechBrain Speech_Emotion_Diarization (probs + aggregates)"
    category = "speech"
    dependencies = ["numpy", "torch", "speechbrain", "dp_models"]
    estimated_duration = 6.0

    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 2.0  # SpeechBrain model requires GPU memory

    def __init__(
        self,
        device: str = "auto",
        model_size: str = "small",
        sample_rate: int = 16000,
        batch_size: int = 16,
        # Feature gating flags (per-feature control, default: all False)
        enable_probs: bool = False,
        enable_ids: bool = True,
        enable_confidence: bool = True,
        enable_mean_probs: bool = False,
        enable_entropy: bool = True,
        enable_dominant: bool = True,
        enable_quality_metrics: bool = False,
        # Silence detection
        silence_peak_threshold: float = 1e-3,
        silence_rms_threshold: float = 1e-4,
        enable_silence_detection: bool = True,
        # Full audio processing mode (Audit v3: disabled)
        process_full_audio: bool = False,
        # Progress reporting callback
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        """
        Инициализация экстрактора эмоциональной диаризации.
        
        Args:
            device: Устройство для обработки
            model_size: small|large (in-process model selection via ModelManager)
            sample_rate: Частота дискретизации
            batch_size: Размер батча для обработки окон
            enable_probs: Включить emotion_probs (per-window probabilities)
            enable_ids: Включить emotion_id (argmax per window)
            enable_confidence: Включить emotion_confidence (max prob per window)
            enable_mean_probs: Включить emotion_mean_probs (средние вероятности)
            enable_entropy: Включить emotion_entropy (энтропия)
            enable_dominant: Включить dominant_emotion_id/prob (доминирующая эмоция)
            enable_quality_metrics: Включить метрики качества (confidence distribution, stability)
            silence_peak_threshold: Порог peak для детекции тишины
            silence_rms_threshold: Порог RMS для детекции тишины
            enable_silence_detection: Включить проверку на тишину
            process_full_audio: Если True, обрабатывает все аудио целиком как один сегмент (использует run() вместо run_segments())
            progress_callback: Callback для прогресса (batch_index, total_batches, message)
        """
        super().__init__(device=device)
        self.model_size = str(model_size or "small").strip().lower()
        if self.model_size not in ("small", "large"):
            raise ValueError(f"emotion_diarization | unsupported model_size={self.model_size}. Expected: small|large")
        self.sample_rate = int(sample_rate)
        self.batch_size = max(1, int(batch_size))
        
        # Feature gating flags
        self.enable_probs = bool(enable_probs)
        self.enable_ids = bool(enable_ids)
        self.enable_confidence = bool(enable_confidence)
        self.enable_mean_probs = bool(enable_mean_probs)
        self.enable_entropy = bool(enable_entropy)
        self.enable_dominant = bool(enable_dominant)
        self.enable_quality_metrics = bool(enable_quality_metrics)
        
        # Silence detection
        self.silence_peak_threshold = float(silence_peak_threshold)
        self.silence_rms_threshold = float(silence_rms_threshold)
        self.enable_silence_detection = bool(enable_silence_detection)
        
        # Audit v3: Segmenter owns sampling; no full-audio fallback in audited contract.
        if bool(process_full_audio):
            raise RuntimeError(
                "emotion_diarization | process_full_audio is disabled in audited mode. "
                "Use Segmenter families.emotion windows and run_segments()."
            )
        self.process_full_audio = False

        # Audit v3: minimal model-facing sequences must be enabled.
        if not self.enable_ids or not self.enable_confidence:
            raise RuntimeError(
                "emotion_diarization | audited contract requires enable_ids=true and enable_confidence=true "
                "(model_facing per-segment sequences)."
            )
        # Audit v3: core aggregates are required (frozen model-facing subset).
        if not self.enable_entropy or not self.enable_dominant:
            raise RuntimeError(
                "emotion_diarization | audited contract requires enable_entropy=true and enable_dominant=true "
                "(core model-facing aggregates)."
            )
        
        # Progress callback
        self.progress_callback = progress_callback
        
        # Log feature flags for debugging
        logger.info(f"emotion_diarization | initialized with feature flags: probs={self.enable_probs}, ids={self.enable_ids}, confidence={self.enable_confidence}, mean_probs={self.enable_mean_probs}, entropy={self.enable_entropy}, dominant={self.enable_dominant}, quality_metrics={self.enable_quality_metrics}")

        self.audio_utils = AudioUtils(device=device, sample_rate=self.sample_rate)

        # ModelManager: resolve in-process model (no-network).
        try:
            from dp_models import get_global_model_manager  # type: ignore

            self._mm = get_global_model_manager()
        except Exception as e:
            raise RuntimeError(f"emotion_diarization | ModelManager is required but failed to init: {e}") from e

        spec_name = f"emotion_diarization_{self.model_size}_inprocess"
        try:
            self.model_spec = self._mm.get_spec(model_name=spec_name)
            _dev, _prec, rt, _eng, wd, _arts = self._mm.resolve(self.model_spec)
            if str(rt) != "inprocess":
                raise RuntimeError(f"emotion_diarization | expected runtime=inprocess in spec {spec_name}, got {rt}")
            self.model_name = str(self.model_spec.model_name)
            self.weights_digest = str(wd)


            # Load model via ModelManager
            resolved_model = self._mm.get(model_name=spec_name)
            self.model = resolved_model.handle
            self.models_used_entry = resolved_model.models_used_entry



            # Get emotion labels from runtime_params if available, otherwise from model
            rp = self.model_spec.runtime_params or {}
            emotion_labels_raw = rp.get("emotion_labels")
            if isinstance(emotion_labels_raw, list) and emotion_labels_raw:
                self.emotion_labels = [str(x) for x in emotion_labels_raw]
            else:
                # Try to get labels from model's label_encoder
                self.emotion_labels = []
                if hasattr(self.model, 'hparams') and hasattr(self.model.hparams, 'label_encoder'):
                    label_encoder = self.model.hparams.label_encoder
                    if hasattr(label_encoder, 'ind2lab'):
                        # Map indices to labels
                        max_idx = max(label_encoder.ind2lab.keys()) if label_encoder.ind2lab else -1
                        self.emotion_labels = [label_encoder.ind2lab.get(i, f"emotion_{i}") for i in range(max_idx + 1)]
                    elif hasattr(label_encoder, 'lab2ind'):
                        # Reverse mapping: get labels from lab2ind
                        self.emotion_labels = sorted(label_encoder.lab2ind.keys())
            
            # Model is already loaded and on device via SpeechBrainProvider
            # SpeechBrain models handle device internally
        except Exception as e:
            raise RuntimeError(f"emotion_diarization | failed to resolve/load model via ModelManager: {e}") from e


    def _validate_emotion_labels(self, num_classes: int) -> tuple[bool, Optional[str]]:
        """
        Полная валидация emotion_labels: согласованность с количеством классов, отсутствие дубликатов, валидность типов.
        
        Args:
            num_classes: количество классов эмоций (из shape emotion_probs)
        
        Returns:
            (is_valid, error_message)
        """
        if not self.emotion_labels:
            return True, None  # Empty labels are valid (optional)
        
        if not isinstance(self.emotion_labels, list):
            return False, f"emotion_diarization | emotion_labels must be a list, got {type(self.emotion_labels)}"
        
        if len(self.emotion_labels) != num_classes:
            return False, f"emotion_diarization | emotion_labels length ({len(self.emotion_labels)}) != num_classes ({num_classes})"
        
        # Check for duplicates
        if len(self.emotion_labels) != len(set(self.emotion_labels)):
            return False, "emotion_diarization | emotion_labels contain duplicates"
        
        # Check types (all should be strings)
        for i, label in enumerate(self.emotion_labels):
            if not isinstance(label, str):
                return False, f"emotion_diarization | emotion_labels[{i}] must be str, got {type(label)}"
        
        return True, None

    def _validate_probs(self, probs: np.ndarray, num_segments: int) -> tuple[bool, Optional[str]]:
        """
        Полная валидация вероятностей: проверка NaN/inf, диапазонов [0,1], валидность argmax, согласованность размеров.
        
        Args:
            probs: массив вероятностей (float32, shape [N, C])
            num_segments: ожидаемое количество сегментов
        
        Returns:
            (is_valid, error_message)
        """
        if probs.size == 0:
            return True, None  # Empty is valid
        
        # Check dtype
        if probs.dtype != np.float32:
            return False, f"emotion_diarization | probs dtype must be float32, got {probs.dtype}"
        
        # Check shape
        if probs.ndim != 2:
            return False, f"emotion_diarization | probs must be 2D [N, C], got shape {probs.shape}"
        
        if probs.shape[0] != num_segments:
            return False, f"emotion_diarization | probs shape[0] ({probs.shape[0]}) != num_segments ({num_segments})"
        
        # Check for NaN/inf
        if np.any(np.isnan(probs)):
            return False, "emotion_diarization | probs contain NaN values"
        if np.any(np.isinf(probs)):
            return False, "emotion_diarization | probs contain inf values"
        
        # Check ranges [0, 1]
        if np.any(probs < 0.0) or np.any(probs > 1.0):
            return False, f"emotion_diarization | probs out of range [0, 1]: min={np.min(probs)}, max={np.max(probs)}"
        
        # Check normalization (sum should be close to 1.0 per row)
        row_sums = np.sum(probs, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            return False, f"emotion_diarization | probs not normalized (sum per row should be 1.0): min_sum={np.min(row_sums)}, max_sum={np.max(row_sums)}"
        
        # Check number of classes (should be reasonable, e.g., 2-20)
        num_classes = probs.shape[1]
        if num_classes < 2 or num_classes > 20:
            return False, f"emotion_diarization | suspicious num_classes: {num_classes} (expected 2-20)"
        
        return True, None

    @staticmethod
    def _rms_and_peak(x: np.ndarray) -> tuple[float, float]:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return 0.0, 0.0
        rms = float(np.sqrt(float(np.mean(x * x)) + 1e-12))
        peak = float(np.max(np.abs(x)) + 1e-12)
        return rms, peak

    def _infer_probs_batch(self, batch: np.ndarray, wav_lens: np.ndarray) -> np.ndarray:
        """
        Выполнить batch inference для получения вероятностей эмоций через SpeechBrain Speech_Emotion_Diarization.
        
        Использует встроенный метод diarize_batch() для батчевой обработки, затем извлекает вероятности
        из logits до применения log_softmax.
        
        Args:
            batch: паддингнутые аудио сегменты, shape [B, max_length] float32
            wav_lens: относительные длины (float32[B], (0,1]) для каждого сегмента ДО паддинга
        
        Returns:
            вероятности эмоций, shape [B, num_emotions] float32
        
        Raises:
            RuntimeError при ошибках inference
        """
        try:
            # Convert numpy array to torch tensor
            # SpeechBrain expects [batch, time] format for mono audio
            batch_tensor = torch.from_numpy(batch).to(self.device)
            
            # wav_lens is provided by caller (true lengths), do NOT infer it from non-zero samples.
            wav_lens = np.asarray(wav_lens, dtype=np.float32).reshape(-1)
            if wav_lens.shape[0] != batch.shape[0]:
                raise RuntimeError(
                    f"emotion_diarization | wav_lens shape mismatch: wav_lens={wav_lens.shape} batch={batch.shape}"
                )
            wav_lens = np.clip(wav_lens, 1e-6, 1.0)
            wav_lens_t = torch.from_numpy(wav_lens).to(self.device)
            
            # Get logits from model before softmax
            # SpeechBrain model structure: encode_batch -> avg_pool -> output_mlp -> log_softmax
            # Note: avg_pool with kernel_size=1 and stride=1 doesn't actually pool, so we need to manually average
            with torch.no_grad():
                # Encode batch (WavLM encoder)
                outputs = self.model.encode_batch(batch_tensor, wav_lens_t)
                
                # outputs shape: [B, T, D] where T is time dimension
                
                # Average pool (may not actually pool if kernel_size=1, stride=1)
                averaged_out = self.model.hparams.avg_pool(outputs)
                # averaged_out shape: [B, T', D] where T' may still be > 1
                
                # Get logits from output_mlp (before log_softmax)
                logits = self.model.mods.output_mlp(averaged_out)
                # logits shape: [B, T', num_emotions] - may still have time dimension
                
                # If logits has time dimension (ndim == 3), average over time to get [B, num_emotions]
                if logits.ndim == 3:
                    # Average over time dimension (dim=1)
                    logits = logits.mean(dim=1)
                elif logits.ndim != 2:
                    raise RuntimeError(f"emotion_diarization | unexpected logits shape: {logits.shape}, expected 2D [B, C] or 3D [B, T, C]")
                
                # Apply softmax to get probabilities (model uses log_softmax, we need regular softmax)
                probs_tensor = torch.softmax(logits, dim=-1)
                # probs_tensor shape: [B, num_emotions]
                
                # Convert to numpy
                probs = probs_tensor.cpu().numpy().astype(np.float32)
            
            # Validate probabilities
            is_valid, error_msg = self._validate_probs(probs, batch.shape[0])
            if not is_valid:
                raise ValueError(f"emotion_diarization | probability validation failed: {error_msg}")
            
            if probs.ndim != 2 or probs.shape[0] != batch.shape[0]:
                raise RuntimeError(f"emotion_diarization | unexpected probs shape from model: {probs.shape}, expected [{batch.shape[0]}, C]")
            
            return probs
        except Exception as e:
            raise RuntimeError(f"emotion_diarization | inference failed: {e}") from e

    def run_segments(
        self, 
        input_uri: str, 
        tmp_path: str, 
        segments: List[Dict[str, Any]]
    ) -> ExtractorResult:
        """
        Segmenter-driven emotion diarization: compute emotion probabilities on provided windows.
        
        Progress reporting: каждые 10% батчей (если progress_callback установлен).
        """
        start_time = time.time()
        timings: Dict[str, float] = {}  # Детальное профилирование этапов
        emotion_diarization_resource_profile: Optional[Dict[str, Any]] = None
        
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)
            if not isinstance(segments, list) or not segments:
                raise ValueError("segments is empty (no-fallback)")

            if resource_profile_enabled():
                try:
                    emotion_diarization_resource_profile = {
                        **prefix_snapshot("at_start", snapshot_process_resources()),
                    }
                except Exception:
                    emotion_diarization_resource_profile = None

            # Time axis (strict alignment)
            N = len(segments)
            starts = np.asarray([float(s.get("start_sec", 0.0) or 0.0) for s in segments], dtype=np.float32)
            ends = np.asarray([float(s.get("end_sec", 0.0) or 0.0) for s in segments], dtype=np.float32)
            centers = np.asarray([float(s.get("center_sec", 0.0) or 0.0) for s in segments], dtype=np.float32)
            seg_mask = np.zeros((N,), dtype=np.bool_)

            # For <5s, return valid empty (Audit v3).
            dur_sec = float(np.max(ends) if ends.size else 0.0)
            if dur_sec < 5.0:
                payload: Dict[str, Any] = {
                    "status": "empty",
                    "empty_reason": "audio_too_short",
                    "segments_total": int(N),
                    "segments_count": int(0),
                    "sample_rate": int(self.sample_rate),
                    "device_used": str(self.device),
                    "emotion_labels": self.emotion_labels,
                    "segment_start_sec": starts,
                    "segment_end_sec": ends,
                    "segment_center_sec": centers,
                    "segment_mask": seg_mask,
                    "emotion_id": np.full((N,), -1, dtype=np.int32),
                    "emotion_confidence": np.full((N,), np.nan, dtype=np.float32),
                    "emotion_contract_version": EMOTION_CONTRACT_VERSION,
                    "_features_enabled": [],
                    "stage_timings_ms": {},
                    "emotion_diarization_resource_profile": emotion_diarization_resource_profile,
                }
                return self._create_result(True, payload=payload, processing_time=time.time() - start_time)

            # Этап 1: Загрузка сегментов
            t_load_start = time.time()
            waves_valid: list[np.ndarray] = []
            lens_valid: list[int] = []
            valid_indices: list[int] = []
            for i, seg in enumerate(segments):
                try:
                    ss = int(seg.get("start_sample"))
                    es = int(seg.get("end_sample"))
                    wav_t, sr = self.audio_utils.load_audio_segment(
                        input_uri, start_sample=ss, end_sample=es, target_sr=self.sample_rate
                    )
                    wav = self.audio_utils.to_numpy(wav_t)
                    wav = wav[0] if wav.ndim == 2 else wav.reshape(-1)
                    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
                    if int(sr) != int(self.sample_rate):
                        raise RuntimeError(f"segment SR mismatch: got {sr} expected {self.sample_rate}")
                    if wav.size <= 0:
                        continue
                    waves_valid.append(wav)
                    lens_valid.append(int(wav.shape[0]))
                    valid_indices.append(i)
                    seg_mask[i] = True
                except Exception as e:
                    # Strict alignment: mark segment as invalid, keep arrays length N.
                    seg_mask[i] = False
            
            t_load_end = time.time()
            timings["load_segments_sec"] = t_load_end - t_load_start
            logger.info(
                f"emotion_diarization | loaded segments: total={N} valid={int(np.sum(seg_mask))} in {timings['load_segments_sec']:.3f}s"
            )

            max_len = int(max(lens_valid) if lens_valid else 0)
            if max_len <= 0:
                raise RuntimeError("emotion_diarization | no valid audio samples in segments")

            # Этап 2: Silence detection (if enabled)
            t_silence_start = time.time()
            # Avoid concatenating all segments: compute streaming RMS/peak (saves memory, faster).
            sumsq = 0.0
            n_samp = 0
            peak_abs = 0.0
            for w in waves_valid:
                if w.size <= 0:
                    continue
                # w is float32; accumulate in float64 scalar.
                peak_abs = max(peak_abs, float(np.max(np.abs(w))))
                sumsq += float(np.dot(w, w))
                n_samp += int(w.size)
            rms = float(np.sqrt(sumsq / float(n_samp))) if n_samp > 0 else 0.0
            peak = float(peak_abs)

            if self.enable_silence_detection:
                if peak < self.silence_peak_threshold and rms < self.silence_rms_threshold:
                    payload: Dict[str, Any] = {
                        "status": "empty",
                        "empty_reason": "audio_silent",
                        "segments_total": int(N),
                        "segments_count": int(0),
                        "sample_rate": int(self.sample_rate),
                        "rms": float(rms),
                        "peak": float(peak),
                        "emotion_labels": self.emotion_labels,
                        "device_used": str(self.device),
                        "emotion_contract_version": EMOTION_CONTRACT_VERSION,
                        "segment_start_sec": starts,
                        "segment_end_sec": ends,
                        "segment_center_sec": centers,
                        "segment_mask": np.zeros((N,), dtype=np.bool_),
                        "emotion_id": np.full((N,), -1, dtype=np.int32),
                        "emotion_confidence": np.full((N,), np.nan, dtype=np.float32),
                        "_features_enabled": [],
                        "stage_timings_ms": {},
                        "emotion_diarization_resource_profile": emotion_diarization_resource_profile,
                    }
                    timings["silence_detection_sec"] = time.time() - t_silence_start
                    return self._create_result(True, payload=payload, processing_time=time.time() - start_time)
                rms_val, peak_val = rms, peak
            else:
                rms_val, peak_val = rms, peak
            
            t_silence_end = time.time()
            timings["silence_detection_sec"] = t_silence_end - t_silence_start

            # Этап 3: Padding для батчинга
            t_pad_start = time.time()
            padded = np.zeros((len(waves_valid), max_len), dtype=np.float32)
            lens_arr = np.asarray(lens_valid, dtype=np.int32).reshape(-1)
            for j, w in enumerate(waves_valid):
                padded[j, : int(w.shape[0])] = w
            t_pad_end = time.time()
            timings["padding_sec"] = t_pad_end - t_pad_start

            # Этап 4: Batch inference
            t_inference_start = time.time()
            # Determine batch size (auto-split if >100 segments)
            effective_batch_size = self.batch_size
            if len(waves_valid) > 100:
                effective_batch_size = min(100, self.batch_size)  # Auto-split large batches
            
            # Process in batches
            probs_chunks: list[np.ndarray] = []
            total_batches = (len(waves_valid) + effective_batch_size - 1) // effective_batch_size
            progress_report_interval = max(1, total_batches // 10) if total_batches >= 10 else 1
            last_reported_pct = -1
            
            logger.info(
                f"emotion_diarization | starting inference: {total_batches} batches, batch_size={effective_batch_size}, segments_valid={len(waves_valid)}"
            )
            
            for batch_idx, start in enumerate(range(0, padded.shape[0], effective_batch_size)):
                batch = padded[start : start + effective_batch_size]
                batch_lens = lens_arr[start : start + effective_batch_size].astype(np.float32, copy=False)
                max_len_b = float(batch.shape[1]) if batch.shape[1] > 0 else 1.0
                wav_lens = (batch_lens / max_len_b).astype(np.float32, copy=False)
                np.clip(wav_lens, 1e-6, 1.0, out=wav_lens)
                batch_probs = self._infer_probs_batch(batch, wav_lens)
                probs_chunks.append(batch_probs)
                
                # Progress reporting
                if self.progress_callback and batch_idx % progress_report_interval == 0:
                    pct = int((batch_idx + 1) * 100 / total_batches)
                    if pct != last_reported_pct:
                        batch_elapsed = time.time() - t_inference_start
                        self.progress_callback(batch_idx + 1, total_batches, f"Inference: {batch_idx + 1}/{total_batches} batches ({pct}%, {batch_elapsed:.1f}s)")
                        last_reported_pct = pct
            
            t_inference_end = time.time()
            timings["inference_sec"] = t_inference_end - t_inference_start
            logger.info(f"emotion_diarization | inference completed: {timings['inference_sec']:.3f}s for {total_batches} batches")

            # Этап 5: Concatenation и нормализация
            t_postprocess_start = time.time()
            probs = np.concatenate(probs_chunks, axis=0) if probs_chunks else np.zeros((0, 0), dtype=np.float32)
            if probs.shape[0] != padded.shape[0]:
                raise RuntimeError(f"emotion_diarization | probs batch mismatch: probs={probs.shape} windows={padded.shape[0]}")

            # Normalize (defensive)
            probs /= (np.sum(probs, axis=1, keepdims=True) + 1e-9)
            
            # Validate emotion_labels
            num_classes = probs.shape[1]
            is_valid_labels, error_msg_labels = self._validate_emotion_labels(num_classes)
            if not is_valid_labels:
                raise ValueError(f"emotion_diarization | emotion_labels validation failed: {error_msg_labels}")

            # Этап 6: Compute aggregates (valid segments only)
            t_aggregates_start = time.time()
            emotion_id_valid = np.argmax(probs, axis=1).astype(np.int32)
            emotion_conf_valid = np.max(probs, axis=1).astype(np.float32)
            mean_probs = np.mean(probs, axis=0).astype(np.float32) if probs.size else np.zeros((0,), dtype=np.float32)
            ent = float(-np.sum(mean_probs * np.log(mean_probs + 1e-9))) if mean_probs.size else 0.0
            dominant_id = int(np.argmax(mean_probs)) if mean_probs.size else -1
            dominant_prob = float(np.max(mean_probs)) if mean_probs.size else 0.0
            t_aggregates_end = time.time()
            timings["aggregates_sec"] = t_aggregates_end - t_aggregates_start

            # Scatter back to strict-aligned arrays
            emotion_id_full = np.full((N,), -1, dtype=np.int32)
            emotion_conf_full = np.full((N,), np.nan, dtype=np.float32)
            if valid_indices:
                vi = np.asarray(valid_indices, dtype=np.int32)
                emotion_id_full[vi] = emotion_id_valid
                emotion_conf_full[vi] = emotion_conf_valid

            payload: Dict[str, Any] = {
                "segments_total": int(N),
                "segments_count": int(np.sum(seg_mask)),
                "sample_rate": int(self.sample_rate),
                "device_used": str(self.device),
                "model_name": getattr(self, "model_name", None),
                "weights_digest": getattr(self, "weights_digest", None),
                "rms": float(rms_val),
                "peak": float(peak_val),
                "emotion_labels": self.emotion_labels,
                "emotion_contract_version": EMOTION_CONTRACT_VERSION,
                "segment_start_sec": starts,
                "segment_end_sec": ends,
                "segment_center_sec": centers,
                "segment_mask": seg_mask,
                "emotion_id": emotion_id_full,
                "emotion_confidence": emotion_conf_full,
                # Core model-facing aggregates (Audit v3: always present)
                "emotion_entropy": float(ent),
                "dominant_emotion_id": int(dominant_id),
                "dominant_emotion_prob": float(dominant_prob),
            }

            # Transitions / distribution / stability / diversity computed on valid segments
            valid_ids = emotion_id_full[seg_mask]
            valid_durations = (ends - starts)[seg_mask].astype(np.float32) if seg_mask.size else np.zeros((0,), dtype=np.float32)
            total_duration = float(np.sum(valid_durations)) if valid_durations.size else 0.0

            transitions = 0
            if valid_ids.size > 1:
                transitions = int(np.sum(valid_ids[1:] != valid_ids[:-1]))
            payload["emotion_transitions_count"] = int(transitions)

            # Stability score
            transitions_freq = float(transitions) / total_duration if total_duration > 0 else 0.0
            payload["emotion_stability_score"] = float(1.0 / (1.0 + transitions_freq))

            # Diversity score
            if mean_probs.size > 1:
                max_entropy = float(np.log(float(num_classes)))
                payload["emotion_diversity_score"] = float(ent / max_entropy) if max_entropy > 0 else 0.0
            else:
                payload["emotion_diversity_score"] = 0.0

            # Optional: save probs / mean_probs
            if self.enable_probs:
                probs_full = np.full((N, int(num_classes)), np.nan, dtype=np.float32)
                if valid_indices:
                    vi = np.asarray(valid_indices, dtype=np.int32)
                    probs_full[vi, :] = probs
                payload["emotion_probs"] = probs_full
            if self.enable_mean_probs:
                payload["emotion_mean_probs"] = mean_probs

            # Optional: distribution objects
            if self.enable_dominant:
                emotion_duration: Dict[int, float] = {}
                emotion_segments_count: Dict[int, int] = {}
                for k, emo_id in enumerate(valid_ids.tolist() if hasattr(valid_ids, "tolist") else list(valid_ids)):
                    emo_id_int = int(emo_id)
                    emotion_duration.setdefault(emo_id_int, 0.0)
                    emotion_segments_count.setdefault(emo_id_int, 0)
                    d = float(valid_durations[k]) if k < int(valid_durations.size) else 0.0
                    emotion_duration[emo_id_int] += d
                    emotion_segments_count[emo_id_int] += 1
                emotion_distribution: Dict[int, float] = {}
                if total_duration > 0:
                    for emo_id_int, d in emotion_duration.items():
                        emotion_distribution[int(emo_id_int)] = float(d / total_duration)
                payload["emotion_distribution"] = emotion_distribution
                payload["emotion_segments_per_emotion"] = {int(k): int(v) for k, v in emotion_segments_count.items()}
                payload["emotion_duration_per_emotion"] = {int(k): float(v) for k, v in emotion_duration.items()}
            
            # Feature gating: quality metrics
            if self.enable_quality_metrics:
                quality_metrics = {}
                conf_clean = emotion_conf_full[seg_mask]
                if conf_clean.size:
                    quality_metrics["confidence_mean"] = float(np.mean(conf_clean))
                    quality_metrics["confidence_std"] = float(np.std(conf_clean))
                    quality_metrics["confidence_min"] = float(np.min(conf_clean))
                    quality_metrics["confidence_max"] = float(np.max(conf_clean))
                    quality_metrics["confidence_median"] = float(np.median(conf_clean))
                if mean_probs.size:
                    quality_metrics["mean_probs_min"] = float(np.min(mean_probs))
                    quality_metrics["mean_probs_max"] = float(np.max(mean_probs))
                    quality_metrics["mean_probs_std"] = float(np.std(mean_probs))
                payload["emotion_quality_metrics"] = quality_metrics
            
            # Track enabled features for meta
            enabled_features = []
            if self.enable_probs:
                enabled_features.append("probs")
            if self.enable_mean_probs:
                enabled_features.append("mean_probs")
            enabled_features.append("ids")
            enabled_features.append("confidence")
            enabled_features.append("entropy")
            enabled_features.append("dominant")
            if self.enable_quality_metrics:
                enabled_features.append("quality_metrics")
            
            payload["_features_enabled"] = enabled_features
            
            t_postprocess_end = time.time()
            timings["postprocess_sec"] = t_postprocess_end - t_postprocess_start
            total_time = time.time() - start_time
            
            # Log detailed profiling
            logger.info(f"emotion_diarization | run_segments completed: segments={len(segments)}, probs_shape={probs.shape if self.enable_probs else 'disabled'}, enabled_features={enabled_features}")
            logger.info(f"emotion_diarization | profiling: load={timings.get('load_segments_sec', 0):.3f}s, silence={timings.get('silence_detection_sec', 0):.3f}s, pad={timings.get('padding_sec', 0):.3f}s, inference={timings.get('inference_sec', 0):.3f}s, aggregates={timings.get('aggregates_sec', 0):.3f}s, postprocess={timings.get('postprocess_sec', 0):.3f}s, total={total_time:.3f}s")

            stage_timings_ms = {k.replace("_sec", "_ms"): float(v) * 1000.0 for k, v in timings.items()}
            stage_timings_ms["total_ms"] = float(total_time) * 1000.0

            if emotion_diarization_resource_profile is not None:
                try:
                    emotion_diarization_resource_profile = {
                        **emotion_diarization_resource_profile,
                        **prefix_snapshot("at_end", snapshot_process_resources()),
                    }
                except Exception:
                    pass

            payload["stage_timings_ms"] = stage_timings_ms
            payload["emotion_diarization_resource_profile"] = emotion_diarization_resource_profile

            return self._create_result(True, payload=payload, processing_time=total_time)

        except Exception as e:
            return self._create_result(False, error=str(e), processing_time=time.time() - start_time)

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Legacy full-audio mode is disabled in audited contract (Audit v3).
        """
        t0 = time.time()
        return self._create_result(
            success=False,
            error="emotion_diarization_extractor | run() is disabled in audited mode. Use run_segments() with Segmenter families.emotion windows.",
            processing_time=time.time() - t0,
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
        """Emotion diarization extractor поддерживает batch processing для сегментов через Triton."""
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
        - Группирует в батчи по max_segments_per_batch (если задан) или использует batch_size
        - Обрабатывает батчи через in-process PyTorch модель для получения вероятностей эмоций
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
        # Audit v3 correctness-first:
        # - Preserve per-file strict alignment + mask semantics by reusing run_segments().
        # - Cross-video batching is intentionally not performed here.
        start_time = time.time()
        if not audio_files_with_segments:
            return []

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
                r = self.run_segments(input_uri, tmp_path, segments)
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

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_size": self.model_size,
            "sample_rate": self.sample_rate,
            "batch_size": self.batch_size,
            "device": self.device,
            "model_name": getattr(self, "model_name", None),
            "weights_digest": getattr(self, "weights_digest", None),
            "models_used_entry": getattr(self, "models_used_entry", None),
        }
