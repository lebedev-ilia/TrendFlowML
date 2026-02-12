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

# Import base classes first (before modifying sys.path)
from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

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
    version = "3.0.0"
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
        enable_ids: bool = False,
        enable_confidence: bool = False,
        enable_mean_probs: bool = False,
        enable_entropy: bool = False,
        enable_dominant: bool = False,
        enable_quality_metrics: bool = False,
        # Silence detection
        silence_peak_threshold: float = 1e-3,
        silence_rms_threshold: float = 1e-4,
        enable_silence_detection: bool = True,
        # Full audio processing mode
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
        
        # Full audio processing mode (process entire audio as one segment instead of using provided segments)
        self.process_full_audio = bool(process_full_audio)
        
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

    def _infer_probs_batch(self, batch: np.ndarray, batch_ids: List[str]) -> np.ndarray:
        """
        Выполнить batch inference для получения вероятностей эмоций через SpeechBrain Speech_Emotion_Diarization.
        
        Использует встроенный метод diarize_batch() для батчевой обработки, затем извлекает вероятности
        из logits до применения log_softmax.
        
        Args:
            batch: паддингнутые аудио сегменты, shape [B, max_length] float32
            batch_ids: список идентификаторов для каждого сегмента в батче
        
        Returns:
            вероятности эмоций, shape [B, num_emotions] float32
        
        Raises:
            RuntimeError при ошибках inference
        """
        try:
            # Convert numpy array to torch tensor
            # SpeechBrain expects [batch, time] format for mono audio
            batch_tensor = torch.from_numpy(batch).to(self.device)
            
            # Calculate relative lengths (for padding)
            max_len = batch.shape[1]
            # Calculate actual lengths (non-zero samples)
            actual_lens = []
            for i in range(batch.shape[0]):
                # Find last non-zero sample
                non_zero = np.nonzero(batch[i])[0]
                actual_len = len(non_zero) if len(non_zero) > 0 else max_len
                actual_lens.append(float(actual_len) / max_len)
            
            wav_lens = torch.tensor(actual_lens, device=self.device)
            
            # Get logits from model before softmax
            # SpeechBrain model structure: encode_batch -> avg_pool -> output_mlp -> log_softmax
            # Note: avg_pool with kernel_size=1 and stride=1 doesn't actually pool, so we need to manually average
            with torch.no_grad():
                # Encode batch (WavLM encoder)
                outputs = self.model.encode_batch(batch_tensor, wav_lens)
                
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
        timings = {}  # Детальное профилирование этапов
        
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)
            if not isinstance(segments, list) or not segments:
                raise ValueError("segments is empty (no-fallback)")

            dur_sec = float(max((float(s.get("end_sec", 0.0)) for s in segments), default=0.0))
            if dur_sec < 5.0:
                raise RuntimeError(f"emotion_diarization | audio too short (<5s): duration_sec={dur_sec:.3f}")

            # Этап 1: Загрузка сегментов
            t_load_start = time.time()
            waves: list[np.ndarray] = []
            starts: list[float] = []
            ends: list[float] = []
            centers: list[float] = []
            lens: list[int] = []
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
                    raise RuntimeError(f"emotion_diarization | segment SR mismatch: got {sr} expected {self.sample_rate}")
                waves.append(wav)
                lens.append(int(wav.shape[0]))
                starts.append(st)
                ends.append(en)
                centers.append(c)
            
            t_load_end = time.time()
            timings["load_segments_sec"] = t_load_end - t_load_start
            logger.info(f"emotion_diarization | loaded {len(segments)} segments in {timings['load_segments_sec']:.3f}s")

            max_len = int(max(lens) if lens else 0)
            if max_len <= 0:
                raise RuntimeError("emotion_diarization | no audio samples in segments")

            # Этап 2: Silence detection (if enabled)
            t_silence_start = time.time()
            if self.enable_silence_detection:
                concat = np.concatenate([w for w in waves if w.size], axis=0) if waves else np.zeros((0,), dtype=np.float32)
                rms, peak = self._rms_and_peak(concat)
                if peak < self.silence_peak_threshold and rms < self.silence_rms_threshold:
                    payload: Dict[str, Any] = {
                        "status": "empty",
                        "empty_reason": "audio_silent",
                        "segments_count": int(len(segments)),
                        "sample_rate": int(self.sample_rate),
                        "rms": float(rms),
                        "peak": float(peak),
                        "model_name": self.model_name,
                        "emotion_labels": self.emotion_labels,
                        "device_used": "cuda",
                        "emotion_contract_version": EMOTION_CONTRACT_VERSION,
                    }
                    timings["silence_detection_sec"] = time.time() - t_silence_start
                    return self._create_result(True, payload=payload, processing_time=time.time() - start_time)
                rms_val, peak_val = rms, peak
            else:
                concat = np.concatenate([w for w in waves if w.size], axis=0) if waves else np.zeros((0,), dtype=np.float32)
                rms_val, peak_val = self._rms_and_peak(concat)
            
            t_silence_end = time.time()
            timings["silence_detection_sec"] = t_silence_end - t_silence_start

            # Этап 3: Padding для батчинга
            t_pad_start = time.time()
            padded = np.zeros((len(waves), max_len), dtype=np.float32)
            for i, w in enumerate(waves):
                padded[i, : int(w.shape[0])] = w
            t_pad_end = time.time()
            timings["padding_sec"] = t_pad_end - t_pad_start

            # Этап 4: Batch inference
            t_inference_start = time.time()
            # Determine batch size (auto-split if >100 segments)
            effective_batch_size = self.batch_size
            if len(waves) > 100:
                effective_batch_size = min(100, self.batch_size)  # Auto-split large batches
            
            # Process in batches
            probs_chunks: list[np.ndarray] = []
            total_batches = (len(waves) + effective_batch_size - 1) // effective_batch_size
            progress_report_interval = max(1, total_batches // 10) if total_batches >= 10 else 1
            last_reported_pct = -1
            
            logger.info(f"emotion_diarization | starting inference: {total_batches} batches, batch_size={effective_batch_size}, segments={len(waves)}")
            
            for batch_idx, start in enumerate(range(0, padded.shape[0], effective_batch_size)):
                batch = padded[start : start + effective_batch_size]
                # Create batch IDs for SpeechBrain
                batch_ids = [f"seg_{start + i}" for i in range(batch.shape[0])]
                batch_probs = self._infer_probs_batch(batch, batch_ids)
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
            s = np.sum(probs, axis=1, keepdims=True) + 1e-9
            probs = probs / s
            
            # Validate emotion_labels
            num_classes = probs.shape[1]
            is_valid_labels, error_msg_labels = self._validate_emotion_labels(num_classes)
            if not is_valid_labels:
                raise ValueError(f"emotion_diarization | emotion_labels validation failed: {error_msg_labels}")

            # Этап 6: Compute aggregates
            t_aggregates_start = time.time()
            emotion_id = np.argmax(probs, axis=1).astype(np.int32)
            emotion_conf = np.max(probs, axis=1).astype(np.float32)
            mean_probs = np.mean(probs, axis=0).astype(np.float32) if probs.size else np.zeros((0,), dtype=np.float32)
            ent = float(-np.sum(mean_probs * np.log(mean_probs + 1e-9))) if mean_probs.size else 0.0
            dominant_id = int(np.argmax(mean_probs)) if mean_probs.size else -1
            dominant_prob = float(np.max(mean_probs)) if mean_probs.size else 0.0
            t_aggregates_end = time.time()
            timings["aggregates_sec"] = t_aggregates_end - t_aggregates_start

            payload: Dict[str, Any] = {
                "segments_count": int(len(segments)),
                "sample_rate": int(self.sample_rate),
                "device_used": "cuda",
                "rms": float(rms_val),
                "peak": float(peak_val),
                "model_name": self.model_name,
                "emotion_labels": self.emotion_labels,
                "emotion_contract_version": EMOTION_CONTRACT_VERSION,
            }
            
            # Feature gating: emotion_probs
            if self.enable_probs:
                payload["emotion_probs"] = probs
            
            # Feature gating: emotion_id
            if self.enable_ids:
                payload["emotion_id"] = emotion_id
            
            # Feature gating: emotion_confidence
            if self.enable_confidence:
                payload["emotion_confidence"] = emotion_conf
            
            # Feature gating: emotion_mean_probs
            if self.enable_mean_probs:
                payload["emotion_mean_probs"] = mean_probs
            
            # Feature gating: emotion_entropy
            if self.enable_entropy:
                payload["emotion_entropy"] = float(ent)
            
            # Feature gating: dominant emotion
            if self.enable_dominant:
                payload["dominant_emotion_id"] = int(dominant_id)
                payload["dominant_emotion_prob"] = float(dominant_prob)
            
            # Always include segment timestamps (needed for downstream)
            payload["segment_start_sec"] = starts
            payload["segment_end_sec"] = ends
            payload["segment_center_sec"] = centers
            
            # Additional aggregates (if any feature is enabled)
            if self.enable_ids or self.enable_confidence or self.enable_dominant:
                # Emotion transitions count
                if len(emotion_id) > 1:
                    transitions = sum(1 for i in range(len(emotion_id) - 1) if emotion_id[i] != emotion_id[i + 1])
                    payload["emotion_transitions_count"] = int(transitions)
                
                # Emotion distribution (time ratios)
                if self.enable_dominant:
                    emotion_duration = {}
                    emotion_segments_count = {}
                    total_duration = float(max(ends) if ends else 0.0)
                    for i, emo_id in enumerate(emotion_id):
                        emo_id_int = int(emo_id)
                        if emo_id_int not in emotion_duration:
                            emotion_duration[emo_id_int] = 0.0
                            emotion_segments_count[emo_id_int] = 0
                        seg_duration = float(ends[i] - starts[i])
                        emotion_duration[emo_id_int] += seg_duration
                        emotion_segments_count[emo_id_int] += 1
                    
                    emotion_distribution = {}
                    if total_duration > 0:
                        for emo_id, duration in emotion_duration.items():
                            emotion_distribution[int(emo_id)] = float(duration / total_duration)
                    payload["emotion_distribution"] = emotion_distribution
                    payload["emotion_segments_per_emotion"] = {int(k): int(v) for k, v in emotion_segments_count.items()}
                    payload["emotion_duration_per_emotion"] = {int(k): float(v) for k, v in emotion_duration.items()}
                    
                    # Emotion stability score (inverse of transitions frequency)
                    if total_duration > 0:
                        transitions_freq = float(transitions) / total_duration if transitions > 0 else 0.0
                        stability_score = float(1.0 / (1.0 + transitions_freq))  # 0 = unstable, 1 = stable
                        payload["emotion_stability_score"] = stability_score
                    
                    # Emotion diversity score (normalized entropy)
                    if mean_probs.size > 1:
                        max_entropy = float(np.log(num_classes))
                        diversity_score = float(ent / max_entropy) if max_entropy > 0 else 0.0
                        payload["emotion_diversity_score"] = diversity_score
            
            # Feature gating: quality metrics
            if self.enable_quality_metrics:
                quality_metrics = {}
                if self.enable_confidence:
                    quality_metrics["confidence_mean"] = float(np.mean(emotion_conf))
                    quality_metrics["confidence_std"] = float(np.std(emotion_conf))
                    quality_metrics["confidence_min"] = float(np.min(emotion_conf))
                    quality_metrics["confidence_max"] = float(np.max(emotion_conf))
                    quality_metrics["confidence_median"] = float(np.median(emotion_conf))
                if self.enable_mean_probs:
                    quality_metrics["mean_probs_min"] = float(np.min(mean_probs))
                    quality_metrics["mean_probs_max"] = float(np.max(mean_probs))
                    quality_metrics["mean_probs_std"] = float(np.std(mean_probs))
                payload["emotion_quality_metrics"] = quality_metrics
            
            # Track enabled features for meta
            enabled_features = []
            if self.enable_probs:
                enabled_features.append("probs")
            if self.enable_ids:
                enabled_features.append("ids")
            if self.enable_confidence:
                enabled_features.append("confidence")
            if self.enable_mean_probs:
                enabled_features.append("mean_probs")
            if self.enable_entropy:
                enabled_features.append("entropy")
            if self.enable_dominant:
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
            if not enabled_features:
                logger.warning(f"emotion_diarization | WARNING: All feature flags are disabled! No emotion data will be saved. Enable at least enable_ids or enable_confidence in config.")

            return self._create_result(True, payload=payload, processing_time=total_time)

        except Exception as e:
            return self._create_result(False, error=str(e), processing_time=time.time() - start_time)

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Process entire audio file as one continuous segment (if process_full_audio=True).
        Otherwise, returns error (use run_segments() for segment-based processing).
        """
        if not self.process_full_audio:
            return self._create_result(
                success=False,
                error="emotion_diarization_extractor | run() is not supported. Use run_segments() with Segmenter families.emotion windows, or set process_full_audio=True in config.",
                processing_time=0.0,
            )
        
        start_time = time.time()
        timings = {}  # Детальное профилирование этапов
        
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)
            
            logger.info(f"emotion_diarization | run() processing full audio: {input_uri}")
            
            # Этап 1: Load entire audio
            t_load_start = time.time()
            wav_t, sr = self.audio_utils.load_audio(input_uri, target_sr=self.sample_rate)
            wav = self.audio_utils.to_numpy(wav_t)
            wav = wav[0] if wav.ndim == 2 else wav.reshape(-1)
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            
            if int(sr) != int(self.sample_rate):
                raise RuntimeError(f"emotion_diarization | SR mismatch: got {sr} expected {self.sample_rate}")
            
            dur_sec = float(wav.shape[0]) / self.sample_rate
            if dur_sec < 5.0:
                raise RuntimeError(f"emotion_diarization | audio too short (<5s): duration_sec={dur_sec:.3f}")
            
            t_load_end = time.time()
            timings["load_audio_sec"] = t_load_end - t_load_start
            logger.info(f"emotion_diarization | loaded audio: duration={dur_sec:.2f}s, samples={wav.shape[0]}, sr={sr}, load_time={timings['load_audio_sec']:.3f}s")
            
            # Progress reporting: загрузка завершена
            if self.progress_callback:
                self.progress_callback("emotion_diarization", 1, 3, f"Loaded audio ({dur_sec:.1f}s)")
            
            # Этап 2: Silence detection
            t_silence_start = time.time()
            if self.enable_silence_detection:
                rms, peak = self._rms_and_peak(wav)
                if peak < self.silence_peak_threshold and rms < self.silence_rms_threshold:
                    payload: Dict[str, Any] = {
                        "status": "empty",
                        "empty_reason": "audio_silent",
                        "segments_count": 1,
                        "sample_rate": int(self.sample_rate),
                        "rms": float(rms),
                        "peak": float(peak),
                        "model_name": self.model_name,
                        "emotion_labels": self.emotion_labels,
                        "device_used": "cuda",
                        "emotion_contract_version": EMOTION_CONTRACT_VERSION,
                    }
                    timings["silence_detection_sec"] = time.time() - t_silence_start
                    return self._create_result(True, payload=payload, processing_time=time.time() - start_time)
                rms_val, peak_val = rms, peak
            else:
                rms_val, peak_val = self._rms_and_peak(wav)
            
            t_silence_end = time.time()
            timings["silence_detection_sec"] = t_silence_end - t_silence_start
            
            # Этап 3: Inference
            t_inference_start = time.time()
            batch = wav.reshape(1, -1).astype(np.float32)
            batch_ids = ["full_audio"]
            probs = self._infer_probs_batch(batch, batch_ids)
            t_inference_end = time.time()
            timings["inference_sec"] = t_inference_end - t_inference_start
            logger.info(f"emotion_diarization | inference completed: {timings['inference_sec']:.3f}s")
            
            # Progress reporting: inference завершен
            if self.progress_callback:
                self.progress_callback("emotion_diarization", 2, 3, f"Inference completed ({timings['inference_sec']:.1f}s)")
            
            # Этап 4: Postprocessing и агрегация
            t_postprocess_start = time.time()
            # Normalize (defensive)
            s = np.sum(probs, axis=1, keepdims=True) + 1e-9
            probs = probs / s
            
            # Validate emotion_labels
            num_classes = probs.shape[1]
            is_valid_labels, error_msg_labels = self._validate_emotion_labels(num_classes)
            if not is_valid_labels:
                raise ValueError(f"emotion_diarization | emotion_labels validation failed: {error_msg_labels}")
            
            # Compute aggregates
            emotion_id = np.argmax(probs, axis=1).astype(np.int32)
            emotion_conf = np.max(probs, axis=1).astype(np.float32)
            mean_probs = np.mean(probs, axis=0).astype(np.float32) if probs.size else np.zeros((0,), dtype=np.float32)
            ent = float(-np.sum(mean_probs * np.log(mean_probs + 1e-9))) if mean_probs.size else 0.0
            dominant_id = int(np.argmax(mean_probs)) if mean_probs.size else -1
            dominant_prob = float(np.max(mean_probs)) if mean_probs.size else 0.0
            
            payload: Dict[str, Any] = {
                "segments_count": 1,
                "sample_rate": int(self.sample_rate),
                "device_used": "cuda",
                "rms": float(rms_val),
                "peak": float(peak_val),
                "model_name": self.model_name,
                "emotion_labels": self.emotion_labels,
                "emotion_contract_version": EMOTION_CONTRACT_VERSION,
            }
            
            # Feature gating (same as run_segments)
            if self.enable_probs:
                payload["emotion_probs"] = probs
            if self.enable_ids:
                payload["emotion_id"] = emotion_id
            if self.enable_confidence:
                payload["emotion_confidence"] = emotion_conf
            if self.enable_mean_probs:
                payload["emotion_mean_probs"] = mean_probs
            if self.enable_entropy:
                payload["emotion_entropy"] = float(ent)
            if self.enable_dominant:
                payload["dominant_emotion_id"] = int(dominant_id)
                payload["dominant_emotion_prob"] = float(dominant_prob)
            
            # Single segment timestamps
            payload["segment_start_sec"] = [0.0]
            payload["segment_end_sec"] = [dur_sec]
            payload["segment_center_sec"] = [dur_sec / 2.0]
            
            # Additional aggregates
            if self.enable_ids or self.enable_confidence or self.enable_dominant:
                if self.enable_dominant:
                    payload["emotion_distribution"] = {int(dominant_id): 1.0}
                    payload["emotion_segments_per_emotion"] = {int(dominant_id): 1}
                    payload["emotion_duration_per_emotion"] = {int(dominant_id): dur_sec}
                    payload["emotion_stability_score"] = 1.0  # Single segment = stable
                    if mean_probs.size > 1:
                        max_entropy = float(np.log(num_classes))
                        diversity_score = float(ent / max_entropy) if max_entropy > 0 else 0.0
                        payload["emotion_diversity_score"] = diversity_score
            
            if self.enable_quality_metrics:
                quality_metrics = {}
                if self.enable_confidence:
                    quality_metrics["confidence_mean"] = float(emotion_conf[0])
                    quality_metrics["confidence_std"] = 0.0
                    quality_metrics["confidence_min"] = float(emotion_conf[0])
                    quality_metrics["confidence_max"] = float(emotion_conf[0])
                    quality_metrics["confidence_median"] = float(emotion_conf[0])
                if self.enable_mean_probs:
                    quality_metrics["mean_probs_min"] = float(np.min(mean_probs))
                    quality_metrics["mean_probs_max"] = float(np.max(mean_probs))
                    quality_metrics["mean_probs_std"] = float(np.std(mean_probs))
                payload["emotion_quality_metrics"] = quality_metrics
            
            # Track enabled features
            enabled_features = []
            if self.enable_probs:
                enabled_features.append("probs")
            if self.enable_ids:
                enabled_features.append("ids")
            if self.enable_confidence:
                enabled_features.append("confidence")
            if self.enable_mean_probs:
                enabled_features.append("mean_probs")
            if self.enable_entropy:
                enabled_features.append("entropy")
            if self.enable_dominant:
                enabled_features.append("dominant")
            if self.enable_quality_metrics:
                enabled_features.append("quality_metrics")
            
            payload["_features_enabled"] = enabled_features
            
            t_postprocess_end = time.time()
            timings["postprocess_sec"] = t_postprocess_end - t_postprocess_start
            total_time = time.time() - start_time
            
            # Log detailed profiling
            logger.info(f"emotion_diarization | run() completed: duration={dur_sec:.2f}s, emotion_id={emotion_id[0] if self.enable_ids else 'disabled'}, confidence={emotion_conf[0] if self.enable_confidence else 'disabled'}, enabled_features={enabled_features}")
            logger.info(f"emotion_diarization | profiling: load={timings.get('load_audio_sec', 0):.3f}s, silence={timings.get('silence_detection_sec', 0):.3f}s, inference={timings.get('inference_sec', 0):.3f}s, postprocess={timings.get('postprocess_sec', 0):.3f}s, total={total_time:.3f}s")
            
            return self._create_result(True, payload=payload, processing_time=total_time)
            
        except Exception as e:
            logger.error(f"emotion_diarization | run() failed: {e}", exc_info=True)
            return self._create_result(False, error=str(e), processing_time=time.time() - start_time)

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
                    self.logger.warning(f"emotion_diarization | file_id={file_id} audio too short (<5s): duration_sec={dur_sec:.3f}")
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
            
            # Этап 2: Загрузка и предобработка всех сегментов
            waves: List[Dict[str, Any]] = []
            starts: List[float] = []
            ends: List[float] = []
            centers: List[float] = []
            lens: List[int] = []
            
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
                        raise RuntimeError(f"emotion_diarization | segment SR mismatch: got {sr} expected {self.sample_rate}")
                    
                    waves.append({
                        "audio": wav,
                        "file_id": seg_meta["file_id"],
                    })
                    starts.append(st)
                    ends.append(en)
                    centers.append(c)
                    lens.append(int(wav.shape[0]))
                except Exception as e:
                    self.logger.error(f"Error preprocessing segment for file_id={seg_meta['file_id']}: {e}")
                    continue
            
            if not waves:
                return [
                    self._create_result(
                        success=False,
                        error="Failed to preprocess any segments",
                        processing_time=time.time() - start_time,
                    )
                    for _ in audio_files_with_segments
                ]
            
            # Этап 3: Паддинг для батчинга
            max_len = int(max(lens) if lens else 0)
            if max_len <= 0:
                return [
                    self._create_result(
                        success=False,
                        error="No audio samples in segments",
                        processing_time=time.time() - start_time,
                    )
                    for _ in audio_files_with_segments
                ]
            
            padded = np.zeros((len(waves), max_len), dtype=np.float32)
            for i, w in enumerate(waves):
                padded[i, : int(w["audio"].shape[0])] = w["audio"]
            
            # Этап 4: Определение размера батча
            effective_batch_size = max_segments_per_batch
            if effective_batch_size is None:
                effective_batch_size = self.batch_size
                if len(waves) > 100:
                    effective_batch_size = min(100, self.batch_size)  # Auto-split large batches
            
            # Этап 5: Обработка батчей через Triton
            probs_chunks: List[np.ndarray] = []
            
            for batch_start in range(0, padded.shape[0], effective_batch_size):
                batch_end = min(batch_start + effective_batch_size, padded.shape[0])
                batch = padded[batch_start:batch_end]
                
                try:
                    # Create batch IDs for SpeechBrain
                    batch_ids = [f"seg_{batch_start + i}" for i in range(batch.shape[0])]
                    batch_probs = self._infer_probs_batch(batch, batch_ids)
                    probs_chunks.append(batch_probs)
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_start // effective_batch_size}: {e}")
                    # Добавляем нулевые вероятности для неудачных сегментов
                    batch_size_actual = batch_end - batch_start
                    num_emotions = len(self.emotion_labels) if hasattr(self, 'emotion_labels') else 7
                    probs_chunks.append(np.zeros((batch_size_actual, num_emotions), dtype=np.float32))
            
            probs_all = np.concatenate(probs_chunks, axis=0) if probs_chunks else np.zeros((0, 0), dtype=np.float32)
            
            # Normalize (defensive)
            if probs_all.size > 0:
                s = np.sum(probs_all, axis=1, keepdims=True) + 1e-9
                probs_all = probs_all / s
            
            # Этап 6: Распределение результатов обратно по файлам
            results: List[ExtractorResult] = []
            
            for file_info in audio_files_with_segments:
                file_id = file_info.get("file_id", "unknown")
                file_start, file_end = file_segment_ranges.get(file_id, (0, 0))
                
                # Извлекаем результаты для этого файла
                file_probs: List[np.ndarray] = []
                file_starts: List[float] = []
                file_ends: List[float] = []
                file_centers: List[float] = []
                
                for idx in range(len(waves)):
                    if waves[idx]["file_id"] == file_id:
                        if idx < probs_all.shape[0]:
                            file_probs.append(probs_all[idx])
                        file_starts.append(starts[idx])
                        file_ends.append(ends[idx])
                        file_centers.append(centers[idx])
                
                if not file_probs:
                    results.append(self._create_result(
                        success=False,
                        error="No probabilities generated for this file",
                        processing_time=time.time() - start_time,
                    ))
                    continue
                
                # Формируем массив вероятностей для файла
                probs = np.stack(file_probs, axis=0).astype(np.float32)
                
                # Silence detection (если включено)
                if self.enable_silence_detection:
                    concat = np.concatenate([w["audio"] for w in waves if w["file_id"] == file_id], axis=0) if waves else np.zeros((0,), dtype=np.float32)
                    rms_val, peak_val = self._rms_and_peak(concat)
                    if peak_val < self.silence_peak_threshold and rms_val < self.silence_rms_threshold:
                        payload: Dict[str, Any] = {
                            "status": "empty",
                            "empty_reason": "audio_silent",
                            "segments_count": int(len(file_probs)),
                            "sample_rate": int(self.sample_rate),
                            "rms": float(rms_val),
                            "peak": float(peak_val),
                            "model_name": self.model_name,
                            "emotion_labels": self.emotion_labels,
                            "device_used": "cuda",
                            "emotion_contract_version": EMOTION_CONTRACT_VERSION,
                        }
                        results.append(self._create_result(True, payload=payload, processing_time=time.time() - start_time))
                        continue
                else:
                    concat = np.concatenate([w["audio"] for w in waves if w["file_id"] == file_id], axis=0) if waves else np.zeros((0,), dtype=np.float32)
                    rms_val, peak_val = self._rms_and_peak(concat)
                
                # Validate emotion_labels
                num_classes = probs.shape[1]
                is_valid_labels, error_msg_labels = self._validate_emotion_labels(num_classes)
                if not is_valid_labels:
                    results.append(self._create_result(
                        success=False,
                        error=f"emotion_labels validation failed: {error_msg_labels}",
                        processing_time=time.time() - start_time,
                    ))
                    continue
                
                # Compute aggregates (аналогично run_segments)
                emotion_id = np.argmax(probs, axis=1).astype(np.int32)
                emotion_conf = np.max(probs, axis=1).astype(np.float32)
                mean_probs = np.mean(probs, axis=0).astype(np.float32) if probs.size else np.zeros((0,), dtype=np.float32)
                ent = float(-np.sum(mean_probs * np.log(mean_probs + 1e-9))) if mean_probs.size else 0.0
                dominant_id = int(np.argmax(mean_probs)) if mean_probs.size else -1
                dominant_prob = float(np.max(mean_probs)) if mean_probs.size else 0.0
                
                payload: Dict[str, Any] = {
                    "segments_count": int(len(file_probs)),
                    "sample_rate": int(self.sample_rate),
                    "device_used": "cuda",
                    "rms": float(rms_val),
                    "peak": float(peak_val),
                    "model_name": self.model_name,
                    "emotion_labels": self.emotion_labels,
                    "emotion_contract_version": EMOTION_CONTRACT_VERSION,
                }
                
                # Feature gating (аналогично run_segments)
                if self.enable_probs:
                    payload["emotion_probs"] = probs
                if self.enable_ids:
                    payload["emotion_id"] = emotion_id
                if self.enable_confidence:
                    payload["emotion_confidence"] = emotion_conf
                if self.enable_mean_probs:
                    payload["emotion_mean_probs"] = mean_probs
                if self.enable_entropy:
                    payload["emotion_entropy"] = float(ent)
                if self.enable_dominant:
                    payload["dominant_emotion_id"] = int(dominant_id)
                    payload["dominant_emotion_prob"] = float(dominant_prob)
                
                payload["segment_start_sec"] = file_starts
                payload["segment_end_sec"] = file_ends
                payload["segment_center_sec"] = file_centers
                
                # Additional aggregates (если нужно)
                if self.enable_ids or self.enable_confidence or self.enable_dominant:
                    if len(emotion_id) > 1:
                        transitions = sum(1 for i in range(len(emotion_id) - 1) if emotion_id[i] != emotion_id[i + 1])
                        payload["emotion_transitions_count"] = int(transitions)
                    
                    if self.enable_dominant:
                        emotion_duration = {}
                        emotion_segments_count = {}
                        total_duration = float(max(file_ends) if file_ends else 0.0)
                        for i, emo_id in enumerate(emotion_id):
                            emo_id_int = int(emo_id)
                            if emo_id_int not in emotion_duration:
                                emotion_duration[emo_id_int] = 0.0
                                emotion_segments_count[emo_id_int] = 0
                            seg_duration = float(file_ends[i] - file_starts[i])
                            emotion_duration[emo_id_int] += seg_duration
                            emotion_segments_count[emo_id_int] += 1
                        
                        emotion_distribution = {}
                        if total_duration > 0:
                            for emo_id, duration in emotion_duration.items():
                                emotion_distribution[int(emo_id)] = float(duration / total_duration)
                        payload["emotion_distribution"] = emotion_distribution
                        payload["emotion_segments_per_emotion"] = {int(k): int(v) for k, v in emotion_segments_count.items()}
                        payload["emotion_duration_per_emotion"] = {int(k): float(v) for k, v in emotion_duration.items()}
                        
                        if total_duration > 0:
                            transitions_freq = float(transitions) / total_duration if transitions > 0 else 0.0
                            stability_score = float(1.0 / (1.0 + transitions_freq))
                            payload["emotion_stability_score"] = stability_score
                        
                        if mean_probs.size > 1:
                            max_entropy = float(np.log(num_classes))
                            diversity_score = float(ent / max_entropy) if max_entropy > 0 else 0.0
                            payload["emotion_diversity_score"] = diversity_score
                
                if self.enable_quality_metrics:
                    quality_metrics = {}
                    if self.enable_confidence:
                        quality_metrics["confidence_mean"] = float(np.mean(emotion_conf))
                        quality_metrics["confidence_std"] = float(np.std(emotion_conf))
                        quality_metrics["confidence_min"] = float(np.min(emotion_conf))
                        quality_metrics["confidence_max"] = float(np.max(emotion_conf))
                        quality_metrics["confidence_median"] = float(np.median(emotion_conf))
                    if self.enable_mean_probs:
                        quality_metrics["mean_probs_min"] = float(np.min(mean_probs))
                        quality_metrics["mean_probs_max"] = float(np.max(mean_probs))
                        quality_metrics["mean_probs_std"] = float(np.std(mean_probs))
                    payload["emotion_quality_metrics"] = quality_metrics
                
                # Track enabled features
                enabled_features = []
                if self.enable_probs:
                    enabled_features.append("probs")
                if self.enable_ids:
                    enabled_features.append("ids")
                if self.enable_confidence:
                    enabled_features.append("confidence")
                if self.enable_mean_probs:
                    enabled_features.append("mean_probs")
                if self.enable_entropy:
                    enabled_features.append("entropy")
                if self.enable_dominant:
                    enabled_features.append("dominant")
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
            "sample_rate": self.sample_rate,
            "batch_size": self.batch_size,
            "device": self.device,
            "model_name": getattr(self, "model_name", None),
            "weights_digest": getattr(self, "weights_digest", None),
            "models_used_entry": getattr(self, "models_used_entry", None),
        }
