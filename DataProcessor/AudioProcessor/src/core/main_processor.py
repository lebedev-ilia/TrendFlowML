"""
Основной процессор для координации работы экстракторов.
"""
import os
import sys
import time
import logging
import warnings
from typing import Dict, Any, List, Optional, Union
import threading
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .base_extractor import BaseExtractor, ExtractorResult
from .audio_utils import AudioUtils
from .audio_file_context import AudioFileContext

logger = logging.getLogger(__name__)

# Suppress noisy torch meshgrid warning about upcoming indexing arg requirement
try:
    warnings.filterwarnings(
        "ignore",
        message=r".*torch\.meshgrid:.*indexing.*",
        category=UserWarning,
        module=r"torch\.functional"
    )
except Exception:
    pass


class MainProcessor:
    """Основной процессор для обработки видео и извлечения аудио признаков."""
    
    def __init__(
        self,
        device: str = "auto",
        max_workers: int = 4,
        gpu_memory_limit: float = 0.8,
        sample_rate: int = 22050,
        asr_model_size: str = "small",
        asr_language: str = "auto",
        asr_temperature: float = 0.0,
        asr_beam_size: int = 5,
        asr_best_of: int = 1,
        asr_enable_fallback_decode: bool = False,
        asr_fallback_temperature: float = 0.4,
        asr_fallback_avg_logprob_threshold: float = -1.0,
        asr_save_segment_text: bool = False,
        asr_enable_token_sequences: bool = False,
        asr_enable_token_counts: bool = False,
        asr_enable_token_total: bool = False,
        asr_enable_token_density: bool = False,
        asr_enable_speech_rate: bool = False,
        asr_enable_lang_distribution: bool = False,
        asr_enable_segments_with_speech: bool = False,
        asr_enable_avg_segment_duration: bool = False,
        asr_enable_token_variance: bool = False,
        diarization_model_size: str = "small",
        diarization_batch_size: Optional[int] = None,
        diarization_clustering_method: str = "agglomerative",
        diarization_speaker_count_method: str = "heuristic",
        diarization_silence_peak_threshold: float = 1e-3,
        diarization_silence_rms_threshold: float = 1e-4,
        diar_enable_speaker_segments: bool = False,
        diar_enable_speaker_embeddings: bool = False,
        diar_enable_speaker_stats: bool = False,
        diar_enable_speaker_durations: bool = False,
        diar_enable_clustering_metrics: bool = False,
        diar_enable_segment_embeddings: bool = False,
        diar_enable_silence_detection: bool = True,
        emotion_model_size: str = "small",
        emotion_batch_size: int = 16,
        emotion_silence_peak_threshold: float = 1e-3,
        emotion_silence_rms_threshold: float = 1e-4,
        emotion_enable_probs: bool = False,
        emotion_enable_ids: bool = False,
        emotion_enable_confidence: bool = False,
        emotion_enable_mean_probs: bool = False,
        emotion_enable_entropy: bool = False,
        emotion_enable_dominant: bool = False,
        emotion_enable_quality_metrics: bool = False,
        emotion_enable_silence_detection: bool = True,
        emotion_process_full_audio: bool = False,
        source_separation_model_size: str = "large",
        sep_batch_size: int = 8,
        sep_silence_peak_threshold: float = 1e-3,
        sep_silence_rms_threshold: float = 1e-4,
        sep_enable_share_sequence: bool = False,
        sep_enable_energy_sequence: bool = False,
        sep_enable_share_mean: bool = False,
        sep_enable_share_std: bool = False,
        sep_enable_quality_metrics: bool = False,
        sep_enable_silence_detection: bool = True,
        speech_analysis_pitch_enabled: bool = False,
        speech_silence_peak_threshold: float = 1e-3,
        speech_silence_rms_threshold: float = 1e-4,
        speech_enable_asr_metrics: bool = False,
        speech_enable_diarization_metrics: bool = False,
        speech_enable_pitch_metrics: bool = False,
        speech_enable_silence_detection: bool = True,
        # Pitch extractor parameters
        pitch_sample_rate: int = 22050,
        pitch_fmin: float = 50.0,
        pitch_fmax: float = 2000.0,
        pitch_hop_length: int = 512,
        pitch_frame_length: int = 2048,
        pitch_backend: str = "classic",
        pitch_channel_mode: str = "first",
        pitch_torchcrepe_batch_size: int = 1,
        pitch_enable_basic_stats: bool = False,
        pitch_enable_stability_metrics: bool = False,
        pitch_enable_delta_features: bool = False,
        pitch_enable_method_stats: bool = False,
        pitch_enable_time_series: bool = False,
        # Spectral extractor parameters
        spectral_sample_rate: int = 22050,
        spectral_hop_length: int = 512,
        spectral_n_fft: int = 2048,
        spectral_average_channels: bool = True,
        spectral_keep_contrast_bands: bool = True,
        spectral_enable_normalization: bool = False,
        spectral_enable_basic_features: bool = False,
        spectral_enable_contrast: bool = False,
        spectral_enable_advanced_features: bool = False,
        spectral_enable_time_series: bool = False,
        # Quality extractor parameters
        quality_sample_rate: int = 22050,
        quality_frame_len_ms: float = 50.0,
        quality_hop_ms: float = 25.0,
        quality_clip_threshold: float = 0.999,
        quality_average_channels: bool = True,
        quality_enable_normalization: bool = False,
        quality_enable_basic_metrics: bool = False,
        quality_enable_dynamic_metrics: bool = False,
        quality_enable_frame_analysis: bool = False,
        quality_enable_time_series: bool = False,
        # MFCC extractor parameters
        mfcc_sample_rate: int = 22050,
        mfcc_n_mfcc: int = 13,
        mfcc_n_fft: int = 2048,
        mfcc_hop_length: int = 512,
        mfcc_n_mels: int = 128,
        mfcc_fmin: float = 0.0,
        mfcc_fmax: Optional[float] = None,
        mfcc_enable_audio_normalization: bool = True,
        mfcc_min_gpu_duration_sec: float = 3.0,
        mfcc_min_gpu_file_size_mb: float = 5.0,
        mfcc_enable_basic_features: bool = False,
        mfcc_enable_deltas: bool = False,
        mfcc_enable_time_series: bool = False,
        mfcc_enable_normalization: bool = False,
        # Mel extractor parameters
        mel_sample_rate: int = 22050,
        mel_n_fft: int = 2048,
        mel_hop_length: int = 512,
        mel_n_mels: int = 128,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None,
        mel_power: float = 2.0,
        mel_mix_to_mono: bool = True,
        mel_enable_audio_normalization: bool = True,
        mel_enable_basic_features: bool = False,
        mel_enable_statistics: bool = False,
        mel_enable_spectral_features: bool = False,
        mel_enable_time_series: bool = False,
        mel_enable_stats_vector: bool = False,
        onset_sample_rate: int = 22050,
        onset_hop_length: int = 512,
        onset_pre_max: int = 3,
        onset_post_max: int = 3,
        onset_pre_avg: int = 3,
        onset_post_avg: int = 5,
        onset_delta: float = 0.2,
        onset_wait: int = 10,
        onset_backend: str = "librosa",
        onset_units: str = "time",
        onset_backtrack: bool = False,
        onset_energy: bool = False,
        onset_normalize: bool = False,
        onset_enable_audio_normalization: bool = False,
        onset_enable_basic_features: bool = False,
        onset_enable_interval_stats: bool = False,
        onset_enable_rhythmic_metrics: bool = False,
        onset_enable_time_series: bool = False,
        chroma_sample_rate: int = 22050,
        chroma_hop_length: int = 512,
        chroma_n_fft: int = 4096,
        chroma_mix_to_mono: bool = True,
        chroma_type: str = "cqt",
        chroma_normalize: Optional[str] = "l1",
        chroma_n_chroma: int = 12,
        chroma_fmin: Optional[float] = None,
        chroma_fmax: Optional[float] = None,
        chroma_n_bins: Optional[int] = None,
        chroma_enable_audio_normalization: bool = False,
        chroma_enable_basic_stats: bool = False,
        chroma_enable_extended_stats: bool = False,
        chroma_enable_stats_vector: bool = False,
        chroma_enable_time_series: bool = False,
        rhythmic_sample_rate: int = 22050,
        rhythmic_hop_length: int = 512,
        rhythmic_backend: str = "librosa",
        rhythmic_start_bpm: Optional[float] = None,
        rhythmic_std_bpm: Optional[float] = None,
        rhythmic_ac_size: int = 4,
        rhythmic_max_tempo: Optional[float] = None,
        rhythmic_enable_audio_normalization: bool = False,
        rhythmic_enable_basic_metrics: bool = False,
        rhythmic_enable_interval_stats: bool = False,
        rhythmic_enable_regularity_metrics: bool = False,
        rhythmic_enable_beat_times: bool = False,
        rhythmic_enable_tempo_metrics: bool = False,
        # Key extractor parameters
        key_sample_rate: int = 22050,
        key_hop_length: int = 512,
        key_chroma_type: str = "cqt",
        key_use_beat_sync: bool = False,
        key_top_k: int = 3,
        key_method: str = "auto",
        key_confidence_threshold: float = 0.3,
        key_enable_audio_normalization: bool = False,
        key_enable_detailed_scores: bool = False,
        key_enable_top_k: bool = False,
        key_enable_time_series: bool = False,
        key_enable_key_changes: bool = False,
        key_enable_stability_metrics: bool = False,
        # Band energy extractor parameters
        band_energy_sample_rate: int = 22050,
        band_energy_n_fft: int = 2048,
        band_energy_hop_length: int = 512,
        band_energy_use_mel_bands: bool = True,
        band_energy_n_mels: int = 3,
        band_energy_method: str = "auto",
        band_energy_average_channels: bool = True,
        band_energy_enable_audio_normalization: bool = False,
        band_energy_enable_basic_stats: bool = False,
        band_energy_enable_extended_stats: bool = False,
        band_energy_enable_time_series: bool = False,
        band_energy_enable_dynamics: bool = False,
        band_energy_enable_balance_metrics: bool = False,
        # Spectral entropy extractor parameters
        spectral_entropy_sample_rate: int = 22050,
        spectral_entropy_n_fft: int = 2048,
        spectral_entropy_hop_length: int = 512,
        spectral_entropy_average_channels: bool = True,
        spectral_entropy_smoothing_window: int = 0,
        spectral_entropy_use_mel: bool = False,
        spectral_entropy_n_mels: int = 128,
        spectral_entropy_enable_audio_normalization: bool = False,
        spectral_entropy_enable_basic_stats: bool = False,
        spectral_entropy_enable_flatness: bool = False,
        spectral_entropy_enable_spread: bool = False,
        spectral_entropy_enable_time_series: bool = False,
        spectral_entropy_enable_extended_stats: bool = False,
        spectral_entropy_enable_dynamics: bool = False,
        # Voice quality extractor parameters
        voice_quality_sample_rate: int = 22050,
        voice_quality_hnr_frame_ms: float = 40.0,
        voice_quality_rms_mask_threshold: float = 0.01,
        voice_quality_f0_fmin: float = 50.0,
        voice_quality_f0_fmax: float = 500.0,
        voice_quality_f0_method: str = "yin",
        voice_quality_average_channels: bool = True,
        voice_quality_enable_audio_normalization: bool = False,
        voice_quality_enable_jitter: bool = False,
        voice_quality_enable_shimmer: bool = False,
        voice_quality_enable_hnr: bool = False,
        voice_quality_enable_f0_stats: bool = False,
        voice_quality_enable_time_series: bool = False,
        # HPSS extractor parameters
        hpss_sample_rate: int = 22050,
        hpss_n_fft: int = 2048,
        hpss_hop_length: int = 512,
        hpss_average_channels: bool = True,
        hpss_kernel_size: int = 31,
        hpss_margin: float = 1.0,
        hpss_power: float = 2.0,
        hpss_enable_audio_normalization: bool = False,
        hpss_enable_energy_metrics: bool = False,
        hpss_enable_waveforms: bool = False,
        hpss_enable_spectral_features: bool = False,
        hpss_enable_time_series: bool = False,
        save_debug_results: bool = False,
        enabled_extractors: Optional[List[str]] = None,
        write_legacy_manifest: bool = True,
        # Batch processing parameters (Stage 4)
        batch_max_video_workers: Optional[int] = None,  # None = auto (os.cpu_count())
        batch_enable_gpu_batching: bool = False,  # Enable GPU batching for segments
        batch_enable_cpu_parallel: bool = False,  # Enable CPU parallelism for CPU extractors
        batch_max_segments_per_gpu_batch: Optional[int] = None,  # Max segments per GPU batch (None = use extractor's default)
    ):
        """
        Инициализация основного процессора.
        
        Args:
            device: Устройство для обработки ('cuda', 'cpu', 'auto')
            max_workers: Максимальное количество воркеров
            gpu_memory_limit: Лимит памяти GPU (0.0-1.0)
            sample_rate: Частота дискретизации
        """
        self.device = device
        self.max_workers = max_workers
        self.gpu_memory_limit = gpu_memory_limit
        self.sample_rate = sample_rate
        self.asr_model_size = str(asr_model_size or "small")
        self.asr_language = str(asr_language or "auto")
        self.asr_temperature = float(asr_temperature)
        self.asr_beam_size = int(asr_beam_size)
        self.asr_best_of = int(asr_best_of)
        self.asr_enable_fallback_decode = bool(asr_enable_fallback_decode)
        self.asr_fallback_temperature = float(asr_fallback_temperature)
        self.asr_fallback_avg_logprob_threshold = float(asr_fallback_avg_logprob_threshold)
        self.asr_save_segment_text = bool(asr_save_segment_text)
        self.asr_enable_token_sequences = bool(asr_enable_token_sequences)
        self.asr_enable_token_counts = bool(asr_enable_token_counts)
        self.asr_enable_token_total = bool(asr_enable_token_total)
        self.asr_enable_token_density = bool(asr_enable_token_density)
        self.asr_enable_speech_rate = bool(asr_enable_speech_rate)
        self.asr_enable_lang_distribution = bool(asr_enable_lang_distribution)
        self.asr_enable_segments_with_speech = bool(asr_enable_segments_with_speech)
        self.asr_enable_avg_segment_duration = bool(asr_enable_avg_segment_duration)
        self.asr_enable_token_variance = bool(asr_enable_token_variance)
        self.diarization_model_size = str(diarization_model_size or "small")
        self.diarization_batch_size = diarization_batch_size
        self.diarization_clustering_method = str(diarization_clustering_method or "agglomerative")
        self.diarization_speaker_count_method = str(diarization_speaker_count_method or "heuristic")
        self.diarization_silence_peak_threshold = float(diarization_silence_peak_threshold)
        self.diarization_silence_rms_threshold = float(diarization_silence_rms_threshold)
        self.diar_enable_speaker_segments = bool(diar_enable_speaker_segments)
        self.diar_enable_speaker_embeddings = bool(diar_enable_speaker_embeddings)
        self.diar_enable_speaker_stats = bool(diar_enable_speaker_stats)
        self.diar_enable_speaker_durations = bool(diar_enable_speaker_durations)
        self.diar_enable_clustering_metrics = bool(diar_enable_clustering_metrics)
        self.diar_enable_segment_embeddings = bool(diar_enable_segment_embeddings)
        self.diar_enable_silence_detection = bool(diar_enable_silence_detection)
        self.emotion_model_size = str(emotion_model_size or "small")
        self.emotion_batch_size = int(emotion_batch_size or 16)
        self.emotion_silence_peak_threshold = float(emotion_silence_peak_threshold)
        self.emotion_silence_rms_threshold = float(emotion_silence_rms_threshold)
        self.emotion_enable_probs = bool(emotion_enable_probs)
        self.emotion_enable_ids = bool(emotion_enable_ids)
        self.emotion_enable_confidence = bool(emotion_enable_confidence)
        self.emotion_enable_mean_probs = bool(emotion_enable_mean_probs)
        self.emotion_enable_entropy = bool(emotion_enable_entropy)
        self.emotion_enable_dominant = bool(emotion_enable_dominant)
        self.emotion_enable_quality_metrics = bool(emotion_enable_quality_metrics)
        self.emotion_enable_silence_detection = bool(emotion_enable_silence_detection)
        self.emotion_process_full_audio = bool(emotion_process_full_audio)
        self.source_separation_model_size = str(source_separation_model_size or "large")
        self.sep_batch_size = int(sep_batch_size)
        self.sep_silence_peak_threshold = float(sep_silence_peak_threshold)
        self.sep_silence_rms_threshold = float(sep_silence_rms_threshold)
        self.sep_enable_share_sequence = bool(sep_enable_share_sequence)
        self.sep_enable_energy_sequence = bool(sep_enable_energy_sequence)
        self.sep_enable_share_mean = bool(sep_enable_share_mean)
        self.sep_enable_share_std = bool(sep_enable_share_std)
        self.sep_enable_quality_metrics = bool(sep_enable_quality_metrics)
        self.sep_enable_silence_detection = bool(sep_enable_silence_detection)
        self.speech_analysis_pitch_enabled = bool(speech_analysis_pitch_enabled)
        self.speech_silence_peak_threshold = float(speech_silence_peak_threshold)
        self.speech_silence_rms_threshold = float(speech_silence_rms_threshold)
        self.speech_enable_asr_metrics = bool(speech_enable_asr_metrics)
        self.speech_enable_diarization_metrics = bool(speech_enable_diarization_metrics)
        self.speech_enable_pitch_metrics = bool(speech_enable_pitch_metrics)
        self.speech_enable_silence_detection = bool(speech_enable_silence_detection)
        # Pitch extractor parameters
        self.pitch_sample_rate = int(pitch_sample_rate)
        self.pitch_fmin = float(pitch_fmin)
        self.pitch_fmax = float(pitch_fmax)
        self.pitch_hop_length = int(pitch_hop_length)
        self.pitch_frame_length = int(pitch_frame_length)
        self.pitch_backend = str(pitch_backend)
        self.pitch_channel_mode = str(pitch_channel_mode)
        self.pitch_torchcrepe_batch_size = int(pitch_torchcrepe_batch_size)
        self.pitch_enable_basic_stats = bool(pitch_enable_basic_stats)
        self.pitch_enable_stability_metrics = bool(pitch_enable_stability_metrics)
        self.pitch_enable_delta_features = bool(pitch_enable_delta_features)
        self.pitch_enable_method_stats = bool(pitch_enable_method_stats)
        self.pitch_enable_time_series = bool(pitch_enable_time_series)
        # Spectral extractor parameters
        self.spectral_sample_rate = int(spectral_sample_rate)
        self.spectral_hop_length = int(spectral_hop_length)
        self.spectral_n_fft = int(spectral_n_fft)
        self.spectral_average_channels = bool(spectral_average_channels)
        self.spectral_keep_contrast_bands = bool(spectral_keep_contrast_bands)
        self.spectral_enable_normalization = bool(spectral_enable_normalization)
        self.spectral_enable_basic_features = bool(spectral_enable_basic_features)
        self.spectral_enable_contrast = bool(spectral_enable_contrast)
        self.spectral_enable_advanced_features = bool(spectral_enable_advanced_features)
        self.spectral_enable_time_series = bool(spectral_enable_time_series)
        # Quality extractor parameters
        self.quality_sample_rate = int(quality_sample_rate)
        self.quality_frame_len_ms = float(quality_frame_len_ms)
        self.quality_hop_ms = float(quality_hop_ms)
        self.quality_clip_threshold = float(quality_clip_threshold)
        self.quality_average_channels = bool(quality_average_channels)
        self.quality_enable_normalization = bool(quality_enable_normalization)
        self.quality_enable_basic_metrics = bool(quality_enable_basic_metrics)
        self.quality_enable_dynamic_metrics = bool(quality_enable_dynamic_metrics)
        self.quality_enable_frame_analysis = bool(quality_enable_frame_analysis)
        self.quality_enable_time_series = bool(quality_enable_time_series)
        # MFCC extractor parameters
        self.mfcc_sample_rate = int(mfcc_sample_rate)
        self.mfcc_n_mfcc = int(mfcc_n_mfcc)
        self.mfcc_n_fft = int(mfcc_n_fft)
        self.mfcc_hop_length = int(mfcc_hop_length)
        self.mfcc_n_mels = int(mfcc_n_mels)
        self.mfcc_fmin = float(mfcc_fmin)
        self.mfcc_fmax = float(mfcc_fmax) if mfcc_fmax is not None else float(mfcc_sample_rate // 2)
        self.mfcc_enable_audio_normalization = bool(mfcc_enable_audio_normalization)
        self.mfcc_min_gpu_duration_sec = float(mfcc_min_gpu_duration_sec)
        self.mfcc_min_gpu_file_size_mb = float(mfcc_min_gpu_file_size_mb)
        self.mfcc_enable_basic_features = bool(mfcc_enable_basic_features)
        self.mfcc_enable_deltas = bool(mfcc_enable_deltas)
        self.mfcc_enable_time_series = bool(mfcc_enable_time_series)
        self.mfcc_enable_normalization = bool(mfcc_enable_normalization)
        # Mel extractor parameters
        self.mel_sample_rate = int(mel_sample_rate)
        self.mel_n_fft = int(mel_n_fft)
        self.mel_hop_length = int(mel_hop_length)
        self.mel_n_mels = int(mel_n_mels)
        self.mel_fmin = float(mel_fmin)
        self.mel_fmax = float(mel_fmax) if mel_fmax is not None else float(mel_sample_rate // 2)
        self.mel_power = float(mel_power)
        self.mel_mix_to_mono = bool(mel_mix_to_mono)
        self.mel_enable_audio_normalization = bool(mel_enable_audio_normalization)
        self.mel_enable_basic_features = bool(mel_enable_basic_features)
        self.mel_enable_statistics = bool(mel_enable_statistics)
        self.mel_enable_spectral_features = bool(mel_enable_spectral_features)
        self.mel_enable_time_series = bool(mel_enable_time_series)
        self.mel_enable_stats_vector = bool(mel_enable_stats_vector)
        self.onset_sample_rate = int(onset_sample_rate)
        self.onset_hop_length = int(onset_hop_length)
        self.onset_pre_max = int(onset_pre_max)
        self.onset_post_max = int(onset_post_max)
        self.onset_pre_avg = int(onset_pre_avg)
        self.onset_post_avg = int(onset_post_avg)
        self.onset_delta = float(onset_delta)
        self.onset_wait = int(onset_wait)
        self.onset_backend = str(onset_backend)
        self.onset_units = str(onset_units)
        self.onset_backtrack = bool(onset_backtrack)
        self.onset_energy = bool(onset_energy)
        self.onset_normalize = bool(onset_normalize)
        self.onset_enable_audio_normalization = bool(onset_enable_audio_normalization)
        self.onset_enable_basic_features = bool(onset_enable_basic_features)
        self.onset_enable_interval_stats = bool(onset_enable_interval_stats)
        self.onset_enable_rhythmic_metrics = bool(onset_enable_rhythmic_metrics)
        self.onset_enable_time_series = bool(onset_enable_time_series)
        self.chroma_sample_rate = int(chroma_sample_rate)
        self.chroma_hop_length = int(chroma_hop_length)
        self.chroma_n_fft = int(chroma_n_fft)
        self.chroma_mix_to_mono = bool(chroma_mix_to_mono)
        self.chroma_type = str(chroma_type)
        self.chroma_normalize = chroma_normalize
        self.chroma_n_chroma = int(chroma_n_chroma)
        self.chroma_fmin = float(chroma_fmin) if chroma_fmin is not None else None
        self.chroma_fmax = float(chroma_fmax) if chroma_fmax is not None else None
        self.chroma_n_bins = int(chroma_n_bins) if chroma_n_bins is not None else None
        self.chroma_enable_audio_normalization = bool(chroma_enable_audio_normalization)
        self.chroma_enable_basic_stats = bool(chroma_enable_basic_stats)
        self.chroma_enable_extended_stats = bool(chroma_enable_extended_stats)
        self.chroma_enable_stats_vector = bool(chroma_enable_stats_vector)
        self.chroma_enable_time_series = bool(chroma_enable_time_series)
        self.rhythmic_sample_rate = int(rhythmic_sample_rate)
        self.rhythmic_hop_length = int(rhythmic_hop_length)
        self.rhythmic_backend = str(rhythmic_backend)
        self.rhythmic_start_bpm = rhythmic_start_bpm
        self.rhythmic_std_bpm = rhythmic_std_bpm
        self.rhythmic_ac_size = int(rhythmic_ac_size)
        self.rhythmic_max_tempo = rhythmic_max_tempo
        self.rhythmic_enable_audio_normalization = bool(rhythmic_enable_audio_normalization)
        self.rhythmic_enable_basic_metrics = bool(rhythmic_enable_basic_metrics)
        self.rhythmic_enable_interval_stats = bool(rhythmic_enable_interval_stats)
        self.rhythmic_enable_regularity_metrics = bool(rhythmic_enable_regularity_metrics)
        self.rhythmic_enable_beat_times = bool(rhythmic_enable_beat_times)
        self.rhythmic_enable_tempo_metrics = bool(rhythmic_enable_tempo_metrics)
        # Key extractor parameters
        self.key_sample_rate = int(key_sample_rate)
        self.key_hop_length = int(key_hop_length)
        self.key_chroma_type = str(key_chroma_type)
        self.key_use_beat_sync = bool(key_use_beat_sync)
        self.key_top_k = int(key_top_k)
        self.key_method = str(key_method)
        self.key_confidence_threshold = float(key_confidence_threshold)
        self.key_enable_audio_normalization = bool(key_enable_audio_normalization)
        self.key_enable_detailed_scores = bool(key_enable_detailed_scores)
        self.key_enable_top_k = bool(key_enable_top_k)
        self.key_enable_time_series = bool(key_enable_time_series)
        self.key_enable_key_changes = bool(key_enable_key_changes)
        self.key_enable_stability_metrics = bool(key_enable_stability_metrics)
        # Band energy extractor parameters
        self.band_energy_sample_rate = int(band_energy_sample_rate)
        self.band_energy_n_fft = int(band_energy_n_fft)
        self.band_energy_hop_length = int(band_energy_hop_length)
        self.band_energy_use_mel_bands = bool(band_energy_use_mel_bands)
        self.band_energy_average_channels = bool(band_energy_average_channels)
        self.band_energy_n_mels = int(band_energy_n_mels)
        self.band_energy_method = str(band_energy_method)
        self.band_energy_enable_audio_normalization = bool(band_energy_enable_audio_normalization)
        self.band_energy_enable_basic_stats = bool(band_energy_enable_basic_stats)
        self.band_energy_enable_extended_stats = bool(band_energy_enable_extended_stats)
        self.band_energy_enable_time_series = bool(band_energy_enable_time_series)
        self.band_energy_enable_dynamics = bool(band_energy_enable_dynamics)
        self.band_energy_enable_balance_metrics = bool(band_energy_enable_balance_metrics)
        # Spectral entropy extractor parameters
        self.spectral_entropy_sample_rate = int(spectral_entropy_sample_rate)
        self.spectral_entropy_n_fft = int(spectral_entropy_n_fft)
        self.spectral_entropy_hop_length = int(spectral_entropy_hop_length)
        self.spectral_entropy_average_channels = bool(spectral_entropy_average_channels)
        self.spectral_entropy_smoothing_window = int(spectral_entropy_smoothing_window)
        self.spectral_entropy_use_mel = bool(spectral_entropy_use_mel)
        self.spectral_entropy_n_mels = int(spectral_entropy_n_mels)
        self.spectral_entropy_enable_audio_normalization = bool(spectral_entropy_enable_audio_normalization)
        self.spectral_entropy_enable_basic_stats = bool(spectral_entropy_enable_basic_stats)
        self.spectral_entropy_enable_flatness = bool(spectral_entropy_enable_flatness)
        self.spectral_entropy_enable_spread = bool(spectral_entropy_enable_spread)
        self.spectral_entropy_enable_time_series = bool(spectral_entropy_enable_time_series)
        self.spectral_entropy_enable_extended_stats = bool(spectral_entropy_enable_extended_stats)
        self.spectral_entropy_enable_dynamics = bool(spectral_entropy_enable_dynamics)
        # Voice quality extractor parameters
        self.voice_quality_sample_rate = int(voice_quality_sample_rate)
        self.voice_quality_hnr_frame_ms = float(voice_quality_hnr_frame_ms)
        self.voice_quality_rms_mask_threshold = float(voice_quality_rms_mask_threshold)
        self.voice_quality_f0_fmin = float(voice_quality_f0_fmin)
        self.voice_quality_f0_fmax = float(voice_quality_f0_fmax)
        self.voice_quality_f0_method = str(voice_quality_f0_method)
        self.voice_quality_average_channels = bool(voice_quality_average_channels)
        self.voice_quality_enable_audio_normalization = bool(voice_quality_enable_audio_normalization)
        self.voice_quality_enable_jitter = bool(voice_quality_enable_jitter)
        self.voice_quality_enable_shimmer = bool(voice_quality_enable_shimmer)
        self.voice_quality_enable_hnr = bool(voice_quality_enable_hnr)
        self.voice_quality_enable_f0_stats = bool(voice_quality_enable_f0_stats)
        self.voice_quality_enable_time_series = bool(voice_quality_enable_time_series)
        # HPSS extractor parameters
        self.hpss_sample_rate = int(hpss_sample_rate)
        self.hpss_n_fft = int(hpss_n_fft)
        self.hpss_hop_length = int(hpss_hop_length)
        self.hpss_average_channels = bool(hpss_average_channels)
        self.hpss_kernel_size = int(hpss_kernel_size)
        self.hpss_margin = float(hpss_margin)
        self.hpss_power = float(hpss_power)
        self.hpss_enable_audio_normalization = bool(hpss_enable_audio_normalization)
        self.hpss_enable_energy_metrics = bool(hpss_enable_energy_metrics)
        self.hpss_enable_waveforms = bool(hpss_enable_waveforms)
        self.hpss_enable_spectral_features = bool(hpss_enable_spectral_features)
        self.hpss_enable_time_series = bool(hpss_enable_time_series)
        self.save_debug_results = save_debug_results
        self.write_legacy_manifest = bool(write_legacy_manifest)
        
        # Batch processing parameters (Stage 4)
        if batch_max_video_workers is None:
            import os
            batch_max_video_workers = os.cpu_count() or 4
        self._batch_max_video_workers = int(batch_max_video_workers)
        self._batch_enable_gpu_batching = bool(batch_enable_gpu_batching)
        self._batch_enable_cpu_parallel = bool(batch_enable_cpu_parallel)
        self._batch_max_segments_per_gpu_batch = batch_max_segments_per_gpu_batch
        
        self.logger = logging.getLogger(f"{__name__}.MainProcessor")
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        
        # Progress callback (will be set later in extractor_runner)
        self._progress_callback = None
        
        # Реестр экстракторов
        self.extractors: Dict[str, BaseExtractor] = {}
        
        # Инициализируем только запрошенные экстракторы (или все по умолчанию)
        self._initialize_extractors(enabled_extractors)

    # -------------------- System Monitor --------------------
    class _SystemMonitor:
        def __init__(self, logger: logging.Logger, sample_interval: float = 5,
                     cpu_threshold: float = 95.0, ram_threshold: float = 95.0,
                     gpu_threshold: float = 95.0):
            self.logger = logger
            self.sample_interval = sample_interval
            self.cpu_thr = cpu_threshold
            self.ram_thr = ram_threshold
            self.gpu_thr = gpu_threshold
            self._stop_event = threading.Event()
            self._thread: Optional[threading.Thread] = None
            self.exceeded: Optional[str] = None

            # maxima
            self.max_cpu: float = 0.0
            self.max_ram: float = 0.0
            self.max_gpu_util: float = 0.0
            self.max_gpu_mem_pct: float = 0.0

        def _get_gpu_stats(self) -> Dict[str, float]:
            util = 0.0
            mem_pct = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    try:
                        import pynvml  # type: ignore
                        pynvml.nvmlInit()
                        h = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util_s = pynvml.nvmlDeviceGetUtilizationRates(h)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                        util = float(util_s.gpu)
                        mem_pct = float(mem.used) / float(mem.total) * 100.0 if mem.total else 0.0
                        pynvml.nvmlShutdown()
                    except Exception:
                        # Fallback by memory percent only
                        total = torch.cuda.get_device_properties(0).total_memory
                        used = torch.cuda.memory_allocated(0)
                        mem_pct = float(used) / float(total) * 100.0 if total else 0.0
                        util = 0.0
            except Exception:
                pass
            return {"gpu_util": util, "gpu_mem_pct": mem_pct}

        def _loop(self):
            if psutil is None:
                # psutil is optional; disable monitoring if not installed.
                return
            # Prime cpu_percent
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass
            while not self._stop_event.wait(self.sample_interval):
                try:
                    cpu = float(psutil.cpu_percent(interval=None))
                    ram = float(psutil.virtual_memory().percent)
                    gpu_stats = self._get_gpu_stats()
                    gpu_util = gpu_stats["gpu_util"]
                    gpu_mem_pct = gpu_stats["gpu_mem_pct"]

                    # update maxima
                    if cpu > self.max_cpu:
                        self.max_cpu = cpu
                    if ram > self.max_ram:
                        self.max_ram = ram
                    if gpu_util > self.max_gpu_util:
                        self.max_gpu_util = gpu_util
                    if gpu_mem_pct > self.max_gpu_mem_pct:
                        self.max_gpu_mem_pct = gpu_mem_pct

                    # periodic info
                    self.logger.info(
                        f"SYS cpu={cpu:.1f}% ram={ram:.1f}% gpu_util={gpu_util:.1f}% gpu_mem={gpu_mem_pct:.1f}%"
                    )

                    # threshold check
                    if self.exceeded is None:
                        if cpu >= self.cpu_thr:
                            self.exceeded = f"CPU usage {cpu:.1f}% >= {self.cpu_thr:.1f}%"
                        elif ram >= self.ram_thr:
                            self.exceeded = f"RAM usage {ram:.1f}% >= {self.ram_thr:.1f}%"
                        elif gpu_util >= self.gpu_thr and gpu_util > 0:
                            self.exceeded = f"GPU util {gpu_util:.1f}% >= {self.gpu_thr:.1f}%"
                        elif gpu_mem_pct >= self.gpu_thr and gpu_mem_pct > 0:
                            self.exceeded = f"GPU mem {gpu_mem_pct:.1f}% >= {self.gpu_thr:.1f}%"

                except Exception:
                    # do not break monitoring on transient errors
                    pass

        def start(self):
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, name="SystemMonitor", daemon=True)
            self._thread.start()

        def stop(self):
            self._stop_event.set()
            if self._thread is not None:
                try:
                    self._thread.join(timeout=1.0)
                except Exception:
                    pass
    
    def _initialize_extractors(self, enabled_extractors: Optional[List[str]] = None):
        """Инициализация только выбранных экстракторов (ленивая загрузка модулей)."""
        try:
            # Фабрики экстракторов: импорт внутри, чтобы не грузить лишние зависимости
            extractor_factories: Dict[str, Any] = {
                "mfcc": lambda: __import__(
                    "src.extractors.mfcc_extractor", fromlist=["MFCCExtractor"]
                ).MFCCExtractor(device=self.device, sample_rate=self.sample_rate),
                "mel": lambda: __import__(
                    "src.extractors.mel_extractor", fromlist=["MelExtractor"]
                ).MelExtractor(
                    device=self.device,
                    sample_rate=self.mel_sample_rate,
                    n_fft=self.mel_n_fft,
                    hop_length=self.mel_hop_length,
                    n_mels=self.mel_n_mels,
                    fmin=self.mel_fmin,
                    fmax=self.mel_fmax,
                    power=self.mel_power,
                    mix_to_mono=self.mel_mix_to_mono,
                    enable_audio_normalization=self.mel_enable_audio_normalization,
                    enable_basic_features=self.mel_enable_basic_features,
                    enable_statistics=self.mel_enable_statistics,
                    enable_spectral_features=self.mel_enable_spectral_features,
                    enable_time_series=self.mel_enable_time_series,
                    enable_stats_vector=self.mel_enable_stats_vector,
                ),
                "clap": lambda: __import__(
                    "src.extractors.clap_extractor", fromlist=["CLAPExtractor"]
                ).CLAPExtractor(device=self.device, sample_rate=48000),
                "asr": lambda: __import__(
                    "src.extractors.asr_extractor", fromlist=["ASRExtractor"]
                ).ASRExtractor(
                    device=self.device,
                    model_size=self.asr_model_size,
                    sample_rate=16000,
                    language=self.asr_language,
                    temperature=self.asr_temperature,
                    beam_size=self.asr_beam_size,
                    best_of=self.asr_best_of,
                    enable_fallback_decode=self.asr_enable_fallback_decode,
                    fallback_temperature=self.asr_fallback_temperature,
                    fallback_avg_logprob_threshold=self.asr_fallback_avg_logprob_threshold,
                    save_segment_text=self.asr_save_segment_text,
                    enable_token_sequences=self.asr_enable_token_sequences,
                    enable_token_counts=self.asr_enable_token_counts,
                    enable_token_total=self.asr_enable_token_total,
                    enable_token_density=self.asr_enable_token_density,
                    enable_speech_rate=self.asr_enable_speech_rate,
                    enable_lang_distribution=self.asr_enable_lang_distribution,
                    enable_segments_with_speech=self.asr_enable_segments_with_speech,
                    enable_avg_segment_duration=self.asr_enable_avg_segment_duration,
                    enable_token_variance=self.asr_enable_token_variance,
                ),
                "speaker_diarization": lambda: __import__(
                    "src.extractors.speaker_diarization_extractor", fromlist=["SpeakerDiarizationExtractor"]
                ).SpeakerDiarizationExtractor(
                    device=self.device,
                    whisper_model_size=self.diarization_model_size,
                    sample_rate=16000,
                    silence_peak_threshold=self.diarization_silence_peak_threshold,
                    silence_rms_threshold=self.diarization_silence_rms_threshold,
                    enable_speaker_segments=self.diar_enable_speaker_segments,
                    enable_speaker_embeddings=self.diar_enable_speaker_embeddings,
                    enable_speaker_stats=self.diar_enable_speaker_stats,
                    enable_speaker_durations=self.diar_enable_speaker_durations,
                    enable_transcript=False,  # Not used in main processor
                    enable_word_segments=False,  # Not used in main processor
                    enable_silence_detection=self.diar_enable_silence_detection,
                ),
                "tempo": lambda: __import__(
                    "src.extractors.tempo_extractor", fromlist=["TempoExtractor"]
                ).TempoExtractor(device=self.device, sample_rate=self.sample_rate),
                "loudness": lambda: __import__(
                    "src.extractors.loudness_extractor", fromlist=["LoudnessExtractor"]
                ).LoudnessExtractor(device=self.device, sample_rate=self.sample_rate),
                "onset": lambda: __import__(
                    "src.extractors.onset_extractor", fromlist=["OnsetExtractor"]
                ).OnsetExtractor(
                    device=self.device,
                    sample_rate=self.onset_sample_rate,
                    hop_length=self.onset_hop_length,
                    pre_max=self.onset_pre_max,
                    post_max=self.onset_post_max,
                    pre_avg=self.onset_pre_avg,
                    post_avg=self.onset_post_avg,
                    delta=self.onset_delta,
                    wait=self.onset_wait,
                    backend=self.onset_backend,
                    units=self.onset_units,
                    backtrack=self.onset_backtrack,
                    energy=self.onset_energy,
                    normalize=self.onset_normalize,
                    enable_audio_normalization=self.onset_enable_audio_normalization,
                    enable_basic_features=self.onset_enable_basic_features,
                    enable_interval_stats=self.onset_enable_interval_stats,
                    enable_rhythmic_metrics=self.onset_enable_rhythmic_metrics,
                    enable_time_series=self.onset_enable_time_series,
                ),
                "chroma": lambda: __import__(
                    "src.extractors.chroma_extractor", fromlist=["ChromaExtractor"]
                ).ChromaExtractor(
                    device=self.device,
                    sample_rate=self.chroma_sample_rate,
                    hop_length=self.chroma_hop_length,
                    n_fft=self.chroma_n_fft,
                    mix_to_mono=self.chroma_mix_to_mono,
                    chroma_type=self.chroma_type,
                    normalize=self.chroma_normalize,
                    n_chroma=self.chroma_n_chroma,
                    fmin=self.chroma_fmin,
                    fmax=self.chroma_fmax,
                    n_bins=self.chroma_n_bins,
                    enable_audio_normalization=self.chroma_enable_audio_normalization,
                    enable_basic_stats=self.chroma_enable_basic_stats,
                    enable_extended_stats=self.chroma_enable_extended_stats,
                    enable_stats_vector=self.chroma_enable_stats_vector,
                    enable_time_series=self.chroma_enable_time_series,
                ),
                "spectral": lambda: __import__(
                    "src.extractors.spectral_extractor", fromlist=["SpectralExtractor"]
                ).SpectralExtractor(
                    device=self.device,
                    sample_rate=self.spectral_sample_rate,
                    hop_length=self.spectral_hop_length,
                    n_fft=self.spectral_n_fft,
                    average_channels=self.spectral_average_channels,
                    keep_contrast_bands=self.spectral_keep_contrast_bands,
                    enable_normalization=self.spectral_enable_normalization,
                    enable_basic_features=self.spectral_enable_basic_features,
                    enable_contrast=self.spectral_enable_contrast,
                    enable_advanced_features=self.spectral_enable_advanced_features,
                    enable_time_series=self.spectral_enable_time_series,
                ),
                "quality": lambda: __import__(
                    "src.extractors.quality_extractor", fromlist=["QualityExtractor"]
                ).QualityExtractor(
                    device=self.device,
                    sample_rate=self.quality_sample_rate,
                    average_channels=self.quality_average_channels,
                    frame_len_ms=self.quality_frame_len_ms,
                    hop_ms=self.quality_hop_ms,
                    clip_threshold=self.quality_clip_threshold,
                    enable_normalization=self.quality_enable_normalization,
                    enable_basic_metrics=self.quality_enable_basic_metrics,
                    enable_dynamic_metrics=self.quality_enable_dynamic_metrics,
                    enable_frame_analysis=self.quality_enable_frame_analysis,
                    enable_time_series=self.quality_enable_time_series,
                ),
                "mfcc": lambda: __import__(
                    "src.extractors.mfcc_extractor", fromlist=["MFCCExtractor"]
                ).MFCCExtractor(
                    device=self.device,
                    sample_rate=self.mfcc_sample_rate,
                    n_mfcc=self.mfcc_n_mfcc,
                    n_fft=self.mfcc_n_fft,
                    hop_length=self.mfcc_hop_length,
                    n_mels=self.mfcc_n_mels,
                    fmin=self.mfcc_fmin,
                    fmax=self.mfcc_fmax,
                    enable_audio_normalization=self.mfcc_enable_audio_normalization,
                    min_gpu_duration_sec=self.mfcc_min_gpu_duration_sec,
                    min_gpu_file_size_mb=self.mfcc_min_gpu_file_size_mb,
                    enable_basic_features=self.mfcc_enable_basic_features,
                    enable_deltas=self.mfcc_enable_deltas,
                    enable_time_series=self.mfcc_enable_time_series,
                    enable_normalization=self.mfcc_enable_normalization,
                ),
                "rhythmic": lambda: __import__(
                    "src.extractors.rhythmic_extractor", fromlist=["RhythmicExtractor"]
                ).RhythmicExtractor(
                    device=self.device,
                    sample_rate=self.rhythmic_sample_rate,
                    hop_length=self.rhythmic_hop_length,
                    average_channels=True,
                    backend=self.rhythmic_backend,
                    start_bpm=self.rhythmic_start_bpm,
                    std_bpm=self.rhythmic_std_bpm,
                    ac_size=self.rhythmic_ac_size,
                    max_tempo=self.rhythmic_max_tempo,
                    enable_audio_normalization=self.rhythmic_enable_audio_normalization,
                    enable_basic_metrics=self.rhythmic_enable_basic_metrics,
                    enable_interval_stats=self.rhythmic_enable_interval_stats,
                    enable_regularity_metrics=self.rhythmic_enable_regularity_metrics,
                    enable_beat_times=self.rhythmic_enable_beat_times,
                    enable_tempo_metrics=self.rhythmic_enable_tempo_metrics,
                    progress_callback=self._progress_callback,
                    artifacts_dir=None,  # Will be set in run_cli.py
                ),
                "key": lambda: __import__(
                    "src.extractors.key_extractor.main", fromlist=["KeyExtractor"]
                ).KeyExtractor(
                    device=self.device,
                    sample_rate=self.key_sample_rate,
                    hop_length=self.key_hop_length,
                    chroma_type=self.key_chroma_type,
                    use_beat_sync=self.key_use_beat_sync,
                    top_k=self.key_top_k,
                    key_method=self.key_method,
                    key_confidence_threshold=self.key_confidence_threshold,
                    enable_audio_normalization=self.key_enable_audio_normalization,
                    enable_detailed_scores=self.key_enable_detailed_scores,
                    enable_top_k=self.key_enable_top_k,
                    enable_time_series=self.key_enable_time_series,
                    enable_key_changes=self.key_enable_key_changes,
                    enable_stability_metrics=self.key_enable_stability_metrics,
                    progress_callback=self._progress_callback,
                    artifacts_dir=None,  # Will be set in run_cli.py
                ),
                "band_energy": lambda: __import__(
                    "src.extractors.band_energy_extractor", fromlist=["BandEnergyExtractor"]
                ).BandEnergyExtractor(
                    device=self.device,
                    sample_rate=self.band_energy_sample_rate,
                    bands=None,  # Use default bands
                    n_fft=self.band_energy_n_fft,
                    hop_length=self.band_energy_hop_length,
                    use_mel_bands=self.band_energy_use_mel_bands,
                    n_mels=self.band_energy_n_mels,
                    band_method=self.band_energy_method,
                    average_channels=self.band_energy_average_channels,
                    enable_audio_normalization=self.band_energy_enable_audio_normalization,
                    enable_basic_stats=self.band_energy_enable_basic_stats,
                    enable_extended_stats=self.band_energy_enable_extended_stats,
                    enable_time_series=self.band_energy_enable_time_series,
                    enable_dynamics=self.band_energy_enable_dynamics,
                    enable_balance_metrics=self.band_energy_enable_balance_metrics,
                    progress_callback=self._progress_callback,
                    artifacts_dir=None,  # Will be set in run_cli.py
                ),
                "spectral_entropy": lambda: __import__(
                    "src.extractors.spectral_entropy_extractor", fromlist=["SpectralEntropyExtractor"]
                ).SpectralEntropyExtractor(
                    device=self.device,
                    sample_rate=self.spectral_entropy_sample_rate,
                    n_fft=self.spectral_entropy_n_fft,
                    hop_length=self.spectral_entropy_hop_length,
                    average_channels=self.spectral_entropy_average_channels,
                    smoothing_window=self.spectral_entropy_smoothing_window,
                    use_mel=self.spectral_entropy_use_mel,
                    n_mels=self.spectral_entropy_n_mels,
                    enable_audio_normalization=self.spectral_entropy_enable_audio_normalization,
                    enable_basic_stats=self.spectral_entropy_enable_basic_stats,
                    enable_flatness=self.spectral_entropy_enable_flatness,
                    enable_spread=self.spectral_entropy_enable_spread,
                    enable_time_series=self.spectral_entropy_enable_time_series,
                    enable_extended_stats=self.spectral_entropy_enable_extended_stats,
                    enable_dynamics=self.spectral_entropy_enable_dynamics,
                    progress_callback=self._progress_callback,
                    artifacts_dir=None,  # Will be set in run_cli.py
                ),
                "voice_quality": lambda: __import__(
                    "src.extractors.voice_quality_extractor", fromlist=["VoiceQualityExtractor"]
                ).VoiceQualityExtractor(
                    device=self.device,
                    sample_rate=self.voice_quality_sample_rate,
                    average_channels=self.voice_quality_average_channels,
                    hnr_frame_ms=self.voice_quality_hnr_frame_ms,
                    rms_mask_threshold=self.voice_quality_rms_mask_threshold,
                    f0_fmin=self.voice_quality_f0_fmin,
                    f0_fmax=self.voice_quality_f0_fmax,
                    f0_method=self.voice_quality_f0_method,
                    enable_jitter=self.voice_quality_enable_jitter,
                    enable_shimmer=self.voice_quality_enable_shimmer,
                    enable_hnr=self.voice_quality_enable_hnr,
                    enable_f0_stats=self.voice_quality_enable_f0_stats,
                    enable_time_series=self.voice_quality_enable_time_series,
                    enable_audio_normalization=self.voice_quality_enable_audio_normalization,
                    progress_callback=self._progress_callback,
                    artifacts_dir=None,  # Will be set in run_cli.py
                ),
                "hpss": lambda: __import__(
                    "src.extractors.hpss_extractor", fromlist=["HPSSExtractor"]
                ).HPSSExtractor(
                    device=self.device,
                    sample_rate=self.hpss_sample_rate,
                    n_fft=self.hpss_n_fft,
                    hop_length=self.hpss_hop_length,
                    average_channels=self.hpss_average_channels,
                    hpss_kernel_size=self.hpss_kernel_size,
                    hpss_margin=self.hpss_margin,
                    hpss_power=self.hpss_power,
                    enable_audio_normalization=self.hpss_enable_audio_normalization,
                    enable_energy_metrics=self.hpss_enable_energy_metrics,
                    enable_waveforms=self.hpss_enable_waveforms,
                    enable_spectral_features=self.hpss_enable_spectral_features,
                    enable_time_series=self.hpss_enable_time_series,
                    progress_callback=self._progress_callback,
                    artifacts_dir=None,  # Will be set in run_cli.py
                ),
                "key": lambda: __import__(
                    "src.extractors.key_extractor", fromlist=["KeyExtractor"]
                ).KeyExtractor(device=self.device, sample_rate=self.sample_rate),
                "band_energy": lambda: __import__(
                    "src.extractors.band_energy_extractor", fromlist=["BandEnergyExtractor"]
                ).BandEnergyExtractor(device=self.device, sample_rate=self.sample_rate),
                "spectral_entropy": lambda: __import__(
                    "src.extractors.spectral_entropy_extractor", fromlist=["SpectralEntropyExtractor"]
                ).SpectralEntropyExtractor(device=self.device, sample_rate=self.sample_rate),
                "source_separation": lambda: __import__(
                    "src.extractors.source_separation_extractor", fromlist=["SourceSeparationExtractor"]
                ).SourceSeparationExtractor(
                    device=self.device,
                    model_size=self.source_separation_model_size,
                    batch_size=self.sep_batch_size,
                    silence_peak_threshold=self.sep_silence_peak_threshold,
                    silence_rms_threshold=self.sep_silence_rms_threshold,
                    enable_share_sequence=self.sep_enable_share_sequence,
                    enable_energy_sequence=self.sep_enable_energy_sequence,
                    enable_share_mean=self.sep_enable_share_mean,
                    enable_share_std=self.sep_enable_share_std,
                    enable_quality_metrics=self.sep_enable_quality_metrics,
                    enable_silence_detection=self.sep_enable_silence_detection,
                ),
                "emotion_diarization": lambda: __import__(
                    "src.extractors.emotion_diarization_extractor", fromlist=["EmotionDiarizationExtractor"]
                ).EmotionDiarizationExtractor(
                    device=self.device,
                    model_size=self.emotion_model_size,
                    sample_rate=16000,
                    batch_size=self.emotion_batch_size,
                    silence_peak_threshold=self.emotion_silence_peak_threshold,
                    silence_rms_threshold=self.emotion_silence_rms_threshold,
                    enable_probs=self.emotion_enable_probs,
                    enable_ids=self.emotion_enable_ids,
                    enable_confidence=self.emotion_enable_confidence,
                    enable_mean_probs=self.emotion_enable_mean_probs,
                    enable_entropy=self.emotion_enable_entropy,
                    enable_dominant=self.emotion_enable_dominant,
                    enable_quality_metrics=self.emotion_enable_quality_metrics,
                    enable_silence_detection=self.emotion_enable_silence_detection,
                    process_full_audio=self.emotion_process_full_audio,
                ),
                "speech_analysis": lambda: __import__(
                    "src.extractors.speech_analysis_extractor", fromlist=["SpeechAnalysisExtractor"]
                ).SpeechAnalysisExtractor(
                    device=self.device,
                    sample_rate=16000,
                    pitch_enabled=bool(self.speech_analysis_pitch_enabled),
                    pitch_backend=("torchcrepe" if str(self.device).lower() == "cuda" else "classic"),
                    enable_asr_metrics=self.speech_enable_asr_metrics,
                    enable_diarization_metrics=self.speech_enable_diarization_metrics,
                    enable_pitch_metrics=self.speech_enable_pitch_metrics,
                    silence_peak_threshold=self.speech_silence_peak_threshold,
                    silence_rms_threshold=self.speech_silence_rms_threshold,
                    enable_silence_detection=self.speech_enable_silence_detection,
                ),
                "pitch": lambda: __import__(
                    "src.extractors.pitch_extractor.main", fromlist=["PitchExtractor"]
                ).PitchExtractor(
                    device=self.device,
                    sample_rate=self.pitch_sample_rate,
                    fmin=self.pitch_fmin,
                    fmax=self.pitch_fmax,
                    hop_length=self.pitch_hop_length,
                    frame_length=self.pitch_frame_length,
                    backend=self.pitch_backend,
                    channel_mode=self.pitch_channel_mode,
                    torchcrepe_batch_size=self.pitch_torchcrepe_batch_size,
                    enable_basic_stats=self.pitch_enable_basic_stats,
                    enable_stability_metrics=self.pitch_enable_stability_metrics,
                    enable_delta_features=self.pitch_enable_delta_features,
                    enable_method_stats=self.pitch_enable_method_stats,
                    enable_time_series=self.pitch_enable_time_series,
                ),
            }

            # Инициализируем только те extractors, которые явно указаны в enabled_extractors
            # Если enabled_extractors is None или пустой список, не инициализируем ничего
            if enabled_extractors is None or len(enabled_extractors) == 0:
                requested = []
            else:
                requested = list(enabled_extractors)
            
            # Обрабатываем виртуальные экстракторы
            # NOTE: 'asr', 'speaker_diarization' и 'pitch' теперь являются реальными extractors, не виртуальными
            if enabled_extractors is not None and len(enabled_extractors) > 0:
                # Виртуальные экстракторы (если есть) реализуются внутри speech_analysis
                # pitch теперь является реальным extractor'ом с собственной фабрикой
                virtuals = []  # Все extractors теперь реальные (asr, speaker_diarization, pitch)
                # Проверяем только виртуальные, которые не имеют реальной фабрики
                ensure_sa = any(v in requested for v in virtuals if v not in extractor_factories)
                if ensure_sa and "speech_analysis" not in requested:
                    requested.append("speech_analysis")
                # Убираем виртуальные из инициализации, они будут опубликованы после run(speech_analysis)
                # If an extractor exists as a real factory (e.g., asr / speaker_diarization / pitch), do NOT treat it as virtual.
                requested = [n for n in requested if not (n in virtuals and n not in extractor_factories)]

            self.extractors = {}
            total_extractors = len([n for n in requested if extractor_factories.get(n) is not None])
            init_start_time = time.time()
            
            for idx, name in enumerate(requested, 1):
                factory = extractor_factories.get(name)
                if factory is None:
                    # Тихо пропускаем отсутствующие фабрики (например, виртуальные, уже обработанные)
                    continue
                try:
                    from ..utils.progress import Colors  # type: ignore
                    use_colors = Colors.supports_color()
                    
                    t0 = time.time()
                    if use_colors:
                        audio_processor_prefix = f"{Colors.BLUE}{Colors.BOLD}AudioProcessor{Colors.RESET} {Colors.GRAY}|{Colors.RESET}"
                        idx_str = f"{Colors.CYAN}[{idx}/{total_extractors}]{Colors.RESET}"
                        name_str = f"{Colors.YELLOW}{name}{Colors.RESET}"
                        print(f"{audio_processor_prefix} {idx_str} Initializing extractor: {name_str}...", file=sys.stderr, flush=True)
                    else:
                        print(f"AudioProcessor | [{idx}/{total_extractors}] Initializing extractor: {name}...", file=sys.stderr, flush=True)
                    
                    self.extractors[name] = factory()
                    t1 = time.time()
                    elapsed = t1 - t0
                    total_elapsed = t1 - init_start_time
                    
                    if use_colors:
                        idx_str = f"{Colors.CYAN}[{idx}/{total_extractors}]{Colors.RESET}"
                        name_str = f"{Colors.YELLOW}{name}{Colors.RESET}"
                        elapsed_str = f"{Colors.GREEN}{elapsed:.2f}s{Colors.RESET}"
                        total_elapsed_str = f"{Colors.GREEN}{total_elapsed:.2f}s{Colors.RESET}"
                        print(f"{audio_processor_prefix} {idx_str} Extractor {name_str} initialized {Colors.GRAY}({Colors.RESET}{elapsed_str}{Colors.GRAY}, total: {Colors.RESET}{total_elapsed_str}{Colors.GRAY}){Colors.RESET}", file=sys.stderr, flush=True)
                    else:
                        print(f"AudioProcessor | [{idx}/{total_extractors}] Extractor {name} initialized ({elapsed:.2f}s, total: {total_elapsed:.2f}s)", file=sys.stderr, flush=True)
                except Exception as err:
                    self.logger.error(f"Не удалось инициализировать экстрактор {name}: {err}")
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации экстракторов: {e}")
            raise
    
    def process_video(
        self,
        video_path: str,
        output_dir: str,
        extractor_names: Optional[List[str]] = None,
        extract_audio: bool = False
    ) -> Dict[str, Any]:
        """
        Обработка видео файла с извлечением аудио и признаков.
        
        Args:
            video_path: Путь к видео файлу
            output_dir: Директория для сохранения результатов
            extractor_names: Список экстракторов для запуска
            extract_audio: (DEPRECATED) AudioProcessor не извлекает аудио из видео. Должно быть False.
            
        Returns:
            Словарь с результатами обработки
        """
        start_time = time.time()
        
        try:
            # Старт системного мониторинга
            # monitor = MainProcessor._SystemMonitor(self.logger, sample_interval=10,
            #                                        cpu_threshold=95.0, ram_threshold=95.0, gpu_threshold=95.0)
            # monitor.start()
            # Создаем выходную директорию
            t_mkdir0 = time.time()
            os.makedirs(output_dir, exist_ok=True)
            t_mkdir1 = time.time()
            
            # self.logger.debug(f"Начинаем обработку видео: {video_path}")
            
            results = {
                "video_path": video_path,
                "output_dir": output_dir,
                "success": False,
                "extracted_audio_path": None,
                "extractor_results": {},
                "processing_time": 0.0,
                "errors": [],
                "timings": {
                    "wall_clock": {
                        "start_ts": start_time,
                        "start_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
                        "end_ts": None,
                        "end_iso": None,
                        "elapsed_s": None,
                    },
                    "mkdir_ms": float((t_mkdir1 - t_mkdir0) * 1000.0),
                    "audio_extract_ms": 0.0,
                    "per_extractor_wall_ms": {},
                    "per_extractor_reported_ms": {},
                    "save_manifest_ms": 0.0,
                }
            }
            
            # Шаг 1: AudioProcessor НЕ извлекает аудио из видео (Segmenter contract).
            # Здесь video_path трактуется как путь к аудио файлу (обычно frames_dir/audio/audio.wav).
            if bool(extract_audio):
                raise RuntimeError("AudioProcessor | extract_audio is deprecated and not supported. Use Segmenter (frames_dir/audio/audio.wav).")
            if self._is_video_file(video_path):
                raise RuntimeError("AudioProcessor | video input is not supported. Provide audio/audio.wav from Segmenter.")
            audio_path = video_path
            results["extracted_audio_path"] = audio_path
            
            # Шаг 2: Запуск экстракторов
            if extractor_names is None:
                # Запускаем все доступные экстракторы
                extractor_names = list(self.extractors.keys())
            
            # NOTE: 'asr', 'speaker_diarization' и 'pitch' теперь являются реальными extractors, не виртуальными
            requested_names = list(extractor_names)
            # pitch теперь является реальным extractor'ом с собственной фабрикой, не виртуальным
            
            # Скрываем подробный список экстракторов в логах запуска
            # self.logger.debug(f"Запускаем экстракторы: {extractor_names}")
            
            # Запускаем экстракторы последовательно (можно распараллелить)
            extractor_times = {}
            for extractor_name in requested_names:
                if extractor_name not in self.extractors:
                    self.logger.warning(f"Экстрактор {extractor_name} пропущен (не доступен)")
                    continue
                
                try:
                    extractor = self.extractors[extractor_name]
                    
                    # Все экстракторы используют аудио файл от Segmenter
                    input_path = audio_path
                    
                    t_e0 = time.time()
                    extractor_result = extractor.run(input_path, output_dir)
                    t_e1 = time.time()
                    
                    results["extractor_results"][extractor_name] = {
                        "success": extractor_result.success,
                        "payload": extractor_result.payload,
                        "error": extractor_result.error,
                        "processing_time": extractor_result.processing_time,
                        "device_used": extractor_result.device_used
                    }
                    
                    # Сохраняем время выполнения
                    extractor_times[extractor_name] = extractor_result.processing_time
                    results["timings"]["per_extractor_wall_ms"][extractor_name] = float((t_e1 - t_e0) * 1000.0)
                    results["timings"]["per_extractor_reported_ms"][extractor_name] = float((extractor_result.processing_time or 0.0) * 1000.0)
                    
                    if not extractor_result.success:
                        error_msg = f"❌ {extractor_name} не удался: {extractor_result.error}"
                        self.logger.error(error_msg)
                        results["errors"].append(error_msg)
                    
                    # Виртуальные результаты из speech_analysis
                    if extractor_name == "speech_analysis" and extractor_result.success:
                        payload = extractor_result.payload or {}
                        # pitch
                        if wants_virtual_pitch:
                            try:
                                pitch_payload = payload.get("pitch_result") or {}
                                pitch_time = float(payload.get("pitch_processing_time") or 0.0)
                                pitch_success = bool(pitch_payload) and (
                                    (pitch_payload.get("f0_count_pyin") or 0) > 0
                                    or (pitch_payload.get("f0_count_yin") or 0) > 0
                                    or (pitch_payload.get("f0_mean") or 0.0) > 0.0
                                )
                                results["extractor_results"]["pitch"] = {
                                    "success": pitch_success,
                                    "payload": pitch_payload if pitch_success else None,
                                    "error": None if pitch_success else "pitch empty/zero values",
                                    "processing_time": pitch_time,
                                    "device_used": payload.get("device_used", "unknown"),
                                }
                                extractor_times["pitch"] = pitch_time
                            except Exception:
                                pass
                        # asr (всегда добавляем, если есть в payload)
                        if wants_virtual_asr or (payload.get("asr_result") is not None):
                            try:
                                asr_time = float(payload.get("asr_processing_time") or 0.0)
                                # Забираем исходный результат ASR из speech_analysis
                                asr_src = payload.get("asr_result") or {}
                                asr_payload = {
                                    "transcription": asr_src.get("transcription", ""),
                                    "language": asr_src.get("language", "unknown"),
                                    "language_probability": asr_src.get("language_probability", 0.0),
                                    # Совместимость: если старое поле duration есть, трактуем как speech_duration
                                    "speech_duration": float(asr_src.get("speech_duration", asr_src.get("duration", 0.0)) or 0.0),
                                    "audio_duration": float(asr_src.get("audio_duration", 0.0) or 0.0),
                                    "model_size": asr_src.get("model_size", "unknown"),
                                    "task": asr_src.get("task", "transcribe"),
                                    "segments": asr_src.get("segments", []) or [],
                                    "sample_rate": asr_src.get("sample_rate", 0) or 0,
                                    # device_used может отсутствовать в asr_src; пробрасываем из speech_analysis
                                    "device_used": asr_src.get("device_used") or payload.get("device_used", "unknown"),
                                }
                                # Подстрахуемся статистикой из aligned_speech
                                aligned = payload.get("aligned_speech") or {}
                                stats = aligned.get("statistics") or {}
                                if not asr_payload["duration"]:
                                    asr_payload["duration"] = float(stats.get("total_duration", 0.0) or 0.0)
                                asr_success = (
                                    bool(asr_payload.get("transcription"))
                                    or len(asr_payload.get("segments", []) or []) > 0
                                    or (asr_payload.get("speech_duration", 0.0) or 0.0) > 0.0
                                )
                                results["extractor_results"]["asr"] = {
                                    "success": asr_success,
                                    "payload": asr_payload if asr_success else None,
                                    "error": None if asr_success else "asr data unavailable in speech_analysis payload",
                                    "processing_time": asr_time,
                                    "device_used": asr_payload.get("device_used", payload.get("device_used", "unknown")),
                                }
                                extractor_times["asr"] = asr_time
                            except Exception:
                                pass
                        # speaker_diarization
                        if wants_virtual_speaker:
                            try:
                                diar_time = float(payload.get("diarization_processing_time") or 0.0)
                                aligned = payload.get("aligned_speech") or {}
                                stats = aligned.get("statistics") or {}
                                diar_src = payload.get("diarization_result") or {}
                                sp_payload = {
                                    "speaker_count": int(aligned.get("total_speakers", 0) or diar_src.get("speaker_count", 0) or 0),
                                    "segment_duration": float(diar_src.get("segment_duration", 0.0) or 0.0),
                                    "clustering_method": diar_src.get("clustering_method", "unknown") or "unknown",
                                    # duration в speaker_diarization — используем длительность из самого диаризационного экстрактора (полная длительность аудио, если так задано там)
                                    "duration": float(diar_src.get("duration", stats.get("total_duration", 0.0)) or 0.0),
                                    "speaker_segments": diar_src.get("speaker_segments", []) or [],
                                    "speaker_stats": (stats.get("speaker_stats") if isinstance(stats.get("speaker_stats"), dict) else diar_src.get("speaker_stats")),
                                    "device_used": payload.get("device_used", "unknown"),
                                    "sample_rate": payload.get("sample_rate", 0) or 0,
                                }
                                sp_success = (sp_payload["speaker_count"] > 0) or (len(sp_payload.get("speaker_segments", []) or []) > 0) or (sp_payload["duration"] > 0.0)
                                results["extractor_results"]["speaker_diarization"] = {
                                    "success": sp_success,
                                    "payload": sp_payload if sp_success else None,
                                    "error": None if sp_success else "speaker diarization data unavailable in speech_analysis payload",
                                    "processing_time": diar_time,
                                    "device_used": payload.get("device_used", "unknown"),
                                }
                                extractor_times["speaker_diarization"] = diar_time
                            except Exception:
                                pass
                        
                except Exception as e:
                    error_msg = f"Ошибка в экстракторе {extractor_name}: {e}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
                    
                    results["extractor_results"][extractor_name] = {
                        "success": False,
                        "payload": None,
                        "error": str(e),
                        "processing_time": 0.0,
                        "device_used": "unknown"
                    }
                    extractor_times[extractor_name] = 0.0

            # Fallback: гарантируем публикацию виртуального ASR, если он присутствует в speech_analysis
            try:
                if (
                    "asr" not in results["extractor_results"]
                    and "speech_analysis" in results["extractor_results"]
                    and results["extractor_results"]["speech_analysis"].get("success")
                ):
                    sa_payload = results["extractor_results"]["speech_analysis"].get("payload") or {}
                    asr_src = sa_payload.get("asr_result") or {}
                    if asr_src:
                        asr_time = float(sa_payload.get("asr_processing_time") or 0.0)
                        asr_payload = {
                            "transcription": asr_src.get("transcription", ""),
                            "language": asr_src.get("language", "unknown"),
                            "language_probability": asr_src.get("language_probability", 0.0),
                            "speech_duration": float(asr_src.get("speech_duration", asr_src.get("duration", 0.0)) or 0.0),
                            "audio_duration": float(asr_src.get("audio_duration", 0.0) or 0.0),
                            "model_size": asr_src.get("model_size", "unknown"),
                            "task": asr_src.get("task", "transcribe"),
                            "segments": asr_src.get("segments", []) or [],
                            "sample_rate": asr_src.get("sample_rate", 0) or 0,
                            "device_used": asr_src.get("device_used") or sa_payload.get("device_used", "unknown"),
                        }
                        asr_success = (
                            bool(asr_payload.get("transcription"))
                            or len(asr_payload.get("segments", []) or []) > 0
                            or (asr_payload.get("speech_duration", 0.0) or 0.0) > 0.0
                        )
                        results["extractor_results"]["asr"] = {
                            "success": asr_success,
                            "payload": asr_payload if asr_success else None,
                            "error": None if asr_success else "asr data unavailable in speech_analysis payload",
                            "processing_time": asr_time,
                            "device_used": asr_payload.get("device_used", sa_payload.get("device_used", "unknown")),
                        }
                        extractor_times["asr"] = asr_time
            except Exception:
                pass
            
            # Остановка монитора и добавление макс. значений в лог
            # Монитор сейчас отключен; оставляем заглушку
            try:
                monitor  # type: ignore[name-defined]
            except Exception:
                pass
            
            # Вычисляем время обработки перед сохранением
            end_time_total = time.time()
            results["processing_time"] = end_time_total - start_time
            results["timings"]["wall_clock"]["end_ts"] = end_time_total
            results["timings"]["wall_clock"]["end_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_time_total))
            results["timings"]["wall_clock"]["elapsed_s"] = float(end_time_total - start_time)
            
            # Определяем общий успех
            successful_extractors = sum(
                1 for result in results["extractor_results"].values() 
                if result["success"]
            )
            
            # Считаем только доступные экстракторы (те, которые реально запускались)
            available_extractors = len(results["extractor_results"])
            
            results["success"] = len(results["errors"]) == 0 and successful_extractors > 0
            
            # Шаг 3: Сохранение результатов
            if self.write_legacy_manifest:
                t_s0 = time.time()
                self._save_results(results, output_dir)
                t_s1 = time.time()
                results["timings"]["save_manifest_ms"] = float((t_s1 - t_s0) * 1000.0)
            
            # Выводим время выполнения по экстракторам
            if extractor_times:
                # self.logger.info("⏱️ Время выполнения экстракторов:")
                for name, time_taken in sorted(extractor_times.items(), key=lambda x: x[1], reverse=True):
                    status = "✅" if results["extractor_results"][name]["success"] else "❌"
                    # self.logger.info(f"  {status} {name}: {time_taken:.2f}s")
                
                # Добавляем время ASR из speech_analysis если оно есть
                if "speech_analysis" in results["extractor_results"] and results["extractor_results"]["speech_analysis"]["success"]:
                    speech_payload = results["extractor_results"]["speech_analysis"].get("payload", {})
                    asr_time = speech_payload.get("asr_processing_time", 0.0)
                    if asr_time > 0:
                        pass
                        # self.logger.info(f"  📝 asr (в speech_analysis): {asr_time:.2f}s")
                
                total_extractor_time = sum(extractor_times.values())
                # self.logger.info(f"📊 Суммарное время экстракторов: {total_extractor_time:.2f}s")
            
            # Максимальные значения ресурсов за время обработки
            # Логи пиков ресурсов оставляем включаемыми позже (монитор отключен)
            try:
                monitor  # type: ignore[name-defined]
            except Exception:
                pass
            
            # self.logger.info(f"🎯 Общее время обработки: {results['processing_time']:.2f}s")
            # self.logger.info(f"✅ Успешных экстракторов: {successful_extractors}/{available_extractors}")
            
            return results
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Критическая ошибка обработки: {e}"
            self.logger.error(error_msg)
            # Пытаемся остановить монитор
            try:
                monitor  # type: ignore[name-defined]
            except Exception:
                pass
            
            return {
                "video_path": video_path,
                "output_dir": output_dir,
                "success": False,
                "extracted_audio_path": None,
                "extractor_results": {},
                "processing_time": processing_time,
                "errors": [error_msg]
            }
    
    def _is_video_file(self, file_path: str) -> bool:
        """Проверка, является ли файл видео."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv'}
        return any(file_path.lower().endswith(ext) for ext in video_extensions)
    
    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """Сохранение результатов в формате manifest."""
        try:
            import json
            import numpy as np
            from datetime import datetime
            
            # Функция для конвертации numpy arrays в списки
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            # Создаем manifest в формате старой версии
            video_id = os.path.basename(results["video_path"]).split('.')[0]
            
            # Конвертируем результаты экстракторов в формат manifest
            extractor_results = []
            for extractor_name, result in results["extractor_results"].items():
                if result["success"] and result["payload"]:
                    # Создаем плоские признаки из payload
                    flat_payload = self._flatten_payload(result["payload"], extractor_name)
                    
                    extractor_result = {
                        "name": extractor_name,
                        "version": "1.0.0",
                        "success": True,
                        "payload": convert_numpy(flat_payload),
                        "error": None,
                        "processing_time": result["processing_time"]
                    }
                else:
                    extractor_result = {
                        "name": extractor_name,
                        "version": "1.0.0",
                        "success": False,
                        "payload": None,
                        "error": result["error"],
                        "processing_time": result["processing_time"]
                    }
                extractor_results.append(extractor_result)
            
            # Создаем manifest
            manifest = {
                "video_id": video_id,
                "task_id": f"audio_processor_{video_id}",
                "dataset": "audio_processor",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "extractors": extractor_results,
                "schema_version": "audio_manifest_v1",
                "total_processing_time": results["processing_time"],
                "manifest_uri": None
            }
            
            # Сохраняем manifest
            manifest_file = os.path.join(output_dir, f"{video_id}_manifest.json")
            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            # self.logger.info(f"Manifest сохранен: {manifest_file}")
            
            # По умолчанию не сохраняем подробные отладочные результаты
            if getattr(self, "save_debug_results", False):
                results_file = os.path.join(output_dir, "processing_results.json")
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(convert_numpy(results), f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения результатов: {e}")
    
    def _flatten_payload(self, payload: Dict[str, Any], extractor_name: str) -> Dict[str, Any]:
        """Преобразование payload в плоские признаки для manifest."""
        flat_payload: Dict[str, Any] = {}
        
        if extractor_name == "mfcc":
            # Для MFCC экстрактора
            if "mfcc_statistics" in payload:
                stats = payload["mfcc_statistics"]
                flat_payload.update({
                    "mfcc_mean": stats.get("mfcc_mean", []) if stats.get("mfcc_mean", None) is not None else [],
                    "mfcc_std": stats.get("mfcc_std", []) if stats.get("mfcc_std", None) is not None else [],
                    "mfcc_min": stats.get("mfcc_min", []) if stats.get("mfcc_min", None) is not None else [],
                    "mfcc_max": stats.get("mfcc_max", []) if stats.get("mfcc_max", None) is not None else [],
                    "delta_mean": stats.get("delta_mean", []) if stats.get("delta_mean", None) is not None else [],
                    "delta_std": stats.get("delta_std", []) if stats.get("delta_std", None) is not None else [],
                    "delta_delta_mean": stats.get("delta_delta_mean", []) if stats.get("delta_delta_mean", None) is not None else [],
                    "delta_delta_std": stats.get("delta_delta_std", []) if stats.get("delta_delta_std", None) is not None else [],
                    "total_features": stats.get("total_features", 0) if stats.get("total_features", None) is not None else 0
                })
        
        elif extractor_name == "mel":
            # Для Mel экстрактора - массивы в .npy файлах, в JSON только статистики и пути
            flat_payload.update({
                "mel_spectrogram_npy": payload.get("mel_spectrogram_npy", None),
                "mel_shape": payload.get("mel_shape", []) if payload.get("mel_shape", None) is not None else [],
                "mel_elements": payload.get("mel_elements", 0) if payload.get("mel_elements", None) is not None else 0,
                "sample_rate": payload.get("sample_rate", 0) if payload.get("sample_rate", None) is not None else 0,
                "n_fft": payload.get("n_fft", 0) if payload.get("n_fft", None) is not None else 0,
                "hop_length": payload.get("hop_length", 0) if payload.get("hop_length", None) is not None else 0,
                "n_mels": payload.get("n_mels", 0) if payload.get("n_mels", None) is not None else 0,
                "fmin": payload.get("fmin", 0.0) if payload.get("fmin", None) is not None else 0.0,
                "fmax": payload.get("fmax", 0.0) if payload.get("fmax", None) is not None else 0.0,
                "power": payload.get("power", 0.0) if payload.get("power", None) is not None else 0.0,
                "duration": payload.get("duration", 0.0) if payload.get("duration", None) is not None else 0.0,
                "device_used": payload.get("device_used", "unknown") if payload.get("device_used", None) is not None else "unknown",
                "total_features": payload.get("mel_elements", 0) if payload.get("mel_elements", None) is not None else 0,
                # Пути к .npy файлам с массивами
                "mel_mean_npy": payload.get("mel_mean_npy", None),
                "mel_std_npy": payload.get("mel_std_npy", None),
                "mel_min_npy": payload.get("mel_min_npy", None),
                "mel_max_npy": payload.get("mel_max_npy", None),
                "freq_mean_npy": payload.get("freq_mean_npy", None),
                "freq_std_npy": payload.get("freq_std_npy", None),
                "spectral_centroid_npy": payload.get("spectral_centroid_npy", None),
                "spectral_bandwidth_npy": payload.get("spectral_bandwidth_npy", None),
                "mel_stats_vector_npy": payload.get("mel_stats_vector_npy", None),
                # Размеры массивов
                "mel_mean_shape": payload.get("mel_mean_shape", []) if payload.get("mel_mean_shape", None) is not None else [],
                "mel_std_shape": payload.get("mel_std_shape", []) if payload.get("mel_std_shape", None) is not None else [],
                "mel_min_shape": payload.get("mel_min_shape", []) if payload.get("mel_min_shape", None) is not None else [],
                "mel_max_shape": payload.get("mel_max_shape", []) if payload.get("mel_max_shape", None) is not None else [],
                "freq_mean_shape": payload.get("freq_mean_shape", []) if payload.get("freq_mean_shape", None) is not None else [],
                "freq_std_shape": payload.get("freq_std_shape", []) if payload.get("freq_std_shape", None) is not None else [],
                "spectral_centroid_shape": payload.get("spectral_centroid_shape", []) if payload.get("spectral_centroid_shape", None) is not None else [],
                "spectral_bandwidth_shape": payload.get("spectral_bandwidth_shape", []) if payload.get("spectral_bandwidth_shape", None) is not None else [],
                "mel_stats_vector_shape": payload.get("mel_stats_vector_shape", []) if payload.get("mel_stats_vector_shape", None) is not None else [],
                # Статистики по массивам
                "mel_mean_stats": payload.get("mel_mean_stats", {}) if payload.get("mel_mean_stats", None) is not None else {},
                "mel_std_stats": payload.get("mel_std_stats", {}) if payload.get("mel_std_stats", None) is not None else {},
                "freq_mean_stats": payload.get("freq_mean_stats", {}) if payload.get("freq_mean_stats", None) is not None else {},
                "spectral_centroid_stats": payload.get("spectral_centroid_stats", {}) if payload.get("spectral_centroid_stats", None) is not None else {},
                "mel_stats_vector_stats": payload.get("mel_stats_vector_stats", {}) if payload.get("mel_stats_vector_stats", None) is not None else {}
            })
        
        elif extractor_name == "clap":
            # Для CLAP экстрактора - массивы статистик в .npy файлах, в JSON только скалярные статистики и пути
            flat_payload.update({
                "clap_embeddings_npy": payload.get("clap_embeddings_npy", None),
                "embeddings_shape": payload.get("embeddings_shape", []) if payload.get("embeddings_shape", None) is not None else [],
                "embeddings_dtype": payload.get("embeddings_dtype", "unknown") if payload.get("embeddings_dtype", None) is not None else "unknown",
                "embedding_dim": payload.get("embedding_dim", 0) if payload.get("embedding_dim", None) is not None else 0,
                "sample_rate": payload.get("sample_rate", 0) if payload.get("sample_rate", None) is not None else 0,
                "model_available": payload.get("model_available", False) if payload.get("model_available", None) is not None else False,
                # Пути к .npy файлам с массивами статистик
                "clap_mean_npy": payload.get("clap_mean_npy", None),
                "clap_std_npy": payload.get("clap_std_npy", None),
                "clap_min_npy": payload.get("clap_min_npy", None),
                "clap_max_npy": payload.get("clap_max_npy", None),
                # Размеры массивов
                "clap_mean_shape": payload.get("clap_mean_shape", []) if payload.get("clap_mean_shape", None) is not None else [],
                "clap_std_shape": payload.get("clap_std_shape", []) if payload.get("clap_std_shape", None) is not None else [],
                "clap_min_shape": payload.get("clap_min_shape", []) if payload.get("clap_min_shape", None) is not None else [],
                "clap_max_shape": payload.get("clap_max_shape", []) if payload.get("clap_max_shape", None) is not None else [],
                # Скалярные статистики
                "clap_norm": payload.get("clap_norm", 0.0) if payload.get("clap_norm", None) is not None else 0.0,
                "clap_non_zero_count": payload.get("clap_non_zero_count", 0) if payload.get("clap_non_zero_count", None) is not None else 0,
                "clap_magnitude_mean": payload.get("clap_magnitude_mean", 0.0) if payload.get("clap_magnitude_mean", None) is not None else 0.0,
                "clap_magnitude_std": payload.get("clap_magnitude_std", 0.0) if payload.get("clap_magnitude_std", None) is not None else 0.0,
                "total_features": payload.get("total_features", 0) if payload.get("total_features", None) is not None else 0
            })
        
        elif extractor_name == "tempo":
            # Для Tempo экстрактора
            flat_payload.update({
                "tempo_bpm": float(payload.get("tempo_bpm", 0.0) or 0.0),
                "tempo_bpm_mean": float(payload.get("tempo_bpm_mean", 0.0) or 0.0),
                "tempo_bpm_median": float(payload.get("tempo_bpm_median", 0.0) or 0.0),
                "tempo_bpm_std": float(payload.get("tempo_bpm_std", 0.0) or 0.0),
                "tempo_confidence": float(payload.get("confidence", 0.0) or 0.0),
                "tempo_estimates_count": (lambda v: int(len(v)) if v is not None else 0)(payload.get("tempo_estimates", None)),
            })

        elif extractor_name == "loudness":
            # Для Loudness экстрактора
            flat_payload.update({
                "loudness_rms": payload.get("rms", 0.0) or 0.0,
                "loudness_peak": payload.get("peak", 0.0) or 0.0,
                "loudness_dbfs": payload.get("dbfs", 0.0) or 0.0,
                # Исключаем None — используем 0.0 как безопасный дефолт
                "loudness_lufs": (payload.get("lufs") if isinstance(payload.get("lufs"), (int, float)) else 0.0),
            })

        elif extractor_name == "onset":
            # Для Onset экстрактора
            flat_payload.update({
                "onset_count": payload.get("onset_count", 0) or 0,
                "onset_avg_interval_sec": payload.get("avg_interval_sec", 0.0) or 0.0,
                "onset_density_per_sec": payload.get("onset_density_per_sec", 0.0) or 0.0,
                "onset_interval_std": payload.get("interval_std", 0.0) or 0.0,
                "onset_interval_min": payload.get("interval_min", 0.0) or 0.0,
                "onset_interval_max": payload.get("interval_max", 0.0) or 0.0,
                "onset_interval_median": payload.get("interval_median", 0.0) or 0.0,
                "onset_insufficient_onsets": payload.get("insufficient_onsets", True),
            })

        elif extractor_name == "chroma":
            # Для Chroma экстрактора
            flat_payload.update({
                "chroma_mean": payload.get("chroma_mean", []) if payload.get("chroma_mean", None) is not None else [],
                "chroma_std": payload.get("chroma_std", []) if payload.get("chroma_std", None) is not None else [],
                "chroma_min": payload.get("chroma_min", []) if payload.get("chroma_min", None) is not None else [],
                "chroma_max": payload.get("chroma_max", []) if payload.get("chroma_max", None) is not None else [],
            })

        elif extractor_name == "spectral":
            # Для Spectral экстрактора
            def _s(key: str) -> Dict[str, float]:
                data = payload.get(key) or {}
                if not isinstance(data, dict):
                    return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
                return {
                    "mean": float(data.get("mean", 0.0) or 0.0),
                    "std": float(data.get("std", 0.0) or 0.0),
                    "min": float(data.get("min", 0.0) or 0.0),
                    "max": float(data.get("max", 0.0) or 0.0),
                }
            flat_payload.update({
                "spectral_centroid_mean": _s("spectral_centroid_stats").get("mean", 0.0),
                "spectral_centroid_std": _s("spectral_centroid_stats").get("std", 0.0),
                "spectral_bandwidth_mean": _s("spectral_bandwidth_stats").get("mean", 0.0),
                "spectral_bandwidth_std": _s("spectral_bandwidth_stats").get("std", 0.0),
                "spectral_flatness_mean": _s("spectral_flatness_stats").get("mean", 0.0),
                "spectral_flatness_std": _s("spectral_flatness_stats").get("std", 0.0),
                "spectral_rolloff_mean": _s("spectral_rolloff_stats").get("mean", 0.0),
                "spectral_rolloff_std": _s("spectral_rolloff_stats").get("std", 0.0),
                "zcr_mean": _s("zcr_stats").get("mean", 0.0),
                "zcr_std": _s("zcr_stats").get("std", 0.0),
            })


        elif extractor_name == "quality":
            # Для Quality экстрактора
            flat_payload.update({
                "quality_dc_offset": float(payload.get("dc_offset", 0.0) or 0.0),
                "quality_clipping_ratio": float(payload.get("clipping_ratio", 0.0) or 0.0),
                "quality_crest_factor_db": float(payload.get("crest_factor_db", 0.0) or 0.0),
                "quality_dynamic_range_db": float(payload.get("dynamic_range_db", 0.0) or 0.0),
                "quality_snr_db": float(payload.get("snr_db", 0.0) or 0.0),
            })

        elif extractor_name == "rhythmic":
            # Для Rhythmic экстрактора
            flat_payload.update({
                "rhythm_tempo_bpm": float(payload.get("rhythm_tempo_bpm", 0.0) or 0.0),
                "rhythm_beats_count": int(payload.get("rhythm_beats_count", 0) or 0),
                "rhythm_avg_period_sec": float(payload.get("rhythm_avg_period_sec", 0.0) or 0.0),
                "rhythm_period_std_sec": float(payload.get("rhythm_period_std_sec", 0.0) or 0.0),
                "rhythm_regularity": float(payload.get("rhythm_regularity", 0.0) or 0.0),
                "rhythm_beat_density": float(payload.get("rhythm_beat_density", 0.0) or 0.0),
            })

        elif extractor_name == "voice_quality":
            # Для VoiceQuality экстрактора
            flat_payload.update({
                "vq_jitter": float(payload.get("vq_jitter", 0.0) or 0.0),
                "vq_shimmer": float(payload.get("vq_shimmer", 0.0) or 0.0),
                "vq_hnr_like_db": float(payload.get("vq_hnr_like_db", 0.0) or 0.0),
            })

        elif extractor_name == "hpss":
            # Для HPSS экстрактора
            flat_payload.update({
                "hpss_harmonic_share": float(payload.get("hpss_harmonic_share", 0.0) or 0.0),
                "hpss_percussive_share": float(payload.get("hpss_percussive_share", 0.0) or 0.0),
                "hpss_energy_total": float(payload.get("hpss_energy_total", 0.0) or 0.0),
            })

        elif extractor_name == "key":
            # Для Key экстрактора
            flat_payload.update({
                "music_key": payload.get("key_name", "unknown") or "unknown",
                "music_mode": payload.get("key_mode", "unknown") or "unknown",
                "music_key_confidence": float(payload.get("key_confidence", 0.0) or 0.0),
            })

        elif extractor_name == "band_energy":
            # Для BandEnergy экстрактора
            flat_payload.update({
                "band_energy_total": float(payload.get("total_energy", 0.0) or 0.0),
                "band_energy_shares": payload.get("band_energy_shares", []) if payload.get("band_energy_shares", None) is not None else [],
            })

        elif extractor_name == "spectral_entropy":
            # Для SpectralEntropy экстрактора
            flat_payload.update({
                "spectral_entropy_mean": float(payload.get("spectral_entropy_mean", 0.0) or 0.0),
                "spectral_entropy_std": float(payload.get("spectral_entropy_std", 0.0) or 0.0),
            })

        elif extractor_name == "source_separation":
            # Для Open-Unmix экстрактора
            flat_payload.update({
                "sep_energy_total": float(payload.get("energy_total", 0.0) or 0.0),
                "sep_share_vocals": float(payload.get("share_vocals", 0.0) or 0.0),
                "sep_share_drums": float(payload.get("share_drums", 0.0) or 0.0),
                "sep_share_bass": float(payload.get("share_bass", 0.0) or 0.0),
                "sep_share_other": float(payload.get("share_other", 0.0) or 0.0),
            })

        elif extractor_name == "emotion_diarization":
            # Для эмоциональной диаризации SpeechBrain
            flat_payload.update({
                "speaker_count": int(payload.get("speaker_count", 0) or 0),
                "duration": float(payload.get("duration", 0.0) or 0.0),
            })
            # Краткие агрегаты по эмоциям
            if isinstance(payload.get("emotion_statistics"), dict):
                flat_payload["emotion_statistics"] = payload.get("emotion_statistics")
            # Сегменты (оставим в полном виде — обычно их не слишком много)
            if payload.get("emotion_segments") is not None:
                flat_payload["emotion_segments"] = payload.get("emotion_segments")
            if payload.get("speaker_segments") is not None:
                flat_payload["speaker_segments"] = payload.get("speaker_segments")
            # Маппинг эмоций к спикерам
            if payload.get("emotion_speaker_mapping") is not None:
                flat_payload["emotion_speaker_mapping"] = payload.get("emotion_speaker_mapping")

        elif extractor_name == "pitch":
            # Для Pitch экстрактора (без CREPE)
            flat_payload.update({
                "f0_mean": float(payload.get("f0_mean", 0.0) or 0.0),
                "f0_std": float(payload.get("f0_std", 0.0) or 0.0),
                "f0_min": float(payload.get("f0_min", 0.0) or 0.0),
                "f0_max": float(payload.get("f0_max", 0.0) or 0.0),
                "f0_median": float(payload.get("f0_median", 0.0) or 0.0),
                "f0_method": payload.get("f0_method", "none") or "none",
                "pitch_variation": float(payload.get("pitch_variation", 0.0) or 0.0),
                "pitch_stability": float(payload.get("pitch_stability", 0.0) or 0.0),
                "pitch_range": float(payload.get("pitch_range", 0.0) or 0.0),
                "f0_mean_pyin": float(payload.get("f0_mean_pyin", 0.0) or 0.0),
                "f0_std_pyin": float(payload.get("f0_std_pyin", 0.0) or 0.0),
                "f0_min_pyin": float(payload.get("f0_min_pyin", 0.0) or 0.0),
                "f0_max_pyin": float(payload.get("f0_max_pyin", 0.0) or 0.0),
                "f0_median_pyin": float(payload.get("f0_median_pyin", 0.0) or 0.0),
                "f0_count_pyin": int(payload.get("f0_count_pyin", 0) or 0),
                "voiced_fraction_pyin": float(payload.get("voiced_fraction_pyin", 0.0) or 0.0),
                "voiced_probability_mean_pyin": float(payload.get("voiced_probability_mean_pyin", 0.0) or 0.0),
                "f0_mean_yin": float(payload.get("f0_mean_yin", 0.0) or 0.0),
                "f0_std_yin": float(payload.get("f0_std_yin", 0.0) or 0.0),
                "f0_min_yin": float(payload.get("f0_min_yin", 0.0) or 0.0),
                "f0_max_yin": float(payload.get("f0_max_yin", 0.0) or 0.0),
                "f0_median_yin": float(payload.get("f0_median_yin", 0.0) or 0.0),
                "f0_count_yin": int(payload.get("f0_count_yin", 0) or 0),
                "device_used": payload.get("device_used", "unknown") or "unknown",
                "sample_rate": payload.get("sample_rate", 0) or 0,
                "total_features": int(payload.get("f0_count_pyin", 0) or 0) + int(payload.get("f0_count_yin", 0) or 0)
            })
            # Пути к npy для torchcrepe (если есть) и счетчики
            if payload.get("f0_series_torchcrepe_npy"):
                flat_payload["f0_series_torchcrepe_npy"] = payload.get("f0_series_torchcrepe_npy")
                flat_payload["f0_count_torchcrepe"] = int(payload.get("f0_count_torchcrepe", 0) or 0)

        elif extractor_name == "asr":
            # Для ASR экстрактора
            flat_payload.update({
                "transcription": payload.get("transcription", "") or "",
                "language": payload.get("language", "unknown") or "unknown",
                "language_probability": float(payload.get("language_probability", 0.0) or 0.0),
                # Разделяем полную длительность аудио и длительность речи
                "audio_duration": float(payload.get("audio_duration", 0.0) or 0.0),
                "speech_duration": float(payload.get("speech_duration", payload.get("duration", 0.0)) or 0.0),
                "model_size": payload.get("model_size", "unknown") or "unknown",
                "task": payload.get("task", "transcribe") or "transcribe",
                "segments_count": len(payload.get("segments", []) or []),
                "device_used": payload.get("device_used", "unknown") or "unknown",
                "sample_rate": payload.get("sample_rate", 0) or 0
            })

        elif extractor_name == "speaker_diarization":
            # Для диаризационного экстрактора
            flat_payload.update({
                "speaker_count": payload.get("speaker_count", 0) or 0,
                "segment_duration": float(payload.get("segment_duration", 0.0) or 0.0),
                "clustering_method": payload.get("clustering_method", "unknown") or "unknown",
                "duration": float(payload.get("duration", 0.0) or 0.0),
                "segments_count": len(payload.get("speaker_segments", []) or []),
                "device_used": payload.get("device_used", "unknown") or "unknown",
                "sample_rate": payload.get("sample_rate", 0) or 0
            })
            # Таймкоды сегментов спикеров
            if payload.get("speaker_segments"):
                flat_payload["speaker_segments"] = payload.get("speaker_segments")
            # Пер-спикерные агрегаты
            if payload.get("speaker_stats"):
                flat_payload["speaker_stats"] = payload.get("speaker_stats")
            # Добавим путь к npy и форму, если сохранены средние эмбеддинги спикеров
            if payload.get("speaker_embeddings_npy"):
                flat_payload["speaker_embeddings_npy"] = payload.get("speaker_embeddings_npy")
                flat_payload["speaker_embeddings_shape"] = payload.get("speaker_embeddings_shape", []) or []
                flat_payload["speaker_ids_order"] = payload.get("speaker_ids_order", []) or []

        elif extractor_name == "speech_analysis":
            # Для комбинированного экстрактора речи
            aligned_speech = payload.get("aligned_speech", {}) or {}
            pitch_result = payload.get("pitch_result", {}) or {}
            
            flat_payload.update({
                "total_speakers": aligned_speech.get("total_speakers", 0) or 0,
                "total_segments": aligned_speech.get("total_segments", 0) or 0,
                "total_duration": aligned_speech.get("statistics", {}).get("total_duration", 0.0) or 0.0,
                "total_words": aligned_speech.get("statistics", {}).get("total_words", 0) or 0,
                "confidence_mean": aligned_speech.get("statistics", {}).get("confidence_stats", {}).get("mean", 0.0) or 0.0,
                "asr_processing_time": payload.get("asr_processing_time", 0.0) or 0.0,
                "diarization_processing_time": payload.get("diarization_processing_time", 0.0) or 0.0,
                "pitch_processing_time": payload.get("pitch_processing_time", 0.0) or 0.0,
                # Pitch данные
                "pitch_mean": pitch_result.get("f0_mean", 0.0) or 0.0,
                "pitch_std": pitch_result.get("f0_std", 0.0) or 0.0,
                "pitch_min": pitch_result.get("f0_min", 0.0) or 0.0,
                "pitch_max": pitch_result.get("f0_max", 0.0) or 0.0,
                "pitch_median": pitch_result.get("f0_median", 0.0) or 0.0,
                "pitch_method": pitch_result.get("f0_method", "none") or "none",
                "pitch_stability": pitch_result.get("pitch_stability", 0.0) or 0.0,
                "pitch_variation": pitch_result.get("pitch_variation", 0.0) or 0.0,
                "device_used": payload.get("device_used", "unknown") or "unknown"
            })
            # Таймкоды выровненных сегментов доступны по пути к JSON
            if payload.get("aligned_segments_json"):
                flat_payload["aligned_segments_json"] = payload.get("aligned_segments_json")
            # Пер-спикерная статистика из анализа речи
            sp_stats = aligned_speech.get("statistics", {}).get("speaker_stats")
            if sp_stats:
                flat_payload["speaker_stats"] = sp_stats
            # Пробрасываем путь к torchcrepe npy и счетчик, если присутствуют внутри pitch_result
            if pitch_result.get("f0_series_torchcrepe_npy"):
                flat_payload["f0_series_torchcrepe_npy"] = pitch_result.get("f0_series_torchcrepe_npy")
                flat_payload["f0_count_torchcrepe"] = int(pitch_result.get("f0_count_torchcrepe", 0) or 0)
        
        return flat_payload
    
    def get_available_extractors(self) -> Dict[str, Dict[str, Any]]:
        """Получение списка доступных экстракторов."""
        return {
            name: extractor.get_info() 
            for name, extractor in self.extractors.items()
        }
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Получение информации о процессоре."""
        return {
            "device": self.device,
            "max_workers": self.max_workers,
            "gpu_memory_limit": self.gpu_memory_limit,
            "sample_rate": self.sample_rate,
            "available_extractors": list(self.extractors.keys()),
            "total_extractors": len(self.extractors)
        }
    
    def run_batch(
        self,
        audio_file_contexts: List[AudioFileContext],
        extractor_names: Optional[List[str]] = None,
        *,
        max_video_workers: Optional[int] = None,
        enable_video_parallel: bool = False,
        enable_gpu_batching: bool = False,
        enable_cpu_parallel: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Батчевая обработка нескольких аудио файлов.
        
        Stage 0-1: Базовая реализация с изоляцией артефактов (последовательная обработка).
        Stage 4: Двухуровневая параллельность и GPU batching.
        
        Args:
            audio_file_contexts: Список AudioFileContext для каждого файла
            extractor_names: Список экстракторов для запуска (None = все доступные)
            max_video_workers: Количество параллельных воркеров для видео (None = использовать self._batch_max_video_workers)
            enable_video_parallel: Включить параллельную обработку нескольких видео (None = использовать self._batch_enable_cpu_parallel)
            enable_gpu_batching: Включить GPU batching для сегментов (None = использовать self._batch_enable_gpu_batching)
            enable_cpu_parallel: Включить CPU параллелизм (None = использовать self._batch_enable_cpu_parallel)
        
        Returns:
            Список словарей с результатами обработки для каждого файла
        """
        if not audio_file_contexts:
            return []
        
        start_time = time.time()
        
        # Используем параметры из аргументов или из self (если не заданы)
        effective_max_video_workers = max_video_workers if max_video_workers is not None else self._batch_max_video_workers
        effective_enable_video_parallel = enable_video_parallel if enable_video_parallel else self._batch_enable_cpu_parallel
        effective_enable_gpu_batching = enable_gpu_batching if enable_gpu_batching else self._batch_enable_gpu_batching
        effective_enable_cpu_parallel = enable_cpu_parallel if enable_cpu_parallel else self._batch_enable_cpu_parallel
        
        if extractor_names is None:
            extractor_names = list(self.extractors.keys())
        
        # Stage 4: Загрузка сегментов для каждого файла (если нужно)
        # Загружаем segments.json для каждого файла, если он еще не загружен
        import json
        for file_ctx in audio_file_contexts:
            if file_ctx.families is None and file_ctx.segments_json_path:
                try:
                    if os.path.exists(file_ctx.segments_json_path):
                        with open(file_ctx.segments_json_path, "r", encoding="utf-8") as f:
                            segments_data = json.load(f)
                            file_ctx.families = segments_data.get("families", {})
                            file_ctx.segments = segments_data.get("segments", [])
                except Exception as e:
                    self.logger.warning(f"Failed to load segments.json for file_id={file_ctx.file_id}: {e}")
        
        # Stage 4: Разделение extractors на GPU (с supports_batch) и CPU
        gpu_batch_extractors: List[str] = []  # Extractors с supports_batch=True
        cpu_extractors: List[str] = []  # Остальные extractors
        
        for extractor_name in extractor_names:
            if extractor_name not in self.extractors:
                continue
            extractor = self.extractors[extractor_name]
            if effective_enable_gpu_batching and getattr(extractor, 'supports_batch', False):
                # Проверяем, поддерживает ли extractor run_segments (нужно для extract_batch_segments)
                if hasattr(extractor, 'run_segments'):
                    gpu_batch_extractors.append(extractor_name)
                else:
                    cpu_extractors.append(extractor_name)
            else:
                cpu_extractors.append(extractor_name)
        
        # Инициализируем результаты для каждого файла
        results: List[Dict[str, Any]] = [
            {
                "file_id": ctx.file_id,
                "video_path": ctx.input_uri,
                "output_dir": ctx.artifacts_dir,
                "success": False,
                "extracted_audio_path": None,
                "extractor_results": {},
                "processing_time": 0.0,
                "errors": [],
                "timings": {
                    "wall_clock": {
                        "start_ts": start_time,
                        "start_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
                        "end_ts": start_time,
                        "end_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
                        "elapsed_s": 0.0,
                    },
                }
            }
            for ctx in audio_file_contexts
        ]
        
        # Создаем индекс file_id -> result index
        file_id_to_result_idx: Dict[str, int] = {
            ctx.file_id: idx for idx, ctx in enumerate(audio_file_contexts)
        }
        
        # Stage 4: Обработка GPU extractors с батчингом (если включено)
        if gpu_batch_extractors and effective_enable_gpu_batching:
            for extractor_name in gpu_batch_extractors:
                extractor = self.extractors[extractor_name]
                
                # Определяем family для этого extractor'а
                # Маппинг extractor -> family (из документации и README)
                family_mapping = {
                    "clap": "clap",
                    "asr": "asr",
                    "speaker_diarization": "diarization",
                    "emotion_diarization": "emotion",
                    "source_separation": "source_separation",
                }
                family_name = family_mapping.get(extractor_name, "primary")
                
                # Собираем сегменты из всех файлов для этого extractor'а
                audio_files_with_segments: List[Dict[str, Any]] = []
                for file_ctx in audio_file_contexts:
                    # Получаем сегменты для этого family
                    segments = []
                    if file_ctx.families:
                        segments = file_ctx.get_segment_family(family_name)
                    elif file_ctx.segments_json_path:
                        # Загружаем segments.json если еще не загружен
                        try:
                            if os.path.exists(file_ctx.segments_json_path):
                                with open(file_ctx.segments_json_path, "r", encoding="utf-8") as f:
                                    segments_data = json.load(f)
                                    file_ctx.families = segments_data.get("families", {})
                                    segments = file_ctx.get_segment_family(family_name)
                        except Exception:
                            pass
                    
                    if segments:
                        audio_files_with_segments.append({
                            "file_id": file_ctx.file_id,
                            "input_uri": file_ctx.input_uri,
                            "tmp_path": file_ctx.tmp_path,
                            "segments": segments,
                        })
                
                if not audio_files_with_segments:
                    self.logger.warning(f"No segments found for extractor={extractor_name}, family={family_name}")
                    continue
                
                # Устанавливаем artifacts_dir для каждого файла перед батчингом
                original_artifacts_dirs: Dict[str, Optional[str]] = {}
                for file_ctx in audio_file_contexts:
                    if file_ctx.file_id in [f["file_id"] for f in audio_files_with_segments]:
                        per_file_artifacts_dir = os.path.join(file_ctx.artifacts_dir, extractor_name, "_artifacts")
                        os.makedirs(per_file_artifacts_dir, exist_ok=True)
                        original_artifacts_dirs[file_ctx.file_id] = getattr(extractor, 'artifacts_dir', None)
                        extractor.artifacts_dir = per_file_artifacts_dir
                
                try:
                    # Вызываем extract_batch_segments для батчинга
                    batch_results = extractor.extract_batch_segments(
                        audio_files_with_segments,
                        max_workers=None,  # Не используется для GPU extractors
                        max_segments_per_batch=self._batch_max_segments_per_gpu_batch,
                    )
                    
                    # Распределяем результаты обратно по файлам
                    for batch_result, file_info in zip(batch_results, audio_files_with_segments):
                        file_id = file_info["file_id"]
                        result_idx = file_id_to_result_idx.get(file_id)
                        if result_idx is not None:
                            results[result_idx]["extractor_results"][extractor_name] = {
                                "success": batch_result.success,
                                "payload": batch_result.payload,
                                "error": batch_result.error,
                                "processing_time": batch_result.processing_time or 0.0,
                                "device_used": batch_result.device_used or "unknown",
                            }
                            if not batch_result.success:
                                results[result_idx]["errors"].append(
                                    f"{extractor_name}: {batch_result.error}"
                                )
                
                finally:
                    # Восстанавливаем оригинальные artifacts_dir
                    if original_artifacts_dirs:
                        # Восстанавливаем для последнего файла (так как artifacts_dir общий)
                        for file_id, original_dir in original_artifacts_dirs.items():
                            if original_dir is not None:
                                extractor.artifacts_dir = original_dir
                                break
        
        # Stage 4: Обработка CPU extractors (с параллелизмом или без)
        if cpu_extractors:
            def process_single_file(file_ctx: AudioFileContext) -> Dict[str, Any]:
                """Обработка одного файла для CPU extractors."""
                try:
                    # Устанавливаем artifacts_dir для каждого extractor'а для этого файла
                    original_artifacts_dirs: Dict[str, Optional[str]] = {}
                    
                    for extractor_name in cpu_extractors:
                        if extractor_name in self.extractors:
                            extractor = self.extractors[extractor_name]
                            original_artifacts_dirs[extractor_name] = getattr(extractor, 'artifacts_dir', None)
                            per_file_artifacts_dir = os.path.join(file_ctx.artifacts_dir, extractor_name, "_artifacts")
                            os.makedirs(per_file_artifacts_dir, exist_ok=True)
                            extractor.artifacts_dir = per_file_artifacts_dir
                    
                    try:
                        # Используем process_video для обработки CPU extractors
                        result = self.process_video(
                            video_path=file_ctx.input_uri,
                            output_dir=file_ctx.artifacts_dir,
                            extractor_names=cpu_extractors,
                            extract_audio=False,
                        )
                        result["file_id"] = file_ctx.file_id
                        return result
                    finally:
                        # Восстанавливаем оригинальные artifacts_dir
                        for extractor_name, original_dir in original_artifacts_dirs.items():
                            if extractor_name in self.extractors:
                                self.extractors[extractor_name].artifacts_dir = original_dir
                
                except Exception as e:
                    self.logger.error(f"Error processing file_id={file_ctx.file_id}: {e}")
                    return {
                        "file_id": file_ctx.file_id,
                        "video_path": file_ctx.input_uri,
                        "output_dir": file_ctx.artifacts_dir,
                        "success": False,
                        "extracted_audio_path": None,
                        "extractor_results": {},
                        "processing_time": 0.0,
                        "errors": [str(e)],
                        "timings": {
                            "wall_clock": {
                                "start_ts": time.time(),
                                "start_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "end_ts": time.time(),
                                "end_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "elapsed_s": 0.0,
                            },
                        }
                    }
            
            # Выбираем режим обработки: параллельный или последовательный
            if effective_enable_video_parallel and len(audio_file_contexts) > 1:
                # Video-level parallelism через ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=effective_max_video_workers) as executor:
                    cpu_results = list(executor.map(process_single_file, audio_file_contexts))
            else:
                # Последовательная обработка
                cpu_results = [process_single_file(ctx) for ctx in audio_file_contexts]
            
            # Объединяем результаты CPU extractors с общими результатами
            for cpu_result in cpu_results:
                file_id = cpu_result.get("file_id")
                result_idx = file_id_to_result_idx.get(file_id)
                if result_idx is not None:
                    # Объединяем extractor_results
                    if "extractor_results" in cpu_result:
                        results[result_idx]["extractor_results"].update(cpu_result["extractor_results"])
                    # Объединяем errors
                    if "errors" in cpu_result:
                        results[result_idx]["errors"].extend(cpu_result["errors"])
                    # Обновляем success (если хотя бы один extractor успешен)
                    if cpu_result.get("success", False):
                        results[result_idx]["success"] = True
        
        # Финальная обработка результатов
        end_time = time.time()
        for result in results:
            result["processing_time"] = end_time - start_time
            result["timings"]["wall_clock"]["end_ts"] = end_time
            result["timings"]["wall_clock"]["end_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_time))
            result["timings"]["wall_clock"]["elapsed_s"] = float(end_time - start_time)
            
            # Определяем общий success
            successful_extractors = sum(
                1 for er in result["extractor_results"].values()
                if er.get("success", False)
            )
            result["success"] = len(result["errors"]) == 0 and successful_extractors > 0
        
        return results
