"""
Парсинг аргументов командной строки для AudioProcessor CLI.
"""
import argparse
from typing import Tuple


def parse_extractors_arg(s: str) -> Tuple[list[str], list[str]]:
    """
    Парсит строку с extractors и возвращает ключи и имена компонентов.
    
    Args:
        s: Строка с extractors через запятую
    
    Returns:
        Tuple[keys, component_names]
    """
    requested = [x.strip() for x in (s or "").split(",") if x.strip()]
    # Не используем дефолтные extractors - если список пустой, возвращаем пустой список
    # Это позволяет использовать список из конфигурации, если он доступен

    # AudioProcessor internal registry keys -> canonical component names
    key_to_component = {
        "clap": "clap_extractor",
        "tempo": "tempo_extractor",
        "loudness": "loudness_extractor",
        "asr": "asr_extractor",
        "speaker_diarization": "speaker_diarization_extractor",
        "emotion_diarization": "emotion_diarization_extractor",
        "source_separation": "source_separation_extractor",
        "speech_analysis": "speech_analysis_extractor",
        "spectral": "spectral_extractor",
        "quality": "quality_extractor",
        "mfcc": "mfcc_extractor",
        "mel": "mel_extractor",
        "onset": "onset_extractor",
        "chroma": "chroma_extractor",
        "rhythmic": "rhythmic_extractor",
        "voice_quality": "voice_quality_extractor",
        "hpss": "hpss_extractor",
        "key": "key_extractor",
        "band_energy": "band_energy_extractor",
        "spectral_entropy": "spectral_entropy_extractor",
        "pitch": "pitch_extractor",
    }
    keys: list[str] = []
    comps: list[str] = []
    for k in requested:
        if k not in key_to_component:
            raise ValueError(f"Unknown audio extractor key: {k}. Expected one of: {sorted(key_to_component.keys())}")
        keys.append(k)
        comps.append(key_to_component[k])
    return keys, comps


def add_asr_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для ASR extractor."""
    parser.add_argument("--asr-model-size", type=str, default="small", choices=["small", "medium", "large"], help="Whisper model size (inprocess via ModelManager)")
    parser.add_argument(
        "--asr-language",
        type=str,
        default="auto",
        help="ASR language hint: 'auto' or language code like 'ru', 'en'. If not auto, passed to Whisper DecodingOptions(language=...).",
    )
    parser.add_argument("--asr-temperature", type=float, default=0.0, help="Whisper decoding temperature (0.0 = deterministic)")
    parser.add_argument("--asr-beam-size", type=int, default=5, help="Whisper decoding beam size")
    parser.add_argument("--asr-best-of", type=int, default=1, help="Whisper decoding best_of (sampling)")
    parser.add_argument(
        "--asr-enable-fallback-decode",
        action="store_true",
        help="Enable fallback decode for difficult audio (re-decode with fallback temperature when avg_logprob is below threshold).",
    )
    parser.add_argument("--asr-fallback-temperature", type=float, default=0.4, help="Fallback decode temperature (used when --asr-enable-fallback-decode)")
    parser.add_argument("--asr-fallback-avg-logprob-threshold", type=float, default=-1.0, help="Fallback trigger: if avg_logprob < threshold, run fallback decode.")
    parser.add_argument(
        "--asr-save-segment-text",
        action="store_true",
        help="Persist per-segment decoded text into ASR NPZ payload (debug/downstream use; may contain raw transcript).",
    )
    # ASR feature gating flags
    parser.add_argument("--asr-enable-token-sequences", action="store_true", help="Enable token_ids_by_segment (sequences)")
    parser.add_argument("--asr-enable-token-counts", action="store_true", help="Enable token_counts (per-segment counts)")
    parser.add_argument("--asr-enable-token-total", action="store_true", help="Enable token_total (aggregate)")
    parser.add_argument("--asr-enable-token-density", action="store_true", help="Enable token_density_per_sec")
    parser.add_argument("--asr-enable-speech-rate", action="store_true", help="Enable speech_rate_wpm")
    parser.add_argument("--asr-enable-lang-distribution", action="store_true", help="Enable lang_distribution")
    parser.add_argument("--asr-enable-segments-with-speech", action="store_true", help="Enable segments_with_speech count")
    parser.add_argument("--asr-enable-avg-segment-duration", action="store_true", help="Enable avg_segment_duration_sec")
    parser.add_argument("--asr-enable-token-variance", action="store_true", help="Enable token_variance")


def add_diarization_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для speaker diarization extractor."""
    parser.add_argument("--diarization-model-size", type=str, default="small", choices=["small", "large"], help="Speaker diarization embedding model size (in-process via ModelManager)")
    parser.add_argument("--diarization-batch-size", type=int, default=None, help="Batch size for diarization inference (None = auto, >100 segments → split)")
    parser.add_argument("--diarization-clustering-method", type=str, default="agglomerative", choices=["agglomerative", "kmeans", "auto"], help="Clustering method (agglomerative=default for training, kmeans=faster, auto=select based on segment count)")
    parser.add_argument("--diarization-speaker-count-method", type=str, default="heuristic", choices=["heuristic", "silhouette", "fixed"], help="Speaker count estimation method (heuristic=default, silhouette=optimal, fixed=use min_speakers)")
    parser.add_argument("--diarization-silence-peak-threshold", type=float, default=1e-3, help="Peak threshold for silence detection")
    parser.add_argument("--diarization-silence-rms-threshold", type=float, default=1e-4, help="RMS threshold for silence detection")
    # Diarization feature gating flags
    parser.add_argument("--diar-enable-speaker-segments", action="store_true", help="Enable speaker_segments (timeline with speaker IDs)")
    parser.add_argument("--diar-enable-speaker-embeddings", action="store_true", help="Enable speaker_embeddings_mean (mean embeddings per speaker)")
    parser.add_argument("--diar-enable-speaker-stats", action="store_true", help="Enable speaker_stats (statistics per speaker)")
    parser.add_argument("--diar-enable-speaker-durations", action="store_true", help="Enable speaker_time_ratios (time ratios per speaker)")
    parser.add_argument("--diar-enable-clustering-metrics", action="store_true", help="Enable clustering_metrics (quality metrics)")
    parser.add_argument("--diar-enable-segment-embeddings", action="store_true", help="Enable segment_embeddings (all individual embeddings)")
    parser.add_argument("--diar-disable-silence-detection", action="store_true", help="Disable silence detection check")


def add_emotion_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для emotion diarization extractor."""
    parser.add_argument("--emotion-model-size", type=str, default="small", choices=["small", "large"], help="Emotion diarization model size (in-process via ModelManager)")
    parser.add_argument("--emotion-batch-size", type=int, default=16, help="Batch size for emotion diarization (default: 16)")
    parser.add_argument("--emotion-silence-peak-threshold", type=float, default=1e-3, help="Peak threshold for silence detection")
    parser.add_argument("--emotion-silence-rms-threshold", type=float, default=1e-4, help="RMS threshold for silence detection")
    # Emotion feature gating flags
    parser.add_argument("--emotion-enable-probs", action="store_true", help="Enable emotion_probs (per-window probabilities)")
    parser.add_argument("--emotion-enable-ids", action="store_true", help="Enable emotion_id (argmax per window)")
    parser.add_argument("--emotion-enable-confidence", action="store_true", help="Enable emotion_confidence (max prob per window)")
    parser.add_argument("--emotion-enable-mean-probs", action="store_true", help="Enable emotion_mean_probs (mean probabilities)")
    parser.add_argument("--emotion-enable-entropy", action="store_true", help="Enable emotion_entropy")
    parser.add_argument("--emotion-enable-dominant", action="store_true", help="Enable dominant_emotion_id/prob")
    parser.add_argument("--emotion-enable-quality-metrics", action="store_true", help="Enable emotion_quality_metrics")
    parser.add_argument("--emotion-disable-silence-detection", action="store_true", help="Disable silence detection check")
    parser.add_argument("--emotion-process-full-audio", action="store_true", help="Process entire audio as one segment (use run() instead of run_segments())")


def add_source_separation_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для source separation extractor."""
    parser.add_argument("--source-separation-model-size", type=str, default="large", choices=["large"], help="Source separation model size (inprocess PyTorch via ModelManager)")
    parser.add_argument("--sep-batch-size", type=int, default=8, help="Batch size for source separation inference")
    parser.add_argument("--sep-silence-peak-threshold", type=float, default=1e-3, help="Peak threshold for silence detection")
    parser.add_argument("--sep-silence-rms-threshold", type=float, default=1e-4, help="RMS threshold for silence detection")
    # Source separation feature gating flags
    parser.add_argument("--sep-enable-share-sequence", action="store_true", help="Enable share_sequence (per-segment shares)")
    parser.add_argument("--sep-enable-energy-sequence", action="store_true", help="Enable energy_sequence (per-segment energies)")
    parser.add_argument("--sep-enable-share-mean", action="store_true", help="Enable share_mean (mean shares)")
    parser.add_argument("--sep-enable-share-std", action="store_true", help="Enable share_std (std shares)")
    parser.add_argument("--sep-enable-quality-metrics", action="store_true", help="Enable source_quality_metrics")
    parser.add_argument("--sep-disable-silence-detection", action="store_true", help="Disable silence detection check")


def add_speech_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для speech analysis extractor."""
    parser.add_argument("--speech-analysis-pitch", action="store_true", help="Enable pitch inside speech_analysis (full-audio, CPU-heavy)")
    parser.add_argument("--speech-silence-peak-threshold", type=float, default=1e-3, help="Peak threshold for silence detection in speech_analysis")
    parser.add_argument("--speech-silence-rms-threshold", type=float, default=1e-4, help="RMS threshold for silence detection in speech_analysis")
    # Speech analysis feature gating flags
    parser.add_argument("--speech-enable-asr-metrics", action="store_true", help="Enable ASR metrics in speech_analysis")
    parser.add_argument("--speech-enable-diarization-metrics", action="store_true", help="Enable diarization metrics in speech_analysis")
    parser.add_argument("--speech-enable-pitch-metrics", action="store_true", help="Enable pitch metrics in speech_analysis")
    parser.add_argument("--speech-disable-silence-detection", action="store_true", help="Disable silence detection in speech_analysis")


def add_pitch_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для pitch extractor."""
    parser.add_argument("--pitch-sample-rate", type=int, default=22050, help="Sample rate for pitch extraction (Hz)")
    parser.add_argument("--pitch-fmin", type=float, default=50.0, help="Minimum frequency for pitch extraction (Hz)")
    parser.add_argument("--pitch-fmax", type=float, default=2000.0, help="Maximum frequency for pitch extraction (Hz)")
    parser.add_argument("--pitch-hop-length", type=int, default=512, help="Hop length for pitch extraction (samples)")
    parser.add_argument("--pitch-frame-length", type=int, default=2048, help="Frame length for pitch extraction (samples)")
    parser.add_argument("--pitch-backend", type=str, default="classic", choices=["classic", "torchcrepe"], help="Backend for pitch extraction")
    parser.add_argument("--pitch-channel-mode", type=str, default="first", choices=["first", "mean", "max"], help="Channel mode for multi-channel audio")
    parser.add_argument("--pitch-torchcrepe-batch-size", type=int, default=1, help="Batch size for torchcrepe")
    # Pitch feature gating flags
    parser.add_argument("--pitch-enable-basic-stats", action="store_true", help="Enable basic pitch statistics (f0_mean, f0_std, f0_min, f0_max, f0_median)")
    parser.add_argument("--pitch-enable-stability-metrics", action="store_true", help="Enable stability metrics (pitch_variation, pitch_stability, pitch_range)")
    parser.add_argument("--pitch-enable-delta-features", action="store_true", help="Enable delta features (f0_delta_mean, f0_delta_std, f0_delta_abs_mean)")
    parser.add_argument("--pitch-enable-method-stats", action="store_true", help="Enable method-specific statistics (PYIN, YIN, torchcrepe)")
    parser.add_argument("--pitch-enable-time-series", action="store_true", help="Enable time series (f0_series_pyin, f0_series_yin, f0_series_torchcrepe)")


def add_spectral_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для spectral extractor."""
    parser.add_argument("--spectral-sample-rate", type=int, default=22050, help="Sample rate for spectral extraction (Hz)")
    parser.add_argument("--spectral-hop-length", type=int, default=512, help="Hop length for spectral extraction (samples)")
    parser.add_argument("--spectral-n-fft", type=int, default=2048, help="FFT window size for spectral extraction (samples)")
    parser.add_argument("--spectral-average-channels", action="store_true", help="Average channels for multi-channel audio")
    parser.add_argument("--spectral-keep-contrast-bands", action="store_true", help="Keep full contrast bands data")
    parser.add_argument("--spectral-enable-normalization", action="store_true", help="Enable audio normalization before processing")
    # Spectral feature gating flags
    parser.add_argument("--spectral-enable-basic-features", action="store_true", help="Enable basic spectral features (centroid, bandwidth, flatness, rolloff, ZCR)")
    parser.add_argument("--spectral-enable-contrast", action="store_true", help="Enable spectral contrast (contrast stats + contrast_bands)")
    parser.add_argument("--spectral-enable-advanced-features", action="store_true", help="Enable advanced features (slope, flatness_db)")
    parser.add_argument("--spectral-enable-time-series", action="store_true", help="Enable time series for all features")


def add_quality_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для quality extractor."""
    parser.add_argument("--quality-sample-rate", type=int, default=22050, help="Sample rate for quality extraction (Hz)")
    parser.add_argument("--quality-frame-len-ms", type=float, default=50.0, help="Frame length for quality analysis (ms)")
    parser.add_argument("--quality-hop-ms", type=float, default=25.0, help="Hop size for quality analysis (ms)")
    parser.add_argument("--quality-clip-threshold", type=float, default=0.999, help="Clipping threshold (0.0-1.0)")
    parser.add_argument("--quality-average-channels", action="store_true", help="Average channels for multi-channel audio")
    parser.add_argument("--quality-enable-normalization", action="store_true", help="Enable audio normalization before processing")
    # Quality feature gating flags
    parser.add_argument("--quality-enable-basic-metrics", action="store_true", help="Enable basic quality metrics (dc_offset, clipping_ratio, crest_factor_db)")
    parser.add_argument("--quality-enable-dynamic-metrics", action="store_true", help="Enable dynamic metrics (dynamic_range_db, snr_db)")
    parser.add_argument("--quality-enable-frame-analysis", action="store_true", help="Enable frame-level analysis metrics")
    parser.add_argument("--quality-enable-time-series", action="store_true", help="Enable time series for all metrics")


def add_voice_quality_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для voice quality extractor."""
    parser.add_argument("--voice-quality-sample-rate", type=int, default=22050, help="Sample rate for voice quality extraction (Hz)")
    parser.add_argument("--voice-quality-hnr-frame-ms", type=float, default=40.0, help="Frame size for HNR computation (ms)")
    parser.add_argument("--voice-quality-rms-mask-threshold", type=float, default=0.01, help="RMS threshold for masking quiet segments")
    parser.add_argument("--voice-quality-f0-fmin", type=float, default=50.0, help="Minimum f0 frequency for estimation (Hz)")
    parser.add_argument("--voice-quality-f0-fmax", type=float, default=500.0, help="Maximum f0 frequency for estimation (Hz)")
    parser.add_argument("--voice-quality-f0-method", type=str, default="yin", choices=["yin", "pyin", "torchcrepe"], help="F0 estimation method")
    parser.add_argument("--voice-quality-average-channels", action="store_true", help="Average channels for multi-channel audio")
    parser.add_argument("--voice-quality-enable-audio-normalization", action="store_true", help="Enable audio normalization before processing")
    # Voice quality feature gating flags
    parser.add_argument("--voice-quality-enable-jitter", action="store_true", help="Enable jitter metric (variability of f0)")
    parser.add_argument("--voice-quality-enable-shimmer", action="store_true", help="Enable shimmer metric (variability of amplitude)")
    parser.add_argument("--voice-quality-enable-hnr", action="store_true", help="Enable HNR-like metric (harmonic-to-noise ratio)")
    parser.add_argument("--voice-quality-enable-f0-stats", action="store_true", help="Enable f0 statistics (mean, std, min, max, stability)")
    parser.add_argument("--voice-quality-enable-time-series", action="store_true", help="Enable time series (f0, amps, hnr values)")


def add_hpss_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для HPSS extractor."""
    parser.add_argument("--hpss-sample-rate", type=int, default=22050, help="Sample rate for HPSS extraction (Hz)")
    parser.add_argument("--hpss-n-fft", type=int, default=2048, help="FFT window size for HPSS")
    parser.add_argument("--hpss-hop-length", type=int, default=512, help="Hop length for STFT in HPSS")
    parser.add_argument("--hpss-average-channels", action="store_true", help="Average channels for multi-channel audio")
    parser.add_argument("--hpss-kernel-size", type=int, default=31, help="Kernel size for HPSS filtering (must be odd)")
    parser.add_argument("--hpss-margin", type=float, default=1.0, help="Margin for HPSS boundaries")
    parser.add_argument("--hpss-power", type=float, default=2.0, help="Power for HPSS normalization")
    parser.add_argument("--hpss-enable-audio-normalization", action="store_true", help="Enable audio normalization before processing")
    # HPSS feature gating flags
    parser.add_argument("--hpss-enable-energy-metrics", action="store_true", help="Enable energy metrics (shares, energies, stability, separation quality, balance score, dominance)")
    parser.add_argument("--hpss-enable-waveforms", action="store_true", help="Enable reconstructed waveforms (harmonic and percussive signals)")
    parser.add_argument("--hpss-enable-spectral-features", action="store_true", help="Enable spectral features from separated components (centroid, bandwidth, rolloff)")
    parser.add_argument("--hpss-enable-time-series", action="store_true", help="Enable time series (harmonic and percussive share series)")


def add_mfcc_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для MFCC extractor."""
    parser.add_argument("--mfcc-sample-rate", type=int, default=22050, help="Sample rate for MFCC extraction (Hz)")
    parser.add_argument("--mfcc-n-mfcc", type=int, default=13, help="Number of MFCC coefficients")
    parser.add_argument("--mfcc-n-fft", type=int, default=2048, help="FFT window size")
    parser.add_argument("--mfcc-hop-length", type=int, default=512, help="Hop length for STFT")
    parser.add_argument("--mfcc-n-mels", type=int, default=128, help="Number of mel filters")
    parser.add_argument("--mfcc-fmin", type=float, default=0.0, help="Minimum frequency (Hz)")
    parser.add_argument("--mfcc-fmax", type=float, default=None, help="Maximum frequency (Hz, None = sample_rate // 2)")
    parser.add_argument("--mfcc-enable-audio-normalization", action="store_true", help="Enable audio normalization before processing (default: True, use --mfcc-disable-audio-normalization to disable)")
    parser.add_argument("--mfcc-disable-audio-normalization", action="store_true", help="Disable audio normalization before processing")
    parser.add_argument("--mfcc-min-gpu-duration-sec", type=float, default=3.0, help="Minimum duration for GPU usage (seconds)")
    parser.add_argument("--mfcc-min-gpu-file-size-mb", type=float, default=5.0, help="Minimum file size for GPU usage (MB)")
    # MFCC feature gating flags
    parser.add_argument("--mfcc-enable-basic-features", action="store_true", help="Enable basic MFCC features (mfcc_features, mfcc_statistics: mean, std, min, max)")
    parser.add_argument("--mfcc-enable-deltas", action="store_true", help="Enable deltas (delta_mean, delta_std, delta_delta_mean, delta_delta_std)")
    parser.add_argument("--mfcc-enable-time-series", action="store_true", help="Enable time series for all features")
    parser.add_argument("--mfcc-enable-normalization", action="store_true", help="Enable MFCC normalization (z-score)")


def add_mel_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для Mel extractor."""
    parser.add_argument("--mel-sample-rate", type=int, default=22050, help="Sample rate for Mel extraction (Hz)")
    parser.add_argument("--mel-n-fft", type=int, default=2048, help="FFT window size")
    parser.add_argument("--mel-hop-length", type=int, default=512, help="Hop length for STFT")
    parser.add_argument("--mel-n-mels", type=int, default=128, help="Number of mel filters")
    parser.add_argument("--mel-fmin", type=float, default=0.0, help="Minimum frequency (Hz)")
    parser.add_argument("--mel-fmax", type=float, default=None, help="Maximum frequency (Hz, None = sample_rate // 2)")
    parser.add_argument("--mel-power", type=float, default=2.0, help="Power for spectrogram (1.0 = magnitude, 2.0 = power)")
    parser.add_argument("--mel-mix-to-mono", action="store_true", default=True, help="Mix to mono (default: True, use --mel-no-mix-to-mono to disable)")
    parser.add_argument("--mel-no-mix-to-mono", action="store_true", help="Disable mix to mono")
    parser.add_argument("--mel-enable-audio-normalization", action="store_true", help="Enable audio normalization before processing (default: True, use --mel-disable-audio-normalization to disable)")
    parser.add_argument("--mel-disable-audio-normalization", action="store_true", help="Disable audio normalization before processing")
    # Mel feature gating flags
    parser.add_argument("--mel-enable-basic-features", action="store_true", help="Enable basic Mel features (mel_spectrogram, mel_shape, mel_elements)")
    parser.add_argument("--mel-enable-statistics", action="store_true", help="Enable statistics (mel_mean, mel_std, mel_min, mel_max, freq_mean, freq_std)")
    parser.add_argument("--mel-enable-spectral-features", action="store_true", help="Enable spectral features (spectral_centroid, spectral_bandwidth)")
    parser.add_argument("--mel-enable-time-series", action="store_true", help="Enable time series for all features")
    parser.add_argument("--mel-enable-stats-vector", action="store_true", help="Enable compact stats vector (mel_stats_vector)")


def add_onset_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для onset extractor."""
    parser.add_argument("--onset-sample-rate", type=int, default=22050, help="Sample rate for onset extraction (Hz)")
    parser.add_argument("--onset-hop-length", type=int, default=512, help="Hop length for onset analysis")
    parser.add_argument("--onset-pre-max", type=int, default=3, help="Number of frames before maximum for peak detector")
    parser.add_argument("--onset-post-max", type=int, default=3, help="Number of frames after maximum for peak detector")
    parser.add_argument("--onset-pre-avg", type=int, default=3, help="Number of frames before for averaging")
    parser.add_argument("--onset-post-avg", type=int, default=5, help="Number of frames after for averaging")
    parser.add_argument("--onset-delta", type=float, default=0.2, help="Minimum difference for onset detection")
    parser.add_argument("--onset-wait", type=int, default=10, help="Minimum number of frames between onsets")
    parser.add_argument("--onset-backend", type=str, default="librosa", choices=["librosa", "essentia"], help="Backend for onset detection (librosa or essentia)")
    parser.add_argument("--onset-units", type=str, default="time", choices=["time", "frames"], help="Units for onsets (time or frames)")
    parser.add_argument("--onset-backtrack", action="store_true", help="Enable backtrack for onset detection")
    parser.add_argument("--onset-energy", action="store_true", help="Use energy detector")
    parser.add_argument("--onset-normalize", action="store_true", help="Normalize onset envelope")
    parser.add_argument("--onset-enable-audio-normalization", action="store_true", help="Enable audio normalization before processing")
    # Onset feature gating flags
    parser.add_argument("--onset-enable-basic-features", action="store_true", help="Enable basic onset features (onset_times, onset_count)")
    parser.add_argument("--onset-enable-interval-stats", action="store_true", help="Enable interval statistics (interval_std, interval_min, etc.)")
    parser.add_argument("--onset-enable-rhythmic-metrics", action="store_true", help="Enable rhythmic metrics (regularity, clustering, etc.)")
    parser.add_argument("--onset-enable-time-series", action="store_true", help="Enable time series (onset_times as time series)")


def add_chroma_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для chroma extractor."""
    parser.add_argument("--chroma-sample-rate", type=int, default=22050, help="Sample rate for chroma extraction (Hz)")
    parser.add_argument("--chroma-hop-length", type=int, default=512, help="Hop length for STFT/CQT")
    parser.add_argument("--chroma-n-fft", type=int, default=4096, help="FFT window size (for STFT mode)")
    parser.add_argument("--chroma-mix-to-mono", action="store_true", default=True, help="Mix to mono (default: True, use --chroma-no-mix-to-mono to disable)")
    parser.add_argument("--chroma-no-mix-to-mono", action="store_true", help="Disable mix to mono")
    parser.add_argument("--chroma-type", type=str, default="cqt", choices=["cqt", "stft"], help="Chroma type (cqt or stft)")
    parser.add_argument("--chroma-normalize", type=str, default="l1", choices=["none", "l1", "l2"], help="Normalization type (none, l1, or l2)")
    parser.add_argument("--chroma-n-chroma", type=int, default=12, help="Number of chroma classes")
    parser.add_argument("--chroma-fmin", type=float, default=None, help="Minimum frequency (Hz, None = default)")
    parser.add_argument("--chroma-fmax", type=float, default=None, help="Maximum frequency (Hz, None = default)")
    parser.add_argument("--chroma-n-bins", type=int, default=None, help="Number of bins for CQT (None = default)")
    parser.add_argument("--chroma-enable-audio-normalization", action="store_true", help="Enable audio normalization before processing")
    # Chroma feature gating flags
    parser.add_argument("--chroma-enable-basic-stats", action="store_true", help="Enable basic statistics (chroma_mean, chroma_std, chroma_min, chroma_max)")
    parser.add_argument("--chroma-enable-extended-stats", action="store_true", help="Enable extended statistics (chroma_median, chroma_p25, chroma_p75)")
    parser.add_argument("--chroma-enable-stats-vector", action="store_true", help="Enable compact stats vector (chroma_stats_vector)")
    parser.add_argument("--chroma-enable-time-series", action="store_true", help="Enable time series (chroma spectrogram)")


def add_rhythmic_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для rhythmic extractor."""
    parser.add_argument("--rhythmic-sample-rate", type=int, default=22050, help="Sample rate for rhythmic extraction (Hz)")
    parser.add_argument("--rhythmic-hop-length", type=int, default=512, help="Hop length for beat tracking")
    parser.add_argument("--rhythmic-backend", type=str, default="librosa", choices=["librosa", "essentia"], help="Backend for beat tracking (librosa or essentia)")
    parser.add_argument("--rhythmic-start-bpm", type=float, default=None, help="Initial BPM for librosa beat tracking")
    parser.add_argument("--rhythmic-std-bpm", type=float, default=None, help="Standard deviation of BPM for librosa")
    parser.add_argument("--rhythmic-ac-size", type=int, default=4, help="Autocorrelation size for librosa")
    parser.add_argument("--rhythmic-max-tempo", type=float, default=None, help="Maximum tempo for librosa")
    parser.add_argument("--rhythmic-enable-audio-normalization", action="store_true", help="Enable audio normalization before processing")
    # Rhythmic feature gating flags
    parser.add_argument("--rhythmic-enable-basic-metrics", action="store_true", help="Enable basic metrics (tempo_bpm, beats_count, beat_density)")
    parser.add_argument("--rhythmic-enable-interval-stats", action="store_true", help="Enable interval statistics (avg_period, std_period, min/max/median)")
    parser.add_argument("--rhythmic-enable-regularity-metrics", action="store_true", help="Enable regularity metrics (regularity, syncopation, etc.)")
    parser.add_argument("--rhythmic-enable-beat-times", action="store_true", help="Enable beat times (beat_times array)")
    parser.add_argument("--rhythmic-enable-tempo-metrics", action="store_true", help="Enable tempo metrics (median_bpm, tempo_variation, etc.)")


def add_key_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для key extractor."""
    parser.add_argument("--key-sample-rate", type=int, default=22050, help="Sample rate for key extraction (Hz)")
    parser.add_argument("--key-hop-length", type=int, default=512, help="Hop length for STFT/CQT")
    parser.add_argument("--key-chroma-type", type=str, default="cqt", choices=["cqt", "stft"], help="Chroma type (cqt or stft)")
    parser.add_argument("--key-use-beat-sync", action="store_true", help="Aggregate chroma per beat (requires beat tracking)")
    parser.add_argument("--key-top-k", type=int, default=3, help="Number of top-K keys to return")
    parser.add_argument("--key-method", type=str, default="auto", choices=["essentia", "librosa", "auto"], help="Key detection method (essentia, librosa, or auto)")
    parser.add_argument("--key-confidence-threshold", type=float, default=0.3, help="Confidence threshold for warnings (0.0-1.0)")
    parser.add_argument("--key-enable-audio-normalization", action="store_true", help="Enable audio normalization before processing")
    # Key feature gating flags
    parser.add_argument("--key-enable-detailed-scores", action="store_true", help="Enable detailed scores (24 key scores)")
    parser.add_argument("--key-enable-top-k", action="store_true", help="Enable top-K alternative keys")
    parser.add_argument("--key-enable-time-series", action="store_true", help="Enable time series (for run_segments)")
    parser.add_argument("--key-enable-key-changes", action="store_true", help="Enable key change detection (for run_segments)")
    parser.add_argument("--key-enable-stability-metrics", action="store_true", help="Enable stability metrics (for run_segments)")


def add_band_energy_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для band energy extractor."""
    parser.add_argument("--band-energy-sample-rate", type=int, default=22050, help="Sample rate for band energy extraction (Hz)")
    parser.add_argument("--band-energy-n-fft", type=int, default=2048, help="FFT window size")
    parser.add_argument("--band-energy-hop-length", type=int, default=512, help="Hop length for STFT")
    parser.add_argument("--band-energy-use-mel-bands", action="store_true", default=True, help="Use mel scale bands (default: True, use --band-energy-no-mel-bands to disable)")
    parser.add_argument("--band-energy-no-mel-bands", action="store_true", help="Disable mel scale bands (use fixed bands)")
    parser.add_argument("--band-energy-n-mels", type=int, default=3, help="Number of mel bands (if use_mel_bands=True)")
    parser.add_argument("--band-energy-method", type=str, default="auto", choices=["essentia", "librosa", "auto"], help="Band energy method (essentia, librosa, or auto)")
    parser.add_argument("--band-energy-average-channels", action="store_true", help="Average channels for multi-channel audio")
    parser.add_argument("--band-energy-enable-audio-normalization", action="store_true", help="Enable audio normalization before processing")
    # Band energy feature gating flags
    parser.add_argument("--band-energy-enable-basic-stats", action="store_true", help="Enable basic statistics (mean, std, median)")
    parser.add_argument("--band-energy-enable-extended-stats", action="store_true", help="Enable extended statistics (min, max, p25, p75)")
    parser.add_argument("--band-energy-enable-time-series", action="store_true", help="Enable time series (band_energy_ts)")
    parser.add_argument("--band-energy-enable-dynamics", action="store_true", help="Enable dynamics metrics (for run_segments)")
    parser.add_argument("--band-energy-enable-balance-metrics", action="store_true", help="Enable balance metrics")


def add_spectral_entropy_arguments(parser: argparse.ArgumentParser) -> None:
    """Добавляет аргументы для spectral entropy extractor."""
    parser.add_argument("--spectral-entropy-sample-rate", type=int, default=22050, help="Sample rate for spectral entropy extraction (Hz)")
    parser.add_argument("--spectral-entropy-n-fft", type=int, default=2048, help="FFT window size")
    parser.add_argument("--spectral-entropy-hop-length", type=int, default=512, help="Hop length for STFT")
    parser.add_argument("--spectral-entropy-average-channels", action="store_true", default=True, help="Average channels for multi-channel audio (default: True)")
    parser.add_argument("--spectral-entropy-no-average-channels", action="store_true", help="Disable channel averaging")
    parser.add_argument("--spectral-entropy-smoothing-window", type=int, default=0, help="Smoothing window size (0 = no smoothing)")
    parser.add_argument("--spectral-entropy-use-mel", action="store_true", help="Use mel scale instead of linear")
    parser.add_argument("--spectral-entropy-n-mels", type=int, default=128, help="Number of mel filters (if use_mel=True)")
    parser.add_argument("--spectral-entropy-enable-audio-normalization", action="store_true", help="Enable audio normalization before processing")
    # Spectral Entropy feature gating flags
    parser.add_argument("--spectral-entropy-enable-basic-stats", action="store_true", help="Enable basic statistics (mean, std) for entropy")
    parser.add_argument("--spectral-entropy-enable-flatness", action="store_true", help="Enable flatness metrics")
    parser.add_argument("--spectral-entropy-enable-spread", action="store_true", help="Enable spread metrics")
    parser.add_argument("--spectral-entropy-enable-time-series", action="store_true", help="Enable time series for all metrics")
    parser.add_argument("--spectral-entropy-enable-extended-stats", action="store_true", help="Enable extended statistics (min, max, p25, p75)")
    parser.add_argument("--spectral-entropy-enable-dynamics", action="store_true", help="Enable dynamics metrics (for run_segments)")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Создает и настраивает парсер аргументов командной строки.
    
    Returns:
        Настроенный ArgumentParser
    """
    parser = argparse.ArgumentParser(description="AudioProcessor CLI (per-run NPZ artifacts)")
    # NOTE: AudioProcessor does NOT extract audio from video. Segmenter provides audio/audio.wav + audio/segments.json.
    # --video-path is optional and only used for convenience (e.g., deriving video_id if --video-id is not provided).
    parser.add_argument("--video-path", type=str, required=False, default=None, help="Optional video path (metadata-only). Audio is always taken from --frames-dir/audio/audio.wav (Segmenter contract).")
    parser.add_argument("--frames-dir", type=str, required=False, default=None, help="Segmenter output dir for this video: <Segmenter/output>/<video_id> (must contain audio/audio.wav and audio/segments.json). Required for single-file mode, optional for batch mode.")
    
    # Output directory arguments (aliases for convenience)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--rs-base", type=str, default=None, help="Base result_store path (will create per-run subdir). Default: ./result_store")
    output_group.add_argument("--output-dir", type=str, default=None, dest="rs_base", help="Alias for --rs-base: output directory for results")
    output_group.add_argument("--result-dir", type=str, default=None, dest="rs_base", help="Alias for --rs-base: result directory for artifacts")
    parser.add_argument(
        "--run-rs-path",
        type=str,
        default=None,
        help="Explicit per-run result_store directory (overrides --rs-base/platform/video/run). "
             "Expected: <rs_base>/<platform_id>/<video_id>/<run_id>",
    )
    parser.add_argument("--platform-id", type=str, default="youtube")
    parser.add_argument("--video-id", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--sampling-policy-version", type=str, default="v1")
    parser.add_argument(
        "--config-hash",
        type=str,
        default=None,
        help="Optional config hash propagated by DataProcessor (for idempotency). If not provided, will be derived from CLI args.",
    )
    parser.add_argument("--dataprocessor-version", type=str, default="unknown")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    # Scheduler-controlled knobs (L2/L3) - legacy, используйте --extractor-parallelism-config для индивидуальных настроек
    parser.add_argument("--segment-parallelism", type=int, default=1, help="Concurrent segment workers (where supported) - legacy, используйте индивидуальные настройки в extractors")
    parser.add_argument("--max-inflight", type=int, default=None, help="Max in-flight segment tasks (safety cap; default: segment_parallelism) - legacy")
    parser.add_argument("--clap-batch-size", type=int, default=1, help="CLAP micro-batch size (may increase VRAM) - legacy, используйте индивидуальные настройки в extractors")
    parser.add_argument("--extractor-parallelism-config", type=str, default=None, help="JSON с индивидуальными настройками parallelism для каждого extractor'а (из global_config.yaml)")
    parser.add_argument("--extractor-config", type=str, default=None, help="JSON с полной конфигурацией extractors (для render флагов и других настроек из global_config.yaml)")
    parser.add_argument(
        "--extractors",
        type=str,
        default="clap,tempo,loudness",
        help=(
            "Comma-separated keys. Baseline Tier-0: clap,tempo,loudness. "
            "Additional: asr,speaker_diarization,emotion_diarization,source_separation,speech_analysis"
        ),
    )
    
    # Добавляем аргументы для всех extractors
    add_asr_arguments(parser)
    add_diarization_arguments(parser)
    add_emotion_arguments(parser)
    add_source_separation_arguments(parser)
    add_speech_analysis_arguments(parser)
    add_pitch_arguments(parser)
    add_spectral_arguments(parser)
    add_quality_arguments(parser)
    add_voice_quality_arguments(parser)
    add_hpss_arguments(parser)
    add_mfcc_arguments(parser)
    add_mel_arguments(parser)
    add_onset_arguments(parser)
    add_chroma_arguments(parser)
    add_rhythmic_arguments(parser)
    add_key_arguments(parser)
    add_band_energy_arguments(parser)
    add_spectral_entropy_arguments(parser)
    
    # Batch processing arguments
    parser.add_argument(
        "--audio-input-dir",
        type=str,
        default=None,
        help="Directory containing multiple video frames_dirs (batch mode). Each subdirectory should contain audio/audio.wav and audio/segments.json",
    )
    parser.add_argument(
        "--audio-input-list",
        type=str,
        default=None,
        help="Path to text file with list of frames_dir paths (one per line) for batch processing",
    )
    parser.add_argument(
        "--batch-max-workers",
        type=int,
        default=None,
        help="Override max_workers for batch processing (null = auto, typically os.cpu_count())",
    )
    parser.add_argument(
        "--no-batch-gpu",
        action="store_true",
        help="Disable GPU batching for batch processing",
    )
    parser.add_argument(
        "--no-batch-cpu-parallel",
        action="store_true",
        help="Disable CPU parallelism for batch processing",
    )
    parser.add_argument(
        "--batch-max-segments-per-gpu-batch",
        type=int,
        default=None,
        help="Limit batch size for GPU extractors (null = no limit)",
    )
    parser.add_argument("--write-legacy-manifest", action="store_true", help="Also write legacy AudioProcessor JSON manifest into tmp dir")
    parser.add_argument(
        "--no-strict-extractors",
        action="store_true",
        help="Graceful degradation instead of fail-fast if extractor initialization fails (for debugging)",
    )
    
    return parser
