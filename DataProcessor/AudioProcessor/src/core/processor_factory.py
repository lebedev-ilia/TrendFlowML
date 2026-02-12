#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, List

from src.core.main_processor import MainProcessor  # type: ignore


def create_main_processor(args: Any, extractor_keys: List[str], extractor_config: Any = None) -> MainProcessor:
    """
    Create MainProcessor instance from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        extractor_keys: List of enabled extractor keys
        extractor_config: Optional extractor config dict from global_config.yaml (for feature flags fallback)
        
    Returns:
        Configured MainProcessor instance
    """
    # Helper to get feature flag with fallback to extractor_config
    def get_key_flag(flag_name: str, default: bool = False) -> bool:
        import logging
        logger = logging.getLogger(__name__)
        
        # First try CLI args
        cli_attr_name = f"key_{flag_name}"
        cli_value = getattr(args, cli_attr_name, None)
        if cli_value is not None:
            result = bool(cli_value)
            logger.info(f"key_extractor | Flag '{flag_name}': {result} (from CLI args: {cli_attr_name})")
            return result
        
        # Fallback to extractor_config
        if extractor_config and "key" in extractor_config:
            key_cfg = extractor_config["key"]
            feature_flags = key_cfg.get("feature_flags", {})
            result = bool(feature_flags.get(flag_name, default))
            logger.info(f"key_extractor | Flag '{flag_name}': {result} (from extractor_config, feature_flags={feature_flags})")
            return result
        
        logger.info(f"key_extractor | Flag '{flag_name}': {default} (default value)")
        return default
    
    # Log final key extractor flags before creating MainProcessor
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"key_extractor | Final flags passed to MainProcessor:")
    logger.info(f"  enable_audio_normalization={get_key_flag('enable_audio_normalization', False)}")
    logger.info(f"  enable_detailed_scores={get_key_flag('enable_detailed_scores', False)}")
    logger.info(f"  enable_top_k={get_key_flag('enable_top_k', False)}")
    logger.info(f"  enable_time_series={get_key_flag('enable_time_series', False)}")
    logger.info(f"  enable_key_changes={get_key_flag('enable_key_changes', False)}")
    logger.info(f"  enable_stability_metrics={get_key_flag('enable_stability_metrics', False)}")
    
    return MainProcessor(
        device=args.device,
        # Note: MainProcessor.max_workers is internal legacy; per-segment concurrency is controlled in this CLI for now.
        asr_model_size=str(args.asr_model_size),
        asr_language=str(getattr(args, "asr_language", "auto")),
        asr_temperature=float(getattr(args, "asr_temperature", 0.0)),
        asr_beam_size=int(getattr(args, "asr_beam_size", 5)),
        asr_best_of=int(getattr(args, "asr_best_of", 1)),
        asr_enable_fallback_decode=bool(getattr(args, "asr_enable_fallback_decode", False)),
        asr_fallback_temperature=float(getattr(args, "asr_fallback_temperature", 0.4)),
        asr_fallback_avg_logprob_threshold=float(getattr(args, "asr_fallback_avg_logprob_threshold", -1.0)),
        asr_save_segment_text=bool(getattr(args, "asr_save_segment_text", False)),
        asr_enable_token_sequences=bool(args.asr_enable_token_sequences),
        asr_enable_token_counts=bool(args.asr_enable_token_counts),
        asr_enable_token_total=bool(args.asr_enable_token_total),
        asr_enable_token_density=bool(args.asr_enable_token_density),
        asr_enable_speech_rate=bool(args.asr_enable_speech_rate),
        asr_enable_lang_distribution=bool(args.asr_enable_lang_distribution),
        asr_enable_segments_with_speech=bool(args.asr_enable_segments_with_speech),
        asr_enable_avg_segment_duration=bool(args.asr_enable_avg_segment_duration),
        asr_enable_token_variance=bool(args.asr_enable_token_variance),
        diarization_model_size=str(args.diarization_model_size),
        diarization_batch_size=args.diarization_batch_size,
        diarization_clustering_method=str(args.diarization_clustering_method),
        diarization_speaker_count_method=str(args.diarization_speaker_count_method),
        diarization_silence_peak_threshold=float(args.diarization_silence_peak_threshold),
        diarization_silence_rms_threshold=float(args.diarization_silence_rms_threshold),
        diar_enable_speaker_segments=bool(args.diar_enable_speaker_segments),
        diar_enable_speaker_embeddings=bool(args.diar_enable_speaker_embeddings),
        diar_enable_speaker_stats=bool(args.diar_enable_speaker_stats),
        diar_enable_speaker_durations=bool(args.diar_enable_speaker_durations),
        diar_enable_clustering_metrics=bool(args.diar_enable_clustering_metrics),
        diar_enable_segment_embeddings=bool(args.diar_enable_segment_embeddings),
        diar_enable_silence_detection=bool(not args.diar_disable_silence_detection),
        emotion_model_size=str(args.emotion_model_size),
        emotion_batch_size=int(args.emotion_batch_size or 16),
        emotion_silence_peak_threshold=float(args.emotion_silence_peak_threshold),
        emotion_silence_rms_threshold=float(args.emotion_silence_rms_threshold),
        emotion_enable_probs=bool(args.emotion_enable_probs),
        emotion_enable_ids=bool(args.emotion_enable_ids),
        emotion_enable_confidence=bool(args.emotion_enable_confidence),
        emotion_enable_mean_probs=bool(args.emotion_enable_mean_probs),
        emotion_enable_entropy=bool(args.emotion_enable_entropy),
        emotion_enable_dominant=bool(args.emotion_enable_dominant),
        emotion_enable_quality_metrics=bool(args.emotion_enable_quality_metrics),
        emotion_enable_silence_detection=bool(not args.emotion_disable_silence_detection),
        emotion_process_full_audio=bool(getattr(args, "emotion_process_full_audio", False)),
        source_separation_model_size=str(args.source_separation_model_size),
        sep_batch_size=int(getattr(args, "sep_batch_size", 8)),
        sep_silence_peak_threshold=float(args.sep_silence_peak_threshold),
        sep_silence_rms_threshold=float(args.sep_silence_rms_threshold),
        sep_enable_share_sequence=bool(args.sep_enable_share_sequence),
        sep_enable_energy_sequence=bool(args.sep_enable_energy_sequence),
        sep_enable_share_mean=bool(args.sep_enable_share_mean),
        sep_enable_share_std=bool(args.sep_enable_share_std),
        sep_enable_quality_metrics=bool(args.sep_enable_quality_metrics),
        sep_enable_silence_detection=bool(not args.sep_disable_silence_detection),
        speech_analysis_pitch_enabled=bool(args.speech_analysis_pitch),
        speech_enable_asr_metrics=bool(getattr(args, "speech_enable_asr_metrics", False)),
        speech_enable_diarization_metrics=bool(getattr(args, "speech_enable_diarization_metrics", False)),
        speech_enable_pitch_metrics=bool(getattr(args, "speech_enable_pitch_metrics", False)),
        speech_silence_peak_threshold=float(getattr(args, "speech_silence_peak_threshold", 1e-3)),
        speech_silence_rms_threshold=float(getattr(args, "speech_silence_rms_threshold", 1e-4)),
        speech_enable_silence_detection=bool(not getattr(args, "speech_disable_silence_detection", False)),
        quality_sample_rate=int(args.quality_sample_rate),
        quality_frame_len_ms=float(args.quality_frame_len_ms),
        quality_hop_ms=float(args.quality_hop_ms),
        quality_clip_threshold=float(args.quality_clip_threshold),
        quality_average_channels=bool(args.quality_average_channels),
        quality_enable_normalization=bool(args.quality_enable_normalization),
        quality_enable_basic_metrics=bool(args.quality_enable_basic_metrics),
        quality_enable_dynamic_metrics=bool(args.quality_enable_dynamic_metrics),
        quality_enable_frame_analysis=bool(args.quality_enable_frame_analysis),
        quality_enable_time_series=bool(args.quality_enable_time_series),
        spectral_sample_rate=int(args.spectral_sample_rate),
        spectral_hop_length=int(args.spectral_hop_length),
        spectral_n_fft=int(args.spectral_n_fft),
        spectral_average_channels=bool(args.spectral_average_channels),
        spectral_keep_contrast_bands=bool(args.spectral_keep_contrast_bands),
        spectral_enable_normalization=bool(args.spectral_enable_normalization),
        spectral_enable_basic_features=bool(args.spectral_enable_basic_features),
        spectral_enable_contrast=bool(args.spectral_enable_contrast),
        spectral_enable_advanced_features=bool(args.spectral_enable_advanced_features),
        spectral_enable_time_series=bool(args.spectral_enable_time_series),
        mfcc_sample_rate=int(args.mfcc_sample_rate),
        mfcc_n_mfcc=int(args.mfcc_n_mfcc),
        mfcc_n_fft=int(args.mfcc_n_fft),
        mfcc_hop_length=int(args.mfcc_hop_length),
        mfcc_n_mels=int(args.mfcc_n_mels),
        mfcc_fmin=float(args.mfcc_fmin),
        mfcc_fmax=float(args.mfcc_fmax) if args.mfcc_fmax is not None else None,
        mfcc_enable_audio_normalization=bool(not getattr(args, 'mfcc_disable_audio_normalization', False)),  # Default: True
        mfcc_min_gpu_duration_sec=float(args.mfcc_min_gpu_duration_sec),
        mfcc_min_gpu_file_size_mb=float(args.mfcc_min_gpu_file_size_mb),
        mfcc_enable_basic_features=bool(args.mfcc_enable_basic_features),
        mfcc_enable_deltas=bool(args.mfcc_enable_deltas),
        mfcc_enable_time_series=bool(args.mfcc_enable_time_series),
        mfcc_enable_normalization=bool(args.mfcc_enable_normalization),
        mel_sample_rate=int(args.mel_sample_rate),
        mel_n_fft=int(args.mel_n_fft),
        mel_hop_length=int(args.mel_hop_length),
        mel_n_mels=int(args.mel_n_mels),
        mel_fmin=float(args.mel_fmin),
        mel_fmax=float(args.mel_fmax) if args.mel_fmax is not None else None,
        mel_power=float(args.mel_power),
        mel_mix_to_mono=bool(not getattr(args, 'mel_no_mix_to_mono', False)),  # Default: True
        mel_enable_audio_normalization=bool(not getattr(args, 'mel_disable_audio_normalization', False)),  # Default: True
        mel_enable_basic_features=bool(args.mel_enable_basic_features),
        mel_enable_statistics=bool(args.mel_enable_statistics),
        mel_enable_spectral_features=bool(args.mel_enable_spectral_features),
        mel_enable_time_series=bool(args.mel_enable_time_series),
        mel_enable_stats_vector=bool(args.mel_enable_stats_vector),
        onset_sample_rate=int(args.onset_sample_rate),
        onset_hop_length=int(args.onset_hop_length),
        onset_pre_max=int(args.onset_pre_max),
        onset_post_max=int(args.onset_post_max),
        onset_pre_avg=int(args.onset_pre_avg),
        onset_post_avg=int(args.onset_post_avg),
        onset_delta=float(args.onset_delta),
        onset_wait=int(args.onset_wait),
        onset_backend=str(args.onset_backend),
        onset_units=str(args.onset_units),
        onset_backtrack=bool(args.onset_backtrack),
        onset_energy=bool(args.onset_energy),
        onset_normalize=bool(args.onset_normalize),
        onset_enable_audio_normalization=bool(args.onset_enable_audio_normalization),
        onset_enable_basic_features=bool(args.onset_enable_basic_features),
        onset_enable_interval_stats=bool(args.onset_enable_interval_stats),
        onset_enable_rhythmic_metrics=bool(args.onset_enable_rhythmic_metrics),
        onset_enable_time_series=bool(args.onset_enable_time_series),
        chroma_sample_rate=int(args.chroma_sample_rate),
        chroma_hop_length=int(args.chroma_hop_length),
        chroma_n_fft=int(args.chroma_n_fft),
        chroma_mix_to_mono=bool(not getattr(args, 'chroma_no_mix_to_mono', False)),  # Default: True
        chroma_type=str(args.chroma_type),
        chroma_normalize=None if args.chroma_normalize == "none" else str(args.chroma_normalize),
        chroma_n_chroma=int(args.chroma_n_chroma),
        chroma_fmin=float(args.chroma_fmin) if args.chroma_fmin is not None else None,
        chroma_fmax=float(args.chroma_fmax) if args.chroma_fmax is not None else None,
        chroma_n_bins=int(args.chroma_n_bins) if args.chroma_n_bins is not None else None,
        chroma_enable_audio_normalization=bool(args.chroma_enable_audio_normalization),
        chroma_enable_basic_stats=bool(args.chroma_enable_basic_stats),
        chroma_enable_extended_stats=bool(args.chroma_enable_extended_stats),
        chroma_enable_stats_vector=bool(args.chroma_enable_stats_vector),
        chroma_enable_time_series=bool(args.chroma_enable_time_series),
        rhythmic_sample_rate=int(args.rhythmic_sample_rate),
        rhythmic_hop_length=int(args.rhythmic_hop_length),
        rhythmic_backend=str(args.rhythmic_backend),
        rhythmic_start_bpm=args.rhythmic_start_bpm,
        rhythmic_std_bpm=args.rhythmic_std_bpm,
        rhythmic_ac_size=int(args.rhythmic_ac_size),
        rhythmic_max_tempo=args.rhythmic_max_tempo,
        rhythmic_enable_audio_normalization=bool(args.rhythmic_enable_audio_normalization),
        rhythmic_enable_basic_metrics=bool(args.rhythmic_enable_basic_metrics),
        rhythmic_enable_interval_stats=bool(args.rhythmic_enable_interval_stats),
        rhythmic_enable_regularity_metrics=bool(args.rhythmic_enable_regularity_metrics),
        rhythmic_enable_beat_times=bool(args.rhythmic_enable_beat_times),
        rhythmic_enable_tempo_metrics=bool(args.rhythmic_enable_tempo_metrics),
        band_energy_sample_rate=int(args.band_energy_sample_rate),
        band_energy_n_fft=int(args.band_energy_n_fft),
        band_energy_hop_length=int(args.band_energy_hop_length),
        band_energy_use_mel_bands=bool(not getattr(args, 'band_energy_no_mel_bands', False)),  # Default: True
        band_energy_n_mels=int(args.band_energy_n_mels),
        band_energy_method=str(args.band_energy_method),
        band_energy_average_channels=bool(getattr(args, 'band_energy_average_channels', True)),  # Default: True
        band_energy_enable_audio_normalization=bool(args.band_energy_enable_audio_normalization),
        band_energy_enable_basic_stats=bool(args.band_energy_enable_basic_stats),
        band_energy_enable_extended_stats=bool(args.band_energy_enable_extended_stats),
        band_energy_enable_time_series=bool(args.band_energy_enable_time_series),
        band_energy_enable_dynamics=bool(args.band_energy_enable_dynamics),
        band_energy_enable_balance_metrics=bool(args.band_energy_enable_balance_metrics),
        spectral_entropy_sample_rate=int(args.spectral_entropy_sample_rate),
        spectral_entropy_n_fft=int(args.spectral_entropy_n_fft),
        spectral_entropy_hop_length=int(args.spectral_entropy_hop_length),
        spectral_entropy_average_channels=bool(args.spectral_entropy_average_channels) if not getattr(args, 'spectral_entropy_no_average_channels', False) else False,
        spectral_entropy_smoothing_window=int(args.spectral_entropy_smoothing_window),
        spectral_entropy_use_mel=bool(args.spectral_entropy_use_mel),
        spectral_entropy_n_mels=int(args.spectral_entropy_n_mels),
        spectral_entropy_enable_audio_normalization=bool(args.spectral_entropy_enable_audio_normalization),
        spectral_entropy_enable_basic_stats=bool(args.spectral_entropy_enable_basic_stats),
        spectral_entropy_enable_flatness=bool(args.spectral_entropy_enable_flatness),
        spectral_entropy_enable_spread=bool(args.spectral_entropy_enable_spread),
        spectral_entropy_enable_time_series=bool(args.spectral_entropy_enable_time_series),
        spectral_entropy_enable_extended_stats=bool(args.spectral_entropy_enable_extended_stats),
        spectral_entropy_enable_dynamics=bool(args.spectral_entropy_enable_dynamics),
        voice_quality_sample_rate=int(args.voice_quality_sample_rate),
        voice_quality_hnr_frame_ms=float(args.voice_quality_hnr_frame_ms),
        voice_quality_rms_mask_threshold=float(args.voice_quality_rms_mask_threshold),
        voice_quality_f0_fmin=float(args.voice_quality_f0_fmin),
        voice_quality_f0_fmax=float(args.voice_quality_f0_fmax),
        voice_quality_f0_method=str(args.voice_quality_f0_method),
        voice_quality_average_channels=bool(args.voice_quality_average_channels),
        voice_quality_enable_audio_normalization=bool(args.voice_quality_enable_audio_normalization),
        voice_quality_enable_jitter=bool(args.voice_quality_enable_jitter),
        voice_quality_enable_shimmer=bool(args.voice_quality_enable_shimmer),
        voice_quality_enable_hnr=bool(args.voice_quality_enable_hnr),
        voice_quality_enable_f0_stats=bool(args.voice_quality_enable_f0_stats),
        voice_quality_enable_time_series=bool(args.voice_quality_enable_time_series),
        key_sample_rate=int(args.key_sample_rate),
        key_hop_length=int(args.key_hop_length),
        key_chroma_type=str(args.key_chroma_type),
        key_use_beat_sync=bool(args.key_use_beat_sync),
        key_top_k=int(args.key_top_k),
        key_method=str(args.key_method),
        key_confidence_threshold=float(args.key_confidence_threshold),
        key_enable_audio_normalization=get_key_flag("enable_audio_normalization", False),
        key_enable_detailed_scores=get_key_flag("enable_detailed_scores", False),
        key_enable_top_k=get_key_flag("enable_top_k", False),
        key_enable_time_series=get_key_flag("enable_time_series", False),
        key_enable_key_changes=get_key_flag("enable_key_changes", False),
        key_enable_stability_metrics=get_key_flag("enable_stability_metrics", False),
        enabled_extractors=extractor_keys,
        save_debug_results=False,
        write_legacy_manifest=bool(args.write_legacy_manifest),
    )

