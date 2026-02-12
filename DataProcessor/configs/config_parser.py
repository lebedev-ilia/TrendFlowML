"""
Парсер глобального конфига для DataProcessor.

Читает единый YAML конфиг и генерирует CLI аргументы для всех процессоров.
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class GlobalConfigParser:
    """Парсер глобального конфига для всех процессоров."""
    
    def __init__(self, config_path: str):
        """
        Инициализация парсера.
        
        Args:
            config_path: Путь к глобальному конфигу YAML
        """
        self.config_path = os.path.abspath(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Загружает конфиг из YAML файла."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        
        return config
    
    def get_global_settings(self) -> Dict[str, Any]:
        """Возвращает глобальные настройки."""
        return self.config.get("global", {})
    
    def get_processor_config(self, processor_name: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает конфиг для процессора.
        
        Args:
            processor_name: Имя процессора (audio, text, visual)
        
        Returns:
            Конфиг процессора или None
        """
        processors = self.config.get("processors", {})
        return processors.get(processor_name)
    
    def is_processor_enabled(self, processor_name: str) -> bool:
        """Проверяет, включен ли процессор."""
        proc_cfg = self.get_processor_config(processor_name)
        if proc_cfg is None:
            return False
        return bool(proc_cfg.get("enabled", False))
    
    def is_processor_required(self, processor_name: str) -> bool:
        """Проверяет, является ли процессор обязательным (required)."""
        proc_cfg = self.get_processor_config(processor_name)
        if proc_cfg is None:
            return False
        return bool(proc_cfg.get("required", False))
    
    def get_audio_extractors_list(self) -> List[str]:
        """Возвращает список включенных audio extractors."""
        audio_cfg = self.get_processor_config("audio")
        if not audio_cfg:
            return []
        
        extractors = audio_cfg.get("extractors", {})
        enabled = [key for key, cfg in extractors.items() if cfg.get("enabled", False)]
        return enabled
    
    def get_audio_cli_args(self) -> List[str]:
        """
        Генерирует CLI аргументы для AudioProcessor на основе конфига.
        
        Returns:
            Список строк для передачи в subprocess (чередование ключ-значение)
        """
        audio_cfg = self.get_processor_config("audio")
        if not audio_cfg:
            return []
        
        args = []
        
        # Device
        device = audio_cfg.get("device", "auto")
        args.extend(["--device", str(device)])
        
        # Extractors parallelism settings (индивидуальные настройки для каждого extractor'а)
        # Собираем настройки parallelism для всех extractors
        extractors_parallelism = {}
        extractors_config = {}  # Полная конфигурация extractors (для render флагов и других настроек)
        extractors = audio_cfg.get("extractors", {})
        
        for extractor_key, extractor_cfg in extractors.items():
            if not extractor_cfg.get("enabled", False):
                continue
            parallelism_cfg = extractor_cfg.get("parallelism", {})
            if parallelism_cfg:
                extractors_parallelism[extractor_key] = parallelism_cfg
            # Сохраняем полную конфигурацию для передачи render флагов
            extractors_config[extractor_key] = extractor_cfg
        
        # Если есть индивидуальные настройки, передаем их через JSON
        if extractors_parallelism:
            import json
            parallelism_json = json.dumps(extractors_parallelism)
            args.extend(["--extractor-parallelism-config", parallelism_json])
        
        # Передаем полную конфигурацию extractors для render флагов
        if extractors_config:
            import json
            extractors_config_json = json.dumps(extractors_config)
            args.extend(["--extractor-config", extractors_config_json])
        
        # Legacy: глобальные настройки scheduler (fallback, если нет индивидуальных)
        # Используются только если нет индивидуальных настроек для соответствующих extractors
        scheduler = audio_cfg.get("scheduler", {})
        # segment_parallelism используется как fallback для CPU extractors, если нет индивидуальных настроек
        if scheduler.get("segment_parallelism") is not None:
            args.extend(["--segment-parallelism", str(int(scheduler["segment_parallelism"]))])
        if scheduler.get("max_inflight") is not None:
            args.extend(["--max-inflight", str(int(scheduler["max_inflight"]))])
        # clap_batch_size используется только если нет индивидуальной настройки для CLAP
        if scheduler.get("clap_batch_size") is not None and "clap" not in extractors_parallelism:
            args.extend(["--clap-batch-size", str(int(scheduler["clap_batch_size"]))])
        
        # Extractors list
        extractors_list = self.get_audio_extractors_list()
        if extractors_list:
            args.extend(["--extractors", ",".join(extractors_list)])
        
        # Batch processing configuration (Stage 5)
        batch_cfg = audio_cfg.get("batch_processing", {})
        if batch_cfg.get("enabled", False):
            # max_video_workers: only add if explicitly set (not null/None)
            if batch_cfg.get("max_video_workers") is not None:
                args.extend(["--batch-max-workers", str(int(batch_cfg["max_video_workers"]))])
            # max_segments_per_gpu_batch: only add if explicitly set
            if batch_cfg.get("max_segments_per_gpu_batch") is not None:
                args.extend(["--batch-max-segments-per-gpu-batch", str(int(batch_cfg["max_segments_per_gpu_batch"]))])
            # GPU batching: add --no-batch-gpu only if explicitly disabled
            if batch_cfg.get("enable_gpu_batching") is False:
                args.append("--no-batch-gpu")
            # CPU parallelism: add --no-batch-cpu-parallel only if explicitly disabled
            if batch_cfg.get("enable_cpu_parallel") is False:
                args.append("--no-batch-cpu-parallel")
        else:
            # If batch_processing.enabled = false, disable both optimizations
            args.append("--no-batch-gpu")
            args.append("--no-batch-cpu-parallel")
        
        # Extractors settings
        extractors = audio_cfg.get("extractors", {})
        
        # ASR settings
        if "asr" in extractors and extractors["asr"].get("enabled"):
            asr_cfg = extractors["asr"]
            if asr_cfg.get("model_size"):
                args.extend(["--asr-model-size", str(asr_cfg["model_size"])])
            # Decode controls (new)
            decode_cfg = asr_cfg.get("decode", {}) if isinstance(asr_cfg.get("decode", {}), dict) else {}
            if decode_cfg.get("language") is not None:
                args.extend(["--asr-language", str(decode_cfg.get("language"))])
            if decode_cfg.get("temperature") is not None:
                args.extend(["--asr-temperature", str(float(decode_cfg.get("temperature")))])
            if decode_cfg.get("beam_size") is not None:
                args.extend(["--asr-beam-size", str(int(decode_cfg.get("beam_size")))])
            if decode_cfg.get("best_of") is not None:
                args.extend(["--asr-best-of", str(int(decode_cfg.get("best_of")))])

            fallback_cfg = asr_cfg.get("fallback_decode", {}) if isinstance(asr_cfg.get("fallback_decode", {}), dict) else {}
            if fallback_cfg.get("enabled") is True:
                args.append("--asr-enable-fallback-decode")
            if fallback_cfg.get("temperature") is not None:
                args.extend(["--asr-fallback-temperature", str(float(fallback_cfg.get("temperature")))])
            if fallback_cfg.get("avg_logprob_threshold") is not None:
                args.extend(["--asr-fallback-avg-logprob-threshold", str(float(fallback_cfg.get("avg_logprob_threshold")))])

            output_cfg = asr_cfg.get("output", {}) if isinstance(asr_cfg.get("output", {}), dict) else {}
            if output_cfg.get("save_segment_text") is True:
                args.append("--asr-save-segment-text")
            
            # ASR feature flags
            flags = asr_cfg.get("feature_flags", {})
            for flag_name, flag_value in flags.items():
                if flag_value:
                    args.append(f"--asr-{flag_name.replace('_', '-')}")
        
        # Diarization settings
        if "speaker_diarization" in extractors and extractors["speaker_diarization"].get("enabled"):
            diar_cfg = extractors["speaker_diarization"]
            if diar_cfg.get("model_size"):
                args.extend(["--diarization-model-size", str(diar_cfg["model_size"])])
            if diar_cfg.get("batch_size") is not None:
                args.extend(["--diarization-batch-size", str(int(diar_cfg["batch_size"]))])
            if diar_cfg.get("clustering_method"):
                args.extend(["--diarization-clustering-method", str(diar_cfg["clustering_method"])])
            if diar_cfg.get("speaker_count_method"):
                args.extend(["--diarization-speaker-count-method", str(diar_cfg["speaker_count_method"])])
            if diar_cfg.get("silence_peak_threshold") is not None:
                args.extend(["--diarization-silence-peak-threshold", str(float(diar_cfg["silence_peak_threshold"]))])
            if diar_cfg.get("silence_rms_threshold") is not None:
                args.extend(["--diarization-silence-rms-threshold", str(float(diar_cfg["silence_rms_threshold"]))])
            
            # Diarization feature flags
            flags = diar_cfg.get("feature_flags", {})
            for flag_name, flag_value in flags.items():
                if flag_value:
                    args.append(f"--diar-{flag_name.replace('_', '-')}")
        
        # Emotion diarization settings
        if "emotion_diarization" in extractors and extractors["emotion_diarization"].get("enabled"):
            emo_cfg = extractors["emotion_diarization"]
            if emo_cfg.get("model_size"):
                args.extend(["--emotion-model-size", str(emo_cfg["model_size"])])
            if emo_cfg.get("batch_size") is not None:
                args.extend(["--emotion-batch-size", str(int(emo_cfg["batch_size"]))])
            if emo_cfg.get("silence_peak_threshold") is not None:
                args.extend(["--emotion-silence-peak-threshold", str(float(emo_cfg["silence_peak_threshold"]))])
            if emo_cfg.get("silence_rms_threshold") is not None:
                args.extend(["--emotion-silence-rms-threshold", str(float(emo_cfg["silence_rms_threshold"]))])
            if emo_cfg.get("process_full_audio") is True:
                args.append("--emotion-process-full-audio")
            
            # Emotion feature flags
            flags = emo_cfg.get("feature_flags", {})
            for flag_name, flag_value in flags.items():
                if flag_value:
                    args.append(f"--emotion-{flag_name.replace('_', '-')}")
        
        # Source separation settings
        if "source_separation" in extractors and extractors["source_separation"].get("enabled"):
            sep_cfg = extractors["source_separation"]
            if sep_cfg.get("model_size"):
                args.extend(["--source-separation-model-size", str(sep_cfg["model_size"])])
            if sep_cfg.get("batch_size") is not None:
                args.extend(["--sep-batch-size", str(int(sep_cfg["batch_size"]))])
            
            # Source separation feature flags
            flags = sep_cfg.get("feature_flags", {})
            for flag_name, flag_value in flags.items():
                if flag_value:
                    args.append(f"--sep-{flag_name.replace('_', '-')}")
        
        # Speech analysis settings
        if "speech_analysis" in extractors and extractors["speech_analysis"].get("enabled"):
            speech_cfg = extractors["speech_analysis"]
            if speech_cfg.get("pitch_enabled"):
                args.append("--speech-analysis-pitch")
            if speech_cfg.get("silence_peak_threshold"):
                args.extend(["--speech-silence-peak-threshold", str(float(speech_cfg["silence_peak_threshold"]))])
            if speech_cfg.get("silence_rms_threshold"):
                args.extend(["--speech-silence-rms-threshold", str(float(speech_cfg["silence_rms_threshold"]))])
            
            # Feature flags
            flags = speech_cfg.get("feature_flags", {})
            if flags.get("enable_asr_metrics"):
                args.append("--speech-enable-asr-metrics")
            if flags.get("enable_diarization_metrics"):
                args.append("--speech-enable-diarization-metrics")
            if flags.get("enable_pitch_metrics"):
                args.append("--speech-enable-pitch-metrics")
            if flags.get("disable_silence_detection"):
                args.append("--speech-disable-silence-detection")
        
        # Spectral extractor settings
        if "spectral" in extractors and extractors["spectral"].get("enabled"):
            spec_cfg = extractors["spectral"]
            if spec_cfg.get("sample_rate"):
                args.extend(["--spectral-sample-rate", str(int(spec_cfg["sample_rate"]))])
            if spec_cfg.get("hop_length"):
                args.extend(["--spectral-hop-length", str(int(spec_cfg["hop_length"]))])
            if spec_cfg.get("n_fft"):
                args.extend(["--spectral-n-fft", str(int(spec_cfg["n_fft"]))])
            if spec_cfg.get("average_channels"):
                args.append("--spectral-average-channels")
            if spec_cfg.get("keep_contrast_bands"):
                args.append("--spectral-keep-contrast-bands")
            
            flags = spec_cfg.get("feature_flags", {})
            for flag_name, flag_value in flags.items():
                if flag_value:
                    args.append(f"--spectral-{flag_name.replace('_', '-')}")
        
        # Mel extractor settings
        if "mel" in extractors and extractors["mel"].get("enabled"):
            mel_cfg = extractors["mel"]
            if mel_cfg.get("sample_rate"):
                args.extend(["--mel-sample-rate", str(int(mel_cfg["sample_rate"]))])
            if mel_cfg.get("hop_length"):
                args.extend(["--mel-hop-length", str(int(mel_cfg["hop_length"]))])
            if mel_cfg.get("n_fft"):
                args.extend(["--mel-n-fft", str(int(mel_cfg["n_fft"]))])
            if mel_cfg.get("n_mels"):
                args.extend(["--mel-n-mels", str(int(mel_cfg["n_mels"]))])
            if mel_cfg.get("average_channels"):
                args.append("--mel-average-channels")
            
            flags = mel_cfg.get("feature_flags", {})
            for flag_name, flag_value in flags.items():
                if flag_value:
                    args.append(f"--mel-{flag_name.replace('_', '-')}")
        
        # Key extractor settings
        if "key" in extractors and extractors["key"].get("enabled"):
            key_cfg = extractors["key"]
            if key_cfg.get("sample_rate"):
                args.extend(["--key-sample-rate", str(int(key_cfg["sample_rate"]))])
            if key_cfg.get("hop_length"):
                args.extend(["--key-hop-length", str(int(key_cfg["hop_length"]))])
            if key_cfg.get("chroma_type"):
                args.extend(["--key-chroma-type", str(key_cfg["chroma_type"])])
            if key_cfg.get("key_method"):
                args.extend(["--key-method", str(key_cfg["key_method"])])
            if key_cfg.get("use_beat_sync"):
                args.append("--key-use-beat-sync")
            if key_cfg.get("top_k"):
                args.extend(["--key-top-k", str(int(key_cfg["top_k"]))])
            
            flags = key_cfg.get("feature_flags", {})
            for flag_name, flag_value in flags.items():
                if flag_value:
                    args.append(f"--key-{flag_name.replace('_', '-')}")
        
        # Band energy extractor settings
        if "band_energy" in extractors and extractors["band_energy"].get("enabled"):
            band_cfg = extractors["band_energy"]
            if band_cfg.get("sample_rate"):
                args.extend(["--band-energy-sample-rate", str(int(band_cfg["sample_rate"]))])
            if band_cfg.get("n_fft"):
                args.extend(["--band-energy-n-fft", str(int(band_cfg["n_fft"]))])
            if band_cfg.get("hop_length"):
                args.extend(["--band-energy-hop-length", str(int(band_cfg["hop_length"]))])
            if band_cfg.get("use_mel_bands"):
                args.append("--band-energy-use-mel-bands")
            if band_cfg.get("n_mels"):
                args.extend(["--band-energy-n-mels", str(int(band_cfg["n_mels"]))])
            if band_cfg.get("average_channels"):
                args.append("--band-energy-average-channels")
            
            flags = band_cfg.get("feature_flags", {})
            for flag_name, flag_value in flags.items():
                if flag_value:
                    args.append(f"--band-energy-{flag_name.replace('_', '-')}")
        
        # Spectral entropy extractor settings
        if "spectral_entropy" in extractors and extractors["spectral_entropy"].get("enabled"):
            entropy_cfg = extractors["spectral_entropy"]
            if entropy_cfg.get("sample_rate"):
                args.extend(["--spectral-entropy-sample-rate", str(int(entropy_cfg["sample_rate"]))])
            if entropy_cfg.get("n_fft"):
                args.extend(["--spectral-entropy-n-fft", str(int(entropy_cfg["n_fft"]))])
            if entropy_cfg.get("hop_length"):
                args.extend(["--spectral-entropy-hop-length", str(int(entropy_cfg["hop_length"]))])
            if entropy_cfg.get("average_channels"):
                args.append("--spectral-entropy-average-channels")
            if entropy_cfg.get("smoothing_window"):
                args.extend(["--spectral-entropy-smoothing-window", str(int(entropy_cfg["smoothing_window"]))])
            if entropy_cfg.get("use_mel"):
                args.append("--spectral-entropy-use-mel")
            if entropy_cfg.get("n_mels"):
                args.extend(["--spectral-entropy-n-mels", str(int(entropy_cfg["n_mels"]))])
            
            flags = entropy_cfg.get("feature_flags", {})
            for flag_name, flag_value in flags.items():
                if flag_value:
                    args.append(f"--spectral-entropy-{flag_name.replace('_', '-')}")
        
        # Voice quality extractor settings
        if "voice_quality" in extractors and extractors["voice_quality"].get("enabled"):
            vq_cfg = extractors["voice_quality"]
            if vq_cfg.get("sample_rate"):
                args.extend(["--voice-quality-sample-rate", str(int(vq_cfg["sample_rate"]))])
            if vq_cfg.get("hnr_frame_ms"):
                args.extend(["--voice-quality-hnr-frame-ms", str(float(vq_cfg["hnr_frame_ms"]))])
            if vq_cfg.get("rms_mask_threshold"):
                args.extend(["--voice-quality-rms-mask-threshold", str(float(vq_cfg["rms_mask_threshold"]))])
            if vq_cfg.get("f0_fmin"):
                args.extend(["--voice-quality-f0-fmin", str(float(vq_cfg["f0_fmin"]))])
            if vq_cfg.get("f0_fmax"):
                args.extend(["--voice-quality-f0-fmax", str(float(vq_cfg["f0_fmax"]))])
            if vq_cfg.get("f0_method"):
                args.extend(["--voice-quality-f0-method", str(vq_cfg["f0_method"])])
            if vq_cfg.get("average_channels"):
                args.append("--voice-quality-average-channels")
            
            flags = vq_cfg.get("feature_flags", {})
            for flag_name, flag_value in flags.items():
                if flag_value:
                    args.append(f"--voice-quality-{flag_name.replace('_', '-')}")
        
        # HPSS extractor settings
        if "hpss" in extractors and extractors["hpss"].get("enabled"):
            hpss_cfg = extractors["hpss"]
            if hpss_cfg.get("kernel_size"):
                args.extend(["--hpss-kernel-size", str(int(hpss_cfg["kernel_size"]))])
            if hpss_cfg.get("margin"):
                args.extend(["--hpss-margin", str(int(hpss_cfg["margin"]))])
            if hpss_cfg.get("power"):
                args.extend(["--hpss-power", str(float(hpss_cfg["power"]))])
            
            flags = hpss_cfg.get("feature_flags", {})
            for flag_name, flag_value in flags.items():
                if flag_value:
                    args.append(f"--hpss-{flag_name.replace('_', '-')}")
        
        # Chroma extractor settings
        if "chroma" in extractors and extractors["chroma"].get("enabled"):
            chroma_cfg = extractors["chroma"]
            if chroma_cfg.get("sample_rate"):
                args.extend(["--chroma-sample-rate", str(int(chroma_cfg["sample_rate"]))])
            if chroma_cfg.get("hop_length"):
                args.extend(["--chroma-hop-length", str(int(chroma_cfg["hop_length"]))])
            if chroma_cfg.get("n_fft"):
                args.extend(["--chroma-n-fft", str(int(chroma_cfg["n_fft"]))])
            if chroma_cfg.get("chroma_type"):
                args.extend(["--chroma-type", str(chroma_cfg["chroma_type"])])
            if chroma_cfg.get("normalize"):
                args.extend(["--chroma-normalize", str(chroma_cfg["normalize"])])
            
            flags = chroma_cfg.get("feature_flags", {})
            for flag_name, flag_value in flags.items():
                if flag_value:
                    args.append(f"--chroma-{flag_name.replace('_', '-')}")
        
        # Onset extractor settings
        if "onset" in extractors and extractors["onset"].get("enabled"):
            onset_cfg = extractors["onset"]
            flags = onset_cfg.get("feature_flags", {})
            for flag_name, flag_value in flags.items():
                if flag_value:
                    args.append(f"--onset-{flag_name.replace('_', '-')}")
        
        # Quality extractor settings
        if "quality" in extractors and extractors["quality"].get("enabled"):
            quality_cfg = extractors["quality"]
            if quality_cfg.get("sample_rate"):
                args.extend(["--quality-sample-rate", str(int(quality_cfg["sample_rate"]))])
            if quality_cfg.get("frame_len_ms"):
                args.extend(["--quality-frame-len-ms", str(float(quality_cfg["frame_len_ms"]))])
            if quality_cfg.get("hop_ms"):
                args.extend(["--quality-hop-ms", str(float(quality_cfg["hop_ms"]))])
            if quality_cfg.get("clip_threshold"):
                args.extend(["--quality-clip-threshold", str(float(quality_cfg["clip_threshold"]))])
            if quality_cfg.get("average_channels"):
                args.append("--quality-average-channels")
            
            flags = quality_cfg.get("feature_flags", {})
            for flag_name, flag_value in flags.items():
                if flag_value:
                    args.append(f"--quality-{flag_name.replace('_', '-')}")
        
        # MFCC extractor settings
        if "mfcc" in extractors and extractors["mfcc"].get("enabled"):
            mfcc_cfg = extractors["mfcc"]
            if mfcc_cfg.get("sample_rate"):
                args.extend(["--mfcc-sample-rate", str(int(mfcc_cfg["sample_rate"]))])
            if mfcc_cfg.get("n_mfcc"):
                args.extend(["--mfcc-n-mfcc", str(int(mfcc_cfg["n_mfcc"]))])
            if mfcc_cfg.get("n_fft"):
                args.extend(["--mfcc-n-fft", str(int(mfcc_cfg["n_fft"]))])
            if mfcc_cfg.get("hop_length"):
                args.extend(["--mfcc-hop-length", str(int(mfcc_cfg["hop_length"]))])
            if mfcc_cfg.get("n_mels"):
                args.extend(["--mfcc-n-mels", str(int(mfcc_cfg["n_mels"]))])
            if mfcc_cfg.get("fmin") is not None:
                args.extend(["--mfcc-fmin", str(float(mfcc_cfg["fmin"]))])
            if mfcc_cfg.get("fmax") is not None:
                args.extend(["--mfcc-fmax", str(float(mfcc_cfg["fmax"]))])
            if mfcc_cfg.get("min_gpu_duration_sec"):
                args.extend(["--mfcc-min-gpu-duration-sec", str(float(mfcc_cfg["min_gpu_duration_sec"]))])
            if mfcc_cfg.get("min_gpu_file_size_mb"):
                args.extend(["--mfcc-min-gpu-file-size-mb", str(float(mfcc_cfg["min_gpu_file_size_mb"]))])
            
            flags = mfcc_cfg.get("feature_flags", {})
            # enable_audio_normalization обрабатывается отдельно (default: True)
            if flags.get("enable_audio_normalization", True):
                args.append("--mfcc-enable-audio-normalization")
            else:
                args.append("--mfcc-disable-audio-normalization")
            for flag_name, flag_value in flags.items():
                if flag_name != "enable_audio_normalization" and flag_value:
                    args.append(f"--mfcc-{flag_name.replace('_', '-')}")
        
        # Mel extractor settings
        if "mel" in extractors and extractors["mel"].get("enabled"):
            mel_cfg = extractors["mel"]
            if mel_cfg.get("sample_rate"):
                args.extend(["--mel-sample-rate", str(int(mel_cfg["sample_rate"]))])
            if mel_cfg.get("n_fft"):
                args.extend(["--mel-n-fft", str(int(mel_cfg["n_fft"]))])
            if mel_cfg.get("hop_length"):
                args.extend(["--mel-hop-length", str(int(mel_cfg["hop_length"]))])
            if mel_cfg.get("n_mels"):
                args.extend(["--mel-n-mels", str(int(mel_cfg["n_mels"]))])
            if mel_cfg.get("fmin") is not None:
                args.extend(["--mel-fmin", str(float(mel_cfg["fmin"]))])
            if mel_cfg.get("fmax") is not None:
                args.extend(["--mel-fmax", str(float(mel_cfg["fmax"]))])
            if mel_cfg.get("power") is not None:
                args.extend(["--mel-power", str(float(mel_cfg["power"]))])
            if mel_cfg.get("mix_to_mono"):
                args.append("--mel-mix-to-mono")
            
            flags = mel_cfg.get("feature_flags", {})
            # enable_audio_normalization обрабатывается отдельно (default: True)
            if flags.get("enable_audio_normalization", True):
                args.append("--mel-enable-audio-normalization")
            else:
                args.append("--mel-disable-audio-normalization")
            for flag_name, flag_value in flags.items():
                if flag_name != "enable_audio_normalization" and flag_value:
                    args.append(f"--mel-{flag_name.replace('_', '-')}")
        
        return args
    
    def get_text_cli_args(self) -> List[str]:
        """
        Генерирует CLI аргументы для TextProcessor на основе конфига.
        
        Returns:
            Список строк для передачи в subprocess
        """
        text_cfg = self.get_processor_config("text")
        if not text_cfg:
            return []
        
        args = []
        
        # Feature flags
        flags = text_cfg.get("feature_flags", {})
        if flags.get("enable_embeddings"):
            args.append("--enable-embeddings")
        if flags.get("include_primary_embedding"):
            args.append("--include-primary-embedding")
        if flags.get("store_raw_payload"):
            args.append("--store-raw-payload")
        if flags.get("no_strict_extractors"):
            args.append("--no-strict-extractors")
        
        # Batch processing configuration (Stage 4)
        batch_cfg = text_cfg.get("batch_processing", {})
        if batch_cfg.get("enabled", True):  # Default: enabled if not specified
            # max_workers: only add if explicitly set (not null/None)
            if batch_cfg.get("max_workers") is not None:
                args.extend(["--batch-max-workers", str(int(batch_cfg["max_workers"]))])
            # GPU batching: add --no-batch-gpu only if explicitly disabled
            if batch_cfg.get("enable_gpu_batching") is False:
                args.append("--no-batch-gpu")
            # CPU parallelism: add --no-batch-cpu-parallel only if explicitly disabled
            if batch_cfg.get("enable_cpu_parallel") is False:
                args.append("--no-batch-cpu-parallel")
        else:
            # If batch_processing.enabled = false, disable both optimizations
            args.append("--no-batch-gpu")
            args.append("--no-batch-cpu-parallel")
        
        # Extractors configuration через --extractor-params-json и --devices-config-json
        extractors = text_cfg.get("extractors", {})
        if extractors:
            import json
            
            # Маппинг имен extractors на классы
            ext_class_map = {
                "lexico_static_features": "LexicalStatsExtractor",
                "tags_extractor": "TagsExtractor",
                "asr_text_proxy_audio_features": "ASRTextProxyExtractor",
                "title_embedder": "TitleEmbedder",
                "description_embedder": "DescriptionEmbedder",
                "hashtag_embedder": "HashtagEmbedder",
                "transcript_chunk_embedder": "TranscriptChunkEmbedder",
                "comments_embedder": "CommentsEmbedder",
                "transcript_aggregator": "TranscriptAggregatorExtractor",
                "comments_aggregator": "CommentsAggregationExtractor",
                "cosine_metrics_extractor": "CosineMetricsExtractor",
                "embedding_pair_topk_extractor": "EmbeddingPairTopKExtractor",
                "embedding_stats_extractor": "EmbeddingStatsExtractor",
                "embedding_shift_indicator_extractor": "EmbeddingShiftIndicatorExtractor",
                "embedding_source_id_extractor": "EmbeddingSourceIdExtractor",
                "speaker_turn_embeddings_aggregator": "SpeakerTurnEmbeddingsAggregatorExtractor",
                "title_to_hashtag_cosine_extractor": "TitleToHashtagCosineExtractor",
                "topk_similar_titles_extractor": "TopKSimilarCorpusTitlesExtractor",
                "title_embedding_cluster_entropy_extractor": "TitleEmbeddingClusterEntropyExtractor",
                "semantic_cluster_extractor": "SemanticClusterExtractor",
                "qa_embedding_pairs_extractor": "QAEmbeddingPairsExtractor",
                "semantics_topics_keyphrases": "SemanticTopicExtractor",
            }
            
            # Определяем устройства для extractors
            # Embedding extractors обычно на GPU, остальные на CPU
            embedding_extractors = {
                "title_embedder", "description_embedder", "hashtag_embedder",
                "transcript_chunk_embedder", "comments_embedder",
                "speaker_turn_embeddings_aggregator", "qa_embedding_pairs_extractor",
            }
            aggregator_extractors = {
                "transcript_aggregator", "comments_aggregator",
                "embedding_stats_extractor", "cosine_metrics_extractor",
                "embedding_shift_indicator_extractor", "embedding_source_id_extractor",
                "title_to_hashtag_cosine_extractor", "title_embedding_cluster_entropy_extractor",
                "embedding_pair_topk_extractor", "semantic_cluster_extractor",
                "topk_similar_titles_extractor",
            }
            
            devices_config = {"cpu": [], "gpu": [], "cpu2": []}
            extractor_params = {}
            
            for ext_name, ext_cfg in extractors.items():
                if not ext_cfg.get("enabled", False):
                    continue
                
                ext_class = ext_class_map.get(ext_name)
                if not ext_class:
                    continue
                
                # Определяем устройство для extractor
                if ext_name in embedding_extractors:
                    device_group = "gpu"
                elif ext_name in aggregator_extractors:
                    device_group = "cpu2"
                else:
                    device_group = "cpu"
                
                devices_config[device_group].append(ext_class)
                
                # Собираем параметры extractor
                ext_params = {}
                
                # Копируем feature_flags и другие параметры
                if "feature_flags" in ext_cfg:
                    ext_params.update(ext_cfg["feature_flags"])
                
                # Копируем остальные параметры (кроме enabled)
                for key, value in ext_cfg.items():
                    if key not in ("enabled", "feature_flags"):
                        ext_params[key] = value
                
                if ext_params:
                    extractor_params[ext_class] = ext_params
                    logger.debug(f"Added params for {ext_class}: {list(ext_params.keys())}")
            
            # Удаляем пустые группы устройств
            devices_config = {k: v for k, v in devices_config.items() if v}
            
            # Передаем devices_config и extractor_params
            if devices_config:
                args.extend(["--devices-config-json", json.dumps(devices_config)])
            
            if extractor_params:
                args.extend(["--extractor-params-json", json.dumps(extractor_params)])
        
        return args
    
    def get_text_devices_config(self) -> Dict[str, List[str]]:
        """
        Возвращает devices_config для TextProcessor на основе конфига.
        
        Returns:
            Словарь вида {"cpu": [...], "gpu": [...], "cpu2": [...]}
        """
        text_cfg = self.get_processor_config("text")
        if not text_cfg:
            return {}
        
        extractors = text_cfg.get("extractors", {})
        if not extractors:
            return {}
        
        # Маппинг имен extractors на классы
        ext_class_map = {
            "lexico_static_features": "LexicalStatsExtractor",
            "tags_extractor": "TagsExtractor",
            "asr_text_proxy_audio_features": "ASRTextProxyExtractor",
            "title_embedder": "TitleEmbedder",
            "description_embedder": "DescriptionEmbedder",
            "hashtag_embedder": "HashtagEmbedder",
            "transcript_chunk_embedder": "TranscriptChunkEmbedder",
            "comments_embedder": "CommentsEmbedder",
            "transcript_aggregator": "TranscriptAggregatorExtractor",
            "comments_aggregator": "CommentsAggregationExtractor",
            "cosine_metrics_extractor": "CosineMetricsExtractor",
            "embedding_pair_topk_extractor": "EmbeddingPairTopKExtractor",
            "embedding_stats_extractor": "EmbeddingStatsExtractor",
            "embedding_shift_indicator_extractor": "EmbeddingShiftIndicatorExtractor",
            "embedding_source_id_extractor": "EmbeddingSourceIdExtractor",
            "speaker_turn_embeddings_aggregator": "SpeakerTurnEmbeddingsAggregatorExtractor",
            "title_to_hashtag_cosine_extractor": "TitleToHashtagCosineExtractor",
            "topk_similar_titles_extractor": "TopKSimilarCorpusTitlesExtractor",
            "title_embedding_cluster_entropy_extractor": "TitleEmbeddingClusterEntropyExtractor",
            "semantic_cluster_extractor": "SemanticClusterExtractor",
            "qa_embedding_pairs_extractor": "QAEmbeddingPairsExtractor",
            "semantics_topics_keyphrases": "SemanticTopicExtractor",
        }
        
        # Определяем устройства для extractors
        embedding_extractors = {
            "title_embedder", "description_embedder", "hashtag_embedder",
            "transcript_chunk_embedder", "comments_embedder",
            "speaker_turn_embeddings_aggregator", "qa_embedding_pairs_extractor",
        }
        aggregator_extractors = {
            "transcript_aggregator", "comments_aggregator",
            "embedding_stats_extractor", "cosine_metrics_extractor",
            "embedding_shift_indicator_extractor", "embedding_source_id_extractor",
            "title_to_hashtag_cosine_extractor", "title_embedding_cluster_entropy_extractor",
            "embedding_pair_topk_extractor", "semantic_cluster_extractor",
            "topk_similar_titles_extractor",
        }
        
        devices_config = {"cpu": [], "gpu": [], "cpu2": []}
        
        for ext_name, ext_cfg in extractors.items():
            if not ext_cfg.get("enabled", False):
                continue
            
            ext_class = ext_class_map.get(ext_name)
            if not ext_class:
                continue
            
            # Определяем устройство для extractor
            if ext_name in embedding_extractors:
                device_group = "gpu"
            elif ext_name in aggregator_extractors:
                device_group = "cpu2"
            else:
                device_group = "cpu"
            
            devices_config[device_group].append(ext_class)
        
        # Удаляем пустые группы устройств
        devices_config = {k: v for k, v in devices_config.items() if v}
        
        return devices_config
    
    def get_text_extractor_params(self) -> Dict[str, Dict[str, Any]]:
        """
        Возвращает extractor_params для TextProcessor на основе конфига.
        
        Returns:
            Словарь вида {"ExtractorClass": {"param1": value1, ...}}
        """
        text_cfg = self.get_processor_config("text")
        if not text_cfg:
            return {}
        
        extractors = text_cfg.get("extractors", {})
        if not extractors:
            return {}
        
        # Маппинг имен extractors на классы
        ext_class_map = {
            "lexico_static_features": "LexicalStatsExtractor",
            "tags_extractor": "TagsExtractor",
            "asr_text_proxy_audio_features": "ASRTextProxyExtractor",
            "title_embedder": "TitleEmbedder",
            "description_embedder": "DescriptionEmbedder",
            "hashtag_embedder": "HashtagEmbedder",
            "transcript_chunk_embedder": "TranscriptChunkEmbedder",
            "comments_embedder": "CommentsEmbedder",
            "transcript_aggregator": "TranscriptAggregatorExtractor",
            "comments_aggregator": "CommentsAggregationExtractor",
            "cosine_metrics_extractor": "CosineMetricsExtractor",
            "embedding_pair_topk_extractor": "EmbeddingPairTopKExtractor",
            "embedding_stats_extractor": "EmbeddingStatsExtractor",
            "embedding_shift_indicator_extractor": "EmbeddingShiftIndicatorExtractor",
            "embedding_source_id_extractor": "EmbeddingSourceIdExtractor",
            "speaker_turn_embeddings_aggregator": "SpeakerTurnEmbeddingsAggregatorExtractor",
            "title_to_hashtag_cosine_extractor": "TitleToHashtagCosineExtractor",
            "topk_similar_titles_extractor": "TopKSimilarCorpusTitlesExtractor",
            "title_embedding_cluster_entropy_extractor": "TitleEmbeddingClusterEntropyExtractor",
            "semantic_cluster_extractor": "SemanticClusterExtractor",
            "qa_embedding_pairs_extractor": "QAEmbeddingPairsExtractor",
            "semantics_topics_keyphrases": "SemanticTopicExtractor",
        }
        
        extractor_params = {}
        
        for ext_name, ext_cfg in extractors.items():
            if not ext_cfg.get("enabled", False):
                continue
            
            ext_class = ext_class_map.get(ext_name)
            if not ext_class:
                continue
            
            # Собираем параметры extractor
            ext_params = {}
            
            # Копируем feature_flags и другие параметры
            if "feature_flags" in ext_cfg:
                ext_params.update(ext_cfg["feature_flags"])
            
            # Копируем остальные параметры (кроме enabled)
            for key, value in ext_cfg.items():
                if key not in ("enabled", "feature_flags"):
                    ext_params[key] = value
            
            if ext_params:
                extractor_params[ext_class] = ext_params
        
        return extractor_params
    
    def get_visual_inline_config(self) -> Optional[Dict[str, Any]]:
        """
        Возвращает inline конфиг VisualProcessor.
        
        Returns:
            Inline конфиг или None
        """
        visual_cfg = self.get_processor_config("visual")
        if not visual_cfg:
            return None
        
        # Приоритет: inline_config > cfg_path
        inline_config = visual_cfg.get("inline_config")
        if inline_config:
            return inline_config
        
        # Fallback на cfg_path (для обратной совместимости)
        cfg_path = visual_cfg.get("cfg_path")
        if cfg_path:
            # Если путь относительный, делаем его абсолютным относительно директории конфига
            if not os.path.isabs(cfg_path):
                config_dir = os.path.dirname(self.config_path)
                cfg_path = os.path.join(config_dir, cfg_path)
            
            # Загружаем конфиг из файла
            if os.path.exists(cfg_path):
                import yaml
                with open(cfg_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
        
        return None
    
    def validate(self) -> List[str]:
        """
        Валидирует конфиг и возвращает список ошибок (пустой если все OK).
        
        Returns:
            Список строк с ошибками валидации
        """
        errors = []
        
        # Проверка обязательных полей
        if "processors" not in self.config:
            errors.append("Missing 'processors' section in config")
            return errors
        
        # Проверка audio processor
        audio_cfg = self.get_processor_config("audio")
        if audio_cfg and audio_cfg.get("enabled"):
            extractors = audio_cfg.get("extractors", {})
            if not extractors:
                errors.append("Audio processor enabled but no extractors configured")
            
            # Проверка зависимостей extractors (через dependency_resolver)
            # Note: dependency_resolver находится в AudioProcessor/src/core/
            # Валидация зависимостей будет выполнена в AudioProcessor/run_cli.py
            # Здесь мы только проверяем базовую структуру конфига
        
        # Проверка text processor
        text_cfg = self.get_processor_config("text")
        if text_cfg and text_cfg.get("enabled"):
            if not text_cfg.get("input_json"):
                errors.append("Text processor enabled but 'input_json' not specified")
        
        # Проверка visual processor
        visual_cfg = self.get_processor_config("visual")
        if visual_cfg:
            inline_config = self.get_visual_inline_config()
            if not inline_config:
                # Если нет inline_config и нет cfg_path, это ошибка
                if not visual_cfg.get("cfg_path"):
                    errors.append("VisualProcessor: either 'inline_config' or 'cfg_path' must be specified")
        
        return errors

