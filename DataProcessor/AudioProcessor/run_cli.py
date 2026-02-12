#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def safe_log_warning(logger_instance, message, *args, **kwargs):
    """Safely log a warning message, catching I/O errors from closed handlers."""
    try:
        # Try to log directly - catch all exceptions to prevent crashes
        logger_instance.warning(message, *args, **kwargs)
    except Exception:
        # Catch ALL exceptions silently - handlers may be closed, streams may be closed,
        # or logging infrastructure may be in an invalid state during shutdown
        pass


def safe_log_error(logger_instance, message, *args, **kwargs):
    """Safely log an error message, catching I/O errors from closed handlers."""
    try:
        # Try to log directly - catch all exceptions to prevent crashes
        logger_instance.error(message, *args, **kwargs)
    except Exception:
        # Catch ALL exceptions silently - handlers may be closed, streams may be closed,
        # or logging infrastructure may be in an invalid state during shutdown
        pass


# Утилиты перенесены в src/utils/cli_utils.py
# Импортируются в main() после настройки sys.path

# Функции перенесены в новые модули:
# - _emit_progress -> src/utils/progress.py
# - _retry_with_backoff -> src/utils/retry.py
# - _run_clap_with_oom_fallback -> src/utils/retry.py
# - _meta -> src/core/npz_saver.py (build_meta)
# - _save_component_npz -> src/core/npz_saver.py (save_component_npz)
# Импортируются в main() после настройки sys.path


# Функция _parse_extractors_arg перенесена в src/core/cli_args.py
# Импортируется в main() после настройки sys.path


def main() -> int:
    ap_root = Path(__file__).resolve().parent
    repo_root = ap_root.parent

    # AudioProcessor/src imports
    src_path = ap_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Repo root imports (dp_models, Segmenter helpers, etc.)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # VisualProcessor utils imports (manifest + validator)
    vp_root = repo_root / "VisualProcessor"
    if str(vp_root) not in sys.path:
        sys.path.insert(0, str(vp_root))

    from utils.manifest import RunManifest, ManifestComponent  # type: ignore
    from utils.artifact_validator import validate_npz  # type: ignore
    
    # Импорты новых модулей для рефакторинга
    from src.utils.cli_utils import utc_iso_now, atomic_write_json  # type: ignore
    from src.utils.progress import emit_progress as _emit_progress, close_progress_bar  # type: ignore
    from src.core.resource_monitor import ResourceMonitor  # type: ignore
    from src.core.extractor_runner import run_extractors  # type: ignore
    from src.core.batch_processor import collect_frames_dirs, create_audio_file_contexts, process_batch_results  # type: ignore
    from src.core.npz_saver import save_component_npz as _save_component_npz  # type: ignore
    from src.core.cli_args import create_argument_parser, parse_extractors_arg as _parse_extractors_arg  # type: ignore
    from src.core.config_hash import build_config_hash  # type: ignore
    from src.core.model_resolver import resolve_model_metadata  # type: ignore
    from src.core.segments_loader import load_and_validate_segments  # type: ignore
    from src.core.processor_factory import create_main_processor  # type: ignore

    parser = create_argument_parser()
    args = parser.parse_args()

    # Stage 5: Check if batch mode is enabled
    batch_mode = bool(args.audio_input_dir or args.audio_input_list)
    
    # Validate arguments for batch mode
    if batch_mode:
        if args.frames_dir:
            raise RuntimeError("AudioProcessor | --frames-dir cannot be used with batch mode (--audio-input-dir or --audio-input-list)")
        if not args.audio_input_dir and not args.audio_input_list:
            raise RuntimeError("AudioProcessor | batch mode requires either --audio-input-dir or --audio-input-list")
    else:
        # Single-file mode: --frames-dir is required
        if not args.frames_dir:
            raise RuntimeError("AudioProcessor | --frames-dir is required for single-file mode (or use --audio-input-dir/--audio-input-list for batch mode)")

    # Derive video_id from --video-id, else from frames_dir name, else from --video-path (if provided).
    # For batch mode, video_id will be derived per file
    if not batch_mode:
        if args.video_id:
            video_id = args.video_id
        else:
            frames_base = os.path.basename(os.path.normpath(args.frames_dir)) if args.frames_dir else ""
            if frames_base:
                video_id = frames_base
            elif args.video_path:
                video_id = os.path.splitext(os.path.basename(args.video_path))[0]
            else:
                raise RuntimeError("AudioProcessor | video_id is required: provide --video-id or a valid --frames-dir")
    else:
        # For batch mode, video_id will be derived per file
        video_id = None
    run_id = args.run_id or uuid.uuid4().hex[:12]

    config_hash = args.config_hash
    if not config_hash:
        config_hash = build_config_hash(args)

    # Stage 5: Collect frames_dirs for batch mode (используем новый модуль)

    # Set default rs_base if not provided
    if not args.rs_base:
        # Default: ./result_store relative to current working directory
        args.rs_base = os.path.join(os.getcwd(), "result_store")
        logger.info(f"AudioProcessor | --rs-base not specified, using default: {args.rs_base}")
    
    # For batch mode, run_rs_path will be created per file
    if batch_mode:
        # For batch mode, we'll create per-file run_rs_path later
        batch_frames_dirs = collect_frames_dirs(args.audio_input_dir, args.audio_input_list)
        run_rs_path = os.path.abspath(args.run_rs_path) if args.run_rs_path else os.path.join(os.path.abspath(args.rs_base), args.platform_id, "batch", run_id)
    else:
        run_rs_path = os.path.abspath(args.run_rs_path) if args.run_rs_path else os.path.join(os.path.abspath(args.rs_base), args.platform_id, video_id, run_id)
    os.makedirs(run_rs_path, exist_ok=True)

    manifest_path = os.path.join(run_rs_path, "manifest.json")
    manifest = RunManifest(
        path=manifest_path,
        run_meta={
            "platform_id": args.platform_id,
            "video_id": video_id,
            "run_id": run_id,
            "config_hash": config_hash,
            "sampling_policy_version": args.sampling_policy_version,
            "dataprocessor_version": str(args.dataprocessor_version),
            "created_at": utc_iso_now(),
        },
    )

    extractor_keys, component_names = _parse_extractors_arg(args.extractors)

    # Получаем список включенных extractors из конфига (если доступен global_config_parser)
    enabled_extractors_from_config = None
    if hasattr(args, "global_config") and args.global_config:
        try:
            from configs.config_parser import GlobalConfigParser
            global_config_parser = GlobalConfigParser(args.global_config)
            enabled_extractors_from_config = global_config_parser.get_audio_extractors_list()
        except Exception as e:
            safe_log_warning(logger, f"AudioProcessor | Failed to get enabled extractors from config: {e}")
    
    # Если extractor_keys пустой и есть конфигурация, используем список из конфигурации
    if not extractor_keys and enabled_extractors_from_config:
        extractor_keys = enabled_extractors_from_config
        # Пересоздаем component_names для новых extractor_keys
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
        }
        component_names = [key_to_component.get(k, f"{k}_extractor") for k in extractor_keys]

    # Dependency resolution: автоматическое упорядочивание extractors и валидация зависимостей
    from src.core.dependency_resolver import (
        resolve_extractor_dependencies,
        validate_feature_flags,
        get_feature_flag_dependencies,
    )
    
    # Получаем список всех доступных extractors из key_to_component mapping
    # Используем тот же mapping, что и в _parse_extractors_arg (определен выше)
    # Извлекаем ключи из уже распарсенного результата _parse_extractors_arg
    # Но для dependency resolver нужен полный список всех доступных extractors
    _all_extractor_keys = [
        "clap", "tempo", "loudness", "asr", "speaker_diarization", "emotion_diarization",
        "source_separation", "speech_analysis", "spectral", "quality", "mfcc", "mel",
        "onset", "chroma", "rhythmic", "voice_quality", "hpss", "key", "band_energy", "spectral_entropy", "pitch"
    ]
    all_available_extractors = _all_extractor_keys
    
    # Разрешаем зависимости между extractors (автоматически добавляем недостающие)
    auto_add_deps = True  # Автоматически добавлять зависимости для оптимизации
    strict_mode = not bool(getattr(args, "no_strict_extractors", False))  # Строгий режим, если не --no-strict-extractors
    
    # Получаем feature flags для speech_analysis из args (для условных зависимостей)
    speech_enable_asr_metrics = bool(getattr(args, "speech_enable_asr_metrics", False))
    speech_enable_diarization_metrics = bool(getattr(args, "speech_enable_diarization_metrics", False))
    speech_enable_pitch_metrics = bool(getattr(args, "speech_enable_pitch_metrics", False))
    # pitch_enabled флаг (из --speech-analysis-pitch)
    speech_pitch_enabled = bool(getattr(args, "speech_analysis_pitch", False))
    
    ordered_extractors, dep_warnings, dep_errors = resolve_extractor_dependencies(
        extractor_keys,
        available_extractors=all_available_extractors,
        auto_add_dependencies=auto_add_deps,
        strict_mode=strict_mode,
        enabled_extractors=enabled_extractors_from_config,  # Передаем список включенных из конфига
        speech_enable_asr_metrics=speech_enable_asr_metrics,
        speech_enable_diarization_metrics=speech_enable_diarization_metrics,
        speech_enable_pitch_metrics=speech_enable_pitch_metrics,
        speech_pitch_enabled=speech_pitch_enabled,
    )
    
    # Выводим предупреждения и ошибки
    if dep_warnings:
        for w in dep_warnings:
            safe_log_warning(logger, f"AudioProcessor | Dependency warning: {w}")
    
    if dep_errors:
        for e in dep_errors:
            safe_log_error(logger, f"AudioProcessor | Dependency error: {e}")
        if strict_mode:
            raise RuntimeError(f"AudioProcessor | Dependency resolution failed: {', '.join(dep_errors)}")
    
    # Обновляем extractor_keys и component_names с учетом упорядочивания
    if ordered_extractors != extractor_keys:
        # Пересоздаем component_names в правильном порядке
        # Используем полный key_to_component mapping из _parse_extractors_arg
        key_to_component_full = {
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
        component_names = [key_to_component_full.get(k, f"{k}_extractor") for k in ordered_extractors]
        extractor_keys = ordered_extractors
        logger.info(f"AudioProcessor | Reordered extractors based on dependencies: {extractor_keys}")

    # Валидация feature flags для каждого extractor'а
    feature_flag_warnings: List[str] = []
    feature_flag_errors: List[str] = []
    
    # Собираем включенные feature flags для каждого extractor'а
    extractor_feature_flags: Dict[str, Set[str]] = {}
    
    # ASR feature flags
    if "asr" in extractor_keys:
        asr_flags = set()
        if getattr(args, "asr_enable_token_sequences", False):
            asr_flags.add("enable_token_sequences")
        if getattr(args, "asr_enable_token_counts", False):
            asr_flags.add("enable_token_counts")
        if getattr(args, "asr_enable_token_total", False):
            asr_flags.add("enable_token_total")
        if getattr(args, "asr_enable_token_density", False):
            asr_flags.add("enable_token_density")
        if getattr(args, "asr_enable_speech_rate", False):
            asr_flags.add("enable_speech_rate")
        if getattr(args, "asr_enable_lang_distribution", False):
            asr_flags.add("enable_lang_distribution")
        if getattr(args, "asr_enable_segments_with_speech", False):
            asr_flags.add("enable_segments_with_speech")
        if getattr(args, "asr_enable_avg_segment_duration", False):
            asr_flags.add("enable_avg_segment_duration")
        if getattr(args, "asr_enable_token_variance", False):
            asr_flags.add("enable_token_variance")
        extractor_feature_flags["asr"] = asr_flags
    
    # Key feature flags
    if "key" in extractor_keys:
        key_flags = set()
        if getattr(args, "key_enable_detailed_scores", False):
            key_flags.add("enable_detailed_scores")
        if getattr(args, "key_enable_top_k", False):
            key_flags.add("enable_top_k")
        if getattr(args, "key_enable_time_series", False):
            key_flags.add("enable_time_series")
        if getattr(args, "key_enable_key_changes", False):
            key_flags.add("enable_key_changes")
        if getattr(args, "key_enable_stability_metrics", False):
            key_flags.add("enable_stability_metrics")
        extractor_feature_flags["key"] = key_flags
    
    # Mel feature flags
    if "mel" in extractor_keys:
        mel_flags = set()
        if getattr(args, "mel_enable_statistics", False):
            mel_flags.add("enable_statistics")
        if getattr(args, "mel_enable_stats_vector", False):
            mel_flags.add("enable_stats_vector")
        extractor_feature_flags["mel"] = mel_flags
    
    # Валидируем feature flags для каждого extractor'а
    for ext_key, flags in extractor_feature_flags.items():
        warnings, errors = validate_feature_flags(ext_key, flags, strict_mode=strict_mode)
        feature_flag_warnings.extend(warnings)
        feature_flag_errors.extend(errors)
    
    # Выводим предупреждения и ошибки для feature flags
    if feature_flag_warnings:
        for w in feature_flag_warnings:
            safe_log_warning(logger, f"AudioProcessor | Feature flag warning: {w}")
    
    if feature_flag_errors:
        for e in feature_flag_errors:
            safe_log_error(logger, f"AudioProcessor | Feature flag error: {e}")
        if strict_mode:
            raise RuntimeError(f"AudioProcessor | Feature flag validation failed: {', '.join(feature_flag_errors)}")

    # Resolve model metadata via ModelManager (if available) for reproducibility.
    model_metadata = resolve_model_metadata(args)
    clap_model_used = model_metadata["clap_model_used"]
    asr_model_used = model_metadata["asr_model_used"]
    tokenizer_model_used = model_metadata["tokenizer_model_used"]
    diar_model_used = model_metadata["diar_model_used"]
    emo_model_used = model_metadata["emo_model_used"]
    sep_model_used = model_metadata["sep_model_used"]

    # Keep all AudioProcessor internal temp outputs inside the run folder (debuggable, but not source-of-truth).
    tmp_dir = os.path.join(run_rs_path, "_tmp_audio")
    os.makedirs(tmp_dir, exist_ok=True)

    # Парсинг конфигурации extractors из global_config.yaml (для render флагов и feature flags)
    # Пытаемся получить из args, если передано через config_parser
    extractor_config: Dict[str, Dict[str, Any]] = {}
    if hasattr(args, 'extractor_config') and args.extractor_config:
        try:
            if isinstance(args.extractor_config, str):
                extractor_config = json.loads(args.extractor_config)
            else:
                extractor_config = args.extractor_config
            logger.info(f"AudioProcessor | Parsed extractor_config: {list(extractor_config.keys())}")
            if "key" in extractor_config:
                key_cfg = extractor_config["key"]
                feature_flags = key_cfg.get("feature_flags", {})
                logger.info(f"AudioProcessor | key extractor config - feature_flags: {feature_flags}")
        except Exception as e:
            safe_log_warning(logger, f"Failed to parse extractor config: {e}, using defaults")
    else:
        logger.info(f"AudioProcessor | No extractor_config in args (hasattr={hasattr(args, 'extractor_config')}, value={getattr(args, 'extractor_config', None)})")

    # Инициализация MainProcessor (выводим в начале)
    from src.utils.progress import Colors  # type: ignore
    use_colors = Colors.supports_color()
    
    if use_colors:
        audio_processor_prefix = f"{Colors.BLUE}{Colors.BOLD}AudioProcessor{Colors.RESET} {Colors.GRAY}|{Colors.RESET}"
        print(f"{audio_processor_prefix} Initializing MainProcessor and extractors...", file=sys.stderr, flush=True)
    else:
        print("AudioProcessor | Initializing MainProcessor and extractors...", file=sys.stderr, flush=True)
    
    t_init_start = time.time()
    processor = create_main_processor(args, extractor_keys, extractor_config=extractor_config)
    t_init_end = time.time()
    init_elapsed = t_init_end - t_init_start
    
    if use_colors:
        time_str = f"{Colors.GREEN}{init_elapsed:.2f}s{Colors.RESET}"
        extractors_str = f"{Colors.YELLOW}{len(processor.extractors)} extractors{Colors.RESET}"
        print(f"{audio_processor_prefix} MainProcessor initialized {Colors.GRAY}({Colors.RESET}{time_str}{Colors.GRAY}, {Colors.RESET}{extractors_str}{Colors.GRAY}){Colors.RESET}", file=sys.stderr, flush=True)
    else:
        print(f"AudioProcessor | MainProcessor initialized ({init_elapsed:.2f}s, {len(processor.extractors)} extractors)", file=sys.stderr, flush=True)

    # Resolve Segmenter-produced audio + segments (new contract).
    frames_dir = os.path.abspath(args.frames_dir) if args.frames_dir else None
    audio_path, segments_payload = load_and_validate_segments(frames_dir, extractor_keys)
    # NOTE: legacy mode (extract audio from video) is removed. Segmenter contract is mandatory.

    started_at = utc_iso_now()
    t0 = time.time()
    
    # Stage timings tracking
    stage_timings: Dict[str, float] = {}
    timings_by_extractor: Dict[str, Dict[str, float]] = {}
    
    # Progress reporting: load_input stage
    t_load_input_start = time.time()
    _emit_progress(
        platform_id=args.platform_id,
        video_id=video_id,
        run_id=run_id,
        component="audio_processor",
        stage_id="load_input",
        stage_name="Loading input",
        progress_pct=5,
        elapsed_sec=0.0,
        total_elapsed_sec=time.time() - t0,
    )
    
    # Парсинг индивидуальных настроек parallelism для каждого extractor'а
    extractor_parallelism: Dict[str, Dict[str, Any]] = {}
    if args.extractor_parallelism_config:
        try:
            extractor_parallelism = json.loads(args.extractor_parallelism_config)
        except Exception as e:
            safe_log_warning(logger, f"Failed to parse extractor parallelism config: {e}, using defaults")
    
    # Legacy: глобальные настройки (fallback, если нет индивидуальных)
    segment_parallelism = max(1, int(args.segment_parallelism or 1))
    max_inflight = int(args.max_inflight) if args.max_inflight is not None else segment_parallelism
    max_inflight = max(1, int(max_inflight))
    clap_batch_size = max(1, int(args.clap_batch_size or 1))
    
    # Инициализация мониторинга ресурсов
    logger.info("AudioProcessor | Initializing resource monitor...")
    resource_monitor = ResourceMonitor()
    resource_monitor.start()
    logger.info("AudioProcessor | Resource monitor started")

    # Complete load_input stage
    t_load_input_end = time.time()
    load_input_elapsed = t_load_input_end - t_load_input_start
    stage_timings["load_input_ms"] = float(load_input_elapsed * 1000.0)
    total_elapsed = t_load_input_end - t0
    
    _emit_progress(
        platform_id=args.platform_id,
        video_id=video_id,
        run_id=run_id,
        component="audio_processor",
        stage_id="load_input",
        stage_name="Loading input",
        progress_pct=5,
        elapsed_sec=load_input_elapsed,
        total_elapsed_sec=total_elapsed,
    )
    
    # Выводим информацию о конфигурации
    logger.info(f"AudioProcessor | Configuration: segment_parallelism={segment_parallelism}, max_inflight={max_inflight}, clap_batch_size={clap_batch_size}")
    logger.info(f"AudioProcessor | Extractors to run: {', '.join(extractor_keys)}")
    
    # Progress reporting: run_extractors stage
    t_run_extractors_start = time.time()
    _emit_progress(
        platform_id=args.platform_id,
        video_id=video_id,
        run_id=run_id,
        component="audio_processor",
        stage_id="run_extractors",
        stage_name="Running extractors",
        progress_pct=10,
        elapsed_sec=0.0,
        total_elapsed_sec=time.time() - t0,
    )

    # Resource metrics will be captured in finally block and stored here
    resource_metrics_global: Dict[str, Optional[float]] = {}
    
    per_extractor_report: Dict[str, Any] = {}
    try:
        # Stage 5: Batch processing mode
        if batch_mode:
            # Создаем AudioFileContext для каждого файла (используем новый модуль)
            audio_file_contexts = create_audio_file_contexts(
                batch_frames_dirs, args.rs_base, args.platform_id, run_id
            )
            
            # Настраиваем параметры batch обработки
            max_video_workers = args.batch_max_workers
            if max_video_workers is None:
                import os as os_module
                max_video_workers = os_module.cpu_count() or 4
            
            enable_gpu_batching = not args.no_batch_gpu
            enable_cpu_parallel = not args.no_batch_cpu_parallel
            enable_video_parallel = enable_cpu_parallel
            
            # Устанавливаем параметры batch обработки в processor
            processor._batch_max_video_workers = max_video_workers
            processor._batch_enable_gpu_batching = enable_gpu_batching
            processor._batch_enable_cpu_parallel = enable_cpu_parallel
            processor._batch_max_segments_per_gpu_batch = args.batch_max_segments_per_gpu_batch
            
            # Запускаем batch обработку
            batch_results = processor.run_batch(
                audio_file_contexts=audio_file_contexts,
                extractor_names=extractor_keys,
                max_video_workers=max_video_workers,
                enable_video_parallel=enable_video_parallel,
                enable_gpu_batching=enable_gpu_batching,
                enable_cpu_parallel=enable_cpu_parallel,
            )
            
            # Обрабатываем результаты batch (используем новый модуль)
            successful_count, failed_count = process_batch_results(batch_results, per_extractor_report)
            
            # Выходим с соответствующим кодом
            if failed_count > 0:
                return 1
            return 0
        
        # Single-file mode (existing logic)
        if frames_dir:
            # New mode: run extractors directly on Segmenter audio, using Segmenter segments.
            strict_extractors = not bool(getattr(args, "no_strict_extractors", False))
            logger.info(f"AudioProcessor | Starting extraction of {len(extractor_keys)} extractors...")
            
            # Запускаем extractors используя новый модуль
            extractor_results, per_extractor_report, timings_by_extractor = run_extractors(
                processor=processor,
                extractor_keys=extractor_keys,
                audio_path=audio_path,
                tmp_dir=tmp_dir,
                segments_payload=segments_payload,
                run_rs_path=run_rs_path,
                platform_id=args.platform_id,
                video_id=video_id,
                run_id=run_id,
                segment_parallelism=segment_parallelism,
                max_inflight=max_inflight,
                clap_batch_size=clap_batch_size,
                extractor_parallelism_config=extractor_parallelism,
                strict_extractors=strict_extractors,
                t_start=t0,  # Передаем t0 для правильного подсчета total_elapsed_sec
            )
            
            results = {
                "extractor_results": extractor_results,
                "extracted_audio_path": audio_path,
            }
            logger.info(f"AudioProcessor | Extraction completed: {len([r for r in extractor_results.values() if r.get('success')])}/{len(extractor_keys)} extractors succeeded")
        # NOTE: legacy mode (audio extraction from video) is removed. Segmenter contract is mandatory.
        finished_at = utc_iso_now()
        duration_ms = int((time.time() - t0) * 1000)
    finally:
        # Останавливаем мониторинг ресурсов
        resource_monitor.stop()
        resource_metrics = resource_monitor.get_metrics()
        
        # Store resource metrics for use in NPZ meta (will be used in loop below)
        resource_metrics_global["cpu_rss_peak_mb"] = resource_metrics.get("cpu_rss_peak_mb")
        resource_metrics_global["gpu_vram_peak_mb"] = resource_metrics.get("gpu_vram_peak_mb")
        
        # Emit scheduler-facing runtime report (best-effort).
        try:
            report = {
                "schema_version": "scheduler_runtime_report_v1",
                "created_at": utc_iso_now(),
                "platform_id": args.platform_id,
                "video_id": video_id,
                "run_id": run_id,
                "config_hash": config_hash,
                "scheduler_knobs": {
                    "audio.segment_parallelism": int(segment_parallelism),
                    "audio.max_inflight": int(max_inflight),
                    "audio.clap_batch_size": int(clap_batch_size),
                },
                "per_processor": {
                    "audio": {
                        "started_at": started_at,
                        "finished_at": finished_at if "finished_at" in locals() else None,
                        "duration_ms": int(duration_ms) if "duration_ms" in locals() else None,
                        "rss_peak_mb": resource_metrics.get("cpu_rss_peak_mb"),
                        "gpu_used_peak_mb": resource_metrics.get("gpu_vram_peak_mb"),
                        "per_extractor": per_extractor_report,
                    }
                },
            }
            report_path = os.path.join(run_rs_path, "_reports", "scheduler_runtime_report.json")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            atomic_write_json(report_path, report)
        except Exception:
            pass

    extractor_results = (results or {}).get("extractor_results") or {}
    extracted_audio_path = (results or {}).get("extracted_audio_path")
    audio_present = bool(isinstance(extracted_audio_path, str) and extracted_audio_path and os.path.exists(extracted_audio_path))
    audio_empty_reason = None if audio_present else "audio_missing_or_extract_failed"
    # Map internal keys -> payloads for saving.
    # Progress reporting: save_npz stage
    t_save_npz_start = time.time()
    _emit_progress(
        platform_id=args.platform_id,
        video_id=video_id,
        run_id=run_id,
        component="audio_processor",
        stage_id="save_npz",
        stage_name="Saving NPZ artifacts",
        progress_pct=80,
        elapsed_sec=0.0,
        total_elapsed_sec=time.time() - t0,
    )

    key_to_component = dict(zip(extractor_keys, component_names))

    overall_ok = True

    for key, component_name in key_to_component.items():
        r = extractor_results.get(key) or {}
        success = bool(r.get("success"))
        payload = r.get("payload") if isinstance(r.get("payload"), dict) else None
        err = r.get("error")

        # Normalize missing result as error.
        if key not in extractor_results:
            success = False
            err = f"missing extractor result for key={key}"

        if not audio_present:
            # Missing audio is a normal empty case (e.g., silent videos).
            status = "empty"
            empty_reason = audio_empty_reason
        else:
            # Allow extractors to declare valid empty outputs explicitly.
            if success and isinstance(payload, dict) and str(payload.get("status") or "") == "empty":
                status = "empty"
                empty_reason = str(payload.get("empty_reason") or "empty")
            else:
                status = "ok" if success else "error"
                empty_reason = None

        # Progress reporting: validate_artifact stage (per component)
        t_validate_start = time.time()
        _emit_progress(
            platform_id=args.platform_id,
            video_id=video_id,
            run_id=run_id,
            component=component_name,
            stage_id="validate_artifact",
            stage_name="Validating artifact",
            progress_pct=85,
            elapsed_sec=0.0,
            total_elapsed_sec=time.time() - t0,
        )
        
        # Save NPZ artifact regardless (ok or error) so downstream can see status in meta.
        producer_version = getattr(processor.extractors.get(key), "version", None) or "unknown"
        
        # Add stage_timings_ms and timings_by_extractor to extra_meta
        component_stage_timings = stage_timings.copy()
        if key in timings_by_extractor:
            component_stage_timings["extractor_wall_ms"] = timings_by_extractor[key].get("wall_ms", 0.0)
            component_stage_timings["extractor_reported_ms"] = timings_by_extractor[key].get("reported_ms", 0.0)
        
        artifact_path = _save_component_npz(
            run_rs_path=run_rs_path,
            component_name=component_name,
            payload=payload,
            status=status,
            error=str(err) if err else None,
            empty_reason=empty_reason,
            producer_version=str(producer_version),
            schema_version="audio_npz_v1",
            extra_meta={
                # Required run identity fields (baseline contract)
                "platform_id": args.platform_id,
                "video_id": video_id,
                "run_id": run_id,
                "config_hash": config_hash,
                "sampling_policy_version": args.sampling_policy_version,
                "dataprocessor_version": str(args.dataprocessor_version),
                "device_used": r.get("device_used", args.device),
                # scheduler knobs (applied)
                "scheduler_knobs": {
                    "segment_parallelism": int(segment_parallelism),
                    "max_inflight": int(max_inflight),
                    "clap_batch_size": int(clap_batch_size),
                },
                # Per-extractor timings and resource metrics
                "stage_timings_ms": component_stage_timings,
                "resource_metrics": resource_metrics_global if resource_metrics_global else {},
            },
        )
        
        t_validate_end = time.time()
        validate_elapsed = t_validate_end - t_validate_start
        if "validate_artifact_ms" not in stage_timings:
            stage_timings["validate_artifact_ms"] = 0.0
        stage_timings["validate_artifact_ms"] += float(validate_elapsed * 1000.0)
        total_elapsed = t_validate_end - t0
        
        _emit_progress(
            platform_id=args.platform_id,
            video_id=video_id,
            run_id=run_id,
            component=component_name,
            stage_id="validate_artifact",
            stage_name="Validating artifact",
            progress_pct=90,
            elapsed_sec=validate_elapsed,
            total_elapsed_sec=total_elapsed,
        )
        
        # Validate artifact (best-effort)
        warnings = []
        v_ok = True
        issues = []
        try:
            v_ok, issues, _ = validate_npz(artifact_path)
        except Exception as e:
            warnings.append(f"NPZ validation warning: {e}")
            v_ok = False
        
        # Extract meta from payload if available
        meta = payload.get("meta") if isinstance(payload, dict) else None
        
        # Set error_code and notes
        error_code = None
        notes = None
        if not v_ok:
            status = "error"
            notes = "artifact validation failed: " + "; ".join(i.message for i in issues[:5]) if issues else "validation failed"
            safe_log_warning(logger, f"AudioProcessor | Validation failed for {component_name}: {notes}")
            overall_ok = False
        
        # Progress reporting: render stage (per component, optional)
        render_path = None
        try:
            # Получаем флаги рендеринга из конфига
            from src.core.extractor_runner import get_extractor_render_flags  # type: ignore
            # Используем extractor_config, если он доступен, иначе значения по умолчанию
            enable_render, enable_html_render = get_extractor_render_flags(
                key, 
                extractor_config if extractor_config else {}, 
                default_enable_render=True, 
                default_enable_html_render=True
            )
            
            # Generate render context if enabled
            if enable_render:
                from src.core.renderer import render_component  # type: ignore
                component_dir = os.path.join(run_rs_path, component_name)
                if os.path.exists(component_dir):
                    artifact_path_for_render = artifact_path
                    try:
                        render = render_component(
                            artifact_path_for_render,
                            component_name,
                            component_dir,
                            enable_render=enable_render,
                            enable_html_render=enable_html_render,
                        )
                        render_path = os.path.join(component_dir, "_render", "render_context.json")
                        logger.info(f"AudioProcessor | Render generated for {component_name} (HTML: {enable_html_render})")
                    except Exception as e:
                        # Best-effort: do not fail run if render fails
                        safe_log_warning(logger, f"Failed to generate render-context for {component_name}: {e}")
            else:
                logger.debug(f"Render disabled for {component_name}, skipping render generation")
        except Exception as e:
            # Best-effort: do not fail run if render fails
            safe_log_warning(logger, f"Failed to generate render-context for {component_name}: {e}")
        
        # Progress reporting: update_manifest stage (per component)
        t_update_manifest_start = time.time()
        _emit_progress(
            platform_id=args.platform_id,
            video_id=video_id,
            run_id=run_id,
            component=component_name,
            stage_id="update_manifest",
            stage_name="Updating manifest",
            progress_pct=95,
            elapsed_sec=0.0,
            total_elapsed_sec=time.time() - t0,
        )
        
        # Build artifacts list (include render if available)
        artifacts = [{"path": artifact_path, "type": "npz"}]
        if render_path and os.path.exists(render_path):
            artifacts.append({"path": render_path, "type": "render"})
        
        manifest.upsert_component(
            ManifestComponent(
                name=component_name,
                kind="audio",
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                artifacts=artifacts,
                error=str(err) if err else None,
                error_code=error_code,
                warnings=warnings,
                notes=notes,
                device_used=(meta or {}).get("device_used") if isinstance(meta, dict) else r.get("device_used", args.device),
                producer_version=(meta or {}).get("producer_version") if isinstance(meta, dict) else None,
                schema_version=(meta or {}).get("schema_version") if isinstance(meta, dict) else None,
            )
        )
        
        t_update_manifest_end = time.time()
        update_manifest_elapsed = t_update_manifest_end - t_update_manifest_start
        if "update_manifest_ms" not in stage_timings:
            stage_timings["update_manifest_ms"] = 0.0
        stage_timings["update_manifest_ms"] += float(update_manifest_elapsed * 1000.0)
        total_elapsed = t_update_manifest_end - t0
        
        _emit_progress(
            platform_id=args.platform_id,
            video_id=video_id,
            run_id=run_id,
            component=component_name,
            stage_id="update_manifest",
            stage_name="Updating manifest",
            progress_pct=98,
            elapsed_sec=update_manifest_elapsed,
            total_elapsed_sec=total_elapsed,
        )
        
        # Track overall success
        if status == "error":
            overall_ok = False
            error_msg = str(err) if err else "unknown error"
            safe_log_warning(logger, f"AudioProcessor | Status is 'error' for {component_name}, setting overall_ok=False")
            safe_log_error(logger, f"AudioProcessor | {component_name} error details: {error_msg}")
        elif status == "empty":
            # Empty is OK (e.g., silent video)
            pass
        else:
            # status == "ok"
            pass
    
    # Complete save_npz stage
    t_save_npz_end = time.time()
    save_npz_elapsed = t_save_npz_end - t_save_npz_start
    stage_timings["save_npz_ms"] = float(save_npz_elapsed * 1000.0)
    total_elapsed = t_save_npz_end - t0
    
    _emit_progress(
        platform_id=args.platform_id,
        video_id=video_id,
        run_id=run_id,
        component="audio_processor",
        stage_id="save_npz",
        stage_name="Saving NPZ artifacts",
        progress_pct=85,
        elapsed_sec=save_npz_elapsed,
        total_elapsed_sec=total_elapsed,
    )
    
    # Progress reporting: complete
    t_complete = time.time()
    final_total_elapsed = t_complete - t0
    _emit_progress(
        platform_id=args.platform_id,
        video_id=video_id,
        run_id=run_id,
        component="audio_processor",
        stage_id="complete",
        stage_name="Complete",
        progress_pct=100,
        elapsed_sec=final_total_elapsed,
        total_elapsed_sec=final_total_elapsed,
    )
    
    # Ensure progress bar is closed
    close_progress_bar()
    
    return 0 if overall_ok else 2

if __name__ == "__main__":
    rc = int(main())
    raise SystemExit(rc)


