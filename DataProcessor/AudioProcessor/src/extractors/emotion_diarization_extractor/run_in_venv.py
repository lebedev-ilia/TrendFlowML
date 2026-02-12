#!/usr/bin/env python3
"""
Wrapper script для запуска emotion_diarization_extractor в персональной venv.
Используется оркестратором для изолированного запуска экстрактора.
"""
import sys
import os

# Подавляем логи загрузки весов ДО всех импортов
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
# Подавляем DEBUG логи от speechbrain
os.environ["SPEECHBRAIN_LOG_LEVEL"] = "ERROR"

import json
import argparse
import numpy as np
from pathlib import Path
import warnings
import logging

# Подавляем warnings
warnings.filterwarnings("ignore")

# Устанавливаем уровень логирования для всех библиотек ДО их импорта
logging.basicConfig(level=logging.ERROR)
# Подавляем все логи от всех библиотек
for logger_name in ["transformers", "huggingface_hub", "accelerate", "speechbrain", "tqdm", "urllib3", "requests"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    logger.propagate = False

# Подавляем DEBUG логи от speechbrain.utils
speechbrain_utils_logger = logging.getLogger("speechbrain.utils")
speechbrain_utils_logger.setLevel(logging.ERROR)
speechbrain_utils_logger.propagate = False

# Подавляем DEBUG логи от speechbrain.utils.checkpoints
speechbrain_checkpoints_logger = logging.getLogger("speechbrain.utils.checkpoints")
speechbrain_checkpoints_logger.setLevel(logging.ERROR)
speechbrain_checkpoints_logger.propagate = False

# Добавляем пути для импорта
ap_root = Path(__file__).resolve().parent.parent.parent.parent
repo_root = ap_root.parent

# AudioProcessor imports (нужно добавить AudioProcessor, чтобы src был доступен как модуль)
if str(ap_root) not in sys.path:
    sys.path.insert(0, str(ap_root))

# Repo root imports (dp_models, Segmenter helpers, etc.)
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Add local speechbrain to path
_extractor_dir = Path(__file__).resolve().parent
_speechbrain_path = _extractor_dir / "speechbrain"
if _speechbrain_path.exists() and str(_speechbrain_path) not in sys.path:
    sys.path.insert(0, str(_speechbrain_path))

from src.core.base_extractor import ExtractorResult  # type: ignore
from src.extractors.emotion_diarization_extractor.main import EmotionDiarizationExtractor  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Run emotion_diarization_extractor in isolated venv")
    parser.add_argument("--audio-path", required=True, help="Path to audio file")
    parser.add_argument("--tmp-dir", required=True, help="Temporary directory")
    parser.add_argument("--segments-json", help="Path to segments JSON file (optional)")
    parser.add_argument("--output-json", required=True, help="Path to output JSON file with result")
    parser.add_argument("--device", default="auto", help="Device (auto|cuda|cpu)")
    parser.add_argument("--model-size", default="small", choices=["small", "large"], help="Model size")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--enable-probs", action="store_true", help="Enable emotion_probs")
    parser.add_argument("--enable-ids", action="store_true", help="Enable emotion_id")
    parser.add_argument("--enable-confidence", action="store_true", help="Enable emotion_confidence")
    parser.add_argument("--enable-mean-probs", action="store_true", help="Enable emotion_mean_probs")
    parser.add_argument("--enable-entropy", action="store_true", help="Enable emotion_entropy")
    parser.add_argument("--enable-dominant", action="store_true", help="Enable dominant_emotion")
    parser.add_argument("--enable-quality-metrics", action="store_true", help="Enable quality_metrics")
    parser.add_argument("--silence-peak-threshold", type=float, default=1e-3, help="Silence peak threshold")
    parser.add_argument("--silence-rms-threshold", type=float, default=1e-4, help="Silence RMS threshold")
    parser.add_argument("--enable-silence-detection", action="store_true", default=True, help="Enable silence detection")
    parser.add_argument("--disable-silence-detection", action="store_true", help="Disable silence detection")
    parser.add_argument("--process-full-audio", action="store_true", help="Process entire audio as one segment (use run() instead of run_segments())")
    
    args = parser.parse_args()
    
    # Load segments if provided
    segments = []
    if args.segments_json and os.path.exists(args.segments_json):
        with open(args.segments_json, "r") as f:
            segments_data = json.load(f)
            # Extract segments from the payload structure
            if isinstance(segments_data, dict):
                families = segments_data.get("families", {})
                emo_family = families.get("emotion", {})
                segments = emo_family.get("segments", [])
            elif isinstance(segments_data, list):
                segments = segments_data
    
    # Create extractor
    extractor = EmotionDiarizationExtractor(
        device=args.device,
        model_size=args.model_size,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        enable_probs=args.enable_probs,
        enable_ids=args.enable_ids,
        enable_confidence=args.enable_confidence,
        enable_mean_probs=args.enable_mean_probs,
        enable_entropy=args.enable_entropy,
        enable_dominant=args.enable_dominant,
        enable_quality_metrics=args.enable_quality_metrics,
        silence_peak_threshold=args.silence_peak_threshold,
        silence_rms_threshold=args.silence_rms_threshold,
        enable_silence_detection=args.enable_silence_detection and not args.disable_silence_detection,
        process_full_audio=args.process_full_audio,
    )
    
    # Run extractor
    try:
        if args.process_full_audio:
            # Process full audio as one segment (ignore provided segments)
            result = extractor.run(args.audio_path, args.tmp_dir)
        elif segments:
            result = extractor.run_segments(args.audio_path, args.tmp_dir, segments)
        else:
            result = extractor.run(args.audio_path, args.tmp_dir)
    except Exception as e:
        # Если произошло исключение, создаем результат с ошибкой
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        from src.core.base_extractor import ExtractorResult  # type: ignore
        result = ExtractorResult(
            name="emotion_diarization_extractor",
            version="3.0.0",
            success=False,
            error=error_msg,
            processing_time=0.0
        )
    
    # Serialize result to JSON
    result_dict = {
        "success": result.success,
        "error": result.error,
        "processing_time": result.processing_time,
        "payload": result.payload if result.payload else {},
    }
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist() if hasattr(obj, 'tolist') else obj.item()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj
    
    result_dict = convert_for_json(result_dict)
    
    # Write result to JSON file
    with open(args.output_json, "w") as f:
        json.dump(result_dict, f, indent=2)
    
    # Exit with error code if failed
    if not result.success:
        # Выводим ошибку в stderr для отладки (но не DEBUG логи)
        if result.error:
            import sys
            sys.stderr.write(f"ERROR: {result.error}\n")
        sys.exit(1)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

