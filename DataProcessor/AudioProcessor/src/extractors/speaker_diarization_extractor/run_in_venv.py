#!/usr/bin/env python3
"""
Wrapper script для запуска speaker_diarization_extractor в персональной venv.
Используется оркестратором для изолированного запуска экстрактора.
"""
import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

# Добавляем пути для импорта
ap_root = Path(__file__).resolve().parent.parent.parent.parent
repo_root = ap_root.parent

# AudioProcessor/src imports
src_path = ap_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Repo root imports (dp_models, Segmenter helpers, etc.)
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.core.base_extractor import ExtractorResult  # type: ignore
from src.extractors.speaker_diarization_extractor.main import SpeakerDiarizationExtractor  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Run speaker_diarization_extractor in isolated venv")
    parser.add_argument("--audio-path", required=True, help="Path to audio file")
    parser.add_argument("--tmp-dir", required=True, help="Temporary directory")
    parser.add_argument("--segments-json", help="Path to segments JSON file (optional)")
    parser.add_argument("--output-json", required=True, help="Path to output JSON file with result")
    parser.add_argument("--device", default="auto", help="Device (auto|cuda|cpu)")
    parser.add_argument("--whisper-model-size", default="small", help="Whisper model size")
    parser.add_argument("--huggingface-token", help="HuggingFace token (or set HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--enable-speaker-segments", action="store_true", help="Enable speaker_segments")
    parser.add_argument("--enable-speaker-embeddings", action="store_true", help="Enable speaker_embeddings")
    parser.add_argument("--enable-speaker-stats", action="store_true", help="Enable speaker_stats")
    parser.add_argument("--enable-speaker-durations", action="store_true", help="Enable speaker_durations")
    parser.add_argument("--enable-transcript", action="store_true", help="Enable transcript")
    parser.add_argument("--enable-word-segments", action="store_true", help="Enable word_segments")
    parser.add_argument("--silence-peak-threshold", type=float, default=1e-3, help="Silence peak threshold")
    parser.add_argument("--silence-rms-threshold", type=float, default=1e-4, help="Silence RMS threshold")
    parser.add_argument("--enable-silence-detection", action="store_true", default=True, help="Enable silence detection")
    parser.add_argument("--disable-silence-detection", action="store_true", help="Disable silence detection")
    
    args = parser.parse_args()
    
    # Load segments if provided
    segments = []
    if args.segments_json and os.path.exists(args.segments_json):
        with open(args.segments_json, "r") as f:
            segments_data = json.load(f)
            # Extract segments from the payload structure
            if isinstance(segments_data, dict):
                families = segments_data.get("families", {})
                diar_family = families.get("diarization", {})
                segments = diar_family.get("segments", [])
            elif isinstance(segments_data, list):
                segments = segments_data
    
    # Create extractor
    extractor = SpeakerDiarizationExtractor(
        device=args.device,
        whisper_model_size=args.whisper_model_size,
        huggingface_token=args.huggingface_token,
        sample_rate=args.sample_rate,
        enable_speaker_segments=args.enable_speaker_segments,
        enable_speaker_embeddings=args.enable_speaker_embeddings,
        enable_speaker_stats=args.enable_speaker_stats,
        enable_speaker_durations=args.enable_speaker_durations,
        enable_transcript=args.enable_transcript,
        enable_word_segments=args.enable_word_segments,
        silence_peak_threshold=args.silence_peak_threshold,
        silence_rms_threshold=args.silence_rms_threshold,
        enable_silence_detection=args.enable_silence_detection and not args.disable_silence_detection,
    )
    
    # Run extractor
    if segments:
        result = extractor.run_segments(args.audio_path, args.tmp_dir, segments)
    else:
        result = extractor.run(args.audio_path, args.tmp_dir)
    
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
        sys.exit(1)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

