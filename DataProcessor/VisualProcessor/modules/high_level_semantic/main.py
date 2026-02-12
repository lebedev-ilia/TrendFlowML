#!/usr/bin/env python3
"""
CLI entrypoint for VisualProcessor module `high_level_semantic` (baseline-ready).

Key policies:
- No-fallback: frame_indices MUST be provided by Segmenter in frames_dir/metadata.json[high_level_semantic.frame_indices]
- Source-of-truth embeddings: consumes `core_clip/embeddings.npz` (does NOT load CLIP weights)
- Scene source: consumes `cut_detection` outputs (no internal scene detection)
- Output: single fixed-name NPZ artifact via BaseModule.save_results()
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.high_level_semantic.hl_semantic import HighLevelSemanticModule
from utils.logger import get_logger

MODULE_NAME = "high_level_semantic"
logger = get_logger(MODULE_NAME)


def run_pipeline(
    *,
    frames_dir: str,
    rs_path: str,
    feature_groups: Optional[str] = None,
    require_cut_detection_model_facing: bool = False,
    require_text_processor: bool = True,
    require_audio_loudness: bool = True,
    require_audio_tempo: bool = True,
    require_audio_clap: bool = False,
    progress_every_frames: int = 50,
    semantic_jump_topk_events: int = 256,
    semantic_jump_min_strength: float = 0.25,
) -> str:
    if not rs_path:
        raise ValueError(f"{MODULE_NAME} | rs_path is required")

    module = HighLevelSemanticModule(
        rs_path=rs_path,
        feature_groups=feature_groups,
        require_cut_detection_model_facing=require_cut_detection_model_facing,
        require_text_processor=require_text_processor,
        require_audio_loudness=require_audio_loudness,
        require_audio_tempo=require_audio_tempo,
        require_audio_clap=require_audio_clap,
        progress_every_frames=progress_every_frames,
        semantic_jump_topk_events=semantic_jump_topk_events,
        semantic_jump_min_strength=semantic_jump_min_strength,
    )
    return module.run(frames_dir=frames_dir, config={})


def main(argv: Optional[List[str]] = None) -> int:
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="High-level semantic features (baseline-ready) — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Директория с кадрами (должна содержать metadata.json)",
    )
    parser.add_argument(
        "--rs-path",
        type=str,
        required=True,
        help="Путь к директории ResultsStore для сохранения результатов",
    )

    parser.add_argument(
        "--feature-groups",
        type=str,
        default=None,
        help="CSV feature groups to enable. Example: core,scenes,events,audio,emotion,text",
    )

    parser.add_argument(
        "--require-cut-detection-model-facing",
        action="store_true",
        help="Fail-fast if cut_detection model-facing NPZ is missing (recommended for best quality).",
    )

    parser.add_argument("--require-text-processor", action="store_true", help="Fail-fast if text_processor/text_features.npz is missing.")
    parser.add_argument("--no-require-text-processor", action="store_true", help="Allow missing text_processor (not recommended).")

    parser.add_argument("--require-audio-loudness", action="store_true", help="Fail-fast if loudness_extractor is missing.")
    parser.add_argument("--no-require-audio-loudness", action="store_true", help="Allow missing loudness_extractor.")

    parser.add_argument("--require-audio-tempo", action="store_true", help="Fail-fast if tempo_extractor is missing.")
    parser.add_argument("--no-require-audio-tempo", action="store_true", help="Allow missing tempo_extractor.")

    parser.add_argument("--require-audio-clap", action="store_true", help="Fail-fast if clap_extractor is missing.")
    parser.add_argument("--no-require-audio-clap", action="store_true", help="Allow missing clap_extractor.")

    parser.add_argument("--progress-every-frames", type=int, default=50, help="Emit progress at least every N frames (unit=frame).")
    parser.add_argument("--semantic-jump-topk-events", type=int, default=256, help="Max semantic-jump events to emit.")
    parser.add_argument("--semantic-jump-min-strength", type=float, default=0.25, help="Min semantic jump strength for event candidates (1-cosine).")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Уровень логирования (DEBUG/INFO/WARN/ERROR)",
    )

    args = parser.parse_args(argv)

    # Настройка уровня логирования
    try:
        import logging as _logging

        _logging.getLogger().setLevel(getattr(_logging, args.log_level.upper(), _logging.INFO))
    except Exception:
        logger.warning("Не удалось установить log-level: %s", args.log_level)

    try:
        # resolve require_* tri-state flags
        require_text_processor = True
        if bool(args.no_require_text_processor):
            require_text_processor = False
        elif bool(args.require_text_processor):
            require_text_processor = True

        require_audio_loudness = True
        if bool(args.no_require_audio_loudness):
            require_audio_loudness = False
        elif bool(args.require_audio_loudness):
            require_audio_loudness = True

        require_audio_tempo = True
        if bool(args.no_require_audio_tempo):
            require_audio_tempo = False
        elif bool(args.require_audio_tempo):
            require_audio_tempo = True

        require_audio_clap = False
        if bool(args.no_require_audio_clap):
            require_audio_clap = False
        elif bool(args.require_audio_clap):
            require_audio_clap = True

        saved_path = run_pipeline(
            frames_dir=args.frames_dir,
            rs_path=args.rs_path,
            feature_groups=args.feature_groups,
            require_cut_detection_model_facing=bool(args.require_cut_detection_model_facing),
            require_text_processor=require_text_processor,
            require_audio_loudness=require_audio_loudness,
            require_audio_tempo=require_audio_tempo,
            require_audio_clap=require_audio_clap,
            progress_every_frames=int(args.progress_every_frames),
            semantic_jump_topk_events=int(args.semantic_jump_topk_events),
            semantic_jump_min_strength=float(args.semantic_jump_min_strength),
        )

        logger.info(f"Обработка завершена. Результаты сохранены: {saved_path}")
        return 0

    except FileNotFoundError as e:
        logger.error("Файл не найден: %s", e)
        return 2
    except ValueError as e:
        logger.error("Некорректные данные: %s", e)
        return 3
    except Exception as e:
        logger.exception("Fatal error в %s: %s", MODULE_NAME, e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
