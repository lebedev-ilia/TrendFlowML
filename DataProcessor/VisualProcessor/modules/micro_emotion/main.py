"""
CLI entrypoint for VisualProcessor module `micro_emotion` (baseline-ready).

Policy:
- No-fallback sampling: uses `metadata["micro_emotion"]["frame_indices"]` provided by Segmenter
- Time axis: `times_s = union_timestamps_sec[frame_indices]`
- Requires `core_face_landmarks` and runs OpenFace only on face-present frames (but output stays aligned to primary indices).
- Writes NPZ via MicroEmotionModule (fixed filename), no separate JSON artifacts in result_store.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Optional, List


_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PATH not in sys.path:
    sys.path.append(_PATH)

from utils.logger import get_logger
from micro_emotion_processor import MicroEmotionModule  # type: ignore


MODULE_NAME = "micro_emotion"
logger = get_logger(MODULE_NAME)


def run_pipeline(
    frames_dir: str,
    rs_path: str,
    feature_groups: str = "default",
    openface_batch_size: int = 64,
    docker_image: str = "openface/openface:latest",
) -> str:
    """
    Основная логика обработки micro_emotion.

    Возвращает путь к сохраненному NPZ.
    """
    module = MicroEmotionModule(
        rs_path=rs_path,
        docker_image=docker_image,
        openface_batch_size=int(openface_batch_size),
        feature_groups=str(feature_groups),
        use_face_detection=True,
        device="cuda",
    )
    return module.run(frames_dir=frames_dir, config={})


def main(argv: Optional[List[str]] = None) -> int:
    """CLI‑вход для модуля micro_emotion (аналогично action_recognition.main)."""

    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Micro Emotion Module - Extracts micro-expressions and Action Units using OpenFace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Директория с кадрами (должна содержать metadata.json)",
    )
    parser.add_argument(
        "--rs-path",
        required=True,
        help="Путь к директории ResultsStore (будут сохранены результаты micro_emotion)",
    )
    parser.add_argument(
        "--feature-groups",
        type=str,
        default="default",
        help="Feature groups to enable (CSV) or preset name (e.g. default, all, compact22).",
    )
    parser.add_argument(
        "--openface-batch-size",
        type=int,
        default=64,
        help="Max frames per OpenFace run (images per batch).",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default="openface/openface:latest",
        help="Docker image for OpenFace FeatureExtraction",
    )
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

        _logging.getLogger().setLevel(
            getattr(_logging, args.log_level.upper(), _logging.INFO)
        )
    except Exception:  # noqa: BLE001
        logger.warning("Не удалось установить log-level: %s", args.log_level)

    try:
        run_pipeline(
            frames_dir=args.frames_dir,
            rs_path=args.rs_path,
            feature_groups=args.feature_groups,
            openface_batch_size=args.openface_batch_size,
            docker_image=args.docker_image,
        )
        return 0
    except FileNotFoundError as e:
        logger.error("Файл не найден: %s", e)
        return 2
    except ValueError as e:
        logger.error("Некорректные данные: %s", e)
        return 3
    except Exception as e:  # noqa: BLE001
        logger.exception("Fatal error в %s: %s", MODULE_NAME, e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())