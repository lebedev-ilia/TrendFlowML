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
from pathlib import Path
from typing import Optional, List

# VisualProcessor must be first; prepending DataProcessor root breaks `from utils.logger`.
_vp = str(Path(__file__).resolve().parents[2])
if _vp not in sys.path:
    sys.path.insert(0, _vp)
elif sys.path[0] != _vp:
    try:
        sys.path.remove(_vp)
    except ValueError:
        pass
    sys.path.insert(0, _vp)
_repo_root = str(Path(_vp).parent)
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from utils.logger import get_logger
from modules.micro_emotion.utils.micro_emotion_processor import MicroEmotionModule  # type: ignore


MODULE_NAME = "micro_emotion"
logger = get_logger(MODULE_NAME)


def run_pipeline(
    frames_dir: str,
    rs_path: str,
    feature_groups: str = "default",
    openface_batch_size: int = 64,
    docker_image: str = "openface/openface:latest",
    *,
    fps: int = 30,
    microexpr_smoothing_sigma: float = 0.05,
    microexpr_delta_threshold: float = 0.4,
    microexpr_max_duration_frames: int = 15,
    microexpr_min_peak_distance_frames: int = 6,
    gaze_centered_threshold: float = 10.0,
    pca_components: int = 3,
    au_confidence_threshold: float = 0.5,
    device: str = "cuda",
    progress_every_frames: int = 50,
) -> str:
    """
    Основная логика обработки micro_emotion.

    Возвращает путь к сохраненному NPZ.
    """
    module = MicroEmotionModule(
        rs_path=rs_path,
        fps=int(fps),
        microexpr_smoothing_sigma=float(microexpr_smoothing_sigma),
        microexpr_delta_threshold=float(microexpr_delta_threshold),
        microexpr_max_duration_frames=int(microexpr_max_duration_frames),
        microexpr_min_peak_distance_frames=int(microexpr_min_peak_distance_frames),
        gaze_centered_threshold=float(gaze_centered_threshold),
        pca_components=int(pca_components),
        au_confidence_threshold=float(au_confidence_threshold),
        use_face_detection=True,
        docker_image=str(docker_image),
        openface_batch_size=int(openface_batch_size),
        device=str(device or "cuda"),
        feature_groups=str(feature_groups),
        progress_every_frames=int(progress_every_frames),
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
        "--fps",
        type=int,
        default=30,
        help="Кадров в секунду (используется для пересчёта временных меток и окон micro-expressions)",
    )
    parser.add_argument(
        "--microexpr-smoothing-sigma",
        type=float,
        default=0.05,
        help="Сглаживание для micro-expressions (в секундах)",
    )
    parser.add_argument(
        "--microexpr-delta-threshold",
        type=float,
        default=0.4,
        help="Порог изменения интенсивности для micro-expression",
    )
    parser.add_argument(
        "--microexpr-max-duration-frames",
        type=int,
        default=15,
        help="Максимальная длительность micro-expression в кадрах",
    )
    parser.add_argument(
        "--microexpr-min-peak-distance-frames",
        type=int,
        default=6,
        help="Минимальное расстояние между пиками micro-expression в кадрах",
    )
    parser.add_argument(
        "--gaze-centered-threshold",
        type=float,
        default=10.0,
        help="Порог для определения взгляда в камеру (градусы)",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=3,
        help="Количество PCA компонент для AU",
    )
    parser.add_argument(
        "--au-confidence-threshold",
        type=float,
        default=0.5,
        help="Порог уверенности AU для флагов надёжности",
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
        "--device",
        type=str,
        default="cuda",
        help="Устройство для обработки (cuda/cpu/auto)",
    )
    parser.add_argument(
        "--progress-every-frames",
        type=int,
        default=50,
        help="Как часто писать прогресс в state_events.jsonl (каждые N кадров)",
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
            fps=args.fps,
            microexpr_smoothing_sigma=args.microexpr_smoothing_sigma,
            microexpr_delta_threshold=args.microexpr_delta_threshold,
            microexpr_max_duration_frames=args.microexpr_max_duration_frames,
            microexpr_min_peak_distance_frames=args.microexpr_min_peak_distance_frames,
            gaze_centered_threshold=args.gaze_centered_threshold,
            pca_components=args.pca_components,
            au_confidence_threshold=args.au_confidence_threshold,
            device=args.device,
            progress_every_frames=args.progress_every_frames,
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