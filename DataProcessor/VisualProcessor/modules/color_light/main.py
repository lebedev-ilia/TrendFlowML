#!/usr/bin/env python3
"""
CLI интерфейс для модуля анализа цвета и освещения видео.

Использует BaseModule для автоматизации работы с метаданными и FrameManager.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.color_light.processor import ColorLightProcessor
from modules.color_light.presentation import write_presentation
from utils.logger import get_logger

MODULE_NAME = "color_light"
logger = get_logger(MODULE_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Анализ цвета и освещения видео — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Директория с кадрами (FrameManager ожидает metadata.json внутри)"
    )
    parser.add_argument(
        "--rs-path",
        required=True,
        help="Папка для результирующего стора (ResultsStore и npz)"
    )
    parser.add_argument(
        "--max-frames-per-scene",
        type=int,
        default=350,
        help="Максимальное количество кадров для обработки на сцену"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Шаг для выборки кадров"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Уровень логирования (DEBUG/INFO/WARN/ERROR)"
    )
    parser.add_argument(
        "--presentation-dir",
        type=str,
        default=None,
        help="Папка для presentation JSON/HTML (вне result_store)"
    )

    args = parser.parse_args(argv)

    # Настройка уровня логирования
    try:
        import logging as _logging
        _logging.getLogger().setLevel(
            getattr(_logging, args.log_level.upper(), _logging.INFO)
        )
    except Exception:
        logger.warning("Не удалось установить log-level: %s", args.log_level)

    try:
        # Инициализация модуля
        processor = ColorLightProcessor(
            rs_path=args.rs_path,
            max_frames_per_scene=args.max_frames_per_scene,
            stride=args.stride
        )

        # Запуск полного цикла через BaseModule.run (с валидацией meta)
        saved_path = processor.run(
            frames_dir=args.frames_dir,
            config={}
        )
        logger.info(
            f"Обработка завершена. Результаты сохранены: {saved_path}"
        )

        # Генерация presentation (если требуется)
        if args.presentation_dir:
            results = processor._load_npz(saved_path)
            # metadata доступна в frames_dir/metadata.json
            metadata = processor.load_metadata(args.frames_dir)
            json_path, html_path = write_presentation(
                results=results,
                metadata=metadata,
                base_dir=args.presentation_dir
            )
            logger.info("Presentation JSON: %s", json_path)
            logger.info("Presentation HTML: %s", html_path)

        # Выводим некоторые ключевые метрики (если есть)
        try:
            results = processor._load_npz(saved_path)
            if isinstance(results, dict) and "video_features" in results:
                vf = results["video_features"]
                logger.info("Ключевые метрики:")
                if "cinematic_lighting_score" in vf and isinstance(vf["cinematic_lighting_score"], (int, float)):
                    logger.info(f"  - Cinematic Lighting Score: {vf['cinematic_lighting_score']:.3f}")
                if "professional_look_score" in vf and isinstance(vf["professional_look_score"], (int, float)):
                    logger.info(f"  - Professional Look Score: {vf['professional_look_score']:.3f}")
                if "style_teal_orange_prob" in vf and isinstance(vf["style_teal_orange_prob"], (int, float)):
                    logger.info(f"  - Teal & Orange Style: {vf['style_teal_orange_prob']:.3f}")
                if "color_distribution_entropy" in vf and isinstance(vf["color_distribution_entropy"], (int, float)):
                    logger.info(f"  - Color Distribution Entropy: {vf['color_distribution_entropy']:.3f}")
        except Exception:
            pass

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

