#!/usr/bin/env python3
"""
CLI интерфейс для модуля анализа поведения людей в видео.

Использует BaseModule для автоматизации работы с метаданными и FrameManager.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.behavioral.behavior_analyzer import BehaviorAnalyzer
from utils.logger import get_logger

MODULE_NAME = "behavioral"
logger = get_logger(MODULE_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Анализ поведения людей в видео — CLI",
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
        "--log-level",
        type=str,
        default="INFO",
        help="Уровень логирования (DEBUG/INFO/WARN/ERROR)"
    )
    parser.add_argument(
        "--ui-json-path",
        type=str,
        default=None,
        help="(Опционально) путь для сохранения UI JSON (экспорт из NPZ)"
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
        analyzer = BehaviorAnalyzer(rs_path=args.rs_path)
        saved_path = analyzer.run(args.frames_dir, config={}, metadata=None)

        if isinstance(args.ui_json_path, str) and args.ui_json_path.strip():
            try:
                analyzer.export_ui_json(saved_path, args.ui_json_path.strip())
                logger.info(f"UI JSON сохранен: {args.ui_json_path.strip()}")
            except Exception as e:
                logger.exception("Ошибка при экспорте UI JSON: %s", e)

        return 0

    except FileNotFoundError as e:
        logger.error("Файл не найден: %s", e)
        import traceback
        logger.error("Traceback:\n%s", traceback.format_exc())
        return 2
    except ValueError as e:
        logger.error("Некорректные данные: %s", e)
        import traceback
        logger.error("Traceback:\n%s", traceback.format_exc())
        return 3
    except Exception as e:
        logger.exception("Fatal error в %s: %s", MODULE_NAME, e)
        import traceback
        logger.error("Full traceback:\n%s", traceback.format_exc())
        return 4


if __name__ == "__main__":
    raise SystemExit(main())