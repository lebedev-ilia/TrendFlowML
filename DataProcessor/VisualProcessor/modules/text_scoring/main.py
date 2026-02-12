"""
CLI для `text_scoring` (BaseModule, NPZ output).

Важно:
- Модуль **не выполняет OCR сам** (никакого EasyOCR/pyTesseract внутри).
- Модуль consumer'ит OCR-артефакт (NPZ) от внешнего компонента (TextProcessor/OCR service).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.text_scoring.text_scoring import TextScoringModule
from utils.logger import get_logger

MODULE_NAME = "text_scoring"
logger = get_logger(MODULE_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Text scoring (consumer of OCR artifact) — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--frames-dir", required=True, help="Директория с кадрами (с metadata.json)")
    parser.add_argument("--rs-path", required=True, help="Папка result_store для артефактов")

    parser.add_argument("--ocr-npz", type=str, default=None, help="Путь к OCR NPZ (optional override)")
    parser.add_argument("--use-face-data", action="store_true", help="Использовать core_face_landmarks как сигнал alignment (optional)")
    parser.add_argument("--alignment-window-seconds", type=float, default=0.5, help="Окно выравнивания (секунды, time-axis)")
    parser.add_argument("--motion-weight", type=float, default=0.0, help="Вес motion (baseline: 0)")
    parser.add_argument("--face-weight", type=float, default=1.0, help="Вес face (baseline: 1 if use_face_data)")
    parser.add_argument("--audio-weight", type=float, default=0.0, help="Вес audio (baseline: 0)")
    parser.add_argument("--min-ocr-confidence", type=float, default=0.4, help="Минимальная OCR confidence")
    parser.add_argument("--retain-raw-ocr-text", action="store_true", help="Сохранять raw OCR текст в NPZ (privacy override)")
    parser.add_argument("--enable-text-peaks", action="store_true", help="Включить noisy text_emphasis_peak_* фичи")
    parser.add_argument("--enable-language-entropy", action="store_true", help="Включить ocr_language_entropy (noisy)")
    parser.add_argument("--enable-text-movement-speed", action="store_true", help="Включить text_movement_speed (noisy)")
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG/INFO/WARN/ERROR")

    args = parser.parse_args(argv)

    try:
        import logging as _logging
        _logging.getLogger().setLevel(getattr(_logging, args.log_level.upper(), _logging.INFO))
    except Exception:
        logger.warning("Не удалось установить log-level: %s", args.log_level)

    try:
        module = TextScoringModule(rs_path=args.rs_path)
        config: Dict[str, Any] = {
            "ocr_npz": args.ocr_npz,
            "use_face_data": bool(args.use_face_data),
            "alignment_window_seconds": float(args.alignment_window_seconds),
            "motion_weight": float(args.motion_weight),
            "face_weight": float(args.face_weight),
            "audio_weight": float(args.audio_weight),
            "min_ocr_confidence": float(args.min_ocr_confidence),
            "retain_raw_ocr_text": bool(args.retain_raw_ocr_text),
            "enable_text_peaks": bool(args.enable_text_peaks),
            "enable_language_entropy": bool(args.enable_language_entropy),
            "enable_text_movement_speed": bool(args.enable_text_movement_speed),
        }
        saved_path = module.run(frames_dir=args.frames_dir, config=config)
        logger.info("Готово. Результаты сохранены: %s", saved_path)
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

