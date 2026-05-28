"""
CLI для `story_structure` (BaseModule, NPZ output).

Baseline-версия: story/energy/coherence по `core_clip` embeddings (без локальных ML моделей).
"""

from __future__ import annotations

import sys
from pathlib import Path
_vp = Path(__file__).resolve().parent  # story_structure/
for _ in range(2):
    _vp = _vp.parent  # -> modules -> VisualProcessor
sys.path.insert(0, str(_vp))

import argparse
import os
from typing import Optional, List, Dict, Any

from modules.story_structure.utils.story_structure import StoryStructureBaselineModule
from utils.logger import get_logger

MODULE_NAME = "story_structure"
logger = get_logger(MODULE_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Story structure (baseline) — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--frames-dir", required=True, help="Директория с кадрами (с metadata.json)")
    parser.add_argument("--rs-path", required=True, help="Папка result_store для артефактов")

    # Эти аргументы остаются для совместимости с config.yaml, но baseline их не использует напрямую.
    parser.add_argument("--clip-model", type=str, default=None, help="legacy/compat (unused in baseline)")
    parser.add_argument("--sentence-model", type=str, default=None, help="legacy/compat (unused in baseline)")
    parser.add_argument("--subtitles", type=str, default=None, help="Comma-separated subtitles/ASR chunks (optional)")
    parser.add_argument("--min-frames", type=int, default=30, help="Fail-fast минимум sampled кадров (N)")
    parser.add_argument("--max-frames", type=int, default=200, help="Fail-fast лимит на число sampled кадров (N)")
    parser.add_argument("--energy-smoothing-sigma", type=float, default=1.0, help="Sigma для сглаживания energy/motion curves")
    parser.add_argument("--text-mode", type=str, default="ocr_clip_text", help="Text mode: none|ocr_clip_text")
    parser.add_argument("--clip-text-model-spec", type=str, default="clip_text_triton", help="dp_models spec for CLIP text encoder (triton)")
    parser.add_argument("--clip-text-batch-size", type=int, default=64, help="Batch size for CLIP text encoder inference")
    parser.add_argument("--ocr-max-chars-per-frame", type=int, default=256, help="Max OCR text chars per frame (tokenization guard)")
    parser.add_argument("--triton-http-url", type=str, default=None, help="Triton HTTP URL (or use TRITON_HTTP_URL env var)")

    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG/INFO/WARN/ERROR")

    args = parser.parse_args(argv)

    try:
        import logging as _logging
        _logging.getLogger().setLevel(getattr(_logging, args.log_level.upper(), _logging.INFO))
    except Exception:
        logger.warning("Не удалось установить log-level: %s", args.log_level)

    subtitles: Optional[List[str]] = None
    if args.subtitles:
        subtitles = [s.strip() for s in str(args.subtitles).split(",") if s.strip()]

    try:
        module = StoryStructureBaselineModule(rs_path=args.rs_path, max_frames=int(args.max_frames))
        config: Dict[str, Any] = {
            # legacy / compat
            "subtitles": subtitles,
            "clip_model": args.clip_model,
            "sentence_model": args.sentence_model,
            # baseline
            "min_frames": int(args.min_frames),
            "max_frames": int(args.max_frames),
            "energy_smoothing_sigma": float(args.energy_smoothing_sigma),
            "text_mode": str(args.text_mode),
            "clip_text_model_spec": str(args.clip_text_model_spec),
            "clip_text_batch_size": int(args.clip_text_batch_size),
            "ocr_max_chars_per_frame": int(args.ocr_max_chars_per_frame),
            "triton_http_url": args.triton_http_url,
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
