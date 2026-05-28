"""
CLI для `uniqueness` (BaseModule, NPZ output).
Baseline-версия: intra-video uniqueness на основе core_clip (без внешних reference-видео).
"""

import argparse
import os
import sys
from typing import Optional, List, Dict, Any

_vp_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _vp_root not in sys.path:
    sys.path.insert(0, _vp_root)
elif sys.path[0] != _vp_root:
    try:
        sys.path.remove(_vp_root)
    except ValueError:
        pass
    sys.path.insert(0, _vp_root)
_repo_root = os.path.dirname(_vp_root)
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

# Import after sys.path is set
from modules.uniqueness.utils.uniqueness import UniquenessBaselineModule  # noqa: E402
from utils.logger import get_logger  # noqa: E402

MODULE_NAME = "uniqueness"
logger = get_logger(MODULE_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Uniqueness (intra-video repetition/diversity proxies) — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--frames-dir", required=True, help="Директория с кадрами (с metadata.json)")
    parser.add_argument("--rs-path", required=True, help="Папка result_store для артефактов")
    parser.add_argument("--repeat-threshold", type=float, default=0.97, help="Порог max-sim для repetition_ratio")
    parser.add_argument("--repeat-threshold-mode", type=str, default="auto", help="auto|fixed (auto=Otsu by distribution)")
    parser.add_argument("--repeat-threshold-min", type=float, default=0.90, help="Clamp min for auto threshold")
    parser.add_argument("--repeat-threshold-max", type=float, default=0.99, help="Clamp max for auto threshold")
    parser.add_argument("--ui-topk", type=int, default=8, help="Top-K repeats to include in meta.ui_payload")
    parser.add_argument("--max-frames", type=int, default=200, help="Fail-fast лимит на число sampled кадров (N) для NxN similarity")
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG/INFO/WARN/ERROR")

    args = parser.parse_args(argv)

    try:
        import logging as _logging
        _logging.getLogger().setLevel(getattr(_logging, args.log_level.upper(), _logging.INFO))
    except Exception:
        logger.warning("Не удалось установить log-level: %s", args.log_level)

    try:
        module = UniquenessBaselineModule(
            rs_path=args.rs_path,
            repeat_threshold=float(args.repeat_threshold),
            max_frames=int(args.max_frames),
        )
        config: Dict[str, Any] = {
            "repeat_threshold": float(args.repeat_threshold),
            "repeat_threshold_mode": str(args.repeat_threshold_mode),
            "repeat_threshold_min": float(args.repeat_threshold_min),
            "repeat_threshold_max": float(args.repeat_threshold_max),
            "ui_topk": int(args.ui_topk),
            "max_frames": int(args.max_frames),
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

