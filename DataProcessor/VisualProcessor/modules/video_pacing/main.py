"""
CLI для `video_pacing` (BaseModule, NPZ output).
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

from modules.video_pacing.utils.video_pacing import VideoPacingModule
from utils.logger import get_logger

MODULE_NAME = "video_pacing"
logger = get_logger(MODULE_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Video pacing (shot/tempo/motion/color pacing) — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--frames-dir", required=True, help="Директория с кадрами (с metadata.json)")
    parser.add_argument("--rs-path", required=True, help="Папка result_store для артефактов")
    parser.add_argument("--downscale-factor", type=float, default=0.25, help="Downscale для дешёвых визуальных метрик")
    parser.add_argument("--min-shot-length-seconds", type=float, default=0.15, help="Минимальная длительность шота (для merge) в секундах")
    parser.add_argument("--shot-detect-k", type=float, default=6.0, help="Робастный порог (MAD-multiplier) для shot boundary detection")
    parser.add_argument("--min-frames", type=int, default=30, help="Fail-fast: минимальное число sampled кадров для анализа (no-fallback)")
    # Feature gating: remove noisy blocks by default (enable explicitly if needed).
    parser.add_argument("--enable-entropy-features", action="store_true", help="Включить энтропии/Gini (может быть шумно при малом числе шотов)")
    parser.add_argument("--enable-histograms", action="store_true", help="Включить histogram-based pacing features (может быть шумно)")
    parser.add_argument("--enable-pace-curve-peaks", action="store_true", help="Включить peak features по длительностям шотов (может быть шумно)")
    parser.add_argument("--enable-periodicity", action="store_true", help="Включить autocorr periodicity features (может быть шумно)")
    parser.add_argument("--enable-bursts", action="store_true", help="Включить burst features (quick cuts / semantic / color bursts)")
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG/INFO/WARN/ERROR")

    args = parser.parse_args(argv)

    try:
        import logging as _logging
        _logging.getLogger().setLevel(getattr(_logging, args.log_level.upper(), _logging.INFO))
    except Exception:
        logger.warning("Не удалось установить log-level: %s", args.log_level)

    try:
        module = VideoPacingModule(rs_path=args.rs_path, downscale_factor=args.downscale_factor)
        config: Dict[str, Any] = {
            "downscale_factor": args.downscale_factor,
            "min_shot_length_seconds": args.min_shot_length_seconds,
            "shot_detect_k": args.shot_detect_k,
            "min_frames": int(args.min_frames),
            "enable_entropy_features": bool(args.enable_entropy_features),
            "enable_histograms": bool(args.enable_histograms),
            "enable_pace_curve_peaks": bool(args.enable_pace_curve_peaks),
            "enable_periodicity": bool(args.enable_periodicity),
            "enable_bursts": bool(args.enable_bursts),
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
 