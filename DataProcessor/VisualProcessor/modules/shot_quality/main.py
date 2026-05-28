"""
CLI для `shot_quality` (production, NPZ output, BaseModule).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

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

from modules.shot_quality.utils.shot_quality import ShotQualityModule
from utils.logger import get_logger

MODULE_NAME = "shot_quality"
logger = get_logger(MODULE_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Shot quality (frame + per-shot aggregates) — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--frames-dir", required=True, help="Директория с кадрами (с metadata.json)")
    parser.add_argument("--rs-path", required=True, help="Папка result_store для артефактов")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Устройство (GPU recommended)")
    parser.add_argument("--preset", type=str, default="default", choices=["fast", "default", "quality"], help="Feature preset")
    parser.add_argument("--analysis-max-dim", type=int, default=320, help="Downscale max dim for CPU-heavy metrics")
    parser.add_argument("--enable-entropy-features", action="store_true", help="Enable entropy-heavy metrics (default via preset)")
    parser.add_argument("--disable-entropy-features", action="store_true", help="Disable entropy-heavy metrics")
    parser.add_argument("--enable-rolling-shutter", action="store_true", help="Enable rolling shutter metric")
    parser.add_argument("--enable-lens-features", action="store_true", help="Enable lens proxy metrics group")
    parser.add_argument("--matmul-chunk-size", type=int, default=2048, help="Chunk size for CLIP matmul")
    parser.add_argument("--progress-every-n-frames", type=int, default=25, help="Progress update cadence (frames)")
    parser.add_argument("--ui-topk", type=int, default=3, help="Top-K shots to include in meta.ui_payload")
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG/INFO/WARN/ERROR")

    args = parser.parse_args(argv)

    try:
        import logging as _logging
        _logging.getLogger().setLevel(getattr(_logging, args.log_level.upper(), _logging.INFO))
    except Exception:
        logger.warning("Не удалось установить log-level: %s", args.log_level)

    try:
        module = ShotQualityModule(rs_path=args.rs_path, device=args.device)
        config: Dict[str, Any] = {
            "preset": args.preset,
            "analysis_max_dim": int(args.analysis_max_dim),
            "matmul_chunk_size": int(args.matmul_chunk_size),
            "progress_every_n_frames": int(args.progress_every_n_frames),
            "ui_topk": int(args.ui_topk),
        }
        if args.enable_entropy_features:
            config["enable_entropy_features"] = True
        if args.disable_entropy_features:
            config["enable_entropy_features"] = False
        if args.enable_rolling_shutter:
            config["enable_rolling_shutter"] = True
        if args.enable_lens_features:
            config["enable_lens_features"] = True
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

