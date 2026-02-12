#!/usr/bin/env python3
"""
CLI for frames_composition (baseline-ready).

This CLI is executed by VisualProcessor orchestrator as a subprocess.
It MUST write NPZ artifacts only (no JSON in result_store except manifest.json).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.frames_composition.balance_composition import FramesCompositionModule
from utils.logger import get_logger

MODULE_NAME = "frames_composition"
logger = get_logger(MODULE_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Frames composition analysis — CLI (NPZ source-of-truth)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--frames-dir", required=True, help="frames_dir (Segmenter output, contains metadata.json)")
    parser.add_argument("--rs-path", required=True, help="Per-run result_store path (<rs_base>/<platform>/<video>/<run>)")

    # Feature gating (scheduler/UI controlled)
    parser.add_argument(
        "--feature-set",
        type=str,
        default="default",
        help="Feature set preset: default|ml|all (default: default).",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature groups to enable (overrides --feature-set). "
        "Groups: anchors,balance,symmetry,negative_space,complexity,leading_lines,depth,objects,faces,style",
    )

    # Internal parallelism
    parser.add_argument("--num-workers", type=int, default=None, help="Per-frame compute workers (CPU).")

    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG/INFO/WARN/ERROR")

    args = parser.parse_args(argv)

    try:
        import logging as _logging

        _logging.getLogger().setLevel(getattr(_logging, str(args.log_level).upper(), _logging.INFO))
    except Exception:
        logger.warning("Could not set log-level: %s", args.log_level)

    try:
        module = FramesCompositionModule(rs_path=args.rs_path)
        cfg: Dict[str, Any] = {
            "feature_set": str(args.feature_set),
            "features": args.features,
            "num_workers": int(args.num_workers) if args.num_workers is not None else None,
        }
        saved_path = module.run(frames_dir=args.frames_dir, config=cfg)
        logger.info("Done. Saved: %s", saved_path)
        return 0
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 2
    except ValueError as e:
        logger.error("Invalid input: %s", e)
        return 3
    except Exception as e:
        logger.exception("Fatal error in %s: %s", MODULE_NAME, e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
