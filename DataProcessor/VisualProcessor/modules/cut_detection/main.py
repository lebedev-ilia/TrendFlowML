#!/usr/bin/env python3
"""
CLI интерфейс для модуля детекции переходов и анализа стиля монтажа.

Использует BaseModule для автоматизации работы с метаданными и FrameManager.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

from modules.cut_detection.cut_detection import CutDetectionPipeline
from utils.logger import get_logger

MODULE_NAME = "cut_detection"
logger = get_logger(MODULE_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Детекция переходов и анализ стиля монтажа — CLI",
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
        "--device",
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help="Устройство для обработки (auto/cpu/cuda)"
    )
    parser.add_argument(
        "--use-clip",
        action='store_true',
        help="Использовать CLIP для классификации переходов"
    )
    parser.add_argument(
        "--use-deep-features",
        action='store_true',
        help="Использовать глубокие признаки для детекции"
    )
    parser.add_argument(
        "--ssim-max-side",
        type=int,
        default=512,
        help="Max side (px) for internal SSIM computation. 0 = no downscale.",
    )
    parser.add_argument(
        "--flow-max-side",
        type=int,
        default=320,
        help="Max side (px) for internal Farneback optical flow computation. 0 = no downscale.",
    )
    parser.add_argument(
        "--hard-cuts-preset",
        type=str,
        default=None,
        choices=["quality", "default", "fast"],
        help=(
            "Hard cuts preset. If provided, sets recommended ssim/flow/cascade defaults "
            "(explicit CLI args like --ssim-max-side/--flow-max-side/--hard-cuts-cascade override)."
        ),
    )
    parser.add_argument(
        "--hard-cuts-cascade",
        action="store_true",
        help="Optional speed mode: compute SSIM/flow/deep only for histogram-gated candidate pairs (default: off).",
    )
    parser.add_argument(
        "--hard-cuts-cascade-keep-top-p",
        type=float,
        default=0.25,
        help="In cascade mode, always keep at least top-p histogram pairs for expensive compute (0..1).",
    )
    parser.add_argument(
        "--hard-cuts-cascade-hist-margin",
        type=float,
        default=0.0,
        help="In cascade mode, also keep pairs with hist_diff >= (hist_thresh - margin).",
    )
    parser.add_argument(
        "--prefer-core-optical-flow",
        action="store_true",
        help="(Deprecated) Prefer reusing core_optical_flow/flow.npz when aligned. Baseline now requires it by default.",
    )
    parser.add_argument(
        "--require-core-optical-flow",
        action="store_true",
        help="(Deprecated) Require core_optical_flow aligned artifact. Baseline now requires it by default.",
    )
    parser.add_argument(
        "--no-require-core-optical-flow",
        action="store_true",
        help="Disable baseline requirement for core_optical_flow reuse (debug only; not for production/baseline).",
    )
    parser.add_argument(
        "--write-model-facing-npz",
        action="store_true",
        help="Write additional model-facing NPZ (dense curves + unified events) for FeatureEncoder input. (Enabled by default; this flag is kept for compatibility.)",
    )
    parser.add_argument(
        "--require-model-facing-npz",
        action="store_true",
        help="Require writing the model-facing NPZ; if write fails -> error (fail-fast).",
    )
    parser.add_argument(
        "--no-write-model-facing-npz",
        action="store_true",
        help="Disable writing the model-facing NPZ (default is to write it).",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Путь к аудио файлу для аудио-анализа (опционально)"
    )
    # Baseline GPU-only: cut_detection MUST NOT load local CLIP weights.
    # CLIP is resolved via dp_models spec and served via Triton.
    parser.add_argument(
        "--clip-image-model-spec",
        type=str,
        default=None,
        help="dp_models Triton spec for CLIP image encoder (e.g., clip_image_triton). Required when --use-clip.",
    )
    parser.add_argument(
        "--triton-http-url",
        type=str,
        default=None,
        help="Triton HTTP URL (can also be set via TRITON_HTTP_URL env var). Used when clip-image-model-spec doesn't have triton_http_url in runtime_params.",
    )
    parser.add_argument(
        "--use-adaptive-thresholds",
        action="store_true",
        default=False,
        help="Use adaptive thresholds for cut detection (default: True if not specified).",
    )
    parser.add_argument(
        "--no-use-adaptive-thresholds",
        action="store_true",
        help="Disable adaptive thresholds for cut detection.",
    )
    parser.add_argument(
        "--use-semantic-clustering",
        action="store_true",
        default=False,
        help="Use semantic clustering for scene grouping (default: False).",
    )
    parser.add_argument(
        "--fade-threshold",
        type=float,
        default=0.02,
        help="Threshold for fade detection (default: 0.02).",
    )
    parser.add_argument(
        "--min-duration-frames",
        type=int,
        default=4,
        help="Minimum duration in frames for soft transitions (default: 4).",
    )
    parser.add_argument(
        "--use-flow-consistency",
        action="store_true",
        default=False,
        help="Use flow consistency for soft cuts (default: True if not specified).",
    )
    parser.add_argument(
        "--no-use-flow-consistency",
        action="store_true",
        help="Disable flow consistency for soft cuts.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Уровень логирования (DEBUG/INFO/WARN/ERROR)"
    )

    args = parser.parse_args(argv)

    # Apply hard-cuts preset defaults (but do not override explicitly provided CLI args).
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    raw_argv_s = " ".join(str(x) for x in raw_argv)
    if args.hard_cuts_preset:
        preset = str(args.hard_cuts_preset).strip().lower()
        preset_map = {
            "quality": {"ssim": 640, "flow": 384, "cascade": False, "keep_top_p": 0.25, "hist_margin": 0.0},
            "default": {"ssim": 512, "flow": 320, "cascade": False, "keep_top_p": 0.25, "hist_margin": 0.0},
            "fast": {"ssim": 384, "flow": 256, "cascade": True, "keep_top_p": 0.25, "hist_margin": 0.0},
        }
        pm = preset_map.get(preset, preset_map["default"])
        if "--ssim-max-side" not in raw_argv_s:
            args.ssim_max_side = int(pm["ssim"])
        if "--flow-max-side" not in raw_argv_s:
            args.flow_max_side = int(pm["flow"])
        if "--hard-cuts-cascade" not in raw_argv_s:
            args.hard_cuts_cascade = bool(pm["cascade"])
        if "--hard-cuts-cascade-keep-top-p" not in raw_argv_s:
            args.hard_cuts_cascade_keep_top_p = float(pm["keep_top_p"])
        if "--hard-cuts-cascade-hist-margin" not in raw_argv_s:
            args.hard_cuts_cascade_hist_margin = float(pm["hist_margin"])

    # Настройка уровня логирования
    try:
        import logging as _logging
        log_level = getattr(_logging, args.log_level.upper(), _logging.INFO)
        # Устанавливаем уровень для root logger
        _logging.getLogger().setLevel(log_level)
        # Устанавливаем уровень для конкретного логгера cut_detection
        logger.setLevel(log_level)
        # Убеждаемся, что логи выводятся в консоль
        for handler in logger.handlers:
            handler.setLevel(log_level)
        logger.info(f"cut_detection | Log level set to: {args.log_level.upper()}")
    except Exception as e:
        logger.warning(f"Не удалось установить log-level: {args.log_level}, error: {e}")

    try:
        # Determine use_adaptive_thresholds: default True if not explicitly set, but can be disabled with --no-use-adaptive-thresholds
        # If --use-adaptive-thresholds is passed, use it; otherwise default to True unless --no-use-adaptive-thresholds is set
        use_adaptive_thresholds = True  # default
        if args.no_use_adaptive_thresholds:
            use_adaptive_thresholds = False
        elif args.use_adaptive_thresholds:
            use_adaptive_thresholds = True
        
        # Determine use_flow_consistency: default True if not explicitly set, but can be disabled with --no-use-flow-consistency
        use_flow_consistency = True  # default
        if args.no_use_flow_consistency:
            use_flow_consistency = False
        elif args.use_flow_consistency:
            use_flow_consistency = True
        
        pipeline = CutDetectionPipeline(
            rs_path=args.rs_path,
            fps=30.0,  # actual fps is taken from frame_manager.meta in process()
            device=args.device,
            clip_zero_shot=args.use_clip,
            use_deep_features=args.use_deep_features,
            use_adaptive_thresholds=use_adaptive_thresholds,
            use_semantic_clustering=bool(args.use_semantic_clustering),
            fade_threshold=float(args.fade_threshold),
            min_duration_frames=int(args.min_duration_frames),
            use_flow_consistency=use_flow_consistency,
            ssim_max_side=int(args.ssim_max_side),
            flow_max_side=int(args.flow_max_side),
            hard_cuts_cascade=bool(args.hard_cuts_cascade),
            hard_cuts_cascade_keep_top_p=float(args.hard_cuts_cascade_keep_top_p),
            hard_cuts_cascade_hist_margin=float(args.hard_cuts_cascade_hist_margin),
            # Baseline: always attempt reuse, and require core_optical_flow unless explicitly disabled.
            prefer_core_optical_flow=True,
            require_core_optical_flow=not bool(args.no_require_core_optical_flow),
            write_model_facing_npz=(not bool(args.no_write_model_facing_npz)) or bool(args.require_model_facing_npz),
            require_model_facing_npz=bool(args.require_model_facing_npz),
            clip_image_model_spec=args.clip_image_model_spec,
            triton_http_url=args.triton_http_url,
        )
        # Backward-compat: if someone still tries to pass clip weights root - fail fast.
        if getattr(args, "clip_download_root", None):
            raise RuntimeError("cut_detection | clip_download_root is deprecated. Use Triton + dp_models spec instead.")

        config = {}
        if args.audio_path:
            config["audio_path"] = args.audio_path
        # CLIP runtime params are resolved via dp_models spec; nothing to inject into config.

        saved_path = pipeline.run(frames_dir=args.frames_dir, config=config)
        logger.info("Обработка завершена. Результаты сохранены: %s", saved_path)
        return 0

    except FileNotFoundError as e:
        logger.exception("Файл не найден: %s", e)
        return 2
    except ValueError as e:
        logger.exception("Некорректные данные: %s", e)
        return 3
    except Exception as e:
        logger.exception("Fatal error в %s: %s", MODULE_NAME, e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())