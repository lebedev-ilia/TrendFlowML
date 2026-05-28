#!/usr/bin/env python3
"""
CLI интерфейс для модуля распознавания действий в видео (SlowFast).

Использует BaseModule для автоматизации работы с метаданными и FrameManager.
"""


from __future__ import annotations

import os
import sys
import argparse
import faulthandler
import signal
import traceback
from pathlib import Path
from typing import Optional, List
import importlib.util

# Включаем faulthandler для диагностики SIGSEGV и других критических ошибок
# faulthandler.enable() автоматически обрабатывает SIGSEGV, SIGFPE, SIGABRT и другие сигналы
faulthandler.enable(all_threads=True, file=sys.stderr)

_vp_root = str(Path(__file__).resolve().parents[2])  # VisualProcessor
_dp_root = str(Path(__file__).resolve().parents[3])  # DataProcessor (dp_models)

if _vp_root not in sys.path:
    sys.path.insert(0, _vp_root)
elif sys.path[0] != _vp_root:
    try:
        sys.path.remove(_vp_root)
    except ValueError:
        pass
    sys.path.insert(0, _vp_root)

if _dp_root not in sys.path:
    sys.path.insert(1, _dp_root)


def _sanitize_pytorch_cuda_alloc_conf() -> None:
    """
    Родительский E2E/worker часто выставляет PYTORCH_CUDA_ALLOC_CONF=...,expandable_segments:True
    (PyTorch 2.4+). Старый torch в .action_recognition_venv падает при инициализации CUDA:
    RuntimeError: Unrecognized CachingAllocator option: expandable_segments
    Убираем эту опцию до import torch.
    """
    key = "PYTORCH_CUDA_ALLOC_CONF"
    raw = os.environ.get(key)
    if not raw or "expandable_segments" not in raw:
        return
    parts = [p.strip() for p in raw.split(",") if p.strip() and "expandable_segments" not in p]
    if parts:
        os.environ[key] = ",".join(parts)
    else:
        del os.environ[key]


_sanitize_pytorch_cuda_alloc_conf()

# Do not "from utils.action_recognition_slowfast": local utils/ shadows VisualProcessor/utils.
_impl = Path(__file__).resolve().parent / "utils" / "action_recognition_slowfast.py"
_spec = importlib.util.spec_from_file_location("action_recognition._slowfast_impl", str(_impl))
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load SlowFast impl: {_impl}")
_ar_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ar_mod)
SlowFastActionRecognizer = _ar_mod.SlowFastActionRecognizer

from utils.logger import get_logger

MODULE_NAME = "action_recognition"
logger = get_logger(MODULE_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Распознавание действий (SlowFast) — CLI",
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
        "--clip-len",
        type=int,
        default=32,
        help="Длина клипа в кадрах (минимум 32 для SlowFast, T_slow=8)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Шаг скользящего окна (по умолчанию clip_len // 2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size для inference"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Размерность эмбеддингов после проекции"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=SlowFastActionRecognizer.DEFAULT_MODEL_NAME,
        help="ModelManager spec name для SlowFast"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Устройство (cuda/cpu). Если не задано — policy из ModelManager"
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=4,
        help="SlowFast alpha (T_fast / T_slow, по умолчанию 4, как в pytorchvideo)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Уровень логирования (DEBUG/INFO/WARN/ERROR)"
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
        last_pct = {"value": -1.0}

        def _progress_cb(payload: dict) -> None:
            try:
                pct = float(payload.get("progress_pct", 0.0))
            except Exception:
                return
            if pct - last_pct["value"] >= 10.0 or pct >= 100.0:
                last_pct["value"] = pct
                logger.info(
                    "Progress: %.1f%% (%d/%d clips)",
                    pct,
                    int(payload.get("processed_clips", 0)),
                    int(payload.get("total_clips", 0)),
                )

        # Инициализация модуля
        recognizer = SlowFastActionRecognizer(
            rs_path=args.rs_path,
            clip_len=args.clip_len,
            stride=args.stride,
            batch_size=args.batch_size,
            embedding_dim=args.embedding_dim,
            model_name=args.model_name,
            device=args.device,
            alpha=args.alpha,
            progress_callback=_progress_cb,
        )

        config = {
            "clip_len": args.clip_len,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "embedding_dim": args.embedding_dim,
            "model_name": args.model_name,
            "alpha": args.alpha,
        }

        saved_path = recognizer.run(
            frames_dir=args.frames_dir,
            config=config,
            metadata=None,
        )

        logger.info("Обработка завершена. Результаты сохранены: %s", saved_path)
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
