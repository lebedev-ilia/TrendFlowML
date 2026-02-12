#!/usr/bin/env python3
"""
Скрипт для загрузки и сохранения моделей Whisper в DP_MODELS_ROOT.

Использование:
    python scripts/download_whisper_models.py --models-root dp_models/bundled_models --sizes small medium large

Модели будут сохранены в:
    dp_models/bundled_models/audio/whisper/small.pt
    dp_models/bundled_models/audio/whisper/medium.pt
    dp_models/bundled_models/audio/whisper/large.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List


def _repo_root() -> str:
    """Определяет корень репозитория."""
    # scripts/download_whisper_models.py lives at <repo>/scripts/...
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _default_models_root(repo_root: str) -> str:
    """Возвращает путь к bundled_models по умолчанию."""
    return os.path.join(repo_root, "dp_models", "bundled_models")


def download_whisper_model(model_size: str, output_path: str) -> None:
    """
    Загружает модель Whisper указанного размера и сохраняет её state_dict.
    
    Args:
        model_size: Размер модели (small, medium, large)
        output_path: Путь для сохранения .pt файла
    """
    try:
        import whisper  # type: ignore
        import torch  # type: ignore
    except ImportError as e:
        raise RuntimeError(f"openai-whisper и torch должны быть установлены: {e}") from e
    
    size = str(model_size).strip().lower()
    if size not in ("tiny", "base", "small", "medium", "large"):
        raise ValueError(f"Неподдерживаемый размер модели: {size}. Ожидается: tiny|base|small|medium|large")
    
    print(f"[download] Загрузка модели Whisper {size}...")
    print(f"[download] ⚠️  Это может занять время и потребовать интернет-соединение...")
    
    # Загружаем модель через whisper API
    # Это скачает веса, если их нет локально
    try:
        model = whisper.load_model(size, device="cpu")
    except Exception as e:
        raise RuntimeError(f"Не удалось загрузить модель Whisper {size}: {e}") from e
    
    print(f"[download] Модель загружена, сохранение state_dict...")
    
    # Сохраняем state_dict модели
    # Whisper модели используют стандартный PyTorch state_dict формат
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем state_dict в формате, который ожидает TorchStateDictProvider
    # Обертываем в словарь с ключом "state_dict" для совместимости
    state_dict = model.state_dict()
    checkpoint = {
        "state_dict": state_dict,
        "model_size": size,
        "model_type": "whisper",
    }
    
    # Сохраняем во временный файл, затем перемещаем атомарно
    tmp_path = output_path + ".tmp"
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, output_path)
        print(f"[download] ✓ Модель сохранена: {output_path}")
        
        # Вычисляем размер файла для информации
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[download]   Размер файла: {file_size_mb:.2f} MB")
    except Exception as e:
        # Удаляем временный файл при ошибке
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise RuntimeError(f"Не удалось сохранить модель: {e}") from e
    
    # Очистка памяти
    del model
    del state_dict
    del checkpoint
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Загрузка и сохранение моделей Whisper в DP_MODELS_ROOT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Загрузить все модели (small, medium, large)
  python scripts/download_whisper_models.py --sizes small medium large
  
  # Загрузить только small модель
  python scripts/download_whisper_models.py --sizes small
  
  # Указать кастомный путь для моделей
  python scripts/download_whisper_models.py --models-root /path/to/models --sizes small medium large
        """,
    )
    
    parser.add_argument(
        "--models-root",
        type=str,
        default=None,
        help="Путь к DP_MODELS_ROOT (по умолчанию: dp_models/bundled_models)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=["tiny", "base", "small", "medium", "large"],
        default=["small", "medium", "large"],
        help="Размеры моделей для загрузки (по умолчанию: small medium large)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Пропустить модели, которые уже существуют",
    )
    
    args = parser.parse_args()
    
    repo_root = _repo_root()
    models_root = os.path.abspath(args.models_root or _default_models_root(repo_root))
    
    print(f"[download] DP_MODELS_ROOT: {models_root}")
    print(f"[download] Размеры моделей: {', '.join(args.sizes)}")
    print()
    
    # Создаем директорию для Whisper моделей
    whisper_dir = os.path.join(models_root, "audio", "whisper")
    os.makedirs(whisper_dir, exist_ok=True)
    
    # Загружаем каждую модель
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for size in args.sizes:
        output_path = os.path.join(whisper_dir, f"{size}.pt")
        
        # Проверяем, существует ли уже модель
        if args.skip_existing and os.path.exists(output_path):
            print(f"[download] ⊘ Модель {size} уже существует, пропускаем: {output_path}")
            skip_count += 1
            continue
        
        try:
            download_whisper_model(size, output_path)
            success_count += 1
            print()
        except Exception as e:
            print(f"[download] ✗ Ошибка при загрузке модели {size}: {e}", file=sys.stderr)
            error_count += 1
            print()
    
    # Итоговая статистика
    print("=" * 60)
    print(f"[download] Итоги:")
    print(f"  Успешно загружено: {success_count}")
    if skip_count > 0:
        print(f"  Пропущено (уже существуют): {skip_count}")
    if error_count > 0:
        print(f"  Ошибок: {error_count}")
    print()
    
    if success_count > 0:
        print(f"[download] ✓ Модели готовы к использованию через ModelManager!")
        print(f"[download] Spec файлы уже настроены в:")
        print(f"  dp_models/spec_catalog/audio/whisper_*_inprocess.yaml")
        print()
        print(f"[download] Для использования установите:")
        print(f"  export DP_MODELS_ROOT={models_root}")
    
    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

