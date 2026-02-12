#!/usr/bin/env python3
"""
Скрипт для загрузки и сохранения моделей Emotion Diarization в DP_MODELS_ROOT.

Использование:
    python scripts/download_emotion_diarization_models.py --models-root dp_models/bundled_models --sizes small large

Модели будут сохранены в:
    dp_models/bundled_models/audio/emotion_diarization/small.pt
    dp_models/bundled_models/audio/emotion_diarization/large.pt

Примечание: Этот скрипт требует, чтобы модели были доступны локально или через указанный источник.
Измените функцию download_emotion_model() под конкретную модель emotion recognition.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

# base example 

from speechbrain.inference.diarization import Speech_Emotion_Diarization

classifier = Speech_Emotion_Diarization.from_hparams(
    source="speechbrain/emotion-diarization-wavlm-large",
)

diary = classifier.diarize_file("/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/dp_output/video1/audio/audio.wav")

print(diary)


# def _repo_root() -> str:
#     """Определяет корень репозитория."""
#     return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


# def _default_models_root(repo_root: str) -> str:
#     """Возвращает путь к bundled_models по умолчанию."""
#     return os.path.join(repo_root, "dp_models", "bundled_models")


# def download_emotion_model(model_size: str, output_path: str) -> None:
#     """
#     Загружает модель emotion diarization указанного размера и сохраняет в output_path.
    
#     Args:
#         model_size: Размер модели ("small" или "large")
#         output_path: Путь для сохранения модели
    
#     Примечание: Эта функция должна быть адаптирована под конкретную модель emotion recognition.
#     Примеры источников:
#     - HuggingFace модель (например, j-hartmann/emotion-english-distilroberta-base)
#     - Локальный файл модели
#     - Кастомная модель из репозитория
#     """
#     print(f"[download] Загрузка emotion diarization модели ({model_size})...")
    
#     from speechbrain.inference.diarization import Speech_Emotion_Diarization

#     classifier = Speech_Emotion_Diarization.from_hparams(
#         source="speechbrain/emotion-diarization-wavlm-large"
#     )

#     # TODO: Адаптируйте эту функцию под конкретную модель emotion recognition
#     # Пример 1: Загрузка из HuggingFace
#     # try:
#     #     from transformers import AutoModel, AutoTokenizer
#     #     model_name = f"your-model-repo/emotion-{model_size}"
#     #     model = AutoModel.from_pretrained(model_name)
#     #     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     #     # Сохраните модель и токенизатор
#     #     model.save_pretrained(output_path)
#     #     tokenizer.save_pretrained(output_path)
#     # except Exception as e:
#     #     raise RuntimeError(f"Failed to download model from HuggingFace: {e}") from e
    
#     # Пример 2: Загрузка из локального файла
#     # local_model_path = f"/path/to/emotion_{model_size}.pt"
#     # if not os.path.exists(local_model_path):
#     #     raise FileNotFoundError(f"Model file not found: {local_model_path}")
#     # import shutil
#     # shutil.copy2(local_model_path, output_path)
    
#     # Пример 3: Загрузка через torch.hub или другой источник
#     # import torch
#     # model = torch.hub.load("repo/name", "model", size=model_size)
#     # torch.save(model.state_dict(), output_path)
    
#     # Временная заглушка: создаем placeholder файл
#     # УДАЛИТЕ ЭТО И РЕАЛИЗУЙТЕ РЕАЛЬНУЮ ЗАГРУЗКУ
#     print(f"[WARNING] download_emotion_model() не реализована!")
#     print(f"[WARNING] Создаю placeholder файл: {output_path}")
#     print(f"[WARNING] Пожалуйста, реализуйте загрузку модели в функции download_emotion_model()")
    
#     # Создаем директорию если не существует
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
#     # Создаем placeholder файл (это НЕ рабочая модель!)
#     with open(output_path, "wb") as f:
#         f.write(b"PLACEHOLDER_MODEL_FILE\n")
#         f.write(f"emotion_diarization_{model_size}\n".encode())
#         f.write(b"This is a placeholder. Replace with actual model.\n")
    
#     print(f"[download] Placeholder создан: {output_path}")
#     print(f"[download] ВАЖНО: Замените placeholder на реальную модель!")


# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Загрузка и сохранение моделей Emotion Diarization в DP_MODELS_ROOT",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Примеры использования:
#   # Загрузить все модели (small, large)
#   python scripts/download_emotion_diarization_models.py --sizes small large
  
#   # Загрузить только small модель
#   python scripts/download_emotion_diarization_models.py --sizes small
  
#   # Указать кастомный путь для моделей
#   python scripts/download_emotion_diarization_models.py --models-root /path/to/models --sizes small large
  
# Примечание: Функция download_emotion_model() должна быть адаптирована под конкретную модель.
#         """,
#     )
    
#     parser.add_argument(
#         "--models-root",
#         type=str,
#         default=None,
#         help="Путь к DP_MODELS_ROOT (по умолчанию: dp_models/bundled_models)",
#     )
#     parser.add_argument(
#         "--sizes",
#         nargs="+",
#         choices=["small", "large"],
#         default=["small", "large"],
#         help="Размеры моделей для загрузки (по умолчанию: small large)",
#     )
#     parser.add_argument(
#         "--skip-existing",
#         action="store_true",
#         help="Пропустить модели, которые уже существуют",
#     )
    
#     args = parser.parse_args()
    
#     repo_root = _repo_root()
#     models_root = os.path.abspath(args.models_root or _default_models_root(repo_root))
    
#     print(f"[download] DP_MODELS_ROOT: {models_root}")
#     print(f"[download] Размеры моделей: {', '.join(args.sizes)}")
#     print()
    
#     # Создаем директорию для Emotion Diarization моделей
#     emotion_dir = os.path.join(models_root, "audio", "emotion_diarization")
#     os.makedirs(emotion_dir, exist_ok=True)
    
#     # Загружаем каждую модель
#     success_count = 0
#     skip_count = 0
#     error_count = 0
    
#     for size in args.sizes:
#         output_path = os.path.join(emotion_dir, f"{size}.pt")
        
#         # Проверяем, существует ли модель
#         if args.skip_existing and os.path.exists(output_path):
#             print(f"[skip] Модель {size} уже существует: {output_path}")
#             skip_count += 1
#             continue
        
#         try:
#             download_emotion_model(size, output_path)
#             print(f"[success] Модель {size} сохранена: {output_path}")
#             success_count += 1
#         except Exception as e:
#             print(f"[error] Ошибка при загрузке модели {size}: {e}")
#             error_count += 1
    
#     print()
#     print(f"[summary] Успешно: {success_count}, Пропущено: {skip_count}, Ошибок: {error_count}")
    
#     if error_count > 0:
#         sys.exit(1)


# if __name__ == "__main__":
#     main()

