#!/usr/bin/env python3
"""
Скрипт для сохранения SentenceTransformer модели локально.

Использование:
    python scripts/save_sentence_transformer_model.py \
        --model-name "intfloat/multilingual-e5-large" \
        --output-dir "dp_models/bundled_models/text/embeddings/multilingual-e5-large"
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Сохранить SentenceTransformer модель локально")
    parser.add_argument(
        "--model-name",
        type=str,
        default="intfloat/multilingual-e5-large",
        help="Имя модели на HuggingFace (например, intfloat/multilingual-e5-large)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dp_models/bundled_models/text/intfloat_multilingual-e5-large",
        help="Директория для сохранения модели (относительно корня репозитория). По умолчанию соответствует пути из spec файла."
    )
    
    args = parser.parse_args()
    
    # Определяем корень репозитория
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / args.output_dir
    
    # Создаем директорию
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Сохранение модели '{args.model_name}' в '{output_dir}'...")
    print(f"⚠️  Внимание: модель большая (~560MB), загрузка может занять время...")
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Ошибка: sentence-transformers не установлен. Установите: pip install sentence-transformers")
        sys.exit(1)
    
    try:
        # Загружаем модель (это скачает её, если её нет локально)
        print(f"Загрузка модели '{args.model_name}'...")
        model = SentenceTransformer(args.model_name)
        
        # Сохраняем модель в указанную директорию
        print(f"Сохранение модели в '{output_dir}'...")
        model.save(str(output_dir))
        
        print(f"\n✓ Модель успешно сохранена в: {output_dir}")
        print(f"\nМодель готова к использованию через ModelManager!")
        print(f"\nSpec файл уже настроен:")
        print(f"  DataProcessor/dp_models/spec_catalog/text/intfloat_multilingual-e5-large.yaml")
        print(f"\nПуть в spec: \"text/intfloat_multilingual-e5-large\"")
        print(f"\nДля использования в конфиге укажите:")
        print(f"  model_name: \"intfloat/multilingual-e5-large\"")
        
    except Exception as e:
        print(f"Ошибка при сохранении модели: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
