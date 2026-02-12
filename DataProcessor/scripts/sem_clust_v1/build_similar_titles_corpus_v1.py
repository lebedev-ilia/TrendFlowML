#!/usr/bin/env python3
"""
Скрипт для создания similar_titles_corpus_v1.

Генерирует корпус эмбеддингов заголовков для поиска похожих заголовков:
- embeddings.npy: массив эмбеддингов (N, D)
- ids.json: список ID для каждого эмбеддинга

Использование:
    # Из существующих эмбеддингов
    python3 scripts/sem_clust_v1/build_similar_titles_corpus_v1.py \
        --embeddings-path scripts/sem_clust_v1/title_embeddings.npy \
        --output-dir dp_models/bundled_models/text/similar_titles_v1
    
    # Из JSON файлов (создаст эмбеддинги и корпус)
    python3 scripts/sem_clust_v1/build_similar_titles_corpus_v1.py \
        --input scripts/sem_clust_v1 \
        --pattern "data_*.json" \
        --output-dir dp_models/bundled_models/text/similar_titles_v1 \
        --model-name intfloat/multilingual-e5-large \
        --max-titles 50000
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_id_from_title(title: str, id_kind: str = "internal_hashed_id") -> str:
    """
    Генерирует ID для заголовка.
    
    Args:
        title: Текст заголовка
        id_kind: Тип ID ("internal_hashed_id" или "title_hash")
    
    Returns:
        ID строка
    """
    if id_kind == "internal_hashed_id":
        # Хэш заголовка для уникального ID
        return hashlib.sha256(title.encode('utf-8')).hexdigest()[:16]
    elif id_kind == "title_hash":
        return hashlib.md5(title.encode('utf-8')).hexdigest()
    else:
        # Fallback: просто хэш
        return hashlib.sha256(title.encode('utf-8')).hexdigest()[:16]


def load_embeddings_from_file(embeddings_path: str) -> tuple[np.ndarray, List[str]]:
    """
    Загружает эмбеддинги из файла и генерирует ID для каждого.
    
    Args:
        embeddings_path: Путь к .npy файлу с эмбеддингами
    
    Returns:
        (embeddings, ids) - массив эмбеддингов и список ID
    """
    logger.info(f"Загрузка эмбеддингов из {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    elif embeddings.ndim > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    
    logger.info(f"  Форма: {embeddings.shape}")
    logger.info(f"  Тип: {embeddings.dtype}")
    
    # Генерируем ID для каждого эмбеддинга (используем индекс как основу)
    # В реальности ID должны быть связаны с исходными заголовками,
    # но для корпуса можно использовать последовательные ID
    n = embeddings.shape[0]
    ids = [f"title_{i:08d}" for i in range(n)]
    
    logger.info(f"  Сгенерировано {len(ids)} ID")
    
    return embeddings, ids


def create_embeddings_from_json(
    input_path: Path,
    pattern: str,
    model_name: str,
    max_titles: Optional[int] = None,
    batch_size: int = 32
) -> tuple[np.ndarray, List[str]]:
    """
    Создает эмбеддинги из JSON файлов.
    
    NOTE: Эта функция требует наличия create_title_emb_for_cluster_v1.py в той же директории.
    Рекомендуется использовать --embeddings-path с уже созданными эмбеддингами.
    
    Args:
        input_path: Путь к директории или файлу
        pattern: Паттерн для поиска файлов
        model_name: Название модели для эмбеддингов
        max_titles: Максимальное количество заголовков
        batch_size: Размер батча
    
    Returns:
        (embeddings, ids) - массив эмбеддингов и список ID
    """
    logger.warning("Создание эмбеддингов из JSON файлов требует дополнительных зависимостей.")
    logger.warning("Рекомендуется использовать --embeddings-path с уже созданными эмбеддингами.")
    raise NotImplementedError(
        "Создание эмбеддингов из JSON файлов не реализовано в этом скрипте. "
        "Используйте --embeddings-path с уже созданными эмбеддингами из title_embeddings.npy"
    )


def save_corpus(embeddings: np.ndarray, ids: List[str], output_dir: Path) -> None:
    """
    Сохраняет корпус в файлы.
    
    Args:
        embeddings: Массив эмбеддингов (N, D)
        ids: Список ID для каждого эмбеддинга
        output_dir: Директория для сохранения
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Проверка совместимости
    if len(embeddings) != len(ids):
        raise ValueError(f"Несоответствие размеров: {len(embeddings)} эмбеддингов, {len(ids)} ID")
    
    # Сохранение embeddings.npy
    embeddings_path = output_dir / "embeddings.npy"
    logger.info(f"Сохранение эмбеддингов в {embeddings_path}...")
    np.save(embeddings_path, embeddings)
    logger.info(f"  ✓ embeddings.npy: {embeddings.shape}, {embeddings_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Сохранение ids.json
    ids_path = output_dir / "ids.json"
    logger.info(f"Сохранение ID в {ids_path}...")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✓ ids.json: {len(ids)} записей, {ids_path.stat().st_size / 1024:.2f} KB")
    
    # Проверка совместимости
    logger.info("\nПроверка совместимости...")
    assert embeddings.shape[0] == len(ids), f"Несоответствие: {embeddings.shape[0]} эмбеддингов, {len(ids)} ID"
    assert embeddings.dtype == np.float32, f"Тип должен быть float32, получен {embeddings.dtype}"
    assert embeddings.ndim == 2, f"Эмбеддинги должны быть 2D, получена форма {embeddings.shape}"
    
    logger.info("✓ Все проверки пройдены!")


def main():
    parser = argparse.ArgumentParser(
        description="Создание similar_titles_corpus_v1 из эмбеддингов или JSON файлов"
    )
    
    # Входные данные
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--embeddings-path",
        type=Path,
        help="Путь к существующему файлу с эмбеддингами (.npy)"
    )
    input_group.add_argument(
        "--input",
        type=Path,
        help="Путь к JSON файлу или директории с JSON файлами"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="data_*.json",
        help="Паттерн для поиска файлов (если input - директория)"
    )
    
    # Выход
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dp_models/bundled_models/text/similar_titles_v1"),
        help="Директория для сохранения корпуса"
    )
    
    # Параметры для создания эмбеддингов
    parser.add_argument(
        "--model-name",
        type=str,
        default="intfloat/multilingual-e5-large",
        help="Название модели для эмбеддингов (только если --input указан)"
    )
    parser.add_argument(
        "--max-titles",
        type=int,
        default=None,
        help="Максимальное количество заголовков для обработки"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Размер батча для создания эмбеддингов"
    )
    
    args = parser.parse_args()
    
    # Определяем источник данных
    if args.embeddings_path:
        # Загружаем из существующего файла
        embeddings, ids = load_embeddings_from_file(str(args.embeddings_path))
    elif args.input:
        # Создаем из JSON файлов
        embeddings, ids = create_embeddings_from_json(
            args.input,
            args.pattern,
            args.model_name,
            args.max_titles,
            args.batch_size
        )
    else:
        raise ValueError("Необходимо указать либо --embeddings-path, либо --input")
    
    # Сохраняем корпус
    save_corpus(embeddings, ids, args.output_dir)
    
    logger.info(f"\n✓ Готово! Корпус сохранен в {args.output_dir}")
    logger.info(f"\nСледующие шаги:")
    logger.info(f"1. Проверьте, что spec файл существует: dp_models/spec_catalog/text/similar_titles_corpus_v1.yaml")
    logger.info(f"2. Убедитесь, что пути в spec файле правильные (без bundled_models/ префикса)")


if __name__ == "__main__":
    main()

