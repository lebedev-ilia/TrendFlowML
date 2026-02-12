#!/usr/bin/env python3
"""
Скрипт для создания эмбеддингов заголовков из одного или нескольких JSON файлов.

Использует потоковую обработку для работы с большими файлами (сотни MB).

Рекомендации по размеру выборки для кластеризации:
- Минимум: 1000 примеров для стабильных кластеров
- Оптимально: 10000-20000 примеров для 32 кластеров
- Максимум: 50000 примеров (больше обычно не улучшает качество)
- 100k+ примеров - избыточно, используйте выборку (--max-titles или --sample-ratio)
"""

import json
import argparse
import random
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def stream_json_items(file_path: Path) -> Iterator[Dict[str, Any]]:
    """
    Потоковое чтение JSON файла.
    
    Поддерживает три формата:
    1. JSON объект с вложенными объектами: {"key1": {...}, "key2": {...}}
    2. JSON массив: [{"title": "..."}, ...]
    3. JSONL: каждая строка - отдельный JSON объект
    """
    logger.info(f"Открытие файла: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Пробуем определить формат по первому символу
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # JSON массив - используем ijson для потоковой обработки
            try:
                import ijson
                logger.info("Используется потоковый парсер ijson для JSON массива")
                parser = ijson.items(f, 'item')
                for item in parser:
                    yield item
            except ImportError:
                logger.warning("ijson не установлен, загружаем весь файл в память")
                logger.warning("Для больших файлов рекомендуется: pip install ijson")
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        yield item
                else:
                    yield data
        elif first_char == '{':
            # JSON объект - может быть словарь с вложенными объектами
            # Например: {"video_id_1": {...}, "video_id_2": {...}}
            # Для такой структуры ijson.items(f, '*') может не работать правильно,
            # поэтому используем fallback с загрузкой всего объекта
            # (файл 374MB - это приемлемо для современных систем)
            logger.info("JSON объект обнаружен, используем fallback метод с загрузкой в память")
            logger.info("Для файла такого размера это приемлемо")
            
            # Для JSON объекта верхнего уровня ijson.items(f, '*') не работает правильно
            # Используем fallback метод сразу (файл 374MB - это приемлемо для современных систем)
            logger.info("Загружаем JSON объект в память...")
            data = json.load(f)
            if isinstance(data, dict):
                # Если это словарь, итерируемся по значениям
                total = len(data)
                logger.info(f"Найдено {total} элементов для обработки")
                for idx, (key, value) in enumerate(data.items(), 1):
                    if isinstance(value, dict):
                        yield value
                    else:
                        yield {"data": value}
                    if idx % 1000 == 0:
                        logger.info(f"Обработано {idx}/{total} элементов ({idx*100//total}%)")
                logger.info(f"Завершена обработка всех {total} элементов")
            else:
                yield data
        else:
            # JSONL формат - читаем построчно
            logger.info("Используется JSONL формат (построчное чтение)")
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Ошибка парсинга строки {line_num}: {e}")
                    continue


def extract_titles(data: Dict[str, Any], title_key: str = "title") -> Iterator[str]:
    """
    Извлекает заголовки из объекта данных.
    
    Поддерживает различные структуры:
    - {"title": "..."}
    - {"metadata": {"title": "..."}}
    - {"data": {"title": "..."}}
    - {"items": [{"title": "..."}, ...]}
    """
    # Прямой доступ к ключу
    if title_key in data:
        title = data[title_key]
        if isinstance(title, str) and title.strip():
            yield title.strip()
            return  # Нашли, не ищем дальше
    
    # Поиск в metadata
    if "metadata" in data and isinstance(data["metadata"], dict):
        if title_key in data["metadata"]:
            title = data["metadata"][title_key]
            if isinstance(title, str) and title.strip():
                yield title.strip()
                return
    
    # Рекурсивный поиск вложенных структур (только если не нашли напрямую)
    for key, value in data.items():
        if key in ("metadata", title_key):  # Пропускаем уже проверенные ключи
            continue
        if isinstance(value, dict):
            yield from extract_titles(value, title_key)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield from extract_titles(item, title_key)


def create_embeddings_batch(
    titles: list[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    normalize: bool = True
) -> np.ndarray:
    """Создает эмбеддинги для батча заголовков."""
    if not titles:
        return np.array([])
    
    embeddings = model.encode(
        titles,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=True
    )
    return embeddings.astype(np.float32)


def process_multiple_files(
    input_files: List[Path],
    title_key: str,
    max_items: Optional[int] = None
) -> Iterator[str]:
    """
    Обрабатывает несколько JSON файлов и извлекает заголовки.
    
    Args:
        input_files: Список путей к JSON файлам
        title_key: Ключ для поиска заголовков
        max_items: Максимальное количество элементов для обработки (None = все)
    
    Yields:
        Заголовки из всех файлов
    """
    total_processed = 0
    total_titles = 0
    
    for file_idx, file_path in enumerate(input_files, 1):
        logger.info(f"\n[{file_idx}/{len(input_files)}] Обработка файла: {file_path.name}")
        
        if not file_path.exists():
            logger.warning(f"Файл не найден: {file_path}, пропускаем")
            continue
        
        file_titles = 0
        for item in stream_json_items(file_path):
            total_processed += 1
            if max_items and total_processed > max_items:
                logger.info(f"Достигнут лимит max_items={max_items}, останавливаем обработку")
                return
            
            for title in extract_titles(item, title_key):
                file_titles += 1
                total_titles += 1
                yield title
            
            if total_processed % 1000 == 0:
                logger.info(f"  Обработано {total_processed} элементов, собрано {total_titles} заголовков")
        
        logger.info(f"  Файл завершен: собрано {file_titles} заголовков из {file_path.name}")
    
    logger.info(f"\nВсего обработано {total_processed} элементов, собрано {total_titles} заголовков из {len(input_files)} файлов")


def find_json_files(input_path: Path, pattern: str = "data_*.json") -> List[Path]:
    """
    Находит все JSON файлы по паттерну.
    
    Args:
        input_path: Путь к файлу или директории
        pattern: Паттерн для поиска файлов (например, "data_*.json")
    
    Returns:
        Список путей к найденным файлам, отсортированный по имени
    """
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob(pattern))
        logger.info(f"Найдено {len(files)} файлов по паттерну '{pattern}' в {input_path}")
        return files
    else:
        raise FileNotFoundError(f"Путь не найден: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Создание эмбеддингов заголовков из одного или нескольких JSON файлов"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("scripts/data_00.json"),
        help="Путь к входному JSON файлу или директории с JSON файлами"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="data_*.json",
        help="Паттерн для поиска файлов (если input - директория). По умолчанию: 'data_*.json'"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/title_embeddings.npy"),
        help="Путь для сохранения эмбеддингов (.npy)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="intfloat/multilingual-e5-large",
        help="Название модели для эмбеддингов"
    )
    parser.add_argument(
        "--title-key",
        type=str,
        default="title",
        help="Ключ для извлечения заголовков (по умолчанию 'title')"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Размер батча для обработки"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Максимальное количество элементов для обработки (для тестирования)"
    )
    parser.add_argument(
        "--max-titles",
        type=int,
        default=None,
        help="Максимальное количество заголовков для обработки (рекомендуется 10000-20000 для 32 кластеров)"
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=None,
        help="Доля заголовков для выборки (например, 0.1 = 10%%, переопределяет --max-titles)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Количество заголовков для обработки за один батч перед сохранением"
    )
    
    args = parser.parse_args()
    
    # Загрузка модели
    logger.info(f"Загрузка модели: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    logger.info(f"Модель загружена, размерность: {model.get_sentence_embedding_dimension()}")
    
    # Поиск файлов для обработки
    input_files = find_json_files(args.input, args.pattern)
    if not input_files:
        logger.error(f"Не найдено файлов по пути {args.input} с паттерном '{args.pattern}'")
        return
    
    # Сбор заголовков из всех файлов
    logger.info(f"\nИзвлечение заголовков из {len(input_files)} файлов...")
    titles = []
    
    # Используем генератор для обработки всех файлов
    title_generator = process_multiple_files(input_files, args.title_key, args.max_items)
    
    for title in title_generator:
        titles.append(title)
        if len(titles) % 1000 == 0:
            logger.info(f"Собрано {len(titles)} заголовков...")
    
    # Удаляем дубликаты (если нужно)
    unique_titles = list(dict.fromkeys(titles))  # Сохраняет порядок
    if len(unique_titles) < len(titles):
        logger.info(f"Удалено {len(titles) - len(unique_titles)} дубликатов")
    
    logger.info(f"\nВсего собрано {len(unique_titles)} уникальных заголовков из {len(input_files)} файлов")
    
    # Применяем выборку, если указано
    if args.sample_ratio is not None:
        sample_size = int(len(unique_titles) * args.sample_ratio)
        logger.info(f"\nПрименяем случайную выборку: {sample_size} из {len(unique_titles)} ({args.sample_ratio*100:.1f}%)")
        unique_titles = random.sample(unique_titles, sample_size)
        logger.info(f"После выборки: {len(unique_titles)} заголовков")
    elif args.max_titles is not None and len(unique_titles) > args.max_titles:
        logger.info(f"\nПрименяем случайную выборку: {args.max_titles} из {len(unique_titles)}")
        unique_titles = random.sample(unique_titles, args.max_titles)
        logger.info(f"После выборки: {len(unique_titles)} заголовков")
    
    if not unique_titles:
        logger.error("Не найдено ни одного заголовка!")
        return
    
    # Создание эмбеддингов по частям
    logger.info(f"\nСоздание эмбеддингов для {len(unique_titles)} заголовков...")
    all_embeddings = []
    
    for i in range(0, len(unique_titles), args.chunk_size):
        chunk = unique_titles[i:i + args.chunk_size]
        logger.info(f"Обработка батча {i//args.chunk_size + 1} ({len(chunk)} заголовков)...")
        
        embeddings = create_embeddings_batch(
            chunk,
            model,
            batch_size=args.batch_size,
            normalize=True
        )
        all_embeddings.append(embeddings)
    
    # Объединение всех эмбеддингов
    logger.info("Объединение эмбеддингов...")
    final_embeddings = np.vstack(all_embeddings)
    
    logger.info(f"Финальная форма эмбеддингов: {final_embeddings.shape}")
    
    # Сохранение
    logger.info(f"Сохранение в {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, final_embeddings)
    
    logger.info(f"✓ Готово! Эмбеддинги сохранены: {args.output}")
    logger.info(f"  Форма: {final_embeddings.shape}")
    logger.info(f"  Размерность: {final_embeddings.shape[1]}")
    logger.info(f"  Количество: {final_embeddings.shape[0]}")


if __name__ == "__main__":
    main()
