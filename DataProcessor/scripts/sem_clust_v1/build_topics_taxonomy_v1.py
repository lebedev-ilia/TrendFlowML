#!/usr/bin/env python3
"""
Скрипт для создания topics_taxonomy_v1.

Генерирует topics.jsonl файл с темами для semantic topic extraction.

Использование:
    # Из JSON файла
    python3 scripts/build_topics_taxonomy_v1.py \
        --input topics.json \
        --output dp_models/bundled_models/text/topics_v1/topics.jsonl
    
    # Из CSV файла
    python3 scripts/build_topics_taxonomy_v1.py \
        --input topics.csv \
        --output dp_models/bundled_models/text/topics_v1/topics.jsonl \
        --format csv
"""

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any


def load_from_json(input_path: str) -> List[Dict[str, Any]]:
    """Загрузить темы из JSON файла."""
    print(f"Загрузка тем из JSON: {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "topics" in data:
        return data["topics"]
    else:
        raise ValueError(f"Неожиданный формат JSON: ожидается list или dict с ключом 'topics'")


def load_from_csv(input_path: str) -> List[Dict[str, Any]]:
    """Загрузить темы из CSV файла."""
    print(f"Загрузка тем из CSV: {input_path}...")
    topics = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic = {
                "id": int(row.get("id", 0)),
                "name": row.get("name", ""),
                "group": row.get("group", "general"),
            }
            
            # Парсинг списков из CSV (формат: "item1,item2,item3")
            def parse_list(value: str) -> List[str]:
                if not value:
                    return []
                return [item.strip() for item in value.split(",") if item.strip()]
            
            topic["aliases_en"] = parse_list(row.get("aliases_en", ""))
            topic["aliases_ru"] = parse_list(row.get("aliases_ru", ""))
            topic["prompts_en"] = parse_list(row.get("prompts_en", ""))
            topic["prompts_ru"] = parse_list(row.get("prompts_ru", ""))
            
            topics.append(topic)
    
    return topics


def validate_topic(topic: Dict[str, Any], index: int) -> None:
    """Валидация структуры темы."""
    required_fields = ["id", "name"]
    for field in required_fields:
        if field not in topic:
            raise ValueError(f"Тема #{index}: отсутствует обязательное поле '{field}'")
    
    if not isinstance(topic["id"], int):
        raise ValueError(f"Тема #{index}: поле 'id' должно быть int, получено {type(topic['id'])}")
    
    if not isinstance(topic["name"], str) or not topic["name"].strip():
        raise ValueError(f"Тема #{index}: поле 'name' должно быть непустой строкой")
    
    # Проверка опциональных полей
    for field in ["aliases_en", "aliases_ru", "prompts_en", "prompts_ru"]:
        if field in topic and not isinstance(topic[field], list):
            raise ValueError(f"Тема #{index}: поле '{field}' должно быть list")
    
    # Проверка, что есть хотя бы один промпт
    prompts_en = topic.get("prompts_en", [])
    prompts_ru = topic.get("prompts_ru", [])
    if not prompts_en and not prompts_ru:
        print(f"Предупреждение: тема #{index} (id={topic['id']}, name={topic['name']}) не имеет промптов")


def save_topics_jsonl(topics: List[Dict[str, Any]], output_path: str) -> None:
    """Сохранить темы в JSONL формат."""
    print(f"\nСохранение {len(topics)} тем в {output_path}...")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for topic in topics:
            # Валидация перед сохранением
            validate_topic(topic, topic["id"])
            
            # Нормализация структуры
            normalized = {
                "id": int(topic["id"]),
                "name": str(topic["name"]),
                "aliases_en": [str(x) for x in topic.get("aliases_en", []) if str(x).strip()],
                "aliases_ru": [str(x) for x in topic.get("aliases_ru", []) if str(x).strip()],
                "prompts_en": [str(x) for x in topic.get("prompts_en", []) if str(x).strip()],
                "prompts_ru": [str(x) for x in topic.get("prompts_ru", []) if str(x).strip()],
                "group": str(topic.get("group", "general")),
            }
            
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
    
    print(f"✓ Сохранено {len(topics)} тем")


def create_example_json() -> str:
    """Создать пример JSON файла."""
    example = {
        "topics": [
            {
                "id": 1,
                "name": "Business & Money",
                "aliases_en": ["finance", "business", "money"],
                "aliases_ru": ["финансы", "бизнес", "деньги"],
                "prompts_en": ["business news", "personal finance tips", "how to make money"],
                "prompts_ru": ["бизнес новости", "личные финансы", "как заработать деньги"],
                "group": "business"
            },
            {
                "id": 2,
                "name": "Technology",
                "aliases_en": ["tech", "gadgets", "software"],
                "aliases_ru": ["технологии", "гаджеты", "софт"],
                "prompts_en": ["technology review", "new gadgets", "software tutorial"],
                "prompts_ru": ["обзор технологий", "новые гаджеты", "урок по софту"],
                "group": "tech"
            }
        ]
    }
    return json.dumps(example, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Создание topics_taxonomy_v1 из JSON или CSV файла",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Пример JSON файла:
{create_example_json()}

Пример CSV файла (заголовки):
id,name,group,aliases_en,aliases_ru,prompts_en,prompts_ru
1,Business & Money,business,"finance,business,money","финансы,бизнес,деньги","business news,personal finance","бизнес новости,личные финансы"
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Путь к входному файлу (JSON или CSV)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dp_models/bundled_models/text/topics_v1/topics.jsonl",
        help="Путь к выходному JSONL файлу"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "auto"],
        default="auto",
        help="Формат входного файла (auto определяет по расширению)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Только валидировать входной файл, не сохранять"
    )
    
    args = parser.parse_args()
    
    # Определение формата
    input_format = args.format
    if input_format == "auto":
        input_path = Path(args.input)
        if input_path.suffix.lower() == ".json":
            input_format = "json"
        elif input_path.suffix.lower() == ".csv":
            input_format = "csv"
        else:
            raise ValueError(f"Не удалось определить формат файла {args.input}. Укажите --format явно.")
    
    # Загрузка данных
    if input_format == "json":
        topics = load_from_json(args.input)
    elif input_format == "csv":
        topics = load_from_csv(args.input)
    else:
        raise ValueError(f"Неподдерживаемый формат: {input_format}")
    
    print(f"Загружено {len(topics)} тем")
    
    # Валидация
    print("\nВалидация тем...")
    for i, topic in enumerate(topics):
        try:
            validate_topic(topic, i)
        except ValueError as e:
            print(f"Ошибка валидации: {e}")
            if not args.validate_only:
                raise
    
    # Проверка уникальности ID
    ids = [t["id"] for t in topics]
    if len(ids) != len(set(ids)):
        duplicates = [id for id in ids if ids.count(id) > 1]
        raise ValueError(f"Найдены дублирующиеся ID: {set(duplicates)}")
    
    print("✓ Все темы валидны")
    
    # Статистика
    total_prompts = sum(len(t.get("prompts_en", [])) + len(t.get("prompts_ru", [])) for t in topics)
    groups = set(t.get("group", "general") for t in topics)
    print(f"\nСтатистика:")
    print(f"  Всего тем: {len(topics)}")
    print(f"  Всего промптов: {total_prompts}")
    print(f"  Групп: {len(groups)} ({', '.join(sorted(groups))})")
    
    if args.validate_only:
        print("\n✓ Валидация пройдена успешно")
        return
    
    # Сохранение
    save_topics_jsonl(topics, args.output)
    
    print(f"\n✓ Готово! Файл сохранен: {args.output}")
    print(f"\nСледующие шаги:")
    print(f"1. Проверьте, что spec файл существует: dp_models/spec_catalog/text/topics_taxonomy_v1.yaml")
    print(f"2. Кеш промптов будет автоматически пересоздан при первом использовании")


if __name__ == "__main__":
    main()

