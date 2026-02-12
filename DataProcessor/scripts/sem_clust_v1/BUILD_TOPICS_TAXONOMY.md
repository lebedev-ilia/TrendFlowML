# Как создать topics_taxonomy_v1

## Обзор

`topics_taxonomy_v1` - это JSONL файл с темами для semantic topic extraction. Каждая строка содержит JSON объект с информацией о теме.

## Формат topics.jsonl

Каждая строка - это JSON объект со следующими полями:

```json
{
  "id": 1,
  "name": "Business & Money",
  "aliases_en": ["finance", "business", "money"],
  "aliases_ru": ["финансы", "бизнес", "деньги"],
  "prompts_en": ["business news", "personal finance tips", "how to make money"],
  "prompts_ru": ["бизнес новости", "личные финансы", "как заработать деньги"],
  "group": "business"
}
```

### Поля

- **id** (обязательное, int): уникальный идентификатор темы
- **name** (обязательное, str): название темы
- **aliases_en** (опциональное, list[str]): алиасы на английском
- **aliases_ru** (опциональное, list[str]): алиасы на русском
- **prompts_en** (опциональное, list[str]): промпты для поиска на английском (рекомендуется 3-10 промптов)
- **prompts_ru** (опциональное, list[str]): промпты для поиска на русском (рекомендуется 3-10 промптов)
- **group** (опциональное, str): группа темы (по умолчанию "general")

**Важно**: Хотя бы один из `prompts_en` или `prompts_ru` должен быть заполнен для каждой темы.

## Создание из JSON файла

### Шаг 1: Создайте JSON файл

Создайте файл `topics.json`:

```json
{
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
```

Или просто массив:

```json
[
  {
    "id": 1,
    "name": "Business & Money",
    ...
  },
  {
    "id": 2,
    "name": "Technology",
    ...
  }
]
```

### Шаг 2: Запустите скрипт

```bash
cd /home/ilya/Рабочий\ стол/TrendFlowML/DataProcessor

python3 scripts/build_topics_taxonomy_v1.py \
    --input topics.json \
    --output dp_models/bundled_models/text/topics_v1/topics.jsonl
```

## Создание из CSV файла

### Шаг 1: Создайте CSV файл

Создайте файл `topics.csv`:

```csv
id,name,group,aliases_en,aliases_ru,prompts_en,prompts_ru
1,Business & Money,business,"finance,business,money","финансы,бизнес,деньги","business news,personal finance tips,how to make money","бизнес новости,личные финансы,как заработать деньги"
2,Technology,tech,"tech,gadgets,software","технологии,гаджеты,софт","technology review,new gadgets,software tutorial","обзор технологий,новые гаджеты,урок по софту"
```

**Примечание**: Списки в CSV разделяются запятыми.

### Шаг 2: Запустите скрипт

```bash
python3 scripts/build_topics_taxonomy_v1.py \
    --input topics.csv \
    --output dp_models/bundled_models/text/topics_v1/topics.jsonl \
    --format csv
```

## Валидация без сохранения

Чтобы только проверить файл без сохранения:

```bash
python3 scripts/build_topics_taxonomy_v1.py \
    --input topics.json \
    --validate-only
```

## Рекомендации

### Количество тем

- **Минимум**: 10-20 тем для базовой категоризации
- **Рекомендуется**: 50-200 тем для хорошего покрытия
- **Максимум**: до 500 тем (больше может снизить качество)

### Промпты

- **Количество**: 3-10 промптов на тему (больше = лучше recall)
- **Качество**: промпты должны быть репрезентативными для темы
- **Языки**: рекомендуется заполнять и `prompts_en`, и `prompts_ru` для мультиязычности

### Группы

Используйте группы для организации тем:
- `business`, `tech`, `entertainment`, `health`, `food`, `travel`, `education`, и т.д.

### ID тем

- ID должны быть уникальными
- Рекомендуется использовать последовательные ID (1, 2, 3, ...)
- Не переиспользуйте ID при обновлении таксономии

## Примеры промптов

### Хорошие промпты:
- Конкретные и релевантные: "makeup tutorial", "crypto trading analysis"
- Разнообразные: "workout routine", "fitness tips", "home workout"
- Естественные фразы: "how to learn python", "travel vlog"

### Плохие промпты:
- Слишком общие: "video", "content", "stuff"
- Не релевантные: "makeup tutorial" для темы "Technology"
- Одинаковые: "tech", "tech", "tech"

## Расширение существующей таксономии

Чтобы добавить новые темы к существующей таксономии:

1. Загрузите текущий `topics.jsonl`:
```bash
python3 -c "
import json
topics = []
with open('dp_models/bundled_models/text/topics_v1/topics.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            topics.append(json.loads(line))
with open('topics_current.json', 'w', encoding='utf-8') as f:
    json.dump(topics, f, ensure_ascii=False, indent=2)
"
```

2. Добавьте новые темы в JSON файл

3. Пересоздайте `topics.jsonl`:
```bash
python3 scripts/build_topics_taxonomy_v1.py \
    --input topics_current.json \
    --output dp_models/bundled_models/text/topics_v1/topics.jsonl
```

## Проверка результатов

После создания файла проверьте его:

```bash
# Подсчет тем
wc -l dp_models/bundled_models/text/topics_v1/topics.jsonl

# Просмотр первых тем
head -n 5 dp_models/bundled_models/text/topics_v1/topics.jsonl | python3 -m json.tool

# Валидация структуры
python3 scripts/build_topics_taxonomy_v1.py \
    --input dp_models/bundled_models/text/topics_v1/topics.jsonl \
    --validate-only
```

## Устранение проблем

**Ошибка: "отсутствует обязательное поле 'id'"**
- Убедитесь, что каждая тема имеет поле `id` (int)

**Ошибка: "Найдены дублирующиеся ID"**
- Проверьте, что все ID уникальны

**Предупреждение: "тема не имеет промптов"**
- Добавьте хотя бы один промпт в `prompts_en` или `prompts_ru`

**Ошибка при загрузке CSV**
- Убедитесь, что списки разделены запятыми
- Проверьте кодировку файла (должна быть UTF-8)

