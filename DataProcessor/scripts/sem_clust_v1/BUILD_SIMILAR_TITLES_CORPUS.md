# Как создать similar_titles_corpus_v1

## Обзор

`similar_titles_corpus_v1` - это корпус эмбеддингов заголовков для поиска похожих заголовков. Состоит из:
- `embeddings.npy`: массив эмбеддингов (N, D), где N - количество заголовков, D - размерность (1024 для multilingual-e5-large)
- `ids.json`: список ID для каждого эмбеддинга (для идентификации заголовков)

## Требования

1. **Эмбеддинги**: файл `.npy` с эмбеддингами формы `(N, D)`
   - Можно использовать уже созданные эмбеддинги из `title_embeddings.npy`
   - Или создать новые из JSON файлов (требует дополнительных скриптов)

2. **Python пакеты**:
   ```bash
   pip install numpy
   ```

## Шаг 1: Подготовка эмбеддингов

Если у вас уже есть эмбеддинги (например, из `create_title_emb_for_cluster_v1.py`), используйте их:

```bash
# Проверьте наличие файла
ls -lh scripts/sem_clust_v1/title_embeddings.npy
```

Если эмбеддингов нет, создайте их сначала (см. `create_title_emb_for_cluster_v1.py`).

## Шаг 2: Запуск скрипта

```bash
cd /home/ilya/Рабочий\ стол/TrendFlowML/DataProcessor

python3 scripts/sem_clust_v1/build_similar_titles_corpus_v1.py \
    --embeddings-path scripts/sem_clust_v1/title_embeddings.npy \
    --output-dir dp_models/bundled_models/text/similar_titles_v1
```

### Параметры

- `--embeddings-path`: путь к файлу с эмбеддингами (`.npy`)
- `--output-dir`: директория для сохранения корпуса (по умолчанию: `dp_models/bundled_models/text/similar_titles_v1`)

## Шаг 3: Проверка результатов

```bash
# Проверьте созданные файлы
ls -lh dp_models/bundled_models/text/similar_titles_v1/

# Должны быть:
# - embeddings.npy (массив эмбеддингов)
# - ids.json (список ID)
```

## Шаг 4: Проверка spec файла

Убедитесь, что spec файл существует и пути правильные:

```bash
cat dp_models/spec_catalog/text/similar_titles_corpus_v1.yaml
```

Пути должны быть **без префикса `bundled_models/`**:
```yaml
local_artifacts:
  - path: text/similar_titles_v1/embeddings.npy
  - path: text/similar_titles_v1/ids.json
```

## Рекомендации

1. **Размер корпуса**: 
   - Минимум: 1,000 заголовков для базового поиска
   - Оптимально: 10,000-50,000 заголовков
   - Максимум: 100,000+ (может быть медленно без FAISS)

2. **ID формат**: 
   - По умолчанию используется `title_{index:08d}` (например, `title_00000001`)
   - Можно изменить функцию `generate_id_from_title()` для другого формата

3. **Производительность**:
   - Для корпусов > 200,000 рекомендуется использовать FAISS
   - Скрипт автоматически создаст HNSW индекс при использовании extractor'а

## Пример полного процесса

```bash
# 1. Создайте эмбеддинги (если еще не созданы)
python3 scripts/sem_clust_v1/create_title_emb_for_cluster_v1.py \
    --input scripts/sem_clust_v1 \
    --pattern "data_*.json" \
    --output scripts/sem_clust_v1/title_embeddings.npy \
    --max-titles 20000

# 2. Создайте корпус
python3 scripts/sem_clust_v1/build_similar_titles_corpus_v1.py \
    --embeddings-path scripts/sem_clust_v1/title_embeddings.npy \
    --output-dir dp_models/bundled_models/text/similar_titles_v1

# 3. Проверьте результаты
ls -lh dp_models/bundled_models/text/similar_titles_v1/
```

## Устранение проблем

**Ошибка: "Несоответствие размеров"**
- Проверьте, что количество эмбеддингов соответствует количеству ID
- Убедитесь, что файл `.npy` не поврежден

**Ошибка: "weights_missing" при использовании extractor'а**
- Проверьте пути в spec файле (должны быть без `bundled_models/`)
- Убедитесь, что файлы существуют в правильной директории

**Медленный поиск**
- Для больших корпусов (>200k) используйте FAISS
- Установите `use_faiss: true` в конфиге extractor'а

