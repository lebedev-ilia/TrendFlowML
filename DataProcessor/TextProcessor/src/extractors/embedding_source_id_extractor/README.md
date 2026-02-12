## `embedding_source_id_extractor` (Embedding Metadata)

### Назначение

Генерирует **переносимый стабильный идентификатор** (`vector_id`) для primary embedding и возвращает privacy-safe метаданные для интеграции с vector store.

Критически важно (production-grade):
- `vector_id` **не зависит от абсолютных путей** и считается по значениям float32-вектора.
- primary embedding выбирается детерминированно из `doc.tp_artifacts`.

**Версия**: 1.2.0  
**Категория**: metadata  
**GPU**: не требуется

### Входы

Экстрактор читает **детерминированные указатели** на эмбеддинги из in-memory реестра `doc.tp_artifacts`, заполненного ранее в рамках этого же run:

- `doc.tp_artifacts["embeddings"]["title"]["relpath"]` (от `title_embedder`)
- `doc.tp_artifacts["transcripts"]["combined"]["agg_mean_relpath"]` (от `transcript_aggregator`, canonical)
- `doc.tp_artifacts["transcript_aggregates"]["combined"]["agg_mean_relpath"]` (legacy alias)
- `doc.tp_artifacts["embeddings"]["description"]["relpath"]` (от `description_embedder`)

**Важно**:
- Экстрактор больше не делает `glob+mtime` и не выбирает “самый новый файл”.
- JSON sidecar `{embedding_path}.meta.json` **не используется** (per-run JSON запрещён).

### Выходы

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Основные результаты

Экстрактор возвращает:

- `result.features_flat` (только numeric scalars, A-policy)
- `result.embedding_source_id` (privacy-safe dict со строками/идентификаторами)

##### `features_flat`

- `tp_embid_present` (0/1)
- one-hot политики:
  - `tp_embid_policy_transcript_first`
  - `tp_embid_policy_title_first`
  - `tp_embid_policy_description_first`
  - `tp_embid_policy_title_only`
  - `tp_embid_policy_transcript_only`
- one-hot выбранного типа источника:
  - `tp_embid_primary_is_transcript`
  - `tp_embid_primary_is_title`
  - `tp_embid_primary_is_description`

##### `embedding_source_id`

- `vector_id`: 24-символьный hex (sha256 по значениям float32)
- `vector_store_uri`: идентификатор индекса/хранилища (без endpoint’ов)
- `model_version`: строка модели (например `sentence-transformers/all-MiniLM-L6-v2`)
- `weights_digest`: digest модели (если доступен из upstream metadata; иначе `"unknown"`)
- `embedding_relpath`: relpath внутри `text_processor/_artifacts/`
- `primary_source`: какой источник был выбран (например `transcript_combined_mean` / `title`)

#### Метаданные

- `device`: устройство обработки (`"cpu"`)
- `version`: версия экстрактора

#### Системные метрики

- `system.pre_init`: снимок системы до инициализации
- `system.post_init`: снимок системы после инициализации
- `system.post_process`: снимок системы после обработки
- `system.peaks.ram_peak_mb`: пиковое использование RAM (MB)
- `system.peaks.gpu_peak_mb`: пиковое использование GPU памяти (MB, всегда 0)

#### Тайминги

- `timings_s.total`: общее время обработки (секунды)

#### Ошибки

- `error`: описание ошибки (если произошла) или `None`
- `result.embedding_source_id.error`: `"no_embedding_found"` если эмбеддинги не найдены

### Алгоритм обработки

#### 1. Поиск первичного эмбеддинга

Источник истины: `doc.tp_artifacts` (embeddings + transcript aggregates).  
Политика выбора задаётся параметром `primary_source_policy`:
- `transcript_first` (default): transcript_combined_mean → title → description
- `title_first`: title → transcript_combined_mean → description
- `description_first`: description → transcript_combined_mean → title
- `title_only`: только title
- `transcript_only`: только transcript mean

#### 2. Генерация стабильного ID

`vector_id` считается по значениям float32-вектора (после `np.load()`), без участия пути:
- `vector_id = sha256(float32_values)[:24]`

#### 3. Извлечение версии модели

Модель берётся из `doc.tp_artifacts["embeddings"][...]["model_name"]` и `weights_digest` (если доступно), иначе используется `model_version` из конфигурации.

### Конфигурация

```python
{
    "vector_store_uri": "faiss://semantic_titles_v1",          # URI хранилища векторов
    "model_version": "unknown",                               # Версия модели по умолчанию (если не найдена в метаданных)
    "primary_source_policy": "transcript_first",              # transcript_first | title_first | description_first | title_only | transcript_only
    "strict_missing_primary": True,                           # если True и primary не найден → RuntimeError
    "artifacts_dir": None                                     # Путь к артефактам (по умолчанию: default_artifacts_dir())
}
```

### Особенности

- **Приоритет источников**: явный приоритет заголовка над транскриптом и описанием
- **Стабильные ID**: комбинация SHA1 и UUID5 обеспечивает уникальность и детерминированность
- **Метаданные модели**: автоматическое извлечение версии модели из различных источников
- **Временные метки**: ISO 8601 формат с UTC (Z)
- **Обработка граничных случаев**: корректная обработка отсутствующих файлов и метаданных

### Архитектура

1. **Поиск эмбеддингов**: поиск по приоритету (title → transcript → description)
2. **Валидация**: проверка наличия файла
3. **Генерация ID**: вычисление SHA1 и UUID5
4. **Извлечение метаданных**: чтение мета-файлов и кеша
5. **Формирование результата**: сбор всех метаданных
6. **Метрики**: сбор системных метрик и таймингов

### Обработка ошибок

- Если `strict_missing_primary=true` и primary embedding не найден → **RuntimeError** (fail-fast).
- Если embedding найден, но файл не читается → **RuntimeError**.

### Performance characteristics

**Resource costs**:
- **CPU**: очень низкие (только файловые операции и хеширование)
- **GPU**: не используется
- **Estimated duration**: ~0.001-0.01 секунд

**Параметры производительности**:
- Размер файла эмбеддинга: влияет на время вычисления SHA1 (обычно незначительно)
- Количество файлов: линейный поиск по шаблонам

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **title_embedder**: создаёт `title_embedding_*.npy` (без `.meta.json`)
- **transcript_aggregator**: создаёт `transcript_*_agg_mean_*.npy`
- **description_embedder**: создаёт `description_embedding_*.npy` (без `.meta.json`)
- **transcript_chunk_embedder**: создаёт метаданные в кеше
- **default_artifacts_dir**: утилита для определения директории артефактов
- **default_cache_dir**: утилита для определения директории кеша

### Примечания

1. **Зависимости**: требует выполнения хотя бы одного из экстракторов эмбеддингов (title, transcript, description)
2. **Стабильность ID**: ID детерминирован для одного и того же файла, но меняется при изменении содержимого
3. **Векторное хранилище**: `vector_store_uri` используется для идентификации хранилища в системе поиска
4. **Метаданные**: `.meta.json` не используется; используйте `manifest.json.models_used` и `model_version`

### Примеры использования

**Успешное извлечение**:
```json
{
  "embedding_source_id": {
    "vector_id": "a1b2c3d4e5f6-550e8400-e29b-41d4-a716-446655440000",
    "vector_store_uri": "faiss://semantic_titles_v1",
    "model_version": "sentence-transformers/all-MiniLM-L6-v2",
    "created_at": "2024-01-15T10:30:45.123456Z",
    "embedding_path": "/path/to/artifacts/title_embedding_abc123.npy"
  }
}
```

**Отсутствующие эмбеддинги**:
```json
{
  "embedding_source_id": {
    "error": "no_embedding_found"
  }
}
```








