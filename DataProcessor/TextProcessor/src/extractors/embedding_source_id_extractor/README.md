## `embedding_source_id_extractor` (Embedding Metadata)

### Назначение

Генерирует **переносимый стабильный идентификатор** (`vector_id`) для primary embedding и возвращает privacy-safe метаданные для интеграции с vector store.

Критически важно (production-grade):
- `vector_id` **не зависит от абсолютных путей** и считается по значениям float32-вектора.
- primary embedding выбирается детерминированно из `doc.tp_artifacts`.

**Версия**: 1.3.0  
**Категория**: metadata  
**GPU**: не требуется  

**Контракт**: [SCHEMA.md](./SCHEMA.md) · machine: [schemas/embedding_source_id_extractor_output_v1.json](../../../schemas/embedding_source_id_extractor_output_v1.json)  
**Диапазоны и валидатор среза** (`text_features.npz`): [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · [`utils/validate_embedding_source_id_extractor_text_npz.py`](utils/validate_embedding_source_id_extractor_text_npz.py)  
**Audit v4:** [`../../../../docs/audit_v4/components/text_processor/embedding_source_id_extractor_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/embedding_source_id_extractor_audit_v4.md) · L2 stats: `scripts/audit_v4_npz_stats.py` → `storage/audit_v4/embedding_source_id_extractor_l2/`

### Входы

Экстрактор читает **детерминированные указатели** на эмбеддинги из in-memory реестра `doc.tp_artifacts`, заполненного ранее в рамках этого же run:

- **Title**: `doc.tp_artifacts["embeddings"]["title"]["relpath"]` (от `title_embedder`)
- **Description**: `doc.tp_artifacts["embeddings"]["description"]["relpath"]` (от `description_embedder`)
- **Transcript** (canonical, приоритет: combined → whisper → youtube_auto):
  - `doc.tp_artifacts["transcripts"]["combined"]["agg_mean_relpath"]` (от `transcript_aggregator`)
  - `doc.tp_artifacts["transcripts"]["whisper"]["agg_mean_relpath"]`
  - `doc.tp_artifacts["transcripts"]["youtube_auto"]["agg_mean_relpath"]`
- **Transcript** (legacy fallback, приоритет: combined → whisper → youtube_auto):
  - `doc.tp_artifacts["transcript_aggregates"]["combined"]["agg_mean_relpath"]`
  - `doc.tp_artifacts["transcript_aggregates"]["whisper"]["agg_mean_relpath"]`
  - `doc.tp_artifacts["transcript_aggregates"]["youtube_auto"]["agg_mean_relpath"]`

**Важно**:
- Экстрактор больше не делает `glob+mtime` и не выбирает “самый новый файл”.
- JSON sidecar `{embedding_path}.meta.json` **не используется** (per-run JSON запрещён).

### Выходы

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Основные результаты

Экстрактор возвращает:

- `result.features_flat` (только numeric scalars, A-policy)
- `result.embedding_source_id` (privacy-safe dict со строками/идентификаторами)

##### `features_flat` (ровно 13 ключей, фиксированный порядок)

- `tp_embid_present` (1 только если вектор загружен и конечен)
- `tp_embid_strict_missing_primary_enabled` — зеркало `strict_missing_primary`
- one-hot политики: `tp_embid_policy_transcript_first` … `tp_embid_policy_transcript_only`
- one-hot типа источника: `tp_embid_primary_is_transcript` / `title` / `description` (без primary — все 0)
- `tp_embid_unsafe_relpath_flag`, `tp_embid_primary_embed_missing_flag`, `tp_embid_nan_inf_flag`

##### `embedding_source_id`

- `vector_id`: 24-символьный hex (sha256 по little-endian float32 байтам C-order)
- `vector_store_uri`, `embedding_relpath`, `primary_source`
- `model_name`: из upstream (может отсутствовать, напр. transcript mean без per-field meta)
- `model_version`: из upstream **`model_version`** или fallback конфига (не подмена **`model_name`**)
- `weights_digest`: из upstream или `"unknown"`
- при ошибке и **`strict_missing_primary=False`**: только **`error`** с кодом: `no_embedding_found` | `unsafe_relpath` | `embedding_file_missing` | `embedding_load_failed` | `embedding_empty` | `embedding_non_finite`

#### Метаданные (верхний уровень `ExtractorResult`)

- `model_name` / `model_version` / `weights_digest`: **`null`** (канон — вложенный `embedding_source_id`)
- `device`, `version` экстрактора

#### Системные метрики

- `_init_metrics` (pre/post init, RAM peak в байтах)
- `system.post_process`, **`gpu_peak_mb`**

#### Тайминги

- `timings_s.total`: общее время обработки (секунды)

#### Ошибки

- `strict_missing_primary=True`: **RuntimeError** при отсутствии primary, unsafe relpath, отсутствии файла, ошибке `np.load`, пустом векторе, NaN/inf
- `strict_missing_primary=False`: **valid empty**, полный **`features_flat`**, в **`embedding_source_id`** ключ **`error`** (см. список кодов выше)

### Алгоритм обработки

#### 1. Поиск первичного эмбеддинга

Источник истины: `doc.tp_artifacts` (embeddings + transcript aggregates).  
Политика выбора задаётся параметром `primary_source_policy`:
- `transcript_first` (default): transcript (combined → whisper → youtube_auto) → title → description
- `title_first`: title → transcript (combined → whisper → youtube_auto) → description
- `description_first`: description → transcript (combined → whisper → youtube_auto) → title
- `title_only`: только title
- `transcript_only`: только transcript (combined → whisper → youtube_auto)

Для transcript внутри источника сначала проверяется canonical путь (`transcripts`), затем legacy (`transcript_aggregates`). Внутри каждого источника приоритет: combined → whisper → youtube_auto.

#### 2. Генерация стабильного ID

`vector_id` считается по значениям float32-вектора (после `np.load()`), без участия пути:
- Вектор конвертируется в little-endian float32 байты
- Вычисляется SHA256 хеш от байтов
- Берутся первые 24 hex-символа (12 байт) из хеша
- `vector_id = sha256(float32_values)[:24]`

#### 3. Метаданные модели

`model_name` / `weights_digest` — из полей upstream рядом с relpath; **`model_version`** — из **`model_version`** в том же dict либо из конфига экстрактора.

### Конфигурация

```python
{
    "vector_store_uri": "faiss://semantic_titles_v1",          # URI хранилища векторов
    "model_version": "unknown",                               # Версия модели по умолчанию (если не найдена в метаданных)
    "primary_source_policy": "transcript_first",              # transcript_first | title_first | description_first | title_only | transcript_only
    "strict_missing_primary": True,                           # True: fail-fast на всех перечисленных в README ветках; False: soft + error
    "artifacts_dir": None                                     # Путь к артефактам (по умолчанию: default_artifacts_dir())
}
```

### Особенности

- **Приоритет источников**: явный приоритет заголовка над транскриптом и описанием
- **Стабильные ID**: SHA256 хеш от float32 значений обеспечивает уникальность и детерминированность (не зависит от путей)
- **Метаданные модели**: автоматическое извлечение версии модели из `doc.tp_artifacts` или конфигурации
- **Обработка граничных случаев**: корректная обработка отсутствующих файлов и метаданных

### Архитектура

1. **Поиск эмбеддингов**: детерминированный выбор по политике приоритета из `doc.tp_artifacts` (transcript → title → description или по настройке)
2. **Валидация**: проверка наличия файла через safe-join
3. **Генерация ID**: вычисление SHA256 хеша от float32 значений вектора (первые 24 hex символа)
4. **Извлечение метаданных**: чтение `model_name` и `weights_digest` из `doc.tp_artifacts` или использование значений по умолчанию
5. **Формирование результата**: сбор всех метаданных в `embedding_source_id` и `features_flat`
6. **Метрики**: сбор системных метрик и таймингов

### Обработка ошибок

См. раздел **Ошибки**: один флаг **`strict_missing_primary`** задаёт жёсткий или мягкий режим для всего пост-path цикла.

### Performance characteristics

**Resource costs**:
- **CPU**: очень низкие (только файловые операции и хеширование)
- **GPU**: не используется
- **Estimated duration**: ~0.001-0.01 секунд

**Параметры производительности**:
- Размер файла эмбеддинга: влияет на время хеширования SHA256 (обычно незначительно)
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

1. **Зависимости**: требует выполнения хотя бы одного из экстракторов эмбеддингов (title, transcript, description) или transcript_aggregator
2. **Стабильность ID**: ID детерминирован для одного и того же вектора (float32 значения), но меняется при изменении содержимого вектора
3. **Векторное хранилище**: `vector_store_uri` используется для идентификации хранилища в системе поиска
4. **Метаданные**: метаданные модели берутся из `doc.tp_artifacts` (например, `embeddings[title][model_name]`), а не из JSON файлов
5. **Privacy-safe**: в результате нет абсолютных путей, только `embedding_relpath` относительно `artifacts_dir`

### Примеры использования

**Успешное извлечение** (_фрагмент payload_):
```json
{
  "embedding_source_id": {
    "vector_id": "a1b2c3d4e5f6789012345678",
    "vector_store_uri": "faiss://semantic_titles_v1",
    "model_name": null,
    "model_version": "sentence-transformers/all-MiniLM-L6-v2",
    "weights_digest": "unknown",
    "embedding_relpath": "title_embedding_abc123.npy",
    "primary_source": "title"
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








