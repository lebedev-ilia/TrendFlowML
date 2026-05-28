## `transcript_chunk_embedder` (Text embeddings)

### Назначение

Извлекает **эмбеддинги по чанкам** из транскрипта видео. Разбивает транскрипт на перекрывающиеся чанки (по предложениям) и генерирует векторные представления для каждого чанка с использованием sentence-transformers моделей. Поддерживает обработку нескольких источников транскрипта (whisper, youtube_auto) независимо.

**Версия**: 1.3.0  
**Категория**: text embeddings  
**GPU**: optional (зависит от модели и устройства)

**Диапазоны, тайминги и валидатор среза** (`text_features.npz`): [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · [`utils/validate_transcript_chunk_embedder_text_npz.py`](utils/validate_transcript_chunk_embedder_text_npz.py) (`--struct`, `--ranges`, `--timings`)

**Контракт `features_flat`**: [SCHEMA.md](SCHEMA.md) · machine: [`../../schemas/transcript_chunk_embedder_output_v1.json`](../../schemas/transcript_chunk_embedder_output_v1.json) · отчёт Audit v3: [`../../docs/audit_v3/components/transcript_chunk_embedder_AUDIT_V3_REPORT.md`](../../docs/audit_v3/components/transcript_chunk_embedder_AUDIT_V3_REPORT.md) · **Audit v4:** [`../../../../docs/audit_v4/components/text_processor/transcript_chunk_embedder_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/transcript_chunk_embedder_audit_v4.md) · **L2 stats:** [`../../../../storage/audit_v4/transcript_chunk_embedder_l2/transcript_chunk_embedder_audit_v4_stats.json`](../../../../storage/audit_v4/transcript_chunk_embedder_l2/transcript_chunk_embedder_audit_v4_stats.json) (tooling: `scripts/audit_v4_npz_stats.py`)

### Входы

Источник истины для прод‑режима (gated):
- `doc.asr.segments[]` (whisper) — если `use_asr=True`
- `doc.transcripts["youtube_auto"]` — если `use_youtube_auto=True` (legacy)

Tokenizer (строго):
- `dp_models` spec `shared_tokenizer_v1` (no fallback)

### Выходы

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Основные результаты

- `features_flat`: ровно **16** числовых скаляров `tp_tchunk_*` на каждой ветке (**`extract`** и **`extract_batch`**), порядок фиксирован контрактом v1.3.0:
  - Базовые: `tp_tchunk_present`, `tp_tchunk_sources_count`, `tp_tchunk_whisper_present`, `tp_tchunk_youtube_auto_present`, `tp_tchunk_whisper_chunks`, `tp_tchunk_youtube_chunks`, `tp_tchunk_embedding_dim`
  - Confidence: `tp_tchunk_conf_present`, `tp_tchunk_conf_mean`, `tp_tchunk_conf_min`, `tp_tchunk_conf_max` — при **`emit_confidence_metrics=False`** значения **0** или **NaN** (ключи остаются)
  - Доп. настройки: `tp_tchunk_batch_size`, `tp_tchunk_max_chunk_tokens_model`, `tp_tchunk_overlap_ratio`, `tp_tchunk_max_chunks_total`, `tp_tchunk_cache_enabled` — при **`emit_extra_metrics=False`** все пять полей **NaN** (ключи остаются)

Пути к `.npy` артефактам **не возвращаются** в `result` (privacy).  
Для передачи между extractor’ами в рамках одного run используется:
- canonical: `doc.tp_artifacts["transcripts"][source]["chunk_embeddings_relpath"]`
- legacy alias: `doc.tp_artifacts["transcript_chunks"][source]["embeddings_relpath"]`

#### Метаданные

- `device`: устройство обработки (`"cpu"` или `"cuda"`)
- `version`: версия экстрактора (`"1.3.0"`)
- `model_name`, `model_version`, `weights_digest`: на верхнем уровне ответа и внутри **`result`** (в т.ч. пустые ветки)
- `system`: системные метрики (pre_init, post_init, post_process, peaks)
- `timings_s.total`: общее время обработки (секунды)
- `error`: ошибка (если есть, иначе `None`)

### Алгоритм

1. **Разбиение на предложения**: использование регулярного выражения для разделения по `.`, `!`, `?`
2. **Формирование чанков**: token-aware chunking с ограничением по токенам (`max_chunk_tokens_model`) через `shared_tokenizer_v1`
3. **Overlap**: сохранение перекрытия между чанками (`overlap_ratio`) для сохранения контекста
4. **Генерация эмбеддингов**: батчевая обработка чанков через sentence-transformers модель
5. **L2-нормализация**: нормализация каждого эмбеддинга к единичной длине
6. **Кеширование**: сохранение векторов в artifacts и метаданных в cache (атомарная запись через .tmp файлы)

### Разбиение на чанки

- **Максимальный размер чанка**: `max_chunk_tokens_model` (по умолчанию 256 токенов)
- **Overlap**: `overlap_ratio` (по умолчанию 0.15 = 15%)
- **Метод**: предложения группируются в чанки до достижения лимита токенов, затем начинается новый чанк с перекрытием
- **Cost cap**: `max_chunks_total` (по умолчанию 256)
- **Token counting**: точный подсчет через `shared_tokenizer_v1` (dp_models), не приблизительный
- **ASR segments**: для whisper источника используется chunking по ASR segments с сохранением confidence mapping

### Конфигурация

```python
{
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",  # Модель для эмбеддингов
    "tokenizer_spec_name": "shared_tokenizer_v1",            # dp_models tokenizer (strict)
    "cache_dir": None,                                        # Путь к кешу (по умолчанию: TextProcessor/cache/transcript_embed)
    "cache_enabled": False,                                   # по умолчанию off
    "cache_ttl_days": 30.0,                                   # TTL кеша в днях (None = без TTL)
    "cache_max_items": 50000,                                 # Максимальное количество элементов в кеше
    "cache_max_bytes": 5000000000,                            # Максимальный размер кеша в байтах
    "cache_cleanup_on_init": True,                            # Очистка кеша при инициализации
    "cache_cleanup_max_seconds": 0.25,                        # Максимальное время на очистку кеша
    "artifacts_dir": None,                                    # Путь к артефактам (по умолчанию: из env)
    "device": "cpu",                                          # "cpu" | "cuda"
    "fp16": True,                                             # Использовать float16 на GPU
    "batch_size": 64,                                         # Размер батча для обработки чанков
    "use_asr": True,                                          # Использовать doc.asr.segments (whisper)
    "use_youtube_auto": False,                                 # Использовать doc.transcripts["youtube_auto"]
    "require_asr": False,                                     # Если True и нет ASR → ошибка (fail-fast)
    "require_any_source": False,                             # Если True и нет ни одного источника → ошибка
    "max_chunk_tokens_model": 256,                            # Максимальное количество токенов в чанке
    "overlap_ratio": 0.15,                                    # Коэффициент перекрытия между чанками
    "max_chunks_total": 256,                                  # Максимальное количество чанков (cost cap)
    "emit_confidence_metrics": True,                           # Включать метрики уверенности ASR
    "emit_extra_metrics": False                               # Включать дополнительные метрики (batch_size, chunking params, cache_enabled)
}
```

### Обработка источников

Экстрактор обрабатывает каждый источник транскрипта независимо:

- **whisper**: если `use_asr=True` и присутствует `doc.asr.segments`
- **youtube_auto**: если `use_youtube_auto=True` и присутствует `doc.transcripts["youtube_auto"]`
- Результаты сохраняются отдельно для каждого источника

### Кеширование

- **Artifacts (per-run)**: сохраняются в `artifacts_dir` как `transcript_{source}_chunk_embeddings.npy` (fixed name)
- **Cache (optional, вне result_store)**: если `cache_enabled=True`, сохраняет:
  - vectors: `<cache_key>.npy`
  - meta: `<cache_key>.meta.json` (без raw текста)
- **Атомарность**: запись через временные файлы `.tmp.npy` и `.tmp.json` с последующим `os.replace()`
- **Хеширование**: SHA256 от `{model_name}||{text}` для уникальной идентификации

### Формат метаданных (cache)

```json
{
    "source": "whisper",
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "model_version": "unknown",
    "weights_digest": "...",
    "device": "cpu",
    "n_chunks": 10,
    "embedding_dim": 384,
    "conf_present": 1.0,
    "conf_mean": 0.95,
    "conf_min": 0.85,
    "conf_max": 1.0
}
```

**Примечание**: для приватности, сырые тексты чанков **не сохраняются** в метаданных. Cache key основан на privacy-safe transcript_id (hash от token_ids или текста).

### Особенности

- **Множественные источники**: независимая обработка whisper и youtube_auto транскриптов
- **Token-aware chunking**: точный подсчет токенов через shared_tokenizer_v1 (dp_models)
- **ASR confidence integration**: использование весов уверенности Whisper при chunking (сохранение mapping)
- **Overlap чанков**: сохранение контекста между соседними чанками
- **L2-нормализация**: все эмбеддинги нормализованы к единичной длине
- **Батчевая обработка**: эффективная обработка больших транскриптов (поддерживается `extract_batch()`)
- **Кеширование**: избежание повторных вычислений для одинаковых текстов (опционально, по умолчанию off)
- **Атомарная запись**: безопасное сохранение файлов через временные файлы
- **Privacy-safe IDs**: transcript_id основан на hash от token_ids или текста (не raw текст)
- **Метрики производительности**: отслеживание времени выполнения и использования памяти/GPU

### Обработка ошибок

- **Отсутствие транскриптов**: valid empty (`tp_tchunk_present=0`, метрики = NaN/0 согласно контракту)
- **Ошибка загрузки модели**: fail-fast (не fallback)
- **Ошибка кеша**: best-effort (пересчёт)

### Архитектура

1. **Инициализация**: загрузка модели через `model_registry` (shared models)
2. **Нормализация текста**: применение `normalize_whitespace()` к транскриптам
3. **Разбиение на чанки**: `_split_into_chunks()` с overlap
4. **Проверка кеша**: поиск существующих эмбеддингов по хешу
5. **Генерация эмбеддингов**: батчевая обработка через модель
6. **Нормализация**: L2-нормализация каждого вектора
7. **Сохранение**: атомарная запись в artifacts и cache
8. **Сбор метрик**: измерение времени выполнения и использования ресурсов

### Performance characteristics

**Resource costs**:
- **CPU**: умеренные (зависит от модели и размера транскрипта)
- **GPU**: опционально (если `device="cuda"` и модель поддерживает GPU)
- **RAM**: зависит от размера транскрипта и batch_size
- **Estimated duration**: ~0.5-5.0 секунд для типичного транскрипта (зависит от модели и устройства)

**Параметры производительности**:
- `batch_size`: большие значения → быстрее, но больше памяти
- `max_chunk_tokens`: меньшие значения → больше чанков → больше вычислений
- `fp16`: уменьшает использование памяти на GPU, может ускорить обработку
- `overlap_ratio`: не влияет на производительность, только на качество

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **VideoDocument**: схема входного документа
- **model_registry**: реестр моделей (shared models)
- **normalize_whitespace**: утилита для нормализации пробелов
- **sentence-transformers**: библиотека для генерации эмбеддингов
- **system_snapshot, process_memory_bytes**: утилиты для сбора метрик

### Примечания

1. **Модель по умолчанию**: `all-MiniLM-L6-v2` (384-мерные эмбеддинги, быстрая)
2. **Размерность эмбеддинга**: зависит от модели (для all-MiniLM-L6-v2: 384)
3. **Tokenizer**: строго через `dp_models` spec `shared_tokenizer_v1` (no fallback)
4. **Зависимость от ASR**: для whisper источника требуется `doc.asr.segments` (AudioProcessor)
5. **Token-aware chunking**: точный подсчет токенов через shared_tokenizer_v1, не приблизительный
6. **ASR confidence**: для whisper источника confidence сохраняется и может использоваться downstream (transcript_aggregator)
7. **Chunking strategy**: для whisper используется chunking по ASR segments с сохранением confidence mapping; для других источников — sentence-based chunking
8. **Overlap**: перекрытие помогает сохранить контекст между чанками, особенно важно для длинных предложений
9. **Кеширование**: кеш работает на основе privacy-safe transcript_id (hash от token_ids или текста) + weights_digest
10. **Batch processing**: поддерживается `extract_batch()` для эффективной обработки нескольких документов
11. **Нормализация**: L2-нормализация выполняется вручную (модель возвращает ненормализованные векторы)
12. **Cost controls**: `max_chunks_total` ограничивает количество чанков для контроля стоимости
13. **Cache cleanup**: автоматическая очистка кеша при инициализации (best-effort, с таймаутом)

