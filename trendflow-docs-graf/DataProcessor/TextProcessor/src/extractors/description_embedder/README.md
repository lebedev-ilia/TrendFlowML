## `description_embedder` (Text Embeddings)

### Назначение

Извлекает **L2-нормализованные эмбеддинги** для описаний видео (description) с использованием sentence transformers (через `dp_models`, no-network).

Ключевая политика качества:
- `description` **может отсутствовать** → это валидная пустота (без фейк-вектора).
- никаких абсолютных путей в `result`/NPZ; `.npy` артефакт — только per-run в `text_processor/_artifacts/`, а relpath передаётся downstream через `doc.tp_artifacts` (in-memory).
- длинные описания обрабатываются **token-aware chunking** через `shared_tokenizer_v1` (dp_models).

**Версия**: 1.2.0  
**Категория**: text embeddings  
**GPU**: поддерживается (cuda), опционально fp16

**Описание фич (19), диапазоны; валидатор среза в NPZ:** [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · `utils/validate_description_embedder_text_npz.py`

**Контракт Audit v3 (`features_flat`)**: [`SCHEMA.md`](SCHEMA.md) · machine: [`../../../schemas/description_embedder_output_v1.json`](../../../schemas/description_embedder_output_v1.json) · отчёт: [`../../../docs/audit_v3/components/description_embedder_AUDIT_V3_REPORT.md`](../../../docs/audit_v3/components/description_embedder_AUDIT_V3_REPORT.md)  
**Audit v4:** [`../../../../docs/audit_v4/components/text_processor/description_embedder_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/description_embedder_audit_v4.md) · L2 stats: `scripts/audit_v4_npz_stats.py` → `storage/audit_v4/description_embedder_l2/`

**Audit v3 preflight**: каноническая модель эмбеддингов — **`intfloat/multilingual-e5-large`** (задаётся профилем / `model_name`; см. `TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md`).

### Входы

- **`doc.description`** (str): текстовое описание видео из `VideoDocument`

### Выходы

Экстрактор возвращает только `result.features_flat` (privacy-safe скаляры для NPZ export):

**Основные метрики**:
- `tp_descemb_present` (0/1) — эмбеддинг вычислен (не "файл существует")
- `tp_descemb_dim` — размерность эмбеддинга
- `tp_descemb_norm_raw` (NaN если `compute_raw_norm=false`) — L2-норма необработанного вектора (до нормализации)
- `tp_descemb_l2_norm` — L2-норма нормализованного вектора (должна быть ~1.0)

**Флаги присутствия и конфигурации**:
- `tp_descemb_description_present` (0/1) — присутствует ли описание в документе
- `tp_descemb_compute_enabled` (0/1) — включено ли вычисление эмбеддинга
- `tp_descemb_write_artifact_enabled` (0/1) — включено ли сохранение артефакта
- `tp_descemb_artifact_written` (0/1) — был ли артефакт успешно записан
- `tp_descemb_cache_enabled` (0/1) — включено ли кеширование
- `tp_descemb_cache_hit` (0/1 или NaN) — попадание в кеш (NaN если кеш отключен)

**Метрики производительности и устройства**:
- `tp_descemb_fp16` (0/1) — использовался ли режим float16
- `tp_descemb_device_cuda` (0/1) — использовалось ли устройство CUDA
- `tp_descemb_model_digest_u24` — первые 24 бита хеша модели (для идентификации)

**Метрики chunking и pooling**:
- `tp_descemb_pooling_length_weighted` (0/1) — использовалась ли стратегия length_weighted_mean
- `tp_descemb_n_chunks` (NaN если не применимо) — количество чанков, на которые был разбит текст
- `tp_descemb_avg_chunk_tokens` (NaN если не применимо) — среднее количество токенов в чанке

**Тайминги** (NaN если не применимо):
- `tp_descemb_chunk_ms` — время разбиения на чанки (миллисекунды)
- `tp_descemb_encode_ms` — время кодирования через модель (миллисекунды)
- `tp_descemb_pool_ms` — время агрегации (pooling) эмбеддингов чанков (миллисекунды)

Для детерминированного доступа downstream-экстракторами в рамках этого же run используется in-memory реестр (только если `write_artifact=true`):
`doc.tp_artifacts["embeddings"]["description"]["relpath"]` (`description_embedding.npy`).

#### Метаданные

- `device`: устройство обработки (`"cpu"` или `"cuda"`)
- `version`: версия экстрактора
- `model_name`, `model_version`, `weights_digest`

#### Системные метрики

- `system.pre_init`: снимок системы до инициализации
- `system.post_init`: снимок системы после инициализации
- `system.post_process`: снимок системы после обработки
- `system.peaks.ram_peak_mb`: пиковое использование RAM (MB)
- `system.peaks.gpu_peak_mb`: пиковое использование GPU памяти (MB)

#### Тайминги

- `timings_s.total`: общее время обработки (секунды)

#### Ошибки

- `error`: описание ошибки (если произошла) или `None`

### Алгоритм обработки

#### 1. Предобработка текста

- Нормализация пробелов через `normalize_whitespace()`
- Если описание пустое, это **valid empty**: `tp_descemb_present=0`, артефакт не создаётся.

#### 2. Кеширование

- **Хеш**: SHA256 от `model_name|weights_digest|tokenizer_digest|chunking/pooling params + "||" + normalized_text`
- **Кеш вектора**: `{cache_dir}/{hash}.npy`
- **Кеш нормы**: `{cache_dir}/{hash}.norm.npy`
- При наличии кеша возвращается сохранённый результат без пересчёта

#### 3. Chunk-and-Aggregate (для длинных текстов)

- **Разбиение**: token-aware chunking по `max_chunk_tokens_model` (через `shared_tokenizer_v1`)
- **Эмбеддинги чанков**: каждый чанк обрабатывается через модель
- **Pooling**: по стратегии `pooling_strategy` (default `length_weighted_mean`) с весами по длине чанка в токенах:
  ```
  weights = [n_tokens(chunk) for chunk in chunks]
  weights = weights / sum(weights)
  pooled = sum(embeddings[i] * weights[i]) / sum(weights)
  ```
- **Нормализация**: L2-нормализация финального вектора

#### 4. Сохранение артефактов

- **Вектор**: `{artifacts_dir}/description_embedding.npy` (fixed per-run имя; без hash в названии)
- **Метаданные**: `.meta.json` sidecar **не используется** (per-run JSON запрещён; model meta фиксируется через `model_version` и `manifest.json.models_used`).

Для детерминированного доступа downstream‑экстракторами в рамках этого же run используется in-memory реестр:
`doc.tp_artifacts["embeddings"]["description"]["relpath"]`.

### Конфигурация

```python
{
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",  # Модель для эмбеддингов
    "cache_dir": None,                                        # Путь к кешу (по умолчанию: default_cache_dir() / "embed_cache")
    "cache_enabled": False,                                   # Включить/выключить кеширование
    "cache_ttl_days": 30.0,                                   # Время жизни кеша в днях (None = без ограничений)
    "cache_max_items": 200000,                               # Максимальное количество элементов в кеше
    "cache_max_bytes": 2000000000,                           # Максимальный размер кеша в байтах
    "cache_cleanup_on_init": True,                           # Очищать кеш при инициализации
    "cache_cleanup_max_seconds": 0.2,                        # Максимальное время на очистку кеша (секунды)
    "device": "cpu",                                          # "cpu" | "cuda"
    "fp16": True,                                             # Использовать float16 на GPU (только для CUDA)
    "batch_size": 32,                                        # Размер батча для обработки чанков
    "artifacts_dir": None,                                   # Путь к артефактам (по умолчанию: default_artifacts_dir())
    "tokenizer_spec_name": "shared_tokenizer_v1",            # dp_models tokenizer spec
    "max_chunk_tokens_model": 512,                            # Максимум токенов модели в чанке
    "pooling_strategy": "length_weighted_mean",              # mean | length_weighted_mean | max | logsumexp
    "compute_embedding": True,                                # считать ли эмбеддинг
    "write_artifact": True,                                   # писать ли `.npy` и регистрировать relpath
    "write_embedding_artifact": True,                         # (deprecated alias для write_artifact)
    "compute_raw_norm": True,                                # считать ли raw norm (иначе NaN)
    "emit_extra_metrics": False                              # зарезервировано: в v1.2.0 не влияет на features_flat (см. SCHEMA.md)
}
```

**Параметры**:
- `model_name`: название модели sentence-transformers для эмбеддингов
- `cache_*`: параметры кеширования (TTL, лимиты, очистка)
- `device`: устройство для обработки (`"cpu"` или `"cuda"`)
- `fp16`: использовать float16 на GPU (экономия памяти, минимальная потеря точности)
- `batch_size`: размер батча для обработки чанков (больше → быстрее на GPU, но больше памяти)
- `tokenizer_spec_name`: спецификация токенизатора из dp_models
- `max_chunk_tokens_model`: максимальное количество токенов в одном чанке (для длинных текстов)
- `pooling_strategy`: стратегия агрегации эмбеддингов чанков:
  - `"mean"`: простое среднее
  - `"length_weighted_mean"`: взвешенное среднее по длине чанка (default)
  - `"max"`: максимум по каждой размерности
  - `"logsumexp"`: стабильный logsumexp pooling
- `compute_embedding`: feature-gating для вычисления эмбеддинга
- `write_artifact`: сохранять ли `.npy` файл и регистрировать relpath в `doc.tp_artifacts`
- `compute_raw_norm`: вычислять ли L2-норму необработанного вектора (до нормализации)
- `emit_extra_metrics`: зарезервирован в v1.2.0 — **не** гейтит тайминги и chunking-статистику в `features_flat` (они заполняются на успешном пути как задокументировано в SCHEMA.md).

### Особенности

- **Кеширование**: автоматическое кеширование по SHA256(content + model_name) для избежания повторных вычислений
- **Длинные тексты**: автоматическое разбиение на чанки с последующей агрегацией
- **Attention-weighted pooling**: взвешивание чанков по их длине для лучшего представления
- **Батчинг**: эффективная обработка нескольких чанков одновременно
- **GPU поддержка**: опциональное использование CUDA с fp16 для ускорения
- **Атомарная запись**: использование временных файлов (.tmp.npy) для безопасного сохранения
- **Метрики**: детальные системные метрики и тайминги

### Архитектура

1. **Инициализация**: загрузка модели через `get_model()` из registry (модели разделяются между экстракторами)
2. **Хеширование**: вычисление SHA256 для проверки кеша
3. **Проверка кеша**: загрузка из кеша если доступно
4. **Разбиение на чанки**: если текст длиннее `max_chunk_tokens`
5. **Эмбеддинги**: батчевая обработка чанков через модель
6. **Агрегация**: attention-weighted pooling по длине чанка
7. **Нормализация**: L2-нормализация финального вектора
8. **Сохранение**: кеширование и сохранение артефактов
9. **Метрики**: сбор системных метрик и таймингов

### Обработка ошибок

- **Пустое описание**: valid empty (не ошибка).
- **Ошибка сохранения артефакта**: RuntimeError (fail-fast), т.к. extractor включён и должен отдать результат.
- **Повреждённый кеш / ошибки кеша**: best-effort, кеш удаляется/пропускается, пересчёт продолжается.

### Performance characteristics

**Resource costs**:
- **CPU**: умеренные (sentence-transformers операции)
- **GPU**: опционально (значительное ускорение при использовании CUDA)
- **Estimated duration**: ~0.1-0.5 секунд для типичного описания (с кешем: ~0.01 секунд)

**Параметры производительности**:
- `batch_size`: большие значения → быстрее на GPU, но больше памяти
- `max_chunk_tokens`: меньшие значения → больше чанков → медленнее, но точнее для очень длинных текстов
- `fp16`: уменьшает использование GPU памяти в 2 раза, минимальная потеря точности

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **model_registry**: реестр моделей для разделения между экстракторами
- **VideoDocument**: схема входного документа
- **sentence-transformers**: библиотека для эмбеддингов
- **normalize_whitespace**: утилита для нормализации текста

### Примечания

1. **Размерность**: зависит от модели (all-MiniLM-L6-v2 → 384)
2. **Нормализация**: финальный вектор всегда L2-нормализован (норма ≈ 1.0)
3. **Кеш**: кеш персистентен между запусками, очистка вручную
4. **Модели**: модели разделяются через registry, не загружаются повторно
5. **Пустые описания**: valid empty (без фейк‑вектора)
---

## Навигация

[SCHEMA](SCHEMA.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
