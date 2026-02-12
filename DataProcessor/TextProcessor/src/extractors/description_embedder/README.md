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

### Входы

- **`doc.description`** (str): текстовое описание видео из `VideoDocument`

### Выходы

Экстрактор возвращает только `result.features_flat` (privacy-safe скаляры для NPZ export):

- `tp_descemb_present` (0/1) — эмбеддинг вычислен (не “файл существует”)
- `tp_descemb_dim`
- `tp_descemb_norm_raw` (NaN если `compute_raw_norm=false`)
- `tp_descemb_l2_norm`

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
    "cache_enabled": False,
    "cache_ttl_days": 30.0,
    "cache_max_items": 200000,
    "cache_max_bytes": 2000000000,
    "cache_cleanup_on_init": True,
    "cache_cleanup_max_seconds": 0.2,
    "device": "cpu",                                          # "cpu" | "cuda"
    "fp16": True,                                             # Использовать float16 на GPU
    "batch_size": 32,                                         # Размер батча для обработки чанков
    "artifacts_dir": None,                                    # Путь к артефактам (по умолчанию: default_artifacts_dir())
    "tokenizer_spec_name": "shared_tokenizer_v1",             # dp_models tokenizer spec
    "max_chunk_tokens_model": 512,                             # Максимум токенов модели в чанке
    "pooling_strategy": "length_weighted_mean",               # mean | length_weighted_mean | max | logsumexp
    "compute_embedding": True,                                 # считать ли эмбеддинг
    "write_artifact": True,                                    # писать ли `.npy` и регистрировать relpath
    "write_embedding_artifact": True,                          # (deprecated alias)
    "compute_raw_norm": True,                                 # считать ли raw norm (иначе NaN)
    "emit_extra_metrics": False                                # доп. метрики в features_flat
}
```

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








