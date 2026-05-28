## `comments_embedder` (Comments Embedding Extractor)

### Назначение

Извлекает L2-нормализованные эмбеддинги для комментариев видео с использованием sentence-transformers модели, **строго через `dp_models` (offline/no-network)**. Поддерживает батчинг, детерминированный отбор/лимиты, optional cache и per-run sub‑artifact.

**Версия**: 1.3.0  
**Категория**: text embedding  
**GPU**: опционально (если указан `device="cuda"`)

**Описание фич, core/extra, диапазоны; валидатор среза в NPZ:** [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · `utils/validate_comments_embedder_text_npz.py`

**Контракт `features_flat`**: [SCHEMA.md](SCHEMA.md) · machine: [`../../schemas/comments_embedder_output_v1.json`](../../schemas/comments_embedder_output_v1.json) · Audit v3: [`../../docs/audit_v3/components/comments_embedder_AUDIT_V3_REPORT.md`](../../docs/audit_v3/components/comments_embedder_AUDIT_V3_REPORT.md) · **Audit v4:** [`../../../../docs/audit_v4/components/text_processor/comments_embedder_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/comments_embedder_audit_v4.md) · L2 stats: `scripts/audit_v4_npz_stats.py` → `storage/audit_v4/comments_embedder_l2/`

### Входы

- **`VideoDocument`** с полем:
  - `comments`: список объектов комментариев с полем `text`

### Выходы

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Эмбеддинги комментариев

Эмбеддинги сохраняются в per-run `text_processor/_artifacts/` как **фиксированное имя**:
- `comments_embeddings.npy` (матрица `N×D`)

Абсолютные пути в result/NPZ не возвращаются.

Возвращаемые скалярные признаки — ровно **18** ключей `tp_commentsemb_*` в фиксированном порядке (см. JSON-схему).

**Core** (всегда осмысленные числа, не гейтятся `emit_extra_metrics`):  
`tp_commentsemb_present`, `tp_commentsemb_count`, `tp_commentsemb_dim`, `tp_commentsemb_n_input`, `tp_commentsemb_n_deduped`, `tp_commentsemb_n_selected`, `tp_commentsemb_total_chars_used`, `tp_commentsemb_truncated_by_total_chars_flag`.

**Extra** (при **`emit_extra_metrics=False`** — все **NaN**):  
`tp_commentsemb_cache_enabled`, `tp_commentsemb_cache_hit`, `tp_commentsemb_fp16`, `tp_commentsemb_device_cuda`, `tp_commentsemb_model_digest_u24`, `tp_commentsemb_compute_enabled`, `tp_commentsemb_write_artifact_enabled`, `tp_commentsemb_artifact_written`, `tp_commentsemb_select_ms`, `tp_commentsemb_encode_ms`.

Особые случаи:
- **`extract_batch`** + **`emit_extra_metrics=True`**: **`tp_commentsemb_cache_hit`** = **NaN** (единый encode, per-doc кеш не используется).
- **`extract_batch`**: **`tp_commentsemb_encode_ms`** и **`timings_s.encode`** — доля общего времени batch-encode, пропорциональная числу **закодированных** комментариев данного документа.

Для детерминированного доступа downstream‑экстракторами в рамках этого же run используется in-memory реестр:
- `doc.tp_artifacts["embeddings"]["comments"]["relpath"]` — путь к файлу с эмбеддингами
- `doc.tp_artifacts["comments"]["selected_indices_relpath"]` — путь к файлу с индексами выбранных комментариев (для выравнивания весов в `CommentsAggregationExtractor`)
- `doc.tp_artifacts["comments"]["selected_indices_count"]` — количество выбранных индексов

#### Метаданные

- `model_name`, `model_version`, `weights_digest` (верхний уровень payload)
- `device`: устройство обработки (`"cpu"` или `"cuda"`)
- `system.peaks.gpu_peak_mb`: max по снимкам GPU (как у других эмбеддеров)

### Алгоритмы

#### 1. Кодирование текстов

**Процесс**:
1. **Батчинг**: тексты разбиваются на батчи размером `batch_size`
2. **Кодирование**: каждый батч кодируется через `model.encode()`
3. **L2 нормализация**: каждый эмбеддинг нормализуется: `emb = emb / ||emb||`
4. **Объединение**: все батчи объединяются в единый массив `(N, D)`

**Параметры кодирования**:
- `show_progress_bar=False`: без прогресс-бара
- `convert_to_numpy=True`: конвертация в numpy массив
- `normalize_embeddings=False`: нормализация выполняется вручную после

#### 2. Сохранение артефактов

**Процесс**:
1. **Отбор/лимиты**: детерминированно выбираем до `max_comments` комментариев (по policy)
2. **Имя файла**: `text_processor/_artifacts/comments_embeddings.npy` (фиксированное per-run имя)
3. **Атомарное сохранение**: сохранение во временный файл `.tmp.npy`, затем переименование
4. **Сохранение индексов**: сохранение `comments_selected_indices.npy` с индексами исходных комментариев (для выравнивания весов в downstream экстракторах)
5. **Метаданные**: JSON sidecar `.meta.json` **не используется** (model meta идёт через `models_used`/manifest).

### Конфигурация

```python
{
    "model_name": "intfloat/multilingual-e5-large",           # Audit v3 preflight; через dp_models offline
    "cache_dir": None,                                       # default_cache_dir()/embed_cache
    "cache_enabled": False,                                  # по умолчанию OFF (чтобы не раздувать кэш)
    "cache_ttl_days": 7.0,
    "cache_max_items": 50000,
    "cache_max_bytes": 5000000000,
    "cache_cleanup_on_init": True,
    "cache_cleanup_max_seconds": 0.2,
    "artifacts_dir": None,                                    # Путь к директории артефактов
    "device": "cpu",                                          # "cpu" | "cuda"
    "fp16": True,                                             # Использование float16 (только для CUDA)
    "batch_size": 64,                                         # Размер батча для кодирования
    "max_comments": 200,                                      # лимит количества комментариев
    "max_total_chars": 20000,                                 # лимит суммарных символов после отбора (cost control)
    "max_chars_per_comment": 400,                             # лимит длины каждого комментария
    "min_chars_per_comment": 3,                               # фильтр слишком коротких
    "dedup_comments": True,                                   # dedup по normalize_whitespace(text)
    "selection_policy": "by_likes_then_recency",              # by_likes_then_recency | by_likes | by_recency | first_k
    "compute_embeddings": True,                               # считать ли эмбеддинги
    "write_artifact": True,                                   # писать ли `.npy` и регистрировать relpath
    "write_embedding_artifact": True,                         # (deprecated alias)
    "emit_extra_metrics": False                                # доп. метрики в features_flat
}
```

**Параметры**:
- `model_name`: имя модели из библиотеки sentence-transformers
- `artifacts_dir`: директория для сохранения эмбеддингов (по умолчанию: `default_artifacts_dir()`)
- `device`: устройство для обработки (`"cpu"` или `"cuda"`)
- `fp16`: использование float16 для экономии памяти (работает только на CUDA)
- `batch_size`: количество текстов в одном батче (больше = быстрее, но больше памяти)

### Архитектура

1. **Инициализация**: загрузка модели через `ModelRegistry` (переиспользование между экстракторами)
2. **Сбор комментариев**: извлечение текстов из `VideoDocument.comments`
3. **Нормализация**: применение `normalize_whitespace` к каждому комментарию
4. **Фильтрация**: удаление пустых комментариев и применение лимитов (`min_chars_per_comment`, `max_chars_per_comment`)
5. **Дедупликация**: удаление дубликатов (если `dedup_comments=true`)
6. **Отбор комментариев**: применение политики отбора (`selection_policy`) с учётом `max_comments` и `max_total_chars`
7. **Проверка наличия**: если комментариев нет, возвращается пустой результат
8. **Кодирование**: батчевое кодирование текстов через SentenceTransformer (с опциональным кешированием)
9. **L2 нормализация**: нормализация каждого эмбеддинга
10. **Сохранение**: атомарное сохранение массива в `.npy` файл и сохранение индексов выбранных комментариев
11. **Регистрация**: запись метаданных в `doc.tp_artifacts` для downstream экстракторов

### Обработка ошибок

- **Отсутствие комментариев**: valid empty (`tp_commentsemb_present=0`, артефакт не создаётся).
- **Ошибка сохранения артефакта**: RuntimeError (fail-fast, если компонент включён).
- **Ошибка модели**: обрабатывается на уровне ModelRegistry

### Особенности

- **ModelRegistry**: использует общий реестр моделей для переиспользования между экстракторами
- **Батчинг**: эффективная обработка больших наборов комментариев (поддерживается `extract_batch()` для обработки нескольких документов одновременно)
- **L2 нормализация**: все эмбеддинги нормализованы для использования в косинусной метрике
- **Атомарное сохранение**: использование временных файлов для безопасного сохранения
- **Хеширование**: SHA256 хеш для идентификации наборов комментариев (для кеширования)
- **Метаданные**: модель фиксируется через `model_version` и агрегируется в `manifest.json.models_used`
- **FP16 поддержка**: опциональное использование float16 на GPU для экономии памяти
- **Inference mode**: использование `torch.no_grad()` для экономии памяти
- **Индексы выравнивания**: сохранение индексов выбранных комментариев для выравнивания весов в downstream экстракторах
- **Batch processing**: поддержка `extract_batch()` для эффективной обработки нескольких документов (общий encode для всех комментариев)

### Performance characteristics

**Resource costs**:
- **CPU**: умеренные (зависит от модели и размера батча)
- **GPU**: опционально (если `device="cuda"`), значительно ускоряет обработку
- **Estimated duration**: 
  - CPU: ~0.5-2.0 секунд на 100 комментариев
  - GPU: ~0.1-0.5 секунд на 100 комментариев

**Параметры производительности**:
- `batch_size`: большие значения → быстрее, но больше памяти
- `fp16`: экономия памяти на GPU (~50%), минимальная потеря точности
- `device`: GPU ускоряет обработку в 5-10 раз для больших батчей

### Зависимости

- `sentence-transformers`: библиотека для работы с моделями эмбеддингов
- `torch`: PyTorch для выполнения моделей
- `numpy`: работа с массивами
- `hashlib`: генерация хешей
- `pathlib`: работа с путями

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **ModelRegistry**: реестр моделей для переиспользования
- **VideoDocument**: схема документа с комментариями
- **text_utils.normalize_whitespace**: нормализация текста
- **path_utils.default_artifacts_dir**: путь к директории артефактов по умолчанию
- **CommentsAggregationExtractor**: использует созданные эмбеддинги для агрегации

### Примечания

1. **Модель по умолчанию** (1.3.0): `intfloat/multilingual-e5-large` (1024-D; см. preflight)
2. **L2 нормализация**: все эмбеддинги нормализованы, поэтому косинусное сходство = скалярное произведение
3. **Хеширование**: хеш зависит от текстов и модели, поэтому изменения приведут к новому файлу
4. **Переиспользование моделей**: ModelRegistry загружает модель один раз и переиспользует между экстракторами
5. **FP16**: работает только на CUDA, на CPU игнорируется
6. **Батчинг**: автоматическое разбиение на батчи, последний батч может быть меньше
7. **Пустые комментарии**: автоматически фильтруются перед обработкой
8. **Метаданные модели**: `.meta.json` не используется (см. `manifest.json.models_used`)
9. **Batch processing**: экстрактор поддерживает `extract_batch()` для обработки нескольких документов одновременно (эффективнее для больших объёмов)
10. **Индексы выравнивания**: сохраняются `selected_indices` для выравнивания весов в `CommentsAggregationExtractor`

### Примеры использования

**Базовое использование**:
```python
extractor = CommentsEmbedder()
result = extractor.extract(video_doc)
# relpath для downstream доступен через doc.tp_artifacts["embeddings"]["comments"]["relpath"]
```

**С GPU и большим батчем**:
```python
extractor = CommentsEmbedder(
    device="cuda",
    batch_size=128,
    fp16=True
)
result = extractor.extract(video_doc)
```

**Кастомная модель**:
```python
extractor = CommentsEmbedder(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
result = extractor.extract(video_doc)
```

