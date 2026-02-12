## `comments_embedder` (Comments Embedding Extractor)

### Назначение

Извлекает L2-нормализованные эмбеддинги для комментариев видео с использованием sentence-transformers модели, **строго через `dp_models` (offline/no-network)**. Поддерживает батчинг, детерминированный отбор/лимиты, optional cache и per-run sub‑artifact.

**Версия**: 1.2.0  
**Категория**: text embedding  
**GPU**: опционально (если указан `device="cuda"`)

### Входы

- **`VideoDocument`** с полем:
  - `comments`: список объектов комментариев с полем `text`

### Выходы

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Эмбеддинги комментариев

Эмбеддинги сохраняются в per-run `text_processor/_artifacts/` как **фиксированное имя**:
- `comments_embeddings.npy` (матрица `N×D`)

Абсолютные пути в result/NPZ не возвращаются.

Возвращаемые скалярные признаки (`result.features_flat`):
- `tp_commentsemb_present` (0/1)
- `tp_commentsemb_count`
- `tp_commentsemb_dim`

Для детерминированного доступа downstream‑экстракторами в рамках этого же run используется in-memory реестр:
`doc.tp_artifacts["embeddings"]["comments"]["relpath"]`.

#### Метаданные

- `model_name`, `model_version`, `weights_digest`
- `device`: устройство обработки (`"cpu"` или `"cuda"`)

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
4. **Метаданные**: JSON sidecar `.meta.json` **не используется** (model meta идёт через `models_used`/manifest).

### Конфигурация

```python
{
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",  # Имя модели SentenceTransformer
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
4. **Фильтрация**: удаление пустых комментариев
5. **Проверка наличия**: если комментариев нет, возвращается пустой результат
6. **Кодирование**: батчевое кодирование текстов через SentenceTransformer
7. **L2 нормализация**: нормализация каждого эмбеддинга
8. **Генерация хеша**: вычисление SHA256 хеша для идентификации набора
9. **Сохранение**: атомарное сохранение массива в `.npy` файл
10. **Метаданные**: `.meta.json` не создаётся (запрещён per-run JSON рядом с артефактами)

### Обработка ошибок

- **Отсутствие комментариев**: valid empty (`tp_commentsemb_present=0`, артефакт не создаётся).
- **Ошибка сохранения артефакта**: RuntimeError (fail-fast, если компонент включён).
- **Ошибка модели**: обрабатывается на уровне ModelRegistry

### Особенности

- **ModelRegistry**: использует общий реестр моделей для переиспользования между экстракторами
- **Батчинг**: эффективная обработка больших наборов комментариев
- **L2 нормализация**: все эмбеддинги нормализованы для использования в косинусной метрике
- **Атомарное сохранение**: использование временных файлов для безопасного сохранения
- **Хеширование**: SHA256 хеш для идентификации наборов комментариев
- **Метаданные**: модель фиксируется через `model_version` и агрегируется в `manifest.json.models_used`
- **FP16 поддержка**: опциональное использование float16 на GPU для экономии памяти
- **Inference mode**: использование `torch.no_grad()` для экономии памяти

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

1. **Модель по умолчанию**: `all-MiniLM-L6-v2` - компактная модель с размерностью 384
2. **L2 нормализация**: все эмбеддинги нормализованы, поэтому косинусное сходство = скалярное произведение
3. **Хеширование**: хеш зависит от текстов и модели, поэтому изменения приведут к новому файлу
4. **Переиспользование моделей**: ModelRegistry загружает модель один раз и переиспользует между экстракторами
5. **FP16**: работает только на CUDA, на CPU игнорируется
6. **Батчинг**: автоматическое разбиение на батчи, последний батч может быть меньше
7. **Пустые комментарии**: автоматически фильтруются перед обработкой
8. **Метаданные модели**: `.meta.json` не используется (см. `manifest.json.models_used`)

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

