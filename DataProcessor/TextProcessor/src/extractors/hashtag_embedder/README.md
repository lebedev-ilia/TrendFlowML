## `hashtag_embedder` (Hashtag Embedding Extractor)

### Назначение

Извлекает L2-нормализованный эмбеддинг для хештегов видео (агрегация по списку `doc.hashtags`) с использованием sentence-transformers модели, **строго через `dp_models` (offline/no-network)**.

Production‑политика:
- **Dependency (optional by default)**: читает `doc.hashtags` (обычно заполняется `TagsExtractor`). Можно сделать обязательным через `require_hashtags=true`.
- **No abs paths**: никаких абсолютных путей в `result`/NPZ; `.npy` хранится в per-run `text_processor/_artifacts/`, а relpath передаётся downstream через `doc.tp_artifacts`.
- **Determinism**: cache-key включает `weights_digest` + canonicalized tags + параметры агрегации (без glob/mtime).

**Версия**: 1.2.0  
**Категория**: text embedding  
**GPU**: опционально (если указан `device="cuda"`)

### Входы

- **`VideoDocument`** с полем:
  - `hashtags`: список строк с хештегами (list[str])

### Выходы

Экстрактор возвращает только `result.features_flat` (privacy-safe скаляры для NPZ export):
- `tp_hashemb_present` (0/1)
- `tp_hashemb_dim`
- `tp_hashemb_tag_count` (кол-во уникальных тэгов после canonicalization/limit)
- `tp_hashemb_l2_norm`

Для детерминированного доступа downstream‑экстракторами в рамках этого же run используется in-memory реестр (только если `write_artifact=true`):
`doc.tp_artifacts["embeddings"]["hashtag"]["relpath"]` (`hashtag_embedding.npy`).

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

#### 1. Сбор хештегов

- Извлечение поля `hashtags` из `VideoDocument`
- Проверка типа: должен быть `list` (иначе fail-fast при `require_hashtags=true`)
- Canonicalization: strip/ casefold / dedup / sort / truncate до `max_tags`
- Если после canonicalization список пустой → valid empty (`tp_hashemb_present=0`, артефакт не создаётся)

#### 2. Кодирование хештегов

**Процесс**:
1. **Батчинг**: хештеги разбиваются на батчи размером `batch_size`
2. **Кодирование**: каждый батч кодируется через `model.encode()`
3. **L2 нормализация**: каждый эмбеддинг нормализуется: `emb = emb / ||emb||`
4. **Объединение**: все батчи объединяются в единый массив `(N, D)`

**Параметры кодирования**:
- `show_progress_bar=False`: без прогресс-бара
- `convert_to_numpy=True`: конвертация в numpy массив
- `normalize_embeddings=False`: нормализация выполняется вручную после

#### 3. Усреднение эмбеддингов

**Формула**:
```
mean_vec = mean(embs, axis=0)  # Усреднение по всем хештегам
mean_vec = mean_vec / ||mean_vec||  # L2 нормализация
```

**Особенности**:
- Равномерное взвешивание: все хештеги имеют одинаковый вес
- Хештеги уникальны (из `TagsExtractor`), поэтому нет необходимости в частотном взвешивании

#### 4. Сохранение артефактов

Артефакт `.npy` сохраняется в per-run `text_processor/_artifacts/`:
- имя: `hashtag_embedding.npy` (фиксированное per-run имя; без hash в названии)
- запись атомарная (`.tmp.npy` → `replace`)

### Конфигурация

```python
{
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",  # Имя модели SentenceTransformer
    "cache_dir": None,                                       # default_cache_dir()/embed_cache
    "cache_enabled": False,
    "cache_ttl_days": 30.0,
    "cache_max_items": 200000,
    "cache_max_bytes": 2000000000,
    "cache_cleanup_on_init": True,
    "cache_cleanup_max_seconds": 0.2,
    "artifacts_dir": None,                                    # Путь к директории артефактов
    "device": "cpu",                                          # "cpu" | "cuda"
    "fp16": True,                                             # Использование float16 (только для CUDA)
    "batch_size": 128,                                        # Размер батча для кодирования
    "require_hashtags": False,                                # если doc.hashtags отсутствует/не list → RuntimeError
    "max_tags": 50,                                           # лимит уникальных тегов (после dedup/sort)
    "max_tag_len": 64,                                        # лимит длины одного тега
    "normalize_casefold": True,                               # casefold() для детерминизма
    "strip_hash_prefix": True,                                # убрать ведущий '#'
    "use_frequencies": False,                                 # учитывать дубликаты как веса (если есть)
    "aggregation": "mean",                                    # mean | max | logsumexp
    "compute_embedding": True,                                # feature-gating: считать ли эмбеддинг
    "write_artifact": True,                                   # feature-gating: писать ли `.npy` (и регистрировать relpath)
    "emit_extra_metrics": False                               # доп. метрики в features_flat
}
```

**Параметры**:
- `model_name`: имя модели из библиотеки sentence-transformers
- `artifacts_dir`: директория для сохранения эмбеддингов (по умолчанию: `default_artifacts_dir()`)
- `device`: устройство для обработки (`"cpu"` или `"cuda"`)
- `fp16`: использование float16 для экономии памяти (работает только на CUDA)
- `batch_size`: количество хештегов в одном батче (больше = быстрее, но больше памяти)

### Особенности

- **ModelRegistry**: использует общий реестр моделей для переиспользования между экстракторами
- **Батчинг**: эффективная обработка больших наборов хештегов
- **L2 нормализация**: финальный эмбеддинг нормализован для использования в косинусной метрике
- **Атомарное сохранение**: использование временных файлов для безопасного сохранения
- **Хеширование**: SHA256 хеш для идентификации наборов хештегов
- **Метаданные**: сохранение информации о модели для отслеживания версий
- **FP16 поддержка**: опциональное использование float16 на GPU для экономии памяти
- **Inference mode**: использование `torch.no_grad()` для экономии памяти
- **Равномерное взвешивание**: все хештеги имеют одинаковый вес при усреднении

### Архитектура

1. **Инициализация**: загрузка модели через `ModelRegistry` (переиспользование между экстракторами)
2. **Сбор хештегов**: извлечение списка хештегов из `VideoDocument.hashtags`
3. **Проверка наличия**: если хештегов нет, возвращается пустой результат
4. **Кодирование**: батчевое кодирование хештегов через SentenceTransformer
5. **L2 нормализация**: нормализация каждого эмбеддинга хештега
6. **Усреднение**: вычисление среднего арифметического всех эмбеддингов
7. **Финальная нормализация**: L2 нормализация усреднённого вектора
8. **Генерация хеша**: вычисление SHA256 хеша для идентификации набора
9. **Сохранение**: атомарное сохранение вектора в `.npy` файл
10. **Метаданные**: `.meta.json` sidecar **не используется** (per-run JSON запрещён; model meta идёт через `model_version` и `manifest.json.models_used`)
11. **Метрики**: сбор системных метрик и таймингов

### Обработка ошибок

- **Отсутствует `doc.hashtags`**: RuntimeError при `strict_missing_hashtags=true` (missing dependency).
- **Пустой список/после canonicalization пусто**: valid empty (не ошибка).
- **Ошибка сохранения артефакта**: RuntimeError (если компонент включён — должен отдать результат).

### Performance characteristics

**Resource costs**:
- **CPU**: умеренные (зависит от модели и размера батча)
- **GPU**: опционально (если `device="cuda"`), значительно ускоряет обработку
- **Estimated duration**: 
  - CPU: ~0.1-0.3 секунд на 50 хештегов
  - GPU: ~0.02-0.1 секунд на 50 хештегов

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
- **VideoDocument**: схема документа с хештегами
- **TagsExtractor**: создаёт список хештегов, используемый этим экстрактором

### Примечания

1. **Модель по умолчанию**: `all-MiniLM-L6-v2` - компактная модель с размерностью 384
2. **L2 нормализация**: финальный эмбеддинг нормализован, поэтому косинусное сходство = скалярное произведение
3. **Хеширование**: хеш зависит от хештегов и модели, поэтому изменения приведут к новому файлу
4. **Переиспользование моделей**: ModelRegistry загружает модель один раз и переиспользует между экстракторами
5. **FP16**: работает только на CUDA, на CPU игнорируется
6. **Батчинг**: автоматическое разбиение на батчи, последний батч может быть меньше
7. **Усреднение**: равномерное взвешивание всех хештегов (хештеги уникальны)
8. **Метаданные модели**: `.meta.json` не используется (см. `manifest.json.models_used`)

### Примеры использования

**Базовое использование (privacy-safe)**:

Экстрактор отдаёт только скаляры в `features_flat`, а relpath эмбеддинга (если записан) лежит в `doc.tp_artifacts`:

```python
extractor = HashtagEmbedder()
res = extractor.extract(video_doc)
features = res["result"]["features_flat"]
rel = video_doc.tp_artifacts["embeddings"]["hashtag"]["relpath"]  # "hashtag_embedding.npy"
```

**С GPU и большим батчем**:
```python
extractor = HashtagEmbedder(
    device="cuda",
    batch_size=256,
    fp16=True
)
result = extractor.extract(video_doc)
```

**Кастомная модель**:
```python
extractor = HashtagEmbedder(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
result = extractor.extract(video_doc)
```

**Загрузка сохранённого эмбеддинга**:
```python
result = extractor.extract(video_doc)
path = result["result"]["hashtag_embedding"]["path"]
embedding = np.load(path)  # Загрузка нормализованного эмбеддинга
```

### Выходные метрики

#### Примечание по abs paths

Абсолютные пути не возвращаются: downstream должен использовать relpath из `doc.tp_artifacts` и свой `artifacts_dir`.

