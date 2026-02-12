## `topk_similar_titles_extractor` (Similarity search)

### Назначение

Находит **Top-K наиболее похожих заголовков** из **статического корпуса** по эмбеддингу текущего заголовка видео.

Ключевые требования (production-grade):
- корпус загружается **строго через `dp_models` (offline, fail-fast)**;
- входной title embedding берётся детерминированно через `doc.tp_artifacts["embeddings"]["title"]["relpath"]` (без `glob+mtime`);
- missing corpus / missing title embedding / mismatch dim → **ошибка (fail-fast)**.

**Версия**: 1.2.0  
**Категория**: similarity search  
**GPU**: не требуется (FAISS работает на CPU)

### Входы

- **`doc`** (Any): документ (используется только для структуры, не требует специфических полей)

**Зависимости**:
- Эмбеддинг заголовка должен быть предварительно создан компонентом `title_embedder` и зарегистрирован в `doc.tp_artifacts["embeddings"]["title"]["relpath"]`.
- Корпус (`embeddings.npy` + `ids.json`) должен быть доступен в `dp_models` через `corpus_spec_name`.

### Выходы

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Основные результаты

- `topk_similar_corpus_titles`: словарь с результатами поиска
  - `corpus`: метаданные корпуса (версия/размерность/хэш)
  - `topk_similar_ids` и `topk_similar_scores` возвращаются **только если** включён `export_topk_mode` (см. конфиг) и с лимитом `max_export_k`

#### Скалярные признаки (`result.features_flat`)

Экстрактор всегда возвращает стабильный набор `tp_topktitles_*` (NaN + flags при empty):
- `tp_topktitles_present`
- `tp_topktitles_disabled_by_policy`, `tp_topktitles_enabled`
- `tp_topktitles_require_title_embedding_enabled`
- `tp_topktitles_k`, `tp_topktitles_dim`, `tp_topktitles_corpus_size`
- `tp_topktitles_backend_faiss`, `tp_topktitles_faiss_available`, `tp_topktitles_require_faiss_enabled`
- `tp_topktitles_export_k_used`, `tp_topktitles_export_k_truncated_flag`, `tp_topktitles_max_export_k`
- `tp_topktitles_top1_score`, `tp_topktitles_topk_mean_score`
- flags: `tp_topktitles_unsafe_relpath_flag`, `tp_topktitles_dim_mismatch_flag`, `tp_topktitles_zero_norm_flag`, `tp_topktitles_nan_inf_flag`

#### Метаданные

- `device`: устройство обработки (всегда `"cpu"`)
- `version`: версия экстрактора (`"1.0.0"`)
- `system`: системные метрики (pre_init, post_init, post_process, peaks)
- `timings_s.total`: общее время обработки (секунды)
- `error`: ошибка (если есть, иначе `None`)
  - в production-grade режиме missing dependencies → **исключение**, а не мягкий `error`-код.

### Алгоритм поиска

#### 1. FAISS (если доступен)

- **Индекс**: HNSW (Hierarchical Navigable Small World) для эффективного приближенного поиска
- **Параметры**: `hnsw_m`, `hnsw_ef_construction`, `hnsw_ef_search`
- **Метрика**: inner product на L2-нормализованных векторах (эквивалент cosine similarity)
- **Нормализация**: все векторы L2-нормализованы перед добавлением в индекс

#### 2. Numpy fallback (если FAISS недоступен)

- **Метод**: прямое вычисление cosine similarity через матричное умножение
- **Формула**: `similarity = (normalized_query @ normalized_corpus.T)`
- **Сортировка**: argsort по убыванию similarity

### Конфигурация

```python
{
    "corpus_spec_name": "similar_titles_corpus_v1",  # dp_models spec name (offline asset)
    "k": 5,                                          # Top-K
    "enabled": True,                                 # feature-gating
    "require_title_embedding": False,                 # если True и нет title embedding → ошибка (fail-fast)
    "export_topk_mode": "ids_and_scores",             # none | ids_only | ids_and_scores
    "max_export_k": 50,                               # лимит для UI/NPZ size
    "require_faiss": False,                           # если True и faiss недоступен → ошибка
    "require_faiss_above_corpus_size": 200_000,       # если corpus >= threshold и faiss недоступен → ошибка
    "allow_numpy_large_corpus": False,                # защита от случайного O(N·D) на больших корпусах
    "max_corpus_for_numpy": 100_000,                  # порог “большого” корпуса для numpy backend
    "hnsw_m": 32,
    "hnsw_ef_construction": 200,
    "hnsw_ef_search": 128,
    "cache_enabled": True,                            # process-level cache индекса/корпуса
    "cache_ttl_s": 3600.0,
    "cache_max_entries": 2
}
```

### Формат входных данных

**Corpus embeddings** (`embeddings.npy`, в dp_models):
- 2D numpy array, dtype: `float32`
- Shape: `[n_docs, embedding_dim]`
- Векторы должны быть L2-нормализованы (автоматически нормализуются при загрузке)

**Corpus ids** (`ids.json`, в dp_models):
- JSON список ID (любого типа: строки, числа, и т.д.)
- Длина списка должна точно совпадать с `n_docs` из embeddings

Пример:
```json
["video_123", "video_456", "video_789", ...]
```

### Поиск эмбеддинга заголовка

Экстрактор берёт title embedding детерминированно:
- `doc.tp_artifacts["embeddings"]["title"]["relpath"]`
- путь резолвится относительно per-run `artifacts_dir` (см. `TP_AUDIT_CRITERIA.md`).

Безопасность:
- relpath резолвится через safe-join (защита от path traversal); при подозрительных значениях выставляется `tp_topktitles_unsafe_relpath_flag`

### Особенности

- **FAISS поддержка**: использует эффективный HNSW индекс для быстрого поиска
- **Numpy fallback**: работает без FAISS (медленнее, но функционально)
- **L2-нормализация**: автоматическая нормализация всех векторов для cosine similarity
- **Детерминированность**: нет `glob+mtime`, нет global dirs, нет абсолютных путей в result
- **Гибкие ID**: поддерживает любые типы ID (строки, числа, и т.д.)
- **Метрики производительности**: отслеживание времени выполнения и использования памяти

### Обработка ошибок

В production-grade режиме missing dependencies → **исключение**:
- корпус не резолвится через `dp_models` (missing files / invalid spec)
- отсутствует `doc.tp_artifacts["embeddings"]["title"]["relpath"]`
- отсутствует файл title embedding в per-run artifacts
- mismatch размерности (corpus_dim != title_dim)
- `require_faiss=true`, но `faiss` недоступен

Valid empty semantics (если `require_title_embedding=false`):
- отсутствует/нечитаем title embedding → `tp_topktitles_present=0`, scores=NaN, списки не выдаются

### Архитектура

1. **Инициализация**: загрузка корпуса из `dp_models` (fail-fast)
2. **Построение индекса**: FAISS HNSW (inner product) или numpy backend
3. **Поиск эмбеддинга заголовка**: чтение relpath из `doc.tp_artifacts`
4. **Нормализация**: L2-нормализация эмбеддинга заголовка
5. **Поиск похожих**: выполнение поиска через FAISS или numpy
6. **Формирование результатов**: scalar summary всегда; списки ids/scores только при `export_topk_lists=true`
7. **Сбор метрик**: измерение времени выполнения и использования памяти

### Performance characteristics

**Resource costs**:
- **CPU**: умеренные (зависит от размера корпуса и метода поиска)
- **RAM**: зависит от размера корпуса (весь корпус загружается в память)
- **Estimated duration**: 
  - FAISS: ~0.001-0.01 секунд для корпуса до 1M документов
  - Numpy: ~0.01-1.0 секунд для корпуса до 100K документов (зависит от размера)

**Параметры производительности**:
- Размер корпуса: линейная зависимость памяти, логарифмическая зависимость времени поиска (FAISS HNSW)
- `k`: не влияет значительно на время поиска (FAISS)
- `k`: влияет на время сортировки (numpy fallback)
- Размерность эмбеддинга: влияет на память и время поиска линейно

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **title_embedder**: компонент, создающий эмбеддинг заголовка (зависимость)
- **FAISS** (опционально): библиотека для эффективного поиска похожих векторов
- **system_snapshot, process_memory_bytes**: утилиты для сбора метрик

### Примечания

1. **Зависимость от title_embedder**: эмбеддинг заголовка должен быть создан заранее
2. **Подготовка корпуса**: корпус — offline asset в `dp_models` (не создаётся этим компонентом)
3. **Размерность эмбеддинга**: должна совпадать между корпусом и эмбеддингом заголовка
4. **FAISS установка**: для лучшей производительности рекомендуется установить `faiss-cpu` или `faiss-gpu`
5. **Нормализация**: все векторы автоматически нормализуются, поэтому cosine similarity эквивалентна inner product
6. **HNSW параметры**: `hnsw_ef_search` — основной рычаг качества/latency на запрос
7. **Scores**: значения в диапазоне [-1, 1] для cosine similarity (после нормализации часто [0, 1])
8. **Кэш**: индекс/корпус кэшируется в процессе по ключу `(spec+weights_digest+backend+hnsw params)` с TTL и max_entries

