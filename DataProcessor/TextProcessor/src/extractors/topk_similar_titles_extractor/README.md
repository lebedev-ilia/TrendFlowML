## `topk_similar_titles_extractor` (Similarity search)

**Версия**: 1.3.0  
**Контракт Audit v3**: [SCHEMA.md](./SCHEMA.md) · machine: [`schemas/topk_similar_titles_extractor_output_v1.json`](../../schemas/topk_similar_titles_extractor_output_v1.json) · **Audit v4:** [`../../../../docs/audit_v4/components/text_processor/topk_similar_titles_extractor_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/topk_similar_titles_extractor_audit_v4.md) · **L2 stats:** [`../../../../storage/audit_v4/topk_similar_titles_extractor_l2/topk_similar_titles_extractor_audit_v4_stats.json`](../../../../storage/audit_v4/topk_similar_titles_extractor_l2/topk_similar_titles_extractor_audit_v4_stats.json) (tooling: `scripts/audit_v4_npz_stats.py`)

**Диапазоны и валидатор среза** (`text_features.npz`): [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · [`utils/validate_topk_similar_titles_extractor_text_npz.py`](utils/validate_topk_similar_titles_extractor_text_npz.py)

### Назначение

Находит **Top-K наиболее похожих заголовков** из **статического корпуса** по эмбеддингу текущего заголовка видео.

Ключевые требования:
- корпус загружается **строго через `dp_models` (offline, fail-fast)**;
- входной title embedding берётся детерминированно через `doc.tp_artifacts["embeddings"]["title"]["relpath"]` (без `glob+mtime`);
- по умолчанию **`require_title_embedding=false`**: отсутствие relpath/файла/ошибка чтения → **valid empty** + флаги (`tp_topktitles_title_embed_missing_flag` и др.); при **`require_title_embedding=true`** — **fail-fast** на тех же условиях.

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
  - `corpus`: метаданные корпуса (corpus_spec_name, corpus_version, corpus_weights_digest, id_kind, corpus_size, dim, backend, hnsw параметры)
  - `topk_similar_ids` и `topk_similar_scores` возвращаются **только если** включён `export_topk_mode` (см. конфиг) и с лимитом `max_export_k`

#### Скалярные признаки (`result.features_flat`)

Экстрактор всегда возвращает стабильный набор `tp_topktitles_*` (NaN + flags при empty):
- `tp_topktitles_present`: флаг наличия результатов
- `tp_topktitles_disabled_by_policy`, `tp_topktitles_enabled`: флаги конфигурации
- `tp_topktitles_require_title_embedding_enabled`: флаг require_title_embedding
- `tp_topktitles_k`: значение k (Top-K)
- `tp_topktitles_dim`: размерность эмбеддингов корпуса
- `tp_topktitles_corpus_size`: размер корпуса
- `tp_topktitles_backend_faiss`: флаг использования FAISS backend (1.0 если используется)
- `tp_topktitles_faiss_available`: флаг доступности FAISS (1.0 если установлен)
- `tp_topktitles_require_faiss_enabled`, `tp_topktitles_require_faiss_above_corpus_size`: флаги конфигурации FAISS
- `tp_topktitles_allow_numpy_large_corpus_enabled`, `tp_topktitles_max_corpus_for_numpy`: флаги конфигурации numpy backend
- `tp_topktitles_cache_enabled`, `tp_topktitles_cache_ttl_s`, `tp_topktitles_cache_max_entries`: флаги конфигурации кеша
- `tp_topktitles_export_topk_mode_ids_only`, `tp_topktitles_export_topk_mode_ids_and_scores`, `tp_topktitles_export_topk_mode_none`: флаги режима экспорта
- `tp_topktitles_export_k_used`: фактическое количество экспортированных результатов
- `tp_topktitles_export_k_truncated_flag`: флаг обрезки результатов (1.0 если k > max_export_k)
- `tp_topktitles_max_export_k`: максимальное количество экспортируемых результатов
- `tp_topktitles_top1_score`: score первого результата (cosine similarity)
- `tp_topktitles_topk_mean_score`: средний score топ-k результатов
- flags: `tp_topktitles_unsafe_relpath_flag`, `tp_topktitles_title_embed_missing_flag` (нет файла / ошибка загрузки `.npy`), `tp_topktitles_dim_mismatch_flag`, `tp_topktitles_zero_norm_flag`, `tp_topktitles_nan_inf_flag`

#### Метаданные

- `device`: устройство обработки (всегда `"cpu"`)
- `version`: версия экстрактора
- `model_name` / `model_version` / `weights_digest`: **`null`** (корпус не ST-модель в рантайме)
- `system`: **`pre_init`/`post_init`** из **`__init__`** (после загрузки корпуса), **`post_process`**, peaks (**`gpu_peak_mb`**)
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
    "require_title_embedding": False,                # если True и нет title embedding → ошибка (fail-fast)
    "export_topk_mode": "ids_and_scores",             # none | ids_only | ids_and_scores
    "max_export_k": 50,                              # лимит для UI/NPZ size
    "require_faiss": False,                           # если True и faiss недоступен → ошибка
    "require_faiss_above_corpus_size": 200_000,      # если corpus >= threshold и faiss недоступен → ошибка
    "allow_numpy_large_corpus": False,                # защита от случайного O(N·D) на больших корпусах
    "max_corpus_for_numpy": 100_000,                 # порог "большого" корпуса для numpy backend
    "hnsw_m": 32,                                    # HNSW параметр: количество связей на уровне
    "hnsw_ef_construction": 200,                      # HNSW параметр: размер кандидатов при построении
    "hnsw_ef_search": 128,                            # HNSW параметр: размер кандидатов при поиске
    "cache_enabled": True,                           # process-level cache индекса/корпуса
    "cache_ttl_s": 3600.0,                           # TTL кеша в секундах
    "cache_max_entries": 2,                           # Максимальное количество записей в кеше (LRU)
    "artifacts_dir": None                            # Путь к артефактам (по умолчанию: из env)
}
```

### Формат входных данных

**Corpus spec** (dp_models):
- Должен иметь `runtime_params` с:
  - `embeddings_relpath`: относительный путь к файлу embeddings.npy
  - `ids_relpath`: относительный путь к файлу ids.json
  - `id_kind` (опционально): строка, описывающая тип ID (для метаданных)

**Corpus embeddings** (`embeddings.npy`, в dp_models):
- 2D numpy array, dtype: `float32`
- Shape: `[n_docs, embedding_dim]`
- Векторы должны быть L2-нормализованы (автоматически нормализуются при загрузке)
- Не должны содержать NaN/inf

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

**Корпус** в `__init__`: не резолвится / битые файлы spec → **RuntimeError** (fail-fast при старте экстрактора).

**Title embedding** в `extract()`:
- при **`require_title_embedding=true`**: нет relpath / unsafe relpath / нет файла / ошибка `np.load` / NaN / dim mismatch / zero norm → **RuntimeError**;
- при **`require_title_embedding=false`**: те же случаи → **valid empty** (`tp_topktitles_present=0`, scores **NaN**, флаги **`tp_topktitles_*`**), без исключения.

**FAISS**: `require_faiss=true`, но пакет недоступен → **RuntimeError** при загрузке корпуса (если выбран faiss path по размеру).

**HNSW**: приближённый top-K; для **точного** порядка/скоров на малом корпусе сравнивайте с numpy backend (см. `SCHEMA.md`).

### Архитектура

1. **Инициализация**: загрузка корпуса из `dp_models` (fail-fast)
2. **Построение индекса**: FAISS HNSW (inner product) или numpy backend
3. **Поиск эмбеддинга заголовка**: чтение relpath из `doc.tp_artifacts`
4. **Нормализация**: L2-нормализация эмбеддинга заголовка
5. **Поиск похожих**: выполнение поиска через FAISS или numpy
6. **Формирование результатов**: `features_flat` всегда; списки ids/scores только при `export_topk_mode` ≠ `none` (и в пределах `max_export_k`)
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

