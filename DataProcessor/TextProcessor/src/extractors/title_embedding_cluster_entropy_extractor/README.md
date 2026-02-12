## `title_embedding_cluster_entropy_extractor` (Text features)

### Назначение

Вычисляет **энтропию распределения** эмбеддинга заголовка по кластерам (по общей таксономии `semantic_clusters_v1`). Компонент проецирует title embedding через PCA, считает cosine к centroid’ам, применяет softmax с температурой к top‑K и вычисляет энтропию (а также нормализованную энтропию и perplexity).

**Версия**: 1.2.0  
**Категория**: text, clustering, entropy  
**GPU**: не требуется (опционально FAISS, если доступен)

### Входы

- **Title embedding**: должен быть создан `title_embedder` и зарегистрирован в:
  - `doc.tp_artifacts["embeddings"]["title"]["relpath"]`
- **Кластера / PCA**: строго через `dp_models` (offline + reproducible):
  - spec: `semantic_clusters_v1` (по умолчанию)

### Выходы

Экстрактор возвращает `result.features_flat` (A-policy, только scalars) + `title_cluster_entropy_meta` (privacy-safe, без путей и raw).

#### `features_flat` (основные)

- `tp_titleclent_present` (0/1)
- `tp_titleclent_entropy_raw` (float, NaN если empty)
- `tp_titleclent_entropy_norm` (float, NaN если empty; \(H/\log(K)\))
- `tp_titleclent_perplexity` (float, NaN если empty; \(e^H\))
- `tp_titleclent_distinct_clusters_topk` (float; NaN если empty)
- `tp_titleclent_top_k_slots` (float; конфиг)
- `tp_titleclent_top_k_used` (float; фактический K = min(top_k_slots, n_clusters))
- `tp_titleclent_temperature` (float)
- `tp_titleclent_title_present` (0/1)
- `tp_titleclent_dim_mismatch_flag` (0/1)
- `tp_titleclent_backend_faiss` (0/1)

#### `title_cluster_entropy_meta`

- `clusters_spec_name`, `clusters_spec_version`
- `clusters_weights_digest`
- `cluster_db_version`
- `backend` (`faiss_ip` | `numpy_cosine`)
- опционально (если `export_topk_distribution=True`): `topk` (ids/probs/scores) — без raw текста

### Алгоритм

1. **Загрузка assets через `dp_models`**: PCA + centroids (fail-fast в `__init__`)
2. **Загрузка title embedding**: детерминированно через `doc.tp_artifacts["embeddings"]["title"]["relpath"]` (без `glob+mtime`)
3. **Проекция PCA**: `reduced = title @ pca` → L2 normalize
4. **Cosine к centroid’ам**: inner product на нормализованных векторах
5. **Top‑K**: берём top‑K центроидов (K=min(top_k_slots, n_clusters))
6. **Softmax( / temperature )** → вероятности
7. **Entropy/perplexity**: \(H=-\sum p\log p\), \(H_{norm}=H/\log(K)\), perplexity=\(e^H\)

### Конфигурация

```python
TitleEmbeddingClusterEntropyExtractor(
    artifacts_dir=None,
    clusters_spec_name="semantic_clusters_v1",
    top_k_slots=5,
    temperature=0.1,
    export_topk_distribution=False,
    require_title_embedding=False,
    use_faiss=True,
    require_faiss=False,
    emit_extra_metrics=False,
)
```

### Valid empty semantics (A-policy)

- Если title embedding отсутствует и `require_title_embedding=False` → `tp_titleclent_present=0` и метрики = NaN.
- Если `require_title_embedding=True` → отсутствие/несовместимость входа = `RuntimeError`.

### Интерпретация энтропии

- **Низкая энтропия** (~0.0-1.0): эмбеддинг близок к одному или нескольким кластерам, четкая принадлежность
- **Средняя энтропия** (~1.0-2.0): эмбеддинг находится между несколькими кластерами
- **Высокая энтропия** (~2.0+): эмбеддинг равномерно распределен между многими кластерами, неопределенность

Максимальная энтропия для K кластеров: `log(K)` (при равномерном распределении).

### Параметр температуры

Температура в softmax контролирует "остроту" распределения:
- **Низкая температура** (0.01-0.1): более острый пик, фокус на наиболее похожих кластерах
- **Высокая температура** (1.0+): более сглаженное распределение, больше неопределенности

### Особенности

- **Без glob/mtime**: используется только `doc.tp_artifacts` (детерминизм)
- **dp_models**: assets загружаются offline, `weights_digest` фиксируется в meta
- **L2-нормализация**: автоматическая нормализация для корректного косинусного сходства
- **Top‑K**: cost control и стабильный schema (`top_k_slots`)

### Обработка ошибок

- **Модели/asset не найдены в `dp_models`**: `RuntimeError` в `__init__` (no-fallback)
- **Отсутствует title embedding**:
  - `require_title_embedding=False` → valid empty (NaN + flags)
  - `require_title_embedding=True` → `RuntimeError`

### Архитектура

1. **Инициализация**: сохранение путей к кластерам и артефактам
2. **Ленивая загрузка центроидов**: загрузка при первом использовании с кешированием
3. **Поиск эмбеддинга**: поиск последнего файла `title_embedding_*.npy` в artifacts_dir
4. **Нормализация**: L2-нормализация эмбеддинга и центроидов
5. **Вычисление сходства**: матричное умножение для косинусного сходства
6. **Топ-K выборка**: выбор K наиболее похожих кластеров
7. **Softmax**: преобразование сходств в вероятности с температурой
8. **Энтропия**: вычисление энтропии Шеннона распределения

### Performance characteristics

**Resource costs**:
- **CPU**: минимальные (только матричные операции numpy)
- **RAM**: зависит от количества кластеров (~4KB на кластер для float32)
- **Estimated duration**: <0.01 секунды для типичных размеров

**Параметры производительности**:
- Время выполнения линейно зависит от количества кластеров
- Топ-K выборка ограничивает вычисления только релевантными кластерами
- Кеширование центроидов исключает повторную загрузку

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **title_embedder**: создает эмбеддинги заголовков (зависимость)
- **numpy**: работа с массивами и матричными операциями

### Примечания

1. **Зависимость от title_embedder**: требует, чтобы `title_embedder` был выполнен ранее
2. **Размерность эмбеддингов**: должна совпадать с размерностью центроидов
3. **Косинусное сходство**: используется для сравнения нормализованных векторов
4. **Температура**: низкие значения (0.1) подчеркивают различия между кластерами
5. **Топ-K**: анализ только топ-K кластеров повышает эффективность и фокусирует внимание на наиболее релевантных
6. **Энтропия как метрика неопределенности**: высокая энтропия может указывать на неопределенность классификации или межкластерное расположение
7. **distinct_clusters_topk**: показывает, сколько уникальных кластеров попало в топ-K (может быть меньше K при повторяющихся индексах, хотя это маловероятно)

### Примеры использования

**Низкая энтропия** (четкая принадлежность):
```python
# Эмбеддинг очень близок к одному кластеру
entropy ≈ 0.2
distinct_clusters_topk = 1
```

**Высокая энтропия** (неопределенность):
```python
# Эмбеддинг равномерно распределен между кластерами
entropy ≈ 1.5 (для top_k=5, максимум ≈ 1.61)
distinct_clusters_topk = 5
```

