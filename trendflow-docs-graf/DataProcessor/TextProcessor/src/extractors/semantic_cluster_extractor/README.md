## `semantic_cluster_extractor` (Semantic Cluster Classifier)

### Назначение

Определяет семантический кластер для видео на основе его эмбеддингов (заголовка, описания или хештегов). Использует предобученные центроиды кластеров и PCA для снижения размерности. Вычисляет ближайший кластер по косинусному сходству и возвращает его ID, сходство и расстояние.

**Версия**: 1.3.0  
**Категория**: clustering, classification  
**GPU**: не требуется (опционально FAISS для ускорения)

**Диапазоны и валидатор среза** (`text_features.npz`): [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · [`utils/validate_semantic_cluster_extractor_text_npz.py`](utils/validate_semantic_cluster_extractor_text_npz.py)

**Контракт Audit v3**: [SCHEMA.md](./SCHEMA.md) · machine: [`schemas/semantic_cluster_extractor_output_v1.json`](../../schemas/semantic_cluster_extractor_output_v1.json) · **Audit v4:** [`../../../../docs/audit_v4/components/text_processor/semantic_cluster_extractor_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/semantic_cluster_extractor_audit_v4.md) · L2 stats: `scripts/audit_v4_npz_stats.py` → `storage/audit_v4/semantic_cluster_extractor_l2/`

### Входы

- **Эмбеддинги** (должны быть созданы соответствующими экстракторами и зарегистрированы в `doc.tp_artifacts["embeddings"]`):
  - `title_embedder` → `doc.tp_artifacts["embeddings"]["title"]["relpath"]`
  - `description_embedder` → `doc.tp_artifacts["embeddings"]["description"]["relpath"]`
  - `hashtag_embedder` → `doc.tp_artifacts["embeddings"]["hashtag"]["relpath"]`

- **Модели / assets** (строго через `dp_models`, offline + no-network):
  - spec: `semantic_clusters_v1` (по умолчанию)
  - assets:
    - `pca.npy`: матрица PCA (orig_dim, reduced_dim)
    - `centroids.npy`: матрица центроидов (n_clusters, reduced_dim)
    - `clusters.jsonl`: словарь кластеров (id → name/group) для UI/интерпретации

### Выходы

**31** фиксированных scalar в `result.features_flat` — см. [SCHEMA.md](./SCHEMA.md). Кратко: зеркала **`require_*` / `use_faiss` / `emit_extra_metrics`**, one-hot **`tp_semclust_config_primary_*`**, **`tp_semclust_*_present`** (успешная загрузка `.npy`), one-hot **`tp_semclust_source_*`**, **`fallback`**, **`backend_faiss`**, флаги **dim / unsafe / embed_missing**, метрики **id/similarity/distance**, блок **extra** (частично **NaN** при **`emit_extra_metrics=False`**).

#### `semantic_cluster_meta` (метаданные, без путей/текста)

Всегда (все ветки): `clusters_spec_name`, `clusters_spec_version`, `clusters_weights_digest`, `cluster_db_version`, **`backend`** (`faiss_ip` | `numpy_cosine`).

Верхний уровень ответа: **`model_*`/`weights_digest`** = **`null`**; **`system.pre_init`/`post_init`** из **`__init__`**, **`gpu_peak_mb`**.

#### Метаданные

- `device`: устройство обработки (всегда `"cpu"`)
- `version`: версия экстрактора

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

#### 1. Загрузка моделей

**Строго через `dp_models`**:
- `ModelManager.resolve(spec)` валидирует локальные файлы и даёт `weights_digest` для воспроизводимости.
- Если assets отсутствуют/невалидны → **`RuntimeError` в `__init__`** (no-fallback).

**FAISS**:
- включается через `use_faiss`
- если `require_faiss=True` и faiss недоступен → `RuntimeError`

#### 2. Выбор эмбеддинга

**Политика источника**:
- `primary_source`: `"title" | "description" | "hashtag"`
- `allow_fallback_sources`: список fallback источников (если `require_primary_source=False`)
- `require_primary_source=True`: отключает fallback полностью

**Процесс**:
1. Детерминированный выбор эмбеддинга через `doc.tp_artifacts["embeddings"]` (без `glob+mtime`)
2. Загрузка и преобразование в `float32`
3. Reshape в одномерный вектор `(-1,)`

**Valid empty (A-policy)**:
- если embedding отсутствует (и `require_embedding=False`) → `tp_semclust_present=0` и метрики = NaN (без “fake” векторов)

#### 3. Проекция через PCA

**Процесс**:
1. Применение PCA: `reduced = vec @ self._pca`
2. Reshape: `reduced.reshape(1, -1)` для матричной операции
3. L2-нормализация: `reduced = reduced / ||reduced||`

**Формула**:
```
reduced = (vec @ PCA) / ||vec @ PCA||
```

#### 4. Поиск ближайшего кластера

**С использованием FAISS** (если доступен):
```python
scores, idx = self._faiss_index.search(reduced.astype("float32"), 1)
sim = scores[0, 0]
cid = idx[0, 0]
```

**Без FAISS** (fallback):
```python
sims = (reduced @ self._centroids.T).reshape(-1)
cid = np.argmax(sims)
sim = sims[cid]
```

**Вычисление расстояния**:
```python
dist = 1.0 - sim
```

Примечание: компонент использует только **nearest-centroid** классификацию (фиксированная таксономия), без HDBSCAN.

### Конфигурация

```python
{
    "artifacts_dir": None,                   # директория per-run sub-artifacts (где лежат embeddings.npy)
    "clusters_spec_name": "semantic_clusters_v1",
    "primary_source": "title",               # title|description|hashtag
    "allow_fallback_sources": None,          # list[str] или None (дефолт зависит от primary_source)
    "require_primary_source": False,
    "require_embedding": False,
    "use_faiss": True,
    "require_faiss": False,
    "emit_extra_metrics": False
}
```

**Параметры**:
- `clusters_spec_name`: dp_models spec с PCA/centroids/dictionary (offline + reproducible)
- `primary_source`: предпочтительный источник эмбеддинга
- `allow_fallback_sources`: какие fallback источники можно использовать
- `require_primary_source`: запретить fallback
- `require_embedding`: сделать отсутствие/несовместимость embedding ошибкой (fail-fast)
- `use_faiss` / `require_faiss`: политика FAISS backend
- `emit_extra_metrics`: доп. наблюдаемость (margin/top2, dims, timings). При включении добавляет:
  - `tp_semclust_margin_top2`: разница между сходством с топ-1 и топ-2 кластерами
  - `tp_semclust_compute_ms`: время вычисления кластера в миллисекундах
  - `tp_semclust_n_clusters`: количество кластеров в таксономии
  - `tp_semclust_model_orig_dim`: исходная размерность модели
  - `tp_semclust_model_reduced_dim`: приведенная размерность после PCA
  - `tp_semclust_embedding_dim`: размерность входного эмбеддинга

### Особенности

- **Множественные источники**: поддержка эмбеддингов заголовка, описания и хештегов
- **Fallback механизм**: управляемый выбор альтернативного источника (флагами `require_primary_source` / `allow_fallback_sources`)
- **PCA снижение размерности**: эффективная работа с большими эмбеддингами
- **FAISS ускорение**: опциональное использование FAISS для быстрого поиска
- **L2 нормализация**: все векторы нормализованы для косинусной метрики
- **Косинусное сходство**: использование inner product для нормализованных векторов
- **Гибкая конфигурация**: настройка политики источника + dp_models spec

### Архитектура

1. **Инициализация**: загрузка PCA и центроидов, создание FAISS индекса (если доступен)
2. **Выбор эмбеддинга**: поиск и загрузка эмбеддинга по приоритетам
3. **Проверка предварительных условий**: проверка наличия моделей и эмбеддинга
4. **Проекция PCA**: снижение размерности и нормализация
5. **Поиск кластера**: поиск ближайшего центроида (FAISS или numpy)
6. **Вычисление метрик**: вычисление сходства и расстояния
7. **Возврат результата**: возврат ID кластера и метрик

### Обработка ошибок

- **Модели/asset не найдены в `dp_models`**: `RuntimeError` в `__init__` (no-fallback)
- **Эмбеддинг отсутствует**:
  - `require_embedding=False` → valid empty (`present=0`, NaN метрики)
  - `require_embedding=True` → `RuntimeError`
- **Dim mismatch**:
  - `require_embedding=False` → valid empty + `tp_semclust_dim_mismatch_flag=1`
  - `require_embedding=True` → `RuntimeError`

### Performance characteristics

**Resource costs**:
- **CPU**: низкие (numpy операции, опционально FAISS)
- **GPU**: не используется
- **Estimated duration**: ~0.01-0.05 секунд

**Параметры производительности**:
- Количество кластеров: влияет на время поиска (линейно без FAISS, логарифмически с FAISS)
- Размерность PCA: влияет на время матричных операций
- FAISS: значительно ускоряет поиск для большого количества кластеров

### Зависимости

- `numpy`: численные операции (матричное умножение, нормализация)
- `faiss` (опционально): быстрый поиск ближайших соседей
- `VideoDocument.tp_artifacts`: in-memory registry для линковки per-run артефактов между extractor’ами
- `pathlib`: работа с путями

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **Title/Description/Hashtag embedders**: создают эмбеддинги для кластеризации
- **path_utils.default_artifacts_dir**: путь к директории артефактов по умолчанию

### Примечания

1. **Это таксономия**: кластера — фиксированные центроиды (стабильные id) + словарь `clusters.jsonl`.
2. **Размерности**: embedding dim должен совпадать с `pca.orig_dim`.
3. **Нормализация**: projected vector и centroids L2-нормализованы → cosine = inner product.
4. **FAISS**: опциональная зависимость, numpy backend разрешён (если `require_faiss=False`).

### Примеры интерпретации результатов

- **semantic_cluster_id = 5**: видео принадлежит кластеру #5
- **semantic_cluster_similarity = 0.85**: высокое сходство с центроидом кластера
- **semantic_cluster_distance = 0.15**: небольшое расстояние до центроида
- **semantic_cluster_similarity < 0.5**: низкое сходство, возможно пограничный случай

### Порядок выполнения экстракторов

Для корректной работы `SemanticClusterExtractor` должны быть выполнены:

1. Экстракторы эмбеддингов (title/description/hashtag embedders)
2. `SemanticClusterExtractor` (использует сохранённые эмбеддинги)

### Требования к моделям

**PCA модель** (`pca.npy`):
- Форма: `(orig_dim, reduced_dim)`
- Тип: `float32`
- Пример: `(384, 128)` для снижения размерности с 384 до 128

**Центроиды** (`centroids.npy`):
- Форма: `(n_clusters, reduced_dim)`
- Тип: `float32`
- Должны быть L2-нормализованы (или будут нормализованы автоматически)
- Пример: `(100, 128)` для 100 кластеров размерности 128
---

## Навигация

[SCHEMA](SCHEMA.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
