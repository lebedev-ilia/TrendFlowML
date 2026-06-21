## `cosine_metrics_extractor` (Cosine Similarity Metrics)

### Назначение

Вычисляет метрики косинусного сходства между различными текстовыми эмбеддингами видео: заголовком, описанием, транскрипцией и комментариями. Загружает эмбеддинги из артефактов, созданных другими экстракторами.

**Версия**: 1.3.0  
**Категория**: similarity metrics  
**GPU**: не требуется

**Описание фич, диапазоны; валидатор среза в NPZ:** [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · `utils/validate_cosine_metrics_extractor_text_npz.py`

**Контракт Audit v3**: [SCHEMA.md](./SCHEMA.md) · machine: [`schemas/cosine_metrics_extractor_output_v1.json`](../../schemas/cosine_metrics_extractor_output_v1.json)  
**Audit v4:** [`../../../../docs/audit_v4/components/text_processor/cosine_metrics_extractor_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/cosine_metrics_extractor_audit_v4.md) · L2 stats: `scripts/audit_v4_npz_stats.py` → `storage/audit_v4/cosine_metrics_extractor_l2/`

### Входы

- **In-memory registry** (`VideoDocument.tp_artifacts`) — должен быть заполнен ранее в рамках этого же run:
  - `doc.tp_artifacts["embeddings"]["title"]["relpath"]` (создаёт `title_embedder`)
  - `doc.tp_artifacts["embeddings"]["description"]["relpath"]` (создаёт `description_embedder`)
  - **Транскрипт (агрегат mean, canonical)**: `doc.tp_artifacts["transcripts"][source]["agg_mean_relpath"]` (создаёт `transcript_aggregator`)
  - **Транскрипт (legacy alias)**: `doc.tp_artifacts["transcript_aggregates"][source]["agg_mean_relpath"]`
  - **Комментарии (по умолчанию)**: агрегаты комментов (создаёт `comments_aggregator`):
    - `doc.tp_artifacts["comments"]["agg_mean_relpath"]` (canonical)
    - `doc.tp_artifacts["comments"]["agg_median_relpath"]` (canonical)
    - **Legacy alias**: `doc.tp_artifacts["embeddings"]["comments_agg_mean"]["relpath"]`
    - **Legacy alias**: `doc.tp_artifacts["embeddings"]["comments_agg_median"]["relpath"]`
  - **Комментарии (опционально, матричный режим)**: `doc.tp_artifacts["embeddings"]["comments"]["relpath"]` (создаёт `comments_embedder`)
  - `transcript_source_priority`: настраиваемый приоритет (default `combined → whisper → youtube_auto`)

**Важно**: экстрактор больше не делает `glob+mtime` и не сканирует директории “в поисках последнего файла” — только детерминированные relpath внутри `text_processor/_artifacts/`.
Также включена защита от path traversal: relpath обязан оставаться внутри `artifacts_dir`.

### Выходы

Экстрактор возвращает `result.features_flat` (**39** фиксированных скаляров, см. JSON-схему):

- `tp_cos_title_desc`
- `tp_cos_title_transcript`
- `tp_cos_desc_transcript`
- `tp_cos_transcript_comments_mean`
- `tp_cos_transcript_comments_median`

Если входной эмбеддинг отсутствует (или косинус не может быть корректно посчитан), соответствующее значение будет `NaN`.

Дополнительно всегда присутствуют флаги присутствия входов:
- `tp_cos_title_present`, `tp_cos_desc_present`, `tp_cos_transcript_present`, `tp_cos_comments_present`

Feature-gating (какие метрики включены по конфигу):
- `tp_cos_title_desc_enabled`
- `tp_cos_title_transcript_enabled`
- `tp_cos_desc_transcript_enabled`
- `tp_cos_transcript_comments_mean_enabled`
- `tp_cos_transcript_comments_median_enabled`

Диагностика/качество входов:
- `tp_cos_dim_mismatch_flag` (1 если ловили несовпадение размерностей/ошибку математики)
- `tp_cos_pair_dim_mismatch_flag` (1 если проблема в title/desc/transcript парах)
- `tp_cos_tc_dim_mismatch_flag` (1 если проблема в transcript↔comments ветке)
- `tp_cos_zero_norm_flag` (1 если встретили вырожденный вектор с нормой ~0 → возвращаем NaN, а не 0.0)
- `tp_cos_unsafe_relpath_flag` (1 если входной relpath небезопасный/вне `artifacts_dir` → трактуем как missing)

- **`tp_cos_transcript_agg_source_whisper` / `youtube_auto` / `combined`**: какой источник выбран для **`agg_mean_relpath`** транскрипта (ровно один **`1.0`** или все нули).
- Зеркала политик **`tp_cos_require_*_enabled`**, **`tp_cos_emit_extra_metrics_enabled`**.
- **`tp_cos_load_ms`**, **`tp_cos_compute_ms`**, **`tp_cos_comments_mode_*`**, **`tp_cos_tc_*`**: тайминги и matrix-диагностика; при **`emit_extra_metrics=false`** тайминги и **`tp_cos_tc_n_comments_used`**, **`tp_cos_tc_sims_std`**, **`tp_cos_tc_sims_p95`** → **NaN** (ключи всегда присутствуют).

Privacy-safe empty причины (скаляры 0/1, только если они релевантны включённым метрикам):
- `tp_cos_empty_no_title`
- `tp_cos_empty_no_desc`
- `tp_cos_empty_no_transcript`
- `tp_cos_empty_no_comments`

### Алгоритмы

#### 1. Загрузка эмбеддингов

**Процесс**:
1. Берём relpath из `doc.tp_artifacts`
2. Загружаем `*.npy` из `artifacts_dir` (per-run `text_processor/_artifacts/`)
3. Приводим к `float32` и нормализуем форму (вектор/матрица)

#### 2. Вычисление косинусного сходства

**Формула**:
```
cosine_similarity(a, b) = (a · b) / (||a|| × ||b||)
```

**Процесс**:
1. **Проверка размерностей**: если размерности различаются → `NaN` + `*_dim_mismatch_flag=1`
2. **Проверка вырожденности**: если \(||a||\) или \(||b||\) слишком мал → `NaN` + `tp_cos_zero_norm_flag=1`
3. **Скалярное произведение**: считаем косинус через \(\frac{a \cdot b}{||a|| \cdot ||b||}\)

#### 3. Агрегация метрик комментариев

**Процесс**:
1. **Вычисление сходств**: косинусное сходство между каждым комментарием и транскрипцией
2. **Среднее**: `mean(similarities)`
3. **Медиана**: `median(similarities)`

**Важно**: в `comments_mode="matrix"` вырожденные строки (норма ~0) помечаются как `NaN` и исключаются из агрегатов через `nanmean/nanmedian`.

### Конфигурация

```python
{
    "artifacts_dir": None,                                    # Путь к директории артефактов (по умолчанию: default_artifacts_dir())
    "transcript_source_priority": ["whisper", "youtube_auto"],  # или добавить \"combined\" первым при необходимости
    "comments_mode": "aggregates",                            # "aggregates" (default) или "matrix"
    "compute_title_desc": True,                               # feature-gating для метрики title↔description
    "compute_title_transcript": True,                         # feature-gating для метрики title↔transcript
    "compute_desc_transcript": True,                          # feature-gating для метрики description↔transcript
    "compute_transcript_comments_mean": True,                # feature-gating для метрики transcript↔comments (mean)
    "compute_transcript_comments_median": True,               # feature-gating для метрики transcript↔comments (median)
    "require_any_metric": False,                             # fail-fast если все compute_* выключены
    "require_title": False,                                   # fail-fast если обязательный title embedding отсутствует
    "require_description": False,                             # fail-fast если обязательный description embedding отсутствует
    "require_transcript": False,                              # fail-fast если обязательный transcript embedding отсутствует
    "require_comments_for_tc": False,                        # fail-fast если включены transcript↔comments метрики, но комменты отсутствуют
    "emit_extra_metrics": False                               # включает дополнительные метрики/тайминги (дороже только в matrix режиме)
}
```

**Параметры**:
- `artifacts_dir`: директория для поиска файлов эмбеддингов
- `transcript_source_priority`: приоритет источников транскрипта (по умолчанию: `"combined,whisper,youtube_auto"`)
- `comments_mode`: режим работы с комментариями: `"aggregates"` (использует агрегаты) или `"matrix"` (использует матрицу эмбеддингов)
- `compute_*`: feature-gating для отдельных метрик (включает/выключает вычисление конкретных метрик)
- `require_*`: fail-fast политики (вызывают RuntimeError если обязательный вход отсутствует)
- `emit_extra_metrics`: включает дополнительные метрики (тайминги, статистики по комментариям в matrix режиме)

### Архитектура

1. **Инициализация**: установка пути к директории артефактов
2. **Загрузка заголовка**: relpath из `doc.tp_artifacts["embeddings"]["title"]["relpath"]`
3. **Загрузка описания**: relpath из `doc.tp_artifacts["embeddings"]["description"]["relpath"]`
4. **Загрузка транскрипции**: поиск по приоритетам (combined → whisper → youtube_auto)
5. **Загрузка комментариев**: relpath из `doc.tp_artifacts["embeddings"]["comments"]["relpath"]`
6. **Вычисление метрик**: попарное вычисление косинусного сходства для доступных пар
7. **Агрегация комментариев**: вычисление среднего и медианы сходств комментариев с транскрипцией
8. **Возврат результата**: возврат всех вычисленных метрик

### Обработка ошибок

- **Отсутствие relpath в `doc.tp_artifacts`**: соответствующие метрики становятся `NaN`
- **Файл не найден/повреждён**: соответствующие метрики становятся `NaN`
- **Пустые массивы**: если массив пустой, метрики не вычисляются
- **Несоответствие размерностей**: векторы нормализуются независимо, размерности должны совпадать
- **Вырожденные вектора (норма ~0)**: метрики становятся `NaN` (а не 0.0), `tp_cos_zero_norm_flag=1`
- **Unsafe relpath**: метрики становятся `NaN`, `tp_cos_unsafe_relpath_flag=1`

### Особенности

- **Детерминированная загрузка**: только relpath из `doc.tp_artifacts` (без `glob+mtime`)
- **Приоритеты транскрипции**: предпочтение combined транскрипции перед отдельными источниками
- **Условное вычисление**: метрики вычисляются только если оба эмбеддинга доступны
- **No fake metrics**: при вырожденных векторах возвращаем `NaN`, чтобы не “подделывать” валидное число
- **Агрегация**: статистики (mean, median) для множественных комментариев
- **Эффективность**: использование numpy операций для быстрого вычисления

### Performance characteristics

**Resource costs**:
- **CPU**: очень низкие (только numpy операции)
- **GPU**: не используется
- **Estimated duration**:
  - пары векторов: ~0.001-0.01s
  - `comments_mode="matrix"`: зависит от `n_comments` и размерности (обычно ~0.01-0.05s)

**Complexity (Big‑O)**:
- Пары (title/desc/transcript): \(O(d)\) на метрику
- Matrix режим transcript↔comments: \(O(n \cdot d)\), где \(n\) — число комментариев, \(d\) — размерность эмбеддинга

**Параметры производительности**:
- Количество комментариев: влияет на время вычисления метрик комментариев
- Размерность эмбеддингов: влияет на время матричных операций

### Зависимости

- `numpy`: численные операции (нормализация, матричное умножение)
- `pathlib`: работа с путями

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **Title/Description embedders**: создают эмбеддинги заголовка и описания
- **Transcript aggregators**: создают агрегированные эмбеддинги транскрипции
- **CommentsEmbedder**: создает эмбеддинги комментариев
- **path_utils.default_artifacts_dir**: путь к директории артефактов по умолчанию

### Примечания

1. **Зависимость от других экстракторов**: требует предварительного создания эмбеддингов другими экстракторами
3. **Нормализация**: все векторы нормализуются перед вычислением, даже если они уже нормализованы (защита)
4. **Размерности**: все эмбеддинги должны иметь одинаковую размерность для корректного вычисления
5. **Пустые результаты**: если эмбеддинги отсутствуют, возвращается пустой словарь метрик
6. **Транскрипция**: приоритет выбора: combined → whisper → youtube_auto (первый найденный)
7. **Комментарии**: если комментариев нет или матрица пустая, метрики комментариев не вычисляются
8. **Косинусное сходство**: значения в диапазоне [-1, 1], но для нормализованных эмбеддингов обычно [0, 1]

### Примеры интерпретации метрик

- **title_description_cosine > 0.7**: заголовок и описание семантически близки
- **title_transcript_cosine > 0.6**: заголовок соответствует содержанию видео
- **description_transcript_cosine > 0.6**: описание соответствует содержанию видео
- **transcript_comments_cosine_mean > 0.5**: комментарии в целом соответствуют содержанию видео
- **transcript_comments_cosine_median > 0.5**: большинство комментариев соответствуют содержанию

### Порядок выполнения экстракторов

Для корректной работы `CosineMetricsExtractor` должны быть выполнены:

1. Экстракторы эмбеддингов заголовка и описания
2. Экстракторы транскрипции и их агрегация
3. `CommentsEmbedder` для создания эмбеддингов комментариев
4. `CosineMetricsExtractor` (последний в цепочке)
---

## Навигация

[SCHEMA](SCHEMA.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
