## `embedding_stats_extractor` (Embedding Statistics Extractor)

### Назначение

Вычисляет статистические метрики по эмбеддингам чанков транскрипта: дисперсию эмбеддингов между чанками и энтропию topic distribution (если доступно). Используется для анализа вариативности представления текста и смешения тем.

**Версия**: 1.1.0  
**Категория**: embedding statistics  
**GPU**: не требуется (CPU-only)

### Входы

- **Chunk embeddings**:
  - **Canonical**: `doc.tp_artifacts["transcripts"][<source>]["chunk_embeddings_relpath"]`
  - **Legacy fallback**: `doc.tp_artifacts["transcript_chunks"][<source>]["embeddings_relpath"]` (ставится `tp_embstats_used_legacy_key_flag=1`)
  - per-run `.npy` в `text_processor/_artifacts/`, produced by `transcript_chunk_embedder`.
- **Topic distribution (optional)**: `doc.tp_artifacts["topics"]["topk_distribution"]["topic_probs"]`, produced in-memory by `semantics_topics_keyphrases` when `enable_topic_distribution=true`.

Важно:
- компонент **не делает** `glob/mtime` и не сканирует файловую систему;
- компонент **не читает** произвольные `*.json` артефакты (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`: JSON — только presentation layer).

### Выходы

Экстрактор возвращает `ExtractorResult` с payload, содержащим **только** `result.features_flat` (скаляры для NPZ).

#### Основные результаты

`features_flat`:

- `tp_embstats_present` (0/1): 1 только если найдено `n_chunks >= min_chunks_required`.
- `tp_embstats_l2_variance` (float): \(||var||_2\) по компонентам (NaN если empty).
- `tp_embstats_topvar_1..tp_embstats_topvar_<top_k_slots>` (float): fixed slots для top component variances (NaN если slot пуст).
- `tp_embstats_topic_entropy` (float): entropy по topic probs (NaN если topic probs отсутствуют/невалидны или выключено).
- `tp_embstats_topic_entropy_norm` (float): нормированная энтропия \(H/\log(K)\) (NaN если не применимо).
- `tp_embstats_topic_perplexity` (float): \(e^{H}\) (NaN если не применимо).
- `tp_embstats_topic_entropy_present` (0/1).
- `tp_embstats_topic_probs_present` (0/1) и `tp_embstats_topic_probs_invalid_flag` (0/1).
- `tp_embstats_source_used_<source>` (0/1): какой transcript source был использован (из `transcript_source_priority`).
- safety flags: `tp_embstats_unsafe_relpath_flag`, `tp_embstats_dim_mismatch_flag`, `tp_embstats_nan_inf_flag`.

Также всегда присутствуют (стабильная схема):
- `tp_embstats_enabled`, `tp_embstats_disabled_by_policy`
- `tp_embstats_require_chunks_enabled`
- `tp_embstats_compute_topic_entropy_enabled`, `tp_embstats_require_topic_distribution_enabled`
- `tp_embstats_n_chunks`, `tp_embstats_dim`
- `tp_embstats_load_ms`, `tp_embstats_compute_ms`

#### Метаданные

- `device`: устройство обработки (`"cpu"`)
- `version`: версия экстрактора

#### Системные метрики

- `system.pre_init`: снимок системы до инициализации
- `system.post_init`: снимок системы после инициализации
- `system.post_process`: снимок системы после обработки
- `system.peaks.ram_peak_mb`: пиковое использование RAM (MB)
- `system.peaks.gpu_peak_mb`: пиковое использование GPU памяти (MB, всегда 0)

#### Тайминги

- `timings_s.total`: общее время обработки (секунды)

#### Ошибки

- `error`: описание ошибки (если произошла) или `None`

### Алгоритм обработки

#### 1. Загрузка чанков эмбеддингов

**Приоритет источников**: настраивается через `transcript_source_priority` (по умолчанию `["whisper", "youtube_auto"]`).

**Процесс**:
- Берём `embeddings_relpath` из `doc.tp_artifacts["transcript_chunks"][source]`
- Загружаем `.npy` из `artifacts_dir/embeddings_relpath`
- Нормализация формы: если `ndim == 1`, преобразуется в `(1, D)`
- Конвертация в `float32`

#### 2. Вычисление дисперсии между чанками

**Формула**:
```
var_vec = np.var(chunks, axis=0, ddof=variance_ddof)  # Дисперсия по каждой компоненте
l2_variance = ||var_vec||_2       # L2-норма вектора дисперсий
topk_variances = sort(var_vec)[-topk:]  # Top-k наибольших дисперсий (экспортируется в fixed slots)
```

**Интерпретация**:
- Высокая `l2_variance` → большая вариативность представления между чанками
- `topk_variances` показывает, какие компоненты эмбеддинга наиболее вариативны

#### 3. Entropy по топикам (опционально)

Если `semantics_topics_keyphrases` включён и `enable_topic_distribution=true`, он запишет
`doc.tp_artifacts["topics"]["topk_distribution"]["topic_probs"]` (list[float]).

**Формула**:
```
entropy = -sum(p_i * log(p_i + eps))  # по top-K probs
```

**Интерпретация**:
- Высокая энтропия → равномерное смешение тем (много тем представлено)
- Низкая энтропия → доминирование одной/нескольких тем

### Конфигурация

```python
{
    "artifacts_dir": None,        # Путь к директории артефактов (по умолчанию: default_artifacts_dir())
    "transcript_source_priority": ["whisper", "youtube_auto"],
    "top_k_slots": 8,             # фиксированное число слотов tp_embstats_topvar_*
    "topk": 8,                    # сколько top-k реально выбирать (<= top_k_slots)
    "min_chunks_required": 2,     # минимум чанков для valid stats
    "variance_ddof": 0,           # ddof для np.var
    "enabled": True,              # feature-gating
    "require_chunks": False,      # если True и чанков нет/мало -> RuntimeError
    "compute_topic_entropy": True,
    "require_topic_distribution": False,  # fail-fast если topic_probs отсутствуют/невалидны
    "emit_extra_metrics": False,  # дополнительные метрики/наблюдаемость
}
```

**Параметры**:
- `artifacts_dir`: директория с артефактами эмбеддингов чанков
- `transcript_source_priority`: приоритет источников transcript chunk embeddings
- `top_k_slots`: фиксированное число слотов `tp_embstats_topvar_*` (стабильная схема)
- `topk`: сколько top-k реально выбирать
- `min_chunks_required`: минимум чанков для valid вычисления дисперсии
- `variance_ddof`: ddof для `np.var`
- `require_chunks`: fail-fast, если required input отсутствует
- `emit_extra_metrics`: включает дополнительные метрики/тайминги (см. выше)

### Особенности

- **Детерминизм**: входы берутся только через `doc.tp_artifacts`, без glob/mtime.
- **Valid empty**: если `n_chunks < min_chunks_required` → `tp_embstats_present=0` и NaN-метрики.
- **Опциональные топики**: topic entropy вычисляется только если `semantics_topics_keyphrases` предоставил `topic_probs` в `doc.tp_artifacts`.
- **Нормализация формы**: автоматическая обработка одномерных массивов

### Архитектура

1. **Инициализация**: установка путей к артефактам и кешу
2. **Выбор источника**: выбор transcript source по `transcript_source_priority` через `doc.tp_artifacts`
3. **Загрузка чанков**: загрузка и нормализация массива эмбеддингов
4. **Вычисление дисперсии**: расчёт L2-нормы дисперсии и top-k компонент
5. **Entropy по топикам (опционально)**: чтение top-K probs из `doc.tp_artifacts["topics"]`
6. **Вычисление энтропии**: расчёт энтропии top-K распределения
7. **Метрики**: сбор системных метрик и таймингов

### Обработка ошибок

- **Отсутствие/недостаточно чанков**:
  - default: `tp_embstats_present=0`, метрики NaN
  - если `require_chunks=true`: `RuntimeError`
- **Отсутствие топиков**: `tp_embstats_topic_entropy=NaN`, `tp_embstats_topic_entropy_present=0`
- **Невалидные topic_probs**: `tp_embstats_topic_probs_invalid_flag=1` и NaN метрики; fail-fast при `require_topic_distribution=true`
- **Ошибка загрузки файла**: игнорируется, пробуется следующий источник
- **Некорректный формат**: обрабатывается через try-except, возвращается None

### Performance characteristics

**Resource costs**:
- **CPU**: низкие (только numpy операции)
- **GPU**: не используется
- **Estimated duration**: ~0.01-0.05 секунд для типичного набора чанков

**Параметры производительности**:
- `topk`: не влияет на производительность (только размер результата)
- Размер массива: линейная сложность O(N*D) для вычисления дисперсии

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **transcript_chunk_embedder**: создаёт эмбеддинги чанков, используемые этим экстрактором
- **description_embedder**: может использовать похожие метаданные для топиков

### Примечания

1. **Зависимость от других экстракторов**: требует предварительного выполнения `transcript_chunk_embedder`
2. **Топики опциональны**: энтропия топиков вычисляется только если `semantics_topics_keyphrases` включён и отдал top-K probs.
3. **Интерпретация метрик**:
   - `l2_variance`: чем выше, тем больше вариативность представления между чанками
   - `topic_entropy`: чем выше, тем более равномерно распределены темы
4. **Формат чанков**: ожидается массив `(N, D)` где N - количество чанков, D - размерность эмбеддинга
5. **Privacy**: компонент не сохраняет raw текст и не пишет JSON артефакты.

### Примеры использования

**Базовое использование**:
```python
extractor = EmbeddingStatsExtractor()
result = extractor.extract(doc)
features = result["result"]["features_flat"]
```

**С кастомными путями**:
```python
extractor = EmbeddingStatsExtractor(
    artifacts_dir="/path/to/artifacts",
    transcript_source_priority=["whisper", "youtube_auto"],
    top_k_slots=8,
    topk=8,
    min_chunks_required=2,
    variance_ddof=0,
    require_chunks=False,
    emit_extra_metrics=False,
)
result = extractor.extract(doc)
```

### Выходные метрики

#### embedding_variance_across_chunks

- **l2_variance** (float | None): L2-норма вектора дисперсий по компонентам эмбеддинга
  - Высокое значение → большая вариативность между чанками
  - Низкое значение → стабильное представление между чанками
- **topk_variances** (list[float]): список top-k наибольших компонентных дисперсий
  - Показывает, какие компоненты эмбеддинга наиболее вариативны
  - Может использоваться для анализа важных измерений

#### embedding_topic_mix_entropy

- **topic_entropy** (float | None): энтропия top-K topic distribution
  - Высокое значение → равномерное смешение тем (много тем)
  - Низкое значение → доминирование одной/нескольких тем
  - None → топики не найдены или недоступны
- **error** (str | None): описание ошибки, если топики не найдены

