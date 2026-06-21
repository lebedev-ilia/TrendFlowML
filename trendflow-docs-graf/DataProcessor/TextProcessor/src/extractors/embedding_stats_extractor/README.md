## `embedding_stats_extractor` (Embedding Statistics Extractor)

### Назначение

Вычисляет статистические метрики **только по матрице эмбеддингов чанков транскрипта** (дисперсия между чанками, top component variances) и **опционально** энтропию по `topic_probs`, уже посчитанным upstream (`semantics_topics_keyphrases`). Title/description/comments здесь **не** используются.

**Версия**: 1.2.0  
**Категория**: embedding statistics  
**GPU**: не требуется (CPU-only)

**Контракт Audit v3**: [SCHEMA.md](./SCHEMA.md) · machine: [`schemas/embedding_stats_extractor_output_v1.json`](../../schemas/embedding_stats_extractor_output_v1.json)  
**Диапазоны и валидатор среза** (`text_features.npz`): [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · [`utils/validate_embedding_stats_extractor_text_npz.py`](utils/validate_embedding_stats_extractor_text_npz.py)  
**Audit v4:** [`../../../../docs/audit_v4/components/text_processor/embedding_stats_extractor_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/embedding_stats_extractor_audit_v4.md) · L2 stats: `scripts/audit_v4_npz_stats.py` → `storage/audit_v4/embedding_stats_extractor_l2/`

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
- `tp_embstats_topvar_1` … `tp_embstats_topvar_8` (float): **ровно 8** слотов; экспорт заполняется до эффективного `top_k_slots` (после клампа ≤ 8), остальные **NaN**. Порядок убывания: `topvar_1` = наибольшая component variance, и т.д.
- `tp_embstats_topic_entropy` (float): entropy по topic probs (NaN если topic probs отсутствуют/невалидны или выключено).
- `tp_embstats_topic_entropy_norm` (float): нормированная энтропия \(H/\log(K)\) (NaN если не применимо).
- `tp_embstats_topic_perplexity` (float): \(e^{H}\) (NaN если не применимо).
- `tp_embstats_topic_entropy_present` (0/1).
- `tp_embstats_topic_probs_present` (0/1) и `tp_embstats_topic_probs_invalid_flag` (0/1).
- `tp_embstats_source_used_whisper`, `tp_embstats_source_used_youtube_auto` (0/1): фиксированные флаги; ровно один 1.0 при успешном выборе источника из приоритета (неизвестные ключи в конфиге отбрасываются).
- safety flags: `tp_embstats_unsafe_relpath_flag`, `tp_embstats_dim_mismatch_flag`, `tp_embstats_nan_inf_flag`.

Также всегда присутствуют (стабильная схема, **39** ключей — см. JSON):
- `tp_embstats_emit_extra_metrics_enabled`
- `tp_embstats_enabled`, `tp_embstats_disabled_by_policy`
- `tp_embstats_schema_topvar_slots_max`, `tp_embstats_top_k_slots_requested`, `tp_embstats_top_k_slots`, `tp_embstats_top_k_slots_clamped`
- `tp_embstats_require_chunks_enabled`, `tp_embstats_compute_topic_entropy_enabled`, `tp_embstats_require_topic_distribution_enabled`
- `tp_embstats_min_chunks_required`, `tp_embstats_topk`, `tp_embstats_variance_ddof`
- `tp_embstats_n_chunks`, `tp_embstats_dim`
- `tp_embstats_load_ms`, `tp_embstats_compute_ms` (**NaN** при `emit_extra_metrics=false`)

#### Метаданные

- `device`: `"cpu"`
- `version`: версия экстрактора
- `model_name` / `model_version` / `weights_digest`: **`null`** (модель не загружается)

#### Системные метрики

- `system.pre_init` / `post_init`: снимки из `_init_metrics` конструктора
- `system.post_process`: снимок после `extract`
- `system.peaks.ram_peak_mb`, `system.peaks.gpu_peak_mb`: пики по снимкам (GPU часто 0 на CPU-only)

#### Тайминги

- `timings_s.total`: общее время обработки (секунды)

#### Ошибки

- `error`: описание ошибки (если произошла) или `None`

### Алгоритм обработки

#### 1. Загрузка чанков эмбеддингов

**Приоритет источников**: `transcript_source_priority`; допустимы только `whisper` и `youtube_auto`; по умолчанию в коде **`["whisper"]`** (Audit v3: ASR-first). Второй источник включают в конфиге при необходимости (например `["whisper", "youtube_auto"]`).

**Процесс**:
- Сначала canonical: `doc.tp_artifacts["transcripts"][source]["chunk_embeddings_relpath"]`, иначе legacy `doc.tp_artifacts["transcript_chunks"][source]["embeddings_relpath"]`
- Загружаем `.npy` из `artifacts_dir/<relpath>`
- Нормализация формы: если `ndim == 1`, преобразуется в `(1, D)`
- Конвертация в `float32`

#### 2. Вычисление дисперсии между чанками

**Формула**:
```
var_vec = np.var(chunks, axis=0, ddof=variance_ddof)  # Дисперсия по каждой компоненте
l2_variance = ||var_vec||_2       # L2-норма вектора дисперсий
topk_variances = sort(var_vec)[-topk:]  # Top-k наибольших дисперсий (отсортированы по возрастанию)
# Заполнение слотов: topvar_1 = topk_variances[-1] (наибольшая), topvar_2 = topk_variances[-2], и т.д.
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
    "transcript_source_priority": ["whisper"],  # при необходимости: ["whisper", "youtube_auto"]
    "top_k_slots": 8,             # эффективное число заполняемых слотов (кламп ≤ 8; в схеме всегда 8 ключей)
    "topk": 8,                    # сколько component variances брать у np.var по оси (может быть > слотов экспорта)
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
- `top_k_slots`: сколько слотов из **8** заполнять (значение из конфига клампится; см. `tp_embstats_top_k_slots_*`)
- `topk`: сколько top-k реально выбирать
- `min_chunks_required`: минимум чанков для valid вычисления дисперсии
- `variance_ddof`: ddof для `np.var`
- `require_chunks`: fail-fast, если required input отсутствует
- `emit_extra_metrics`: при **True** заполняет `tp_embstats_load_ms` и `tp_embstats_compute_ms`; при **False** → **NaN**

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
- **transcript_chunk_embedder**: создаёт матрицу чанков (`chunk_embeddings_relpath` в `tp_artifacts`)
- **semantics_topics_keyphrases** (опционально): пишет `topic_probs` в `tp_artifacts` in-memory для энтропии

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
---

## Навигация

[SCHEMA](SCHEMA.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
