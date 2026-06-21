## `transcript_aggregator` (Text embeddings aggregation)

### Назначение

Агрегирует **эмбеддинги чанков транскрипта** в единые векторные представления. Использует два метода агрегации: взвешенное среднее (weighted mean) с экспоненциальным затуханием и опциональными весами уверенности ASR, а также max pooling. Обрабатывает несколько источников транскрипта (whisper, youtube_auto) и создает комбинированные агрегаты.

**Версия**: 1.3.0  
**Категория**: text embeddings aggregation  
**GPU**: не требуется (tensor ops обычно на CPU)

**Диапазоны, тайминги и валидатор среза** (`text_features.npz`): [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · [`utils/validate_transcript_aggregator_text_npz.py`](utils/validate_transcript_aggregator_text_npz.py) (`--struct`, `--ranges`, `--timings`)

**Контракт `features_flat`**: [SCHEMA.md](SCHEMA.md) · machine: [`../../schemas/transcript_aggregator_output_v1.json`](../../schemas/transcript_aggregator_output_v1.json) · Audit v3: [`../../docs/audit_v3/components/transcript_aggregator_AUDIT_V3_REPORT.md`](../../docs/audit_v3/components/transcript_aggregator_AUDIT_V3_REPORT.md) · **Audit v4:** [`../../../../docs/audit_v4/components/text_processor/transcript_aggregator_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/transcript_aggregator_audit_v4.md) · **L2 stats:** [`../../../../storage/audit_v4/transcript_aggregator_l2/transcript_aggregator_audit_v4_stats.json`](../../../../storage/audit_v4/transcript_aggregator_l2/transcript_aggregator_audit_v4_stats.json) (tooling: `scripts/audit_v4_npz_stats.py`)

**Табличный слой / набор B:** девять extra-полей (`n_chunks`, `mean_std`, `max_std` по источникам) при **дефолтном `emit_extra_metrics=false`** в merged NPZ — **NaN** (это ожидаемо, не «битый» табличный слой). Для корреляций по числу чанков и std задайте в профиле прогона **`emit_extra_metrics=true`**; **`compute_std=true`** нужен отдельно для колонок **`*_mean_std`** / **`*_max_std`**.

### Входы

**Зависимости (обязательный контракт)**:
- `TranscriptChunkEmbedder` должен быть выполнен **раньше** в том же run и должен заполнить:
  - canonical: `doc.tp_artifacts["transcripts"][source]["chunk_embeddings_relpath"]`
  - legacy alias: `doc.tp_artifacts["transcript_chunks"][source]["embeddings_relpath"]`
- Источник транскрипта для чанков — AudioProcessor (`doc.asr`) и/или legacy `doc.transcripts` (см. README `transcript_chunk_embedder`).

### Выходы (privacy-safe)

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Основные результаты

- `features_flat`: ровно **19** ключей `tp_tragg_*` (фиксированный порядок — JSON-схема).
  - **10 core**: `present`, `present_whisper`, `present_youtube` (источник **`youtube_auto`**), `present_combined`, `decay_rate`, флаги `compute_*`, `write_artifacts`.
  - **9 extra** (`whisper` / `youtube_auto` / `combined`: `n_chunks`, `mean_std`, `max_std`): при **`emit_extra_metrics=False`** все **NaN**; при **`compute_std=False`** поля **`*_mean_std`** / **`*_max_std`** — **NaN**.

**Важно**: компонент **не возвращает абсолютные пути** к `.npy` в `result`.  
Сгенерированные `.npy` живут в per-run `text_processor/_artifacts/` и перечисляются в `manifest.json.artifacts[]`.

#### Метаданные

- `device`: устройство из конфига (инференс модели не выполняется)
- `version`: версия экстрактора (**`1.3.0`**)
- `model_name`, `model_version`, `weights_digest`: **должны совпадать** с **`TranscriptChunkEmbedder`** (resolve через **`dp_models`** в **`__init__`**, без загрузки весов для forward)
- `system`: pre/post init (resolve), post_process, peaks (**`gpu_peak_mb`** для единообразия)
- `timings_s`: словарь с таймингами
  - `load`: время загрузки эмбеддингов (секунды)
  - `aggregate`: время агрегации (секунды)
  - `total`: общее время обработки (секунды)
- `error`: ошибка (если есть, иначе `None`)

### Алгоритмы агрегации

#### 1. Weighted Mean (взвешенное среднее)

- **Экспоненциальное затухание**: веса уменьшаются экспоненциально от начала к концу (`decay_rate`)
- **ASR confidence**: если доступны веса уверенности Whisper из `doc.tp_artifacts["transcripts"][source]["chunk_confidence"]`, они умножаются на веса затухания
- **Нормализация весов**: веса нормализуются к сумме 1.0
- **L2-нормализация**: итоговый вектор нормализуется к единичной длине
- **Streaming computation**: вычисление выполняется потоково для экономии памяти

Формула весов:
```
weights[i] = exp(-decay_rate * i) * confidence[i]  (если confidence доступен)
weights[i] = exp(-decay_rate * i)                  (иначе)
weights = weights / sum(weights)
```

#### 2. Max Pooling

- **Максимум по измерениям**: для каждого измерения эмбеддинга выбирается максимальное значение среди всех чанков
- **L2-нормализация**: итоговый вектор нормализуется к единичной длине

### Обработка источников

Экстрактор обрабатывает источники независимо и создает комбинированные агрегаты:

1. **whisper**: если доступны эмбеддинги чанков whisper
   - Использует `doc.tp_artifacts["transcripts"]["whisper"]["chunk_confidence"]` для взвешивания (если доступно)
2. **youtube_auto**: если доступны эмбеддинги чанков youtube_auto
   - Не использует веса уверенности (недоступны для этого источника)
3. **combined**: если доступен хотя бы один источник и `compute_combined=True`
   - Объединяет все чанки из всех источников в порядке `sources`
   - Не использует веса уверенности для комбинированного агрегата

### Конфигурация

```python
{
    "artifacts_dir": None,                                    # Путь к артефактам (по умолчанию: из env)
    "model_name": "intfloat/multilingual-e5-large",          # Audit v3; должно совпадать с transcript_chunk_embedder
    "device": "cpu",                                          # Устройство (не используется, всегда CPU)
    "decay_rate": 0.01,                                       # Коэффициент экспоненциального затухания для weighted mean
    "compute_std": False,                                     # Если true — считает std (дороже), иначе std=NaN
    "compute_mean": True,                                     # Вычислять weighted mean агрегат
    "compute_max": True,                                      # Вычислять max pooling агрегат
    "compute_combined": True,                                 # Вычислять комбинированный агрегат из всех источников
    "write_artifacts": True,                                  # Сохранять агрегированные векторы в artifacts
    "require_chunks": False,                                  # Если True и нет chunk embeddings → ошибка (fail-fast)
    "sources": ["whisper", "youtube_auto"],                   # Список источников для обработки
    "emit_extra_metrics": False                               # Включать дополнительные метрики (n_chunks, std по источникам)
}
```

### Формат сохраненных файлов (per-run sub-artifacts, fixed names)

- **Weighted mean**: `transcript_{source}_agg_mean.npy`
- **Max pooling**: `transcript_{source}_agg_max.npy`
- **Combined** (если включено): `transcript_combined_agg_mean.npy`, `transcript_combined_agg_max.npy`
- **Атомарность**: запись через временные файлы `.tmp.npy` с последующим `os.replace()`
Примечание: уникальность результата обеспечивается `run_id` в пути result_store; поэтому content-hash в filename не используется.

### Особенности

- **Два метода агрегации**: weighted mean и max pooling для разных применений
- **Экспоненциальное затухание**: приоритет ранним чанкам (важно для последовательного контента)
- **ASR confidence**: использование весов уверенности Whisper для улучшения качества
- **Множественные источники**: независимая обработка whisper и youtube_auto
- **Комбинированные агрегаты**: создание объединенных представлений из всех источников
- **L2-нормализация**: все агрегированные векторы нормализованы к единичной длине
- **Метрики**: отслеживание количества чанков и стандартного отклонения
- **Атомарная запись**: безопасное сохранение файлов через временные файлы

### Обработка ошибок

- **Отсутствие эмбеддингов**: если эмбеддинги чанков не найдены, источник пропускается
- **Пустые/отсутствующие эмбеддинги**: valid empty (источник absent), **без fake vectors** (векторы не пишутся)
- **Ошибка загрузки**: источник пропускается (если `require_chunks=False`), иначе fail-fast

### Архитектура

1. **Загрузка эмбеддингов**: детерминированно через `doc.tp_artifacts` (без glob/mtime)
2. **Агрегация**: weighted mean / max pooling (по флагам)
3. **Агрегация per source**: для каждого источника выполняется weighted mean и max pooling
4. **Комбинированная агрегация**: объединение чанков из всех источников и агрегация
5. **Сохранение**: атомарная запись агрегированных векторов в artifacts
6. **Сбор метрик**: измерение времени выполнения и использования памяти

### Performance characteristics

**Resource costs**:
- **CPU**: минимальные (только tensor операции на CPU)
- **RAM**: зависит от количества чанков (загружаются все эмбеддинги в память)
- **Estimated duration**: ~0.01-0.1 секунд для типичной агрегации

**Параметры производительности**:
- `decay_rate`: не влияет на производительность, только на качество агрегации
- Количество чанков: линейная зависимость времени от количества чанков
- Размерность эмбеддинга: не влияет значительно (операции векторизованы)

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **VideoDocument**: схема входного документа
- **transcript_chunk_embedder**: компонент, создающий эмбеддинги чанков (зависимость)
- **normalize_whitespace**: утилита для нормализации пробелов
- **system_snapshot, process_memory_bytes**: утилиты для сбора метрик

### Примечания

1. **Зависимость от transcript_chunk_embedder**: эмбеддинги чанков должны быть созданы заранее
2. **Совпадение model_name**: `model_name` должен совпадать с тем, что использовался в `transcript_chunk_embedder` для правильного хеширования
3. **Decay rate**: меньшие значения (например, 0.01) дают более равномерное взвешивание, большие (например, 0.1) сильнее приоритизируют ранние чанки
4. **ASR confidence**: доступно только для whisper источника, если поле `whisper_confidence` присутствует в `transcripts`
5. **Комбинированные агрегаты**: создаются только если доступен хотя бы один источник
6. **Размерность**: по умолчанию используется 384 (для all-MiniLM-L6-v2), но автоматически определяется из загруженных эмбеддингов
7. **Max pooling**: полезен для выделения наиболее выраженных признаков из всех чанков
8. **Weighted mean**: полезен для создания общего представления с учетом важности ранних частей и уверенности ASR
---

## Навигация

[SCHEMA](SCHEMA.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
