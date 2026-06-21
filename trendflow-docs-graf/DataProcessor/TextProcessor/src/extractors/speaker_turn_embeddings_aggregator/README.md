## `speaker_turn_embeddings_aggregator` (Speaker Turn Embeddings Aggregator)

### Назначение

Агрегирует эмбеддинги **speaker turns** в per‑speaker агрегаты (mean/max). Компонент предназначен для downstream‑метрик и UI‑индикаторов “multi‑speaker / speaker diversity”, при этом соблюдает A‑policy: no raw, determinism, dp_models.

**Версия**: 1.3.0  
**Категория**: embedding aggregation, speaker analysis  
**GPU**: поддерживается (cuda), опционально fp16

**Диапазоны и валидатор среза** (`text_features.npz`): [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · [`utils/validate_speaker_turn_embeddings_aggregator_text_npz.py`](utils/validate_speaker_turn_embeddings_aggregator_text_npz.py)

**Контракт `features_flat`**: [SCHEMA.md](SCHEMA.md) · machine: [`../../schemas/speaker_turn_embeddings_aggregator_output_v1.json`](../../schemas/speaker_turn_embeddings_aggregator_output_v1.json) · Audit v3: [`../../../docs/audit_v3/components/speaker_turn_embeddings_aggregator_AUDIT_V3_REPORT.md`](../../../docs/audit_v3/components/speaker_turn_embeddings_aggregator_AUDIT_V3_REPORT.md) · **Audit v4:** [`../../../../docs/audit_v4/components/text_processor/speaker_turn_embeddings_aggregator_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/speaker_turn_embeddings_aggregator_audit_v4.md) · L2 stats: `scripts/audit_v4_npz_stats.py` → `storage/audit_v4/speaker_turn_embeddings_aggregator_l2/`

### Входы

Поддерживаются два режима входа:

1) **Предпочтительный (prod‑ready, AudioProcessor‑driven)**:
- `doc.speaker_diarization["speaker_segments"]`: список сегментов со спикером и временем:
  - `{speaker_id, start_sec, end_sec}`
- `doc.asr["segments"]`: ASR сегменты с таймингами:
  - `{text, start_sec, end_sec}`

Компонент детерминированно сопоставляет ASR‑сегменты diar‑сегментам по overlap во времени и получает набор текстов на спикера (raw нигде не сохраняется).

2) **Legacy (поддержка для совместимости)**:
- `doc.speakers: Dict[str, Dict]` со структурами вида `{name, description}`.
  - Используется только как input; имена/тексты не сохраняются и не попадают в filenames.

**Valid empty**: если входа нет и `require_input=False` → `tp_spkemb_present=0` и валидная пустота.

### Выходы

Экстрактор возвращает:

- `result.features_flat` (только scalars, A‑policy)
- `result.speaker_embeddings_meta` (privacy‑safe метаданные модели)

#### `features_flat`

Ровно **17** ключей `tp_spkemb_*` в фиксированном порядке (см. JSON-схему).

**Core + конфиг + режим** (всегда числа, не гейтятся `emit_extra_metrics`):

- Метрики: `tp_spkemb_present`, `tp_spkemb_speakers_total`, `tp_spkemb_speakers_embedded`, `tp_spkemb_turns_total`
- Конфиг: `tp_spkemb_write_artifacts`, `tp_spkemb_compute_mean`, `tp_spkemb_compute_max`
- Вход: `tp_spkemb_input_present`, `tp_spkemb_input_mode_diar_asr`, `tp_spkemb_input_mode_legacy_doc_speakers`, `tp_spkemb_asr_present`, `tp_spkemb_diar_present`

**Tuning** (при **`emit_extra_metrics=False`** — все **NaN**, ключи на месте):

`tp_spkemb_batch_size`, `tp_spkemb_max_speakers`, `tp_spkemb_max_turns_per_speaker`, `tp_spkemb_min_chars_per_turn`, `tp_spkemb_max_chars_per_turn`

#### Артефакты (`*.npy`, per-run, без raw/hash)

Если `write_artifacts=True`, для каждого спикера создаются:
- `speaker_<speaker_id>_mean.npy` (если `compute_mean=True`)
- `speaker_<speaker_id>_max.npy` (если `compute_max=True`)

`speaker_id` имеет вид `spk000`, `spk001`, ... (детерминированно).

#### In-memory registry (`doc.tp_artifacts`)

Canonical:
- `doc.tp_artifacts["speakers"]["embeddings"][speaker_id] = {mean_relpath?, max_relpath?, count_turns}`

Legacy alias (для back-compat):
- `doc.tp_artifacts["speaker_embeddings"][speaker_id] = ...`

#### Метаданные

- `device`: устройство обработки (`"cpu"` или `"cuda"`)
- `version`: версия экстрактора (**1.3.0**)
- `model_name`, `model_version`, `weights_digest`: верхний уровень payload (и дубли в `result.speaker_embeddings_meta`)

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

#### 1. Сбор speaker turn текстов

В прод‑режиме: diarization + ASR → тексты на спикера по таймингам.  
В legacy‑режиме: `doc.speakers` → группировка по `name`, при этом `speaker_id` назначается без использования raw‑хешей.

#### 2. Кодирование текстов

**Процесс**:
1. **Нормализация**: применение `normalize_whitespace()` к каждому тексту
2. **Батчинг**: обработка текстов батчами размером `batch_size`
3. **Кодирование**: использование sentence-transformers модели для кодирования
4. **Нормализация**: L2-нормализация каждого эмбеддинга
5. **Объединение**: объединение всех батчей в единую матрицу (N, D)

**Формула нормализации**:
```
emb_norm = emb / ||emb||
```

#### 3. Агрегация эмбеддингов

**Для каждого спикера**:

**Среднее (Mean)**:
```python
mean_emb = np.mean(embs, axis=0)
mean_emb = mean_emb / ||mean_emb||  # L2 нормализация
```

**Максимум (Max)**:
```python
max_emb = np.max(embs, axis=0)
max_emb = max_emb / ||max_emb||  # L2 нормализация
```

**Особенности**:
- Оба вектора L2-нормализованы после агрегации
- Если эмбеддингов нет, спикер пропускается

#### 4. Сохранение артефактов (опционально)

**Для каждого спикера и типа агрегации**:

**Файлы**:
- `speaker_<speaker_id>_mean.npy`: средний эмбеддинг (вектор)
- `speaker_<speaker_id>_max.npy`: максимальный эмбеддинг (вектор)

**Безопасность**:
- `.meta.json` sidecar не создаётся
- в filenames нет speaker names и нет raw‑derived hashes
- атомарная запись: `.tmp.npy` → `os.replace()` через `Path.replace()`

### Конфигурация

```python
{
    "model_name": "intfloat/multilingual-e5-large",          # Audit v3 preflight; через dp_models
    "artifacts_dir": None,                                    # Путь к артефактам (по умолчанию: default_artifacts_dir())
    "device": "cpu",                                          # "cpu" | "cuda"
    "fp16": True,                                             # Использовать float16 на GPU
    "batch_size": 64,                                         # Размер батча
    "compute_mean": True,                                     # Вычислять средний эмбеддинг
    "compute_max": True,                                      # Вычислять max pooling эмбеддинг
    "write_artifacts": True,                                  # Сохранять артефакты в файлы
    "require_input": False,                                   # Требовать наличие входных данных
    "max_speakers": 16,                                       # Максимальное количество спикеров
    "max_turns_per_speaker": 64,                              # Максимальное количество реплик на спикера
    "min_chars_per_turn": 5,                                  # Минимальная длина реплики в символах
    "max_chars_per_turn": 600,                                # Максимальная длина реплики в символах
    "dedup_turn_texts": True,                                 # Удалять дубликаты реплик
    "emit_extra_metrics": False,                              # Выдавать дополнительные метрики
}
```

**Параметры**:
- `model_name`: название модели sentence-transformers (резолвится через dp_models)
- `artifacts_dir`: директория для сохранения артефактов (по умолчанию: `default_artifacts_dir()`)
- `device`: устройство обработки (cpu или cuda)
- `fp16`: использование float16 на GPU (уменьшает память, минимальная потеря точности)
- `batch_size`: размер батча для обработки текстов
- `compute_mean`: вычислять средний эмбеддинг реплик для каждого спикера
- `compute_max`: вычислять max pooling эмбеддинг реплик для каждого спикера
- `write_artifacts`: сохранять артефакты (`.npy` файлы) на диск
- `require_input`: требовать наличие входных данных (иначе RuntimeError)
- `max_speakers`: максимальное количество спикеров для обработки
- `max_turns_per_speaker`: максимальное количество реплик на спикера
- `min_chars_per_turn`: минимальная длина реплики в символах (более короткие отбрасываются)
- `max_chars_per_turn`: максимальная длина реплики в символах (более длинные обрезаются)
- `dedup_turn_texts`: удалять дубликаты реплик (case-insensitive сравнение)
- `emit_extra_metrics`: выдавать дополнительные метрики в `features_flat`:
  - `tp_spkemb_batch_size`: размер батча
  - `tp_spkemb_max_speakers`: максимальное количество спикеров
  - `tp_spkemb_max_turns_per_speaker`: максимальное количество реплик на спикера
  - `tp_spkemb_min_chars_per_turn`: минимальная длина реплики
  - `tp_spkemb_max_chars_per_turn`: максимальная длина реплики

### Особенности

- **Группировка по спикерам**: автоматическая группировка текстов по именам спикеров
- **Две стратегии агрегации**: среднее (учитывает все описания) и максимум (выделяет пиковые значения)
- **L2 нормализация**: все агрегированные векторы нормализованы для использования в косинусной метрике
- **Батчинг**: эффективная обработка множества текстов
- **GPU поддержка**: опциональное использование CUDA с fp16 для ускорения
- **Атомарная запись**: использование временных файлов для безопасного сохранения
- **Метаданные**: `.meta.json` не используется; модель фиксируется через `model_version` и `manifest.json.models_used`
- **Хеширование**: использование хешей для идентификации наборов текстов

### Архитектура

1. **Инициализация**: загрузка модели через `get_model()` из registry
2. **Проверка данных**: проверка наличия и корректности `doc.speakers`
3. **Группировка**: группировка текстов по именам спикеров
4. **Кодирование**: батчевая обработка текстов через модель для каждого спикера
5. **Агрегация**: вычисление mean и max эмбеддингов
6. **Нормализация**: L2-нормализация агрегированных векторов
7. **Сохранение**: сохранение эмбеддингов и метаданных
8. **Метрики**: сбор системных метрик и таймингов

### Обработка ошибок

- **Отсутствие входа**:
  - `require_input=False` → valid empty (`tp_spkemb_present=0`)
  - `require_input=True` → `RuntimeError`
- **dp_models/torch отсутствуют**: `RuntimeError` (fail-fast)

### Performance characteristics

**Resource costs**:
- **CPU**: умеренные (sentence-transformers операции)
- **GPU**: опционально (значительное ускорение при использовании CUDA)
- **Estimated duration**: ~0.1-2.0 секунд в зависимости от количества спикеров и текстов

**Параметры производительности**:
- Количество спикеров: линейно влияет на время обработки
- Количество текстов на спикера: влияет на время кодирования
- `batch_size`: большие значения → быстрее на GPU, но больше памяти
- `fp16`: уменьшает использование GPU памяти в 2 раза, минимальная потеря точности

### Зависимости

- `numpy`: численные операции (агрегация, нормализация)
- `torch`: для работы с моделями (если используется GPU)
- `sentence-transformers`: библиотека для эмбеддингов
- `hashlib`: генерация хешей для идентификации наборов текстов
- `pathlib`: работа с путями к файлам
- `json`: сохранение метаданных

### Связанные компоненты

- **BaseExtractor**: базовый интерфейс экстрактора
- **model_registry**: реестр моделей для разделения между экстракторами
- **VideoDocument**: схема документа со спикерами
- **text_utils.normalize_whitespace**: нормализация текста
- **path_utils.default_artifacts_dir**: путь к директории артефактов по умолчанию

### Примечания

1. **No-raw**: raw speaker texts используются только для вычисления эмбеддингов и не сохраняются.
2. **Determinism**: имена файлов фиксированные в пределах run, без content‑hash.
3. **dp_models**: модель резолвится offline; `weights_digest` доступен в `speaker_embeddings_meta`.

### Примеры использования

**Структура входных данных**:
```python
doc.speakers = {
    "speaker_1": {
        "name": "Иван",
        "description": "Привет, это Иван. Я расскажу о машинном обучении."
    },
    "speaker_2": {
        "name": "Мария",
        "description": "А я Мария. Давайте обсудим нейронные сети."
    }
}
```

**Результат**:
```python
{
    "speaker_embeddings": {
        "Иван": {
            "mean": {"path": "...", "count_turns": 1},
            "max": {"path": "...", "count_turns": 1}
        },
        "Мария": {
            "mean": {"path": "...", "count_turns": 1},
            "max": {"path": "...", "count_turns": 1}
        }
    }
}
```

### Интерпретация результатов

**Mean эмбеддинг**:
- Представляет "среднее" семантическое содержание всех описаний спикера
- Подходит для общего представления спикера
- Устойчив к выбросам

**Max эмбеддинг**:
- Представляет "пиковые" значения по каждой компоненте
- Подчёркивает наиболее выраженные характеристики
- Может быть более чувствителен к выбросам

**Использование**:
- Mean: для общего сравнения спикеров, поиска похожих спикеров
- Max: для выделения уникальных характеристик, поиска спикеров с похожими пиковыми признаками

### Порядок выполнения экстракторов

`SpeakerTurnEmbeddingsAggregator` может выполняться независимо, но для полного анализа рекомендуется:

1. `SpeakerTurnEmbeddingsAggregator` - агрегация эмбеддингов спикеров
2. Компоненты для сравнения спикеров (используют сохранённые эмбеддинги)
3. Компоненты для анализа диалогов
---

## Навигация

[SCHEMA](SCHEMA.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
