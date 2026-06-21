## `key_extractor` (Audio signal processing extractor)

### Назначение

Определяет **тональность (ключ)** аудио — основной тональный центр и лад (мажор/минор). Использует шаблоны Krumhansl-Schmuckler для корреляции с хрома-профилем. Audit v3: только `run_segments()` (run отключён), default `key_method=librosa`, strict alignment по сегментам.

**Версия**: 2.1.1 (Audit v4.2 observability)  
**Категория**: music_theory  
**GPU**: не требуется  
**Schema**: `key_extractor_npz_v1`

### Входы (строго, no‑fallback)

- **`audio/audio.wav`** (Segmenter) — полный аудио файл
- **`audio/segments.json`** (Segmenter contract `audio_segments_v1`) — для `run_segments()`:
  - `families.key.segments[]` — сегменты для обработки определения тональности

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/key_extractor/key_extractor_features.npz` (**фиксированное имя**)

Схема: **`key_extractor_npz_v1`** (`schemas/key_extractor_npz_v1.json`, `SCHEMA.md`).

#### Audit v4 — NPZ (верхний уровень)

| Ключ | Заметка |
|------|---------|
| `feature_names` / `feature_values` | **Только числа**: `sample_rate`, `hop_length`, `duration`, `key_id`, `key_confidence` (после фикса савера). Ранее строки `key_name` / `key_mode` / `method` / категории утекали в float-вектор → **NaN** (см. `npz_savers/key.py`). |
| `key_id_by_segment`, `key_confidence_by_segment` | Строгое **N** с Segmenter. |
| `key_scores` | `(24,)` при `detailed_scores`; иначе нули. |
| `meta` | `key_name`, `key_mode`, `key_method`, `key_id`, `chroma_reused`, `key_confidence_*` (категория/причина/warning), … |

#### Audit v4.2 observability:
- `meta.stage_timings_ms` (dict): покомпонентные тайминги (ms), пишутся в NPZ meta
- `meta.key_resource_profile` (dict|None): best-effort snapshot RSS/VMS/VRAM (если включено)
  - включение: `AP_KEY_RESOURCE_PROFILE=1`

#### Полезные поля payload:

**Основной результат (всегда включен):**
- `key_name`, `key_mode`, `method`, `key_confidence_category`, `key_low_confidence_warning`, `key_confidence_reason` — в **payload** экстрактора как строки/bool; в **NPZ** хранятся в **`meta`** (не в `feature_values`, чтобы избежать NaN от `as_float`).
- `key_id`: глобальный id 0…23 (в NPZ и **`meta`** при `key_id >= 0`)
- `key_confidence`: уверенность (float, 0…1) — и в payload, и в **tabular** NPZ

**Детальные оценки (feature-gated):**
- `key_scores`: оценки для всех 24 возможных тональностей (list[float], 24 значения, если `enable_detailed_scores=True`)
  - Формат: [maj_C, min_C, maj_C#, min_C#, ..., maj_B, min_B]
  - Нормализованы к диапазону [0, 1]

**Топ-K альтернативных тональностей (feature-gated):**
- `key_top_k`: топ-K тональностей с наивысшими оценками (list[dict], если `enable_top_k=True`)
  - Каждый элемент: `{"key": str, "mode": str, "score": float}`
  - Количество определяется параметром `top_k` (по умолчанию 3)

**Данные сегментов (всегда доступны при `run_segments()`):**
- `segment_centers_sec`: центры сегментов в секундах (float32[N], всегда доступны при `run_segments()`)
- `segment_durations_sec`: длительности сегментов в секундах (float32[N], всегда доступны при `run_segments()`)

**Временные серии (feature-gated, для `run_segments()`, требуют `enable_time_series=True`):**
- `key_names_sequence`: последовательность тональностей по сегментам (list[str], если `enable_time_series=True`)
- `key_modes_sequence`: последовательность ладов по сегментам (list[str], если `enable_time_series=True`)
- `key_confidences_sequence`: последовательность уверенностей по сегментам (float32[N], если `enable_time_series=True`)

**Детекция смены тональности (feature-gated, для `run_segments()`):**
- `key_transitions`: список переходов между тональностями (list[dict], если `enable_key_changes=True`)
  - Каждый элемент: `{"transition_index": int, "from_key": str, "to_key": str, "transition_time_sec": float}`
- `key_transitions_count`: количество переходов (int, если `enable_key_changes=True`)
- `key_transitions_rate`: частота переходов (transitions/sec, float, если `enable_key_changes=True`)

**Метрики стабильности (feature-gated, для `run_segments()`):**
- `key_stability_score`: доля времени в доминирующей тональности (float, 0.0-1.0, если `enable_stability_metrics=True`)
- `key_confidence_mean`: средняя уверенность (float, если `enable_stability_metrics=True`)
- `key_confidence_std`: стандартное отклонение уверенности (float, если `enable_stability_metrics=True`)
- `key_confidence_min`: минимальная уверенность (float, если `enable_stability_metrics=True`)
- `key_confidence_max`: максимальная уверенность (float, если `enable_stability_metrics=True`)
- `key_distribution`: распределение времени по тональностям (dict[str, float], если `enable_stability_metrics=True`)
- `key_diversity`: количество уникальных тональностей (int, если `enable_stability_metrics=True`)
- `key_detection_quality`: метрика качества (confidence × stability, float, если `enable_stability_metrics=True`)

**Метаданные:**
- `sample_rate`: частота дискретизации (int)
- `hop_length`: размер hop для STFT/CQT (int)
- `duration`: длительность аудио в секундах (float, для `run_segments()` - сумма длительностей сегментов)
- `device_used`: устройство обработки (str)
- `key_contract_version`: версия контракта ("key_contract_v1")
- `segments_count`: количество обработанных сегментов (int, только для `run_segments()`)

### Feature Dependencies

**Зависимости между фичами:**
- `key_top_k` зависит от `enable_top_k` и требует `enable_detailed_scores=True` (использует `key_scores` для вычисления топ-K)
- `key_transitions` и `key_transitions_count` зависят от `enable_key_changes` и `enable_time_series` (требуют временных серий)
- Все метрики стабильности зависят от `enable_stability_metrics` и `enable_time_series` (требуют временных серий)
- `key_detection_quality` зависит от `key_stability_score` и `key_confidence` (требует `enable_stability_metrics`)

**Зависимости от других extractors:**
- **chroma_extractor** (опционально): может использовать предвычисленный `chroma` из `shared_features` для оптимизации (избегает повторного вычисления хрома)

### Модели

ML модели **не используются** (signal processing + music theory templates). `models_used[]` пустой.

**Библиотеки:**
- **librosa**: основная библиотека для хрома-фич и оценки тональности
- **essentia** (опционально): если доступен, используется `essentia.standard.KeyExtractor` как альтернативный метод

**Алгоритмы:**
- **Krumhansl-Schmuckler profiles**: психоакустические профили для мажора и минора
- **Pearson-like correlation**: корреляция хрома-профиля с шаблонами (zero-mean cosine similarity)

### Конфигурация

#### Через global_config.yaml

```yaml
audio:
  extractors:
    key:
      enabled: false
      sample_rate: 22050
      hop_length: 512
      chroma_type: "cqt"  # cqt|stft
      key_method: "auto"  # auto|essentia|librosa
      use_beat_sync: false
      top_k: 5
      parallelism:
        segment_workers: 8
        max_inflight: null
      feature_flags:
        enable_audio_normalization: false
        enable_detailed_scores: false
        enable_top_k: false
        enable_time_series: false
        enable_key_changes: false
        enable_stability_metrics: false
```

#### Через Python API

```python
{
    "sample_rate": 22050,           # Частота дискретизации
    "hop_length": 512,              # Размер hop для STFT/CQT
    "chroma_type": "cqt",           # "cqt" | "stft" (CQT предпочтительнее для музыки)
    "use_beat_sync": False,         # Агрегировать хрома по битам (требует beat tracking)
    "top_k": 3,                     # Количество топ-тональностей для key_top_k
    "key_method": "librosa",         # "essentia" | "librosa" | "auto" (Audit v3 default: librosa)
    "key_confidence_threshold": 0.3,  # Порог уверенности для предупреждений (0.0-1.0)
    "enable_audio_normalization": False,  # Нормализация аудио перед обработкой
    # Feature gating flags (все по умолчанию False)
    "enable_detailed_scores": False,  # Включить детальные оценки (24 значения)
    "enable_top_k": False,            # Включить топ-K альтернативных тональностей
    "enable_time_series": False,     # Включить временные серии (для run_segments)
    "enable_key_changes": False,      # Включить детекцию смены тональности
    "enable_stability_metrics": False,  # Включить метрики стабильности
    "device": "auto"                  # "auto" | "cuda" | "cpu"
}
```

### Feature Gating

Все фичи контролируются через персональные флаги (по умолчанию все выключены, кроме базовых полей):

- `--key-enable-detailed-scores`: Включить детальные оценки (`key_scores` — 24 значения для всех тональностей)
- `--key-enable-top-k`: Включить топ-K альтернативных тональностей (`key_top_k`)
- `--key-enable-time-series`: Включить временные серии (`key_names_sequence`, `key_modes_sequence`, и т.д.)
- `--key-enable-key-changes`: Включить детекцию смены тональности (`key_transitions`, `key_transitions_count`, `key_transitions_rate`)
- `--key-enable-stability-metrics`: Включить метрики стабильности (`key_stability_score`, `key_confidence_mean/std/min/max`, `key_distribution`, `key_diversity`, `key_detection_quality`)

**Зависимости фичей:**
- `enable_key_changes` требует `enable_time_series` (для отслеживания изменений по сегментам)
- `enable_stability_metrics` требует `enable_time_series` (для вычисления статистик по сегментам)

### Алгоритм работы

#### Метод выбора (явный, no-fallback):

1. **Явный выбор метода** через `key_method`:
   - **"essentia"**: только Essentia, fail-fast если недоступна
   - **"librosa"**: только librosa (Krumhansl)
   - **"auto"**: Essentia с fallback на librosa (если Essentia недоступна или ошибка)

#### Essentia путь (если выбран "essentia" или "auto"):

1. **Проверка Essentia**: если библиотека доступна, используется `essentia.standard.KeyExtractor`
2. **Прямое определение**: Essentia возвращает key, mode и confidence
3. **Валидация**: проверка корректности выходных данных (key_name в VALID_KEYS, key_mode в VALID_MODES)
4. **Возврат результата**: если успешно, возвращается результат

#### Librosa путь (Krumhansl):

1. **Загрузка аудио**: через `AudioUtils.load_audio()` с ресемплированием до `sample_rate`
2. **Опциональная нормализация аудио**: если `enable_audio_normalization=True`
3. **Сведение в моно**: усреднение каналов для стерео аудио
4. **Проверка минимальной длительности**: fail-fast если аудио < 1 секунды
5. **Вычисление хрома**:
   - **Оптимизация**: если `shared_features` содержит `chroma`, используется он
   - **CQT mode** (по умолчанию): `librosa.feature.chroma_cqt()` — более точный для музыки
   - **STFT mode**: `librosa.feature.chroma_stft()` — для STFT-based хрома
6. **Beat synchronization** (опционально):
   - Если `use_beat_sync=True`, выполняется beat tracking
   - Хрома агрегируется по битам через `librosa.util.sync()`
7. **Агрегация хрома**: усреднение по времени для получения Pitch Class Profile (PCP)
8. **Нормализация PCP**: нормализация к сумме 1.0
9. **Корреляция с шаблонами**:
   - Для каждой из 12 тональностей вычисляется корреляция с мажорным и минорным профилями Krumhansl
   - Используется zero-mean cosine similarity (Pearson-like correlation)
10. **Выбор лучшей тональности**: argmax по всем 24 комбинациям (12 тональностей × 2 лада)
11. **Топ-K**: сортировка и выбор топ-K тональностей (если `enable_top_k=True`)
12. **Категоризация уверенности**: определение категории и причины низкой уверенности

#### Segment-based обработка (`run_segments()`):

1. **Загрузка сегментов**: из `families.key.segments[]` от Segmenter
2. **Фильтрация сегментов**: сегменты короче 0.5 секунды пропускаются (слишком короткие для точного определения тональности)
3. **Обработка каждого сегмента**: определение тональности для каждого сегмента
4. **Агрегация результатов**: определение доминирующей тональности (наиболее частая)
5. **Временные серии** (если `enable_time_series=True`): сохранение последовательностей по сегментам
6. **Детекция смены тональности** (если `enable_key_changes=True`): поиск переходов между тональностями
7. **Метрики стабильности** (если `enable_stability_metrics=True`): вычисление статистик по сегментам

### Особенности

- **Явный выбор метода**: контроль через `key_method` (essentia/librosa/auto), no-fallback policy
- **Krumhansl profiles**: психоакустические профили основаны на исследованиях восприятия тональности
- **Shared chroma**: может использовать предвычисленный хрома из `shared_features` для оптимизации
- **Beat synchronization**: опциональная синхронизация с битами для более точного определения
- **Tuning estimation**: автоматическая оценка строя для коррекции хрома-фич
- **Top-K results**: полезны для анализа альтернативных интерпретаций тональности
- **Confidence categorization**: автоматическая категоризация уверенности (high/medium/low/very_low)
- **Low confidence warnings**: предупреждения о низкой уверенности с указанием причины
- **Segment-based processing**: поддержка `run_segments()` для отслеживания изменений тональности
- **Key change detection**: детекция смены тональности по сегментам
- **Stability metrics**: метрики стабильности тональности для ML/аналитики

### Error Handling

Детальные error codes:
- `audio_load_failed`: Ошибка загрузки аудио
- `audio_too_short`: Аудио слишком короткое (< 1 секунды)
- `chroma_computation_failed`: Ошибка вычисления хрома
- `key_detection_failed`: Ошибка определения тональности
- `essentia_unavailable`: Essentia недоступна (если выбран метод "essentia")
- `invalid_parameters`: Невалидные параметры

**No-fallback policy:**
- Отсутствие обязательного входа → fail-fast с `RuntimeError`
- Пустой список segments → `ValueError("segments is empty (no-fallback)")`
- Невалидные параметры → `ValueError` с описанием
- Ошибки модели/инференса → `status="error"` с error_code

### Parallelization

#### Внутренний параллелизм (внутри компонента)

- **Segment-level parallelism**: поддерживается через `segment_parallelism` и `max_inflight` (для `run_segments()`)
  - Использует `ThreadPoolExecutor` для параллельной обработки сегментов внутри одного файла
  - Количество потоков контролируется параметром `segment_parallelism` (по умолчанию 1)
  - Thread-safety: экстрактор thread-safe для параллельной обработки сегментов
  - Ограничения: параллелизм ограничен I/O операциями (загрузка аудио) и CPU вычислениями (хрома, корреляция)

#### Внешний параллелизм (выше компонента)

- **Batch processing**: поддерживается через `extract_batch_segments()` с CPU parallelism для обработки сегментов из нескольких видео одновременно
  - Использует `ThreadPoolExecutor` для параллельной обработки файлов
  - Количество воркеров контролируется параметром `max_workers` (по умолчанию `os.cpu_count()`)
  - Каждый файл обрабатывается изолированно через `run_segments()`
  - Изоляция артефактов: каждый файл имеет свой `artifacts_dir` для сохранения `.npy` файлов
  - Требования к изоляции: разные `run_id`, разные `result_store` пути, разные `artifacts_dir`
  - Ограничения: нет shared mutable state между файлами (кроме read-only моделей/корпусов)

#### Комбинированный подход

- Внутренний параллелизм (сегменты внутри файла) + внешний параллелизм (несколько файлов)
- Рекомендуется: использовать внешний параллелизм для обработки нескольких файлов, внутренний параллелизм для обработки сегментов внутри одного файла
- GPU: не используется (CPU-only)

### Performance characteristics

**Resource costs:**
- **CPU**: O(N * log(N)) для CQT/STFT, где N — длина аудио
- **Память**: O(12 * frames) для хрома-спектрограммы
- **Estimated duration**: ~1.0 секунд для типичного аудио (для `run()`)
- **Beat sync overhead**: если `use_beat_sync=True`, дополнительное время на beat tracking
- **Segment-based overhead**: для `run_segments()`, время пропорционально количеству сегментов

**Единица обработки:**
- `run()`: весь аудио файл
- `run_segments()`: сегменты от Segmenter (`families.key.segments[]`)

**Batch processing:**
- **CPU parallelism**: поддерживается через `extract_batch_segments()` с `ThreadPoolExecutor`
- **Масштабирование**: линейное ускорение при увеличении количества CPU ядер (до лимита I/O)
- **Изоляция**: каждый файл обрабатывается изолированно, нет shared mutable state между файлами
- **Артефакты**: каждый файл имеет свой `artifacts_dir` для сохранения `.npy` файлов (per-run storage)

**Параметры производительности:**
- `chroma_type`: CQT обычно быстрее STFT для длинных аудио
- `use_beat_sync`: добавляет overhead на beat tracking
- `key_method`: Essentia может быть медленнее librosa, но точнее
- Размер аудио: линейная зависимость от длительности
- Количество сегментов: линейная зависимость для `run_segments()`

### Примеры использования

#### Загрузка результатов

```python
import numpy as np

data = np.load("key_extractor_features.npz", allow_pickle=True)
payload = data["payload"].item()

# Основной результат
key_name = payload["key_name"]  # "C", "C#", "D", etc.
key_mode = payload["key_mode"]  # "major" или "minor"
confidence = payload["key_confidence"]
category = payload["key_confidence_category"]

print(f"Key: {key_name} {key_mode} (confidence={confidence:.3f}, category={category})")
```

#### Анализ альтернативных тональностей

```python
# Если включен enable_top_k
if "key_top_k" in payload:
    top_k = payload["key_top_k"]
    print("Top-K keys:")
    for i, entry in enumerate(top_k, 1):
        print(f"  {i}. {entry['key']} {entry['mode']} (score={entry['score']:.3f})")
```

#### Анализ уверенности

```python
confidence = payload["key_confidence"]
category = payload["key_confidence_category"]
warning = payload["key_low_confidence_warning"]
reason = payload["key_confidence_reason"]

if warning:
    print(f"⚠ Low confidence warning: {reason}")
    print(f"Category: {category}")
```

#### Анализ изменений тональности (для run_segments)

```python
# Если включены enable_time_series и enable_key_changes
if "key_transitions" in payload:
    transitions = payload["key_transitions"]
    print(f"Key changes: {payload['key_transitions_count']} transitions")
    print(f"Transition rate: {payload['key_transitions_rate']:.4f} transitions/sec")
    for trans in transitions:
        print(f"  {trans['transition_time_sec']:.2f}s: {trans['from_key']} → {trans['to_key']}")
```

#### Анализ метрик стабильности (для run_segments)

```python
# Если включены enable_stability_metrics
if "key_stability_score" in payload:
    print(f"Key stability: {payload['key_stability_score']:.3f}")
    print(f"Confidence: mean={payload['key_confidence_mean']:.3f}, std={payload['key_confidence_std']:.3f}")
    print(f"Key diversity: {payload['key_diversity']} unique keys")
    print(f"Detection quality: {payload['key_detection_quality']:.3f}")
    
    # Распределение времени по тональностям
    distribution = payload["key_distribution"]
    print("Key distribution:")
    for key_str, proportion in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {key_str}: {proportion:.2%}")
```

#### Использование с shared_features

```python
# Если chroma_extractor был запущен ранее
shared_features = {
    "chroma": chroma,  # shape: (12, frames)
}

# shared_features поддерживается только в run_segments()
result = key_extractor.run_segments(
    input_uri, 
    tmp_path, 
    segments=segments,
    shared_features=shared_features
)
# Key extractor использует предвычисленный хрома, избегая повторного вычисления
```

### Связанные компоненты

- **Segmenter**: предоставляет аудио файл и сегменты (`families.key.segments[]`)
- **librosa**: основная библиотека для хрома-фич и оценки строя
- **essentia** (опционально): альтернативный метод определения тональности
- **chroma_extractor**: может предоставлять `chroma` в `shared_features` для оптимизации

### Visualization

**Рекомендуемые типы визуализации:**

1. **Timeline визуализация** (для `run_segments()`):
   - График изменений тональности по времени
   - Цветовое кодирование по тональностям
   - Маркеры переходов между тональностями

2. **Confidence timeline**:
   - График уверенности по времени
   - Индикация категорий уверенности (high/medium/low/very_low)
   - Предупреждения о низкой уверенности

3. **Key distribution**:
   - Круговая диаграмма распределения времени по тональностям
   - Процентное соотношение каждой тональности

4. **Top-K comparison**:
   - Сравнение топ-K альтернативных тональностей
   - Bar chart с оценками

5. **Stability metrics**:
   - Индикатор стабильности тональности
   - Статистики уверенности (mean, std, min, max)

**Интерактивные элементы:**
- Tooltips с детальной информацией о тональностях
- Zoom для детального просмотра timeline
- Фильтры по категориям уверенности
- Переключение между различными метриками

### Примечания

1. **Essentia vs librosa**: Essentia обычно более точный, но требует установки библиотеки. Явный выбор через `key_method`.
2. **CQT vs STFT**: CQT (Constant-Q Transform) предпочтительнее для музыки, STFT — альтернатива
3. **Krumhansl profiles**: психоакустические профили основаны на исследованиях восприятия тональности
4. **Beat sync**: синхронизация с битами может улучшить точность для ритмичной музыки
5. **Tuning estimation**: автоматическая оценка строя помогает для нестандартных строев
6. **Top-K results**: полезны для анализа альтернативных интерпретаций тональности
7. **Confidence interpretation**: низкая уверенность может указывать на атональную музыку или недостаточно музыкального контента
8. **Shared chroma**: оптимизация для случаев, когда хрома уже вычислен другим экстрактором
9. **Segment-based processing**: полезно для длинных видео и отслеживания изменений тональности
10. **Key change detection**: автоматическая детекция смены тональности для анализа музыкальной структуры
11. **Stability metrics**: метрики стабильности полезны для ML моделей и аналитики
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
