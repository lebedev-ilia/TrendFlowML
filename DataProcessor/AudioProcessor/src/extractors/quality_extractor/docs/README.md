## `quality_extractor` (Audio quality metrics)

### Назначение

Извлекает **базовые метрики качества аудио** для оценки технического состояния записи. Легковесный экстрактор без тяжелых зависимостей, предназначенный для быстрой оценки качества аудио сигнала.

**Версия**: 2.0.1 (Audit v4.2 observability)  
**Категория**: quality  
**GPU**: не требуется (CPU-only)

### Входы

- **`audio/audio.wav`** (Segmenter contract) — полное аудио для `run()`
- **`audio/segments.json`** (Segmenter contract, family `primary`) — сегменты для `run_segments()`

### Выходы

NPZ: `result_store/.../quality_extractor/quality_extractor_features.npz`, схема **`quality_extractor_npz_v2`** (`schemas/quality_extractor_npz_v2.json`, `docs/SCHEMA.md`).

#### Audit v4 — заметки по NPZ

- **Tabular:** только числа; строковый **`device_used`** ранее попадал в tabular и давал **NaN** через `as_float` — **исправлено в `npz_savers/quality.py`**, значение в **`meta.device_used`**.

#### Audit v4.2 observability:
- `meta.stage_timings_ms` (dict): тайминги этапов (ms), пишутся в NPZ meta
- `meta.quality_resource_profile` (dict|None): best-effort snapshot RSS/VMS (если включено)
  - включение: `AP_QUALITY_RESOURCE_PROFILE=1`

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Базовые метрики (feature-gated: `--quality-disable-basic-metrics` для отключения; Audit v3: включено по умолчанию)

- **`dc_offset`**: среднее смещение постоянной составляющей (DC offset)
  - **Интерпретация**: близко к 0 = хорошо, большое значение = проблема с записью
  - **Единицы**: нормализованные амплитуды (-1.0 до 1.0)
  - **Диапазон**: [-1.0, 1.0]
  - **Примечание**: `abs(dc_offset)` — использовать при необходимости; `dc_offset_abs` удалён (Audit v3)
  
- **`clipping_ratio`**: доля отсечённых сэмплов (clipping)
  - **Интерпретация**: 0.0 = нет клиппинга, >0.0 = есть перегрузка
  - **Единицы**: доля от общего количества сэмплов (0.0-1.0)
  - **Порог**: настраивается через `clip_threshold` (по умолчанию 0.999)
  - **Диапазон**: [0.0, 1.0]
  
- **`crest_factor_db`**: отношение пика к RMS в децибелах
  - **Интерпретация**: высокое значение = более динамичный сигнал
  - **Единицы**: децибелы (dB)
  - **Формула**: 20 * log10(peak / RMS)
  - **Диапазон**: ≥ 0 dB
  - **Типичные значения**: речь: 10-20 dB, музыка: 15-30 dB

#### Динамические метрики (feature-gated: `--quality-enable-dynamic-metrics`)

- **`dynamic_range_db`**: динамический диапазон (разница между 95 и 5 перцентилями уровня)
  - **Интерпретация**: больше = шире динамический диапазон
  - **Единицы**: децибелы (dB)
  - **Вычисление**: 95-й перцентиль - 5-й перцентиль уровней кадров
  - **Диапазон**: ≥ 0 dB
  - **Хорошее качество**: обычно >40 dB
  - **Примечание**: `snr_db` удалён (Audit v3: был дубликатом dynamic_range_db)

#### Анализ кадров (feature-gated: `--quality-enable-frame-analysis`)

- **`frame_levels_distribution`**: распределение уровней кадров (mean, std, min, max, median)
  - Статистики уровней кадров в dB
  - Полезно для анализа динамики сигнала

#### Дополнительные метрики для ML/аналитики (всегда включены, если включены basic_metrics)

- **`clipping_segments_count`**: количество сегментов с клиппингом (для `run_segments()`)
- **`crest_factor_median`**: медиана crest factor по кадрам (для `run_segments()`)
- **`dynamic_range_stability`**: стабильность dynamic range (для `run_segments()`, если включены dynamic_metrics)
- **`quality_score`**: композитная оценка качества на основе всех метрик (0.0-1.0, analytics tier)
  - Нормализует все метрики и вычисляет среднее
  - Высокие значения → лучше качество

#### Временные серии (feature-gated: `--quality-enable-time-series`)

- **`frame_levels_db_series`**: временная серия уровней кадров в dB (float32[])
- **`frame_rms_series`**: временная серия RMS по кадрам (float32[])
- **`clipping_segments_series`**: временная серия флагов клиппинга по кадрам (float32[], 0.0-1.0)
- **`dc_offset_series`**: временная серия DC offset (float32[], для `run_segments()`)
- **`clipping_ratio_series`**: временная серия clipping ratio (float32[], для `run_segments()`)
- **`crest_factor_db_series`**: временная серия crest factor (float32[], для `run_segments()`)
- **`dynamic_range_db_series`**: временная серия dynamic range (float32[], для `run_segments()`)
- **`snr_db_series`**: временная серия SNR (float32[], для `run_segments()`)
- **`segment_centers_sec`**: центры сегментов в секундах (float32[], для `run_segments()`)
- **`segment_durations_sec`**: длительности сегментов в секундах (float32[], для `run_segments()`)

**Примечание**: Большие временные серии (>1000 элементов) сохраняются в `.npy` файлы в `_artifacts/` и регистрируются в `manifest.json`.

#### Метаданные

- `device_used`: устройство обработки (`"cpu"`, `"cuda"` или `"auto"`) — в **NPZ** в **`meta`**, не в `feature_values`
- `sample_rate`: частота дискретизации аудио (Hz)
- `duration`: длительность аудио (секунды)
- `segments_count`: количество сегментов (для `run_segments()`)
- `quality_contract_version`: версия контракта (`"quality_contract_v1"`)
- `_features_enabled`: список включённых групп фичей (для отладки)
- `stage_timings_ms`: детальные метрики времени выполнения этапов обработки (dict):
  - `load_audio_ms`: время загрузки аудио (мс)
  - `extract_metrics_ms`: время извлечения метрик (мс)
  - `save_artifacts_ms`: время сохранения артефактов (мс)
  - `validate_output_ms`: время валидации выходных данных (мс)
  - `total_ms`: общее время обработки (мс)

### Feature Dependencies

- **`dynamic_range_stability`** зависит от **`dynamic_range_db`** (требует включения `--quality-enable-dynamic-metrics`)
- **`quality_score`** зависит от всех базовых и динамических метрик (analytics tier)
- **`frame_levels_distribution`** зависит от анализа кадров (требует включения `--quality-enable-frame-analysis`)
- **`clipping_segments_series`** зависит от **`clipping_ratio`** (требует включения basic_metrics и `--quality-enable-frame-analysis`)

### Конфигурация

#### CLI аргументы

```bash
# Параметры обработки
--quality-sample-rate 22050          # Частота дискретизации (Hz)
--quality-frame-len-ms 50.0          # Длина кадра для анализа уровней (мс)
--quality-hop-ms 25.0                 # Шаг между кадрами (мс)
--quality-clip-threshold 0.999        # Порог для определения клиппинга (0.0-1.0)
--quality-average-channels            # Усреднять каналы для многоканального аудио
--quality-enable-normalization        # Включить нормализацию аудио перед обработкой

# Feature gating (Audit v3: basic_metrics включены по умолчанию)
--quality-disable-basic-metrics     # Отключить базовые метрики (Audit v3: по умолчанию включены)
--quality-enable-dynamic-metrics    # Включить динамические метрики (dynamic_range_db)
--quality-enable-frame-analysis      # Включить анализ кадров (frame-level метрики)
--quality-enable-time-series         # Включить временные серии для всех метрик
```

#### Python API

```python
from src.extractors.quality_extractor import QualityExtractor

extractor = QualityExtractor(
    device="auto",
    sample_rate=22050,
    average_channels=True,
    frame_len_ms=50.0,
    hop_ms=25.0,
    clip_threshold=0.999,
    enable_normalization=False,
    enable_basic_metrics=True,
    enable_dynamic_metrics=False,
    enable_frame_analysis=False,
    enable_time_series=False,
    progress_callback=None,
    artifacts_dir=None,
)
```

### Алгоритмы

Все метрики вычисляются с использованием **numpy**:

1. **DC Offset**: среднее значение всех сэмплов (`np.mean(audio_samples)`)
2. **Clipping Ratio**: доля сэмплов, превышающих порог (`np.mean(np.abs(audio_samples) >= threshold)`)
3. **Crest Factor**: отношение пикового значения к RMS (`20 * log10(peak / rms)`)
4. **Dynamic Range**: разница между 95-м и 5-м перцентилями уровней кадров (Audit v3: snr_db удалён как дубликат)

### Обработка ошибок

Экстрактор использует **no-fallback policy** (fail-fast):

- **Ошибка загрузки аудио**: `quality_audio_load_failed`
- **Ошибка вычисления DC offset**: `quality_dc_offset_failed`
- **Ошибка вычисления clipping**: `quality_clipping_failed`
- **Ошибка вычисления crest factor**: `quality_crest_factor_failed`
- **Ошибка вычисления dynamic range**: `quality_dynamic_range_failed`
- **Ошибка анализа кадров**: `quality_frame_analysis_failed`
- **Ошибка валидации**: `quality_validation_failed`
- **Неизвестная ошибка**: `quality_unknown`

Все ошибки включают детальный `error_code` в сообщении об ошибке.

### Валидация

#### Валидация параметров (fail-fast)

- `sample_rate > 0`
- `frame_len_ms > 0`
- `hop_ms > 0`
- `hop_ms <= frame_len_ms`
- `clip_threshold ∈ [0, 1]`

#### Валидация выходных данных

- Проверка диапазонов значений (например, clipping_ratio ∈ [0, 1], crest_factor_db ≥ 0, dynamic_range_db ≥ 0)
- Проверка NaN/inf в значениях
- Проверка типов и размерностей

### Обработка многоканального аудио

Экстрактор автоматически преобразует многоканальное аудио в моно:

- Если `average_channels=True`: усредняет все каналы (по умолчанию)
- Если `average_channels=False`: использует первый канал

### Нормализация аудио

Опциональная нормализация аудио перед обработкой (включается через `--quality-enable-normalization`):

- Использует `AudioUtils.normalize_audio()` для нормализации амплитуды
- Может улучшить стабильность и точность метрик качества
- **Внимание**: нормализация может скрыть проблемы с исходным аудио (например, низкий уровень записи)

### Sampling / units-of-processing requirements

**Важно**: `quality_extractor` **не генерирует сегменты сам** — Segmenter является единственным владельцем sampling.

**Требования к сегментам**:
- Компонент использует семейство сегментов из `audio/segments.json`:
  - **`families.quality.segments[]`**: окна для анализа качества (обязательно для `run_segments()`)
- Сегменты должны иметь обязательные поля: `start_sample`, `end_sample`, `start_sec`, `end_sec`, `center_sec`
- Отсутствие обязательного семейства → fail-fast (`raise RuntimeError`)

**Sampling policy (Segmenter contract)**:
- Segmenter строит families по **универсальной нелинейной кривой** (sampling curve):
  - Параметры в `families.quality.sampling_curve`: `type="ease_out_power"`, `k∈(0,1]`, `linear_until_sec`, `cap_duration_sec`
  - На коротких видео можно близко к 1:1 (секунда→окно), на длинных рост замедляется и упирается в `max_windows`
- См. `docs/contracts/SEGMENTER_CONTRACT.md` для деталей sampling policy

**Минимальные требования**:
- Минимальная длительность сегмента: **50 мс** (для точности метрик)
- Минимальное количество сегментов: **1 сегмент** (иначе ошибка `segments_invalid`)

### Segmenter Contract

Экстрактор поддерживает работу на сегментах от Segmenter:

- **`run()`**: работает на полном аудио (`audio/audio.wav`)
- **`run_segments()`**: работает на сегментах из `audio/segments.json` (family `quality`)

Для `run_segments()`:
- Читает `families.quality.segments[]` из `audio/segments.json`
- Использует `start_sample/end_sample` для загрузки сегментов через `AudioUtils.load_audio_segment()`
- Агрегирует результаты по всем сегментам (статистики и временные серии)

### Progress Reporting

Экстрактор поддерживает progress reporting через callback:

- Для `run()`: обновление прогресса для каждой метрики (DC offset, clipping, crest factor, dynamic range, SNR, frame analysis)
- Для `run_segments()`: обновление прогресса каждые 10% сегментов

### Per-run Storage

Большие временные серии (>1000 элементов) сохраняются в `.npy` файлы:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/quality_extractor/_artifacts/*.npy`
- Регистрация в `manifest.json.components[].artifacts[]` (type=`"npy"`)

### Visualization

#### Рекомендации для UI/сайта

1. **Timeline визуализация**:
   - Отображать временные серии метрик (dc_offset, clipping_ratio, crest_factor_db, dynamic_range_db, snr_db) на временной шкале
   - Использовать line charts с разными цветами для каждой метрики
   - Добавить tooltips с точными значениями при наведении
   - Выделять сегменты с клиппингом (clipping_ratio > 0)

2. **Метрики качества**:
   - Отображать базовые метрики (dc_offset, clipping_ratio, crest_factor_db) как отдельные карточки
   - Использовать gauge charts для динамических метрик (dynamic_range_db, snr_db)
   - Показывать quality_score как общую оценку качества (0.0-1.0)

3. **Распределения**:
   - Гистограммы для frame_levels_distribution
   - Box plots для сравнения метрик между сегментами

4. **Предупреждения**:
   - Выделять проблемы: высокий DC offset (>0.01), клиппинг (clipping_ratio > 0.01), низкий SNR (<20 dB)
   - Использовать цветовую индикацию (зелёный = хорошо, жёлтый = предупреждение, красный = проблема)

5. **Интерактивные элементы**:
   - Фильтры для выбора метрик
   - Zoom для детального просмотра временных серий
   - Сравнение метрик между разными видео

#### Локальный HTML renderer для дебага

Используйте `render_quality_extractor_html()` для генерации HTML страницы с результатами:

```python
from src.core.renderer import render_quality_extractor_html

render_quality_extractor_html(
    npz_path="result_store/.../quality_extractor/quality_extractor_features.npz",
    output_path="quality_debug.html"
)
```

HTML страница включает:
- Summary (sample_rate, device, duration, segments_count)
- Таблицы метрик для всех включённых групп
- Дополнительные метрики
- Временные серии (если включены)
- Raw JSON данные

### Models

**Важно**: `quality_extractor` не использует ML-модели, а выполняет только сигнальную обработку через numpy.

**Используемые библиотеки**:
- **numpy**: все вычисления выполняются через numpy (векторизованные операции)
- **AudioUtils**: загрузка и предобработка аудио (нормализация, конвертация форматов)

**Runtime**: `inprocess` (CPU-only)  
**Engine**: `numpy` (сигнальная обработка)  
**Precision**: `fp32`  
**Device**: `cpu`  
**Triton**: ❌ Нет (in-process signal processing)

**Загрузка**: нет моделей, только библиотеки (numpy, AudioUtils)

### Parallelization

**Внутренний параллелизм** (внутри компонента):
- **CPU parallelism**: компонент поддерживает batch processing через `extract_batch_segments()` с использованием `ThreadPoolExecutor`
  - Параллельная обработка сегментов из нескольких видео одновременно
  - Количество воркеров контролируется через `max_workers` (по умолчанию `os.cpu_count()`)
- **Векторизация**: использует `numpy.lib.stride_tricks.as_strided` для эффективного формирования кадров

**Внешний параллелизм** (выше компонента):
- **Batch processing**: компонент batch-safe и может обрабатывать несколько файлов параллельно через `extract_batch_segments()`
  - Каждый файл обрабатывается изолированно через `run_segments()`
  - Изоляция данных: каждый файл имеет свой `tmp_path` и `artifacts_dir`
- **Video-level parallelism**: компонент может обрабатываться параллельно на разных видео (разные `run_id`)
  - Требования к изоляции: разные `run_id`, разные `result_store` пути
  - Thread-safety: компонент thread-safe (read-only shared state)

**Ограничения**:
- Компонент не поддерживает GPU batching (CPU-only)
- Параллелизм ограничен CPU (ThreadPoolExecutor)
- Требования к памяти: низкие (только numpy массивы)

### Performance characteristics

**Resource costs**:
- **CPU**: минимальные (только numpy операции)
- **GPU**: не используется
- **Estimated duration**: ~0.5 секунды для типичного аудио файла (полное аудио)
- **Estimated duration (segments)**: ~0.05-0.1 секунды на сегмент (зависит от длины сегмента)

**Параметры производительности**:
- `frame_len_ms`: меньшие значения → больше кадров → точнее, но медленнее
- `hop_ms`: меньшие значения → больше перекрытие → точнее, но медленнее
- `enable_frame_analysis`: `False` → меньше вычислений, быстрее
- `enable_time_series`: `False` → меньше данных в payload, быстрее передача
- `enable_normalization`: `True` → дополнительная обработка, немного медленнее
- Векторизация через `numpy.lib.stride_tricks.as_strided` для эффективности

**Источник данных**: `docs/models_docs/resource_costs/quality_extractor_costs_v1.json` (если доступен)

### Связанные компоненты

- **AudioUtils**: загрузка и предобработка аудио
- **BaseExtractor**: базовый интерфейс экстрактора
- **numpy**: все вычисления выполняются через numpy
- **Segmenter**: источник сегментов для `run_segments()`

### Примечания

1. **DC Offset**: идеальное значение близко к 0, но небольшие отклонения нормальны (<0.01)
2. **Clipping**: порог 0.999 означает, что сэмплы с абсолютным значением >= 0.999 считаются клиппированными
3. **Crest Factor**: типичные значения для речи: 10-20 dB, для музыки: 15-30 dB
4. **Dynamic Range**: хорошее качество обычно имеет диапазон >40 dB
5. **SNR**: грубая оценка, не заменяет профессиональные инструменты измерения SNR
6. **Короткие файлы**: для очень коротких файлов (<50 мс) некоторые метрики могут быть неточными
7. **Векторизация**: использует stride_tricks для эффективного формирования кадров, с fallback на цикл при ошибках
8. **Feature gating**: все фичи opt-in (по умолчанию все выключены) для контроля размера NPZ и стоимости вычислений
9. **Временные серии**: большие серии (>1000 элементов) автоматически сохраняются в `.npy` файлы для экономии памяти
10. **Contract versioning**: используется `quality_contract_version="quality_contract_v1"` для валидации совместимости с downstream extractors
