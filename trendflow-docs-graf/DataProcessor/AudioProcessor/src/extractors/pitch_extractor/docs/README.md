## `pitch_extractor` (Spectral features)

### Назначение

Извлекает **основную частоту (f0)** из аудио сигнала с использованием нескольких алгоритмов: PYIN, YIN и опционально torchcrepe (CREPE). Автоматически выбирает лучший метод на основе качества оценки и вычисляет статистические метрики высоты тона.

**Версия**: 2.0.1 (Audit v4.2 observability)  
**Категория**: spectral  
**GPU**: optional (torchcrepe может использовать GPU, но не требуется)

**Audit v4 (эмпирика по NPZ):** [`DataProcessor/docs/audit_v4/components/audio_processor/pitch_extractor_audit_v4.md`](../../../../../docs/audit_v4/components/audio_processor/pitch_extractor_audit_v4.md)

### Каталог полей NPZ (кратко)

| Поле / группа | Где в NPZ | Как получают | Единицы / смысл |
|---------------|-----------|--------------|-----------------|
| `sample_rate`, `hop_length`, `frame_length`, `fmin`, `fmax` | `feature_names` / `feature_values` | Параметры экстрактора и STFT/pyin | Гц / сэмплы |
| `duration` | tabular | Длительность обработанного аудио | с |
| `segments_count` | tabular | Число окон Segmenter (`run_segments`) | — |
| `backend` | **`meta.backend`** (не float в таблице) | `"classic"` (librosa pyin+yin) или `"torchcrepe"` | строка |
| `f0_mean`, `f0_std`, `f0_min`, `f0_max`, `f0_median` | tabular при `basic_stats` | Статистики по **выбранному** f0 в `run()`; в `run_segments` — по ряду средних f0 по сегментам (см. ниже) | Гц |
| `pitch_contour_smoothness` | tabular при `basic_stats` | `1/(1 + std(second_diff(f0)))` по одномерному ряду f0 | 0…1, выше = гладче |
| `pitch_jump_count` | tabular при `basic_stats` | Число соседних пар с \|Δf0\| > ~2 полутонов (порог от среднего f0) | целое (в NPZ как float32) |
| `pitch_octave_distribution` | массив-скаляр `object` | `numpy.histogram` по бинам [50,100,200,400,800,1600] Гц, доли | dict `octave_0`… |
| `pitch_skewness`, `pitch_kurtosis` | tabular при `basic_stats` | Асимметрия и эксцесс распределения f0 (`scipy.stats` или моменты) | — |
| `segment_*` | векторы длины N | Края окон из Segmenter; `segment_mask` — был ли в сегменте валидный pitch | с, bool |
| `f0_method` | `meta.f0_method` | Имя лучшего метода (`pyin`/`yin`/…) или `aggregated` для сегментов | str |
| Dense f0 по времени | не в NPZ по умолчанию | При `enable_time_series`: ряд уходит в `.npy`, путь в `meta.extra.f0_series_npy` | Гц по кадрам анализа |

#### Audit v4.2 observability:
- `meta.stage_timings_ms` (dict): тайминги этапов (ms), пишутся в NPZ meta
- `meta.pitch_resource_profile` (dict|None): best-effort snapshot RSS/VMS/VRAM (если включено)
  - включение: `AP_PITCH_RESOURCE_PROFILE=1`

**Режим сегментов:** для каждого окна считается полный pyin/yin, из результата берётся скаляр `f0_mean`; дальнейшие метрики (smoothness, jumps, histogram) считаются уже по **нескольких** таких точках — это быстро и устойчиво, но не то же самое, что полный контур по всему клипу.

### Входы

- **`audio/audio.wav`** (любой аудио файл, поддерживаемый AudioUtils)
- **`audio/segments.json`** (опционально, для `run_segments()`): family `pitch` с сегментами

**Режимы работы**:
- **`run()`**: работа на полном аудио (legacy mode)
- **`run_segments()`**: работа на сегментах от Segmenter (рекомендуется для production)

### Выходы

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Обязательные поля (всегда присутствуют)

- `device_used`: устройство обработки (`"cpu"`, `"cuda"` или `"auto"`)
- `sample_rate`: частота дискретизации аудио (Hz)
- `pitch_contract_version`: версия контракта для валидации совместимости (str, `"pitch_contract_v1"`)

#### Feature-gated поля (включаются через флаги)

**`--pitch-enable-basic-stats`**:
- `f0_mean`: среднее значение основной частоты (Hz) (float)
- `f0_std`: стандартное отклонение основной частоты (Hz) (float)
- `f0_min`: минимальное значение f0 (Hz) (float)
- `f0_max`: максимальное значение f0 (Hz) (float)
- `f0_median`: медианное значение f0 (Hz) (float)
- `f0_method`: выбранный метод (`"pyin"`, `"yin"`, `"torchcrepe"` или `"none"`) (str)
- `pitch_contour_smoothness`: гладкость контура pitch (0 = негладкий, 1 = гладкий) (float)
- `pitch_jump_count`: количество больших скачков pitch (>2 semitones) (int; в NPZ может храниться как float32)
- `pitch_skewness`: асимметрия распределения pitch (float)
- `pitch_kurtosis`: эксцесс распределения pitch (float)
- `pitch_octave_distribution`: распределение pitch по октавам (dict[octave_id, ratio])

**`--pitch-enable-stability-metrics`**:
- `pitch_variation`: вариация высоты тона (std отклонение разностей) (float)
- `pitch_stability`: стабильность высоты тона (1 / (1 + variation), 0 = нестабильная, 1 = стабильная) (float)
- `pitch_range`: диапазон высоты тона (max - min, Hz) (float)

**`--pitch-enable-delta-features`**:
- `f0_delta_mean`: среднее изменение f0 между кадрами (Hz) (float)
- `f0_delta_std`: стандартное отклонение изменений f0 (Hz) (float)
- `f0_delta_abs_mean`: среднее абсолютное изменение f0 (Hz) (float)

**`--pitch-enable-method-stats`**:
- `f0_mean_pyin`, `f0_std_pyin`, `f0_min_pyin`, `f0_max_pyin`, `f0_median_pyin`: статистики PYIN (float)
- `f0_count_pyin`: количество валидных кадров PYIN (int)
- `voiced_fraction_pyin`: доля озвученных кадров PYIN (0.0-1.0) (float)
- `voiced_probability_mean_pyin`: средняя вероятность озвученности PYIN (float)
- `f0_mean_yin`, `f0_std_yin`, `f0_min_yin`, `f0_max_yin`, `f0_median_yin`: статистики YIN (float)
- `f0_count_yin`: количество валидных кадров YIN (int)
- `f0_mean_torchcrepe`, `f0_std_torchcrepe`, `f0_min_torchcrepe`, `f0_max_torchcrepe`, `f0_median_torchcrepe`: статистики torchcrepe (float, если используется)
- `f0_count_torchcrepe`: количество валидных кадров torchcrepe (int, если используется)

**`--pitch-enable-time-series`**:
- `f0_series_pyin`: временная серия f0 PYIN (list[float])
- `f0_series_yin`: временная серия f0 YIN (list[float])
- `f0_series_torchcrepe`: временная серия f0 torchcrepe (list[float] или путь к .npy файлу)
- `f0_series`: агрегированная временная серия f0 (для `run_segments()`) (list[float])
- `segment_centers_sec`: центры сегментов в секундах (для `run_segments()`) (list[float])
- `segment_durations_sec`: длительности сегментов в секундах (для `run_segments()`) (list[float])

#### Специальные случаи

**Пустое аудио** (status="empty"):
- `status`: `"empty"`
- `empty_reason`: `"audio_silent"` (если применимо)
- Остальные поля присутствуют (без метрик)

### Feature Dependencies

**Зависимости между фичами**:
- `stability_metrics` зависят от `basic_stats` (требуют `f0_mean`, `f0_std`, `f0_min`, `f0_max`)
- `delta_features` зависят от `basic_stats` (требуют временную серию f0)
- `method_stats` независимы от других фичей
- `time_series` независимы от других фичей

**Зависимости от других extractors**:
- Нет зависимостей от других extractors

**Contract version для совместимости**:
- `pitch_contract_version="pitch_contract_v1"` используется для валидации совместимости с downstream extractors (например, `speech_analysis_extractor`)

### Алгоритмы

#### 1. PYIN (Probabilistic YIN)

- **Библиотека**: librosa
- **Особенности**: наиболее устойчивый метод, использует вероятностную модель
- **Выход**: f0, флаг озвученности, вероятности озвученности
- **Использование**: основной метод по умолчанию (classic backend)

#### 2. YIN

- **Библиотека**: librosa
- **Особенности**: классический алгоритм автокорреляции
- **Выход**: f0
- **Использование**: fallback метод (classic backend)

#### 3. torchcrepe (CREPE)

- **Библиотека**: torchcrepe (опционально)
- **Особенности**: нейросетевая модель, более точная, но требует PyTorch
- **Вход**: аудио ресемплированное до 16 kHz
- **Выход**: f0 с фильтрацией по периодичности
- **Использование**: если `backend="torchcrepe"` и библиотека установлена

### Выбор лучшего метода

Экстрактор автоматически выбирает лучший метод на основе взвешенной оценки (только для classic backend, torchcrepe используется приоритетно если выбран):

```
score = 0.6 * f0_mean + 0.3 * voiced_fraction * 100 + 0.1 * f0_count
```

Выбирается метод с наибольшим score (PYIN vs YIN). Если torchcrepe доступен и выбран как backend, он используется приоритетно (fail-fast, no-fallback).

### Конфигурация

#### Параметры модели

```python
{
    "device": "auto",                  # "auto" | "cuda" | "cpu"
    "sample_rate": 22050,              # Частота дискретизации (Hz)
    "fmin": 50.0,                      # Минимальная частота f0 (Hz, ≥20)
    "fmax": 2000.0,                    # Максимальная частота f0 (Hz, ≤8000)
    "hop_length": 512,                 # Размер шага между кадрами (samples, >0)
    "frame_length": 2048,              # Размер окна анализа (samples, >0)
    "backend": "classic",              # "classic" | "torchcrepe"
    "channel_mode": "first",           # "first" | "mean" | "max" (для многоканального аудио)
    "torchcrepe_batch_size": 1,        # Размер батча для torchcrepe
    "artifacts_dir": None,             # Директория для сохранения .npy файлов (per-run storage)
}
```

#### Python API

```python
from src.extractors.pitch_extractor import PitchExtractor

extractor = PitchExtractor(
    device="auto",
    sample_rate=22050,
    fmin=50.0,
    fmax=2000.0,
    hop_length=512,
    frame_length=2048,
    backend="classic",
    channel_mode="first",
    torchcrepe_batch_size=1,
    enable_basic_stats=False,
    enable_stability_metrics=False,
    enable_delta_features=False,
    enable_method_stats=False,
    enable_time_series=False,
    progress_callback=None,
    artifacts_dir=None,
)
```

#### Feature Gating (персональные флаги)

По умолчанию (Audit v3 / `MainProcessor`) **`basic_stats` включён**: `f0_*`, дополнительные метрики контура/октав/skew/kurtosis, octave distribution. Остальное — opt-in:

- `--pitch-disable-basic-stats` / `--pitch-enable-basic-stats`: выключить или явно включить базовые статистики и связанные аналитические метрики
- `--pitch-enable-stability-metrics`: включить метрики стабильности (pitch_variation, pitch_stability, pitch_range)
- `--pitch-enable-delta-features`: включить delta-признаки (f0_delta_mean, f0_delta_std, f0_delta_abs_mean)
- `--pitch-enable-method-stats`: включить статистики по каждому методу (PYIN, YIN, torchcrepe)
- `--pitch-enable-time-series`: включить временные серии (f0_series_pyin, f0_series_yin, f0_series_torchcrepe)

**Рекомендации для обучения моделей**:
- Включить все фичи для максимального качества и полноты данных

### Архитектура

#### 1. Валидация параметров (fail-fast)

1. Проверка диапазонов: `fmin > 0`, `fmax > fmin`, `hop_length > 0`, `frame_length > 0`, `sample_rate > 0`
2. Проверка разумных значений: `fmin ≥ 20 Hz`, `fmax ≤ 8000 Hz`
3. Fail-fast при невалидных параметрах

#### 2. Обработка аудио

**`run()` (полное аудио)**:
1. Загрузка аудио через `AudioUtils.load_audio()` с ресемплированием до `sample_rate`
2. Нормализация: приведение к моно каналу согласно `channel_mode`
3. Извлечение f0: запуск выбранного backend (torchcrepe или classic: PYIN + YIN)
4. Выбор лучшего метода (только для classic backend)
5. Вычисление статистик (feature-gated)
6. Сохранение больших временных серий в .npy (per-run storage)
7. Валидация выходных данных
8. Формирование payload

**`run_segments()` (сегменты от Segmenter)**:
1. Валидация сегментов (fail-fast при пустых сегментах)
2. Обработка каждого сегмента:
   - Загрузка сегмента через `AudioUtils.load_audio_segment()`
   - Нормализация аудио
   - Извлечение pitch для сегмента
   - Агрегация f0_mean из сегмента
3. Progress reporting: каждые 10% сегментов
4. Агрегация результатов (feature-gated)
5. Валидация выходных данных
6. Формирование payload

#### 3. Вспомогательные методы

- `_validate_parameters()`: валидация входных параметров (fail-fast)
- `_validate_output()`: полная валидация выходных данных (диапазоны, NaN/inf, консистентность)
- `_classify_error()`: классификация ошибок с детальными error codes
- `_extract_pitch_features()`: извлечение pitch признаков с использованием выбранного backend (no-fallback)
- `_extract_torchcrepe()`: извлечение f0 через torchcrepe
- `_calc_stats()`: вычисление статистик для f0 массива (feature-gated)
- `_calc_additional_metrics()`: вычисление дополнительных метрик для ML/аналитики
- `_save_time_series_artifacts()`: сохранение больших временных серий в .npy (per-run storage)

#### 4. Обработка ошибок

**Политика NO FALLBACK**:
- Если выбранный backend падает → ошибка (fail-fast)
- Если все методы вернули пустые результаты → ошибка
- Валидация параметров → ошибка при невалидных параметрах
- Валидация выходных данных → ошибка при невалидных данных

**Детальные error codes**:
- `pitch_audio_load_failed`: ошибка загрузки аудио
- `pitch_torchcrepe_failed`: ошибка torchcrepe (если выбран как backend)
- `pitch_pyin_failed`: ошибка PYIN (если используется classic backend)
- `pitch_yin_failed`: ошибка YIN (если используется classic backend)
- `pitch_all_methods_failed`: все методы вернули пустые результаты
- `pitch_validation_failed`: ошибка валидации выходных данных
- `pitch_unknown`: неизвестная ошибка

### Обработка ошибок

- **Пустой вход**: возвращает ошибку с `error_code="pitch_all_methods_failed"`
- **Ошибка метода**: fail-fast с детальным error_code (no-fallback)
- **torchcrepe недоступен**: ошибка с `error_code="pitch_torchcrepe_failed"` (если выбран как backend)
- **NaN значения**: автоматически фильтруются перед вычислением статистик, но валидируются в выходных данных
- **Невалидные параметры**: fail-fast с `error_code="pitch_validation_failed"`

### Особенности

- **Множественные методы**: использует несколько алгоритмов для повышения надежности (classic backend)
- **Автоматический выбор**: выбирает лучший метод на основе качества оценки (только для classic backend)
- **Опциональный torchcrepe**: может использовать нейросетевую модель при наличии (приоритетно, если выбран)
- **Эффективное хранение**: большие временные серии сохраняются в .npy файлы (per-run storage)
- **Стабильность**: вычисляет метрики стабильности и вариации высоты тона
- **Delta-признаки**: вычисляет изменения f0 между кадрами для анализа динамики
- **Feature gating**: все фичи opt-in через персональные флаги
- **Contract versioning**: версия контракта для валидации совместимости с downstream extractors
- **Полная валидация**: проверка параметров и выходных данных
- **Progress reporting**: обновление прогресса для каждого метода и сегмента
- **No-fallback policy**: fail-fast при ошибках методов
- **Per-run storage**: .npy файлы сохраняются в per-run storage и регистрируются в manifest.json
- **Дополнительные метрики**: pitch_contour_smoothness, pitch_jump_count, pitch_octave_distribution, pitch_centroid, pitch_skewness, pitch_kurtosis

### Performance characteristics

**Resource costs**:
- **CPU**: умеренные (librosa операции)
- **GPU**: опционально (только для torchcrepe)
- **Estimated duration**: ~1.0-2.0 секунд для типичного аудио файла (полное аудио), ~0.1-0.5 секунд на сегмент (run_segments)

**Параметры производительности**:
- `hop_length`: меньшие значения → больше кадров → выше точность, но медленнее
- `frame_length`: большие значения → лучше для низких частот, но медленнее
- `torchcrepe_batch_size`: большие значения → быстрее на GPU, но больше памяти

### Visualization

**Рекомендации для UI/сайта**:

1. **Timeline визуализация**:
   - Отображение `f0_series` по времени (line chart)
   - Цветовая кодировка по методам (PYIN, YIN, torchcrepe)
   - Для `run_segments()`: отображение `f0_series` по `segment_centers_sec`

2. **Статистики**:
   - Отображение `f0_mean`, `f0_std`, `f0_min`, `f0_max` (stat cards)
   - Визуализация `pitch_stability` (progress bar или gauge)
   - Отображение `pitch_range` (range indicator)

3. **Распределения**:
   - Визуализация `pitch_octave_distribution` (bar chart или pie chart)
   - Histogram распределения f0 значений

4. **Метрики стабильности**:
   - Отображение `pitch_variation` и `pitch_stability` (stat cards)
   - Визуализация `pitch_jump_count` (counter)

5. **Delta features**:
   - Визуализация `f0_delta_mean` и `f0_delta_std` (stat cards)
   - Timeline с изменениями f0 (first derivative)

6. **Сравнение методов**:
   - Side-by-side сравнение PYIN, YIN, torchcrepe (если `enable_method_stats`)
   - Визуализация `voiced_fraction_pyin` (progress bar)

7. **HTML renderer для дебага**:
   - Локальный HTML renderer доступен через `render_pitch_extractor_html()`
   - Включает все метрики, распределения, статистики, временные серии
   - Интерактивные графики с Plotly
   - Только для локального дебага, не в production артефактах

**Пример использования HTML renderer**:
```python
from src.core.renderer import render_pitch_extractor_html

html_path = render_pitch_extractor_html(
    npz_path="result_store/.../pitch_extractor/pitch_extractor_features.npz",
    output_path="debug_pitch.html"
)
```

### Связанные компоненты

- **AudioUtils**: загрузка и предобработка аудио
- **BaseExtractor**: базовый интерфейс экстрактора
- **librosa**: библиотека для PYIN и YIN алгоритмов
- **torchcrepe** (опционально): нейросетевая модель для оценки f0
- **Segmenter**: предоставляет сегменты для анализа (для `run_segments()`)
- **speech_analysis_extractor**: использует pitch_extractor для анализа речи

### Использование

#### Полное аудио (legacy mode)

```python
result = extractor.run(
    input_uri="path/to/audio.wav",
    tmp_path="/tmp"
)
```

#### Сегменты от Segmenter (рекомендуется)

```python
result = extractor.run_segments(
    input_uri="path/to/audio.wav",
    tmp_path="/tmp",
    segments=[...]  # от Segmenter families.pitch
)
```

### Примечания

1. **Диапазон частот**: `fmin` и `fmax` должны соответствовать типу аудио (речь: 50-500 Hz, музыка: шире)
2. **Ресемплирование**: torchcrepe требует 16 kHz, автоматически ресемплирует
3. **Периодичность**: torchcrepe фильтрует результаты по периодичности (threshold=0.1)
4. **Валидные кадры**: методы могут возвращать NaN для неозвученных участков, они автоматически фильтруются
5. **Выбор метода**: если все методы вернули пустые результаты, устанавливается `f0_method="none"` и возвращается ошибка
6. **No-fallback policy**: если выбранный backend падает, возвращается ошибка (не используется fallback на другие методы)
7. **Feature gating**: все фичи opt-in через персональные флаги (default: все False)
8. **Contract versioning**: версия контракта `pitch_contract_v1` используется для валидации совместимости с downstream extractors
9. **Per-run storage**: большие временные серии сохраняются в `.npy` файлы в `result_store/<platform_id>/<video_id>/<run_id>/pitch_extractor/_artifacts/` и регистрируются в `manifest.json`
10. **Валидация**: полная валидация параметров (fail-fast) и выходных данных (диапазоны, NaN/inf, консистентность)
11. **Progress reporting**: обновление прогресса для каждого метода (PYIN, YIN, torchcrepe) и для каждого сегмента (run_segments)
12. **Дополнительные метрики**: pitch_contour_smoothness, pitch_jump_count, pitch_octave_distribution, pitch_skewness, pitch_kurtosis для ML/аналитики
13. **Batch processing**: в текущей версии не поддерживается (нет методов `extract_batch_segments()` и `supports_batch`)
14. **Stage timings**: в текущей версии не сохраняются в `stage_timings_ms` (в отличие от других экстракторов)
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
