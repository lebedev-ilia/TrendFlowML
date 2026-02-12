## `rhythmic_extractor` (Audio Tier‑1, optional)

### Назначение

Извлекает **ритмические метрики** из аудио сигнала: beat tracking (отслеживание битов), регулярность ритма, плотность ударов, статистику интервалов между ударами и дополнительные ML/analytics метрики. Использует librosa или Essentia для beat tracking (явный выбор backend, no-fallback policy).

**Версия**: 2.0.0  
**Категория**: rhythm  
**GPU**: не требуется (CPU-only обработка)

### Входы (строго, no‑fallback)

- **`audio/audio.wav`** (Segmenter)
- **`audio/segments.json`** (Segmenter contract, family `rhythmic`) — сегменты для `run_segments()`

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/rhythmic_extractor/rhythmic_extractor_features.npz` (**фиксированное имя**)

Схема: `audio_npz_v1` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).

#### Полезные поля payload (feature-gated):

**Basic metrics** (`--rhythmic-enable-basic-metrics`):
- `rhythm_tempo_bpm`: темп в ударах в минуту (BPM) (float, типичные значения: 60-180)
- `rhythm_beats_count`: количество обнаруженных битов (int)
- `rhythm_beat_density`: плотность ударов (количество ударов в секунду) (float)

**Interval stats** (`--rhythmic-enable-interval-stats`):
- `rhythm_avg_period_sec`: средний период между ударами в секундах (float)
- `rhythm_period_std_sec`: стандартное отклонение периодов в секундах (float)
- `rhythm_median_period_sec`: медианный период между ударами в секундах (float)
- `rhythm_min_period_sec`: минимальный период между ударами в секундах (float)
- `rhythm_max_period_sec`: максимальный период между ударами в секундах (float)

**Regularity metrics** (`--rhythmic-enable-regularity-metrics`):
- `rhythm_regularity`: коэффициент регулярности ритма (0-1, где 1 = идеально регулярный) (float)
- `rhythm_syncopation_score`: мера синкопированности (float)
- `rhythm_polyrhythm_score`: мера полиритмичности (float)
- `rhythm_beat_strength_mean`: средняя сила ударов (float)
- `rhythm_beat_strength_std`: стандартное отклонение силы ударов (float)
- `rhythm_metrical_stability`: метрическая стабильность (float)

**Tempo metrics** (`--rhythmic-enable-tempo-metrics`):
- `rhythm_median_bpm`: медианный темп, вычисленный из медианного периода (BPM) (float)
- `rhythm_tempo_variation`: вариация темпа (коэффициент вариации интервалов) (float)
- `rhythm_beat_consistency`: консистентность ударов (0-1, где 1 = идеально консистентный) (float)
- `rhythm_tempo_mean`: средний темп по сегментам (для `run_segments()`) (float)
- `rhythm_tempo_std`: стандартное отклонение темпа по сегментам (для `run_segments()`) (float)
- `rhythm_tempo_min`: минимальный темп по сегментам (для `run_segments()`) (float)
- `rhythm_tempo_max`: максимальный темп по сегментам (для `run_segments()`) (float)

**Beat times** (`--rhythmic-enable-beat-times`):
- `beat_times`: массив временных меток ударов в секундах (numpy array, dtype: float32)
- Если `beat_times.size > 10000`, сохраняется в `.npy` файл в `_artifacts/beat_times.npy`

**Per-segment data** (для `run_segments()`):
- `segment_centers_sec`: центры сегментов в секундах (float32[L])
- `segment_durations_sec`: длительности сегментов в секундах (float32[L])
- `segment_beat_times`: список массивов временных меток ударов для каждого сегмента (List[List[float]])
- `segments_count`: количество сегментов (int)

**Метаданные:**
- `sample_rate`: частота дискретизации (int)
- `hop_length`: размер hop для анализа (int, по умолчанию 512)
- `duration`: длительность аудио в секундах (float)
- `device_used`: устройство обработки (str, обычно "cpu")
- `backend`: используемый backend ("librosa" | "essentia")
- `rhythmic_contract_version`: версия контракта ("rhythmic_contract_v1")
- `_features_enabled`: список включенных фичей (List[str])

### Модели

ML модели **не используются** (signal processing only). `models_used[]` пустой.

**Библиотеки:**
- **librosa**: основная библиотека для beat tracking (default backend)
- **essentia** (опционально): более точный алгоритм beat tracking (если доступен)

### Feature Dependencies

- **Нет зависимостей от других extractors**: `rhythmic_extractor` работает независимо.
- **Отличие от `tempo_extractor`**: `rhythmic_extractor` фокусируется на beat tracking и регулярности, `tempo_extractor` — на BPM через sliding windows.
- **Отличие от `onset_extractor`**: `rhythmic_extractor` фокусируется на отслеживании битов (beat tracking), `onset_extractor` — на обнаружении атак звука (onset detection).

### Конфигурация

```python
{
    "device": "auto",              # "auto" | "cpu" (GPU не используется)
    "sample_rate": 22050,           # Частота дискретизации
    "hop_length": 512,              # Размер hop для анализа
    "average_channels": true,       # Усреднять каналы для многоканального аудио
    "backend": "librosa",           # "librosa" | "essentia" (no-fallback policy)
    # Librosa parameters (optional)
    "start_bpm": null,              # Начальный BPM для librosa beat tracking (None = auto)
    "std_bpm": null,                # Стандартное отклонение BPM для librosa (None = auto)
    "ac_size": 4,                   # Размер автокорреляции для librosa (1-16)
    "max_tempo": null,              # Максимальный темп для librosa (None = auto)
    # Feature gating flags (all default: False)
    "enable_basic_metrics": false,          # Включить базовые метрики (tempo_bpm, beats_count, beat_density)
    "enable_interval_stats": false,         # Включить статистики интервалов
    "enable_regularity_metrics": false,     # Включить метрики регулярности
    "enable_beat_times": false,             # Включить временные метки ударов
    "enable_tempo_metrics": false,          # Включить метрики темпа
    "enable_audio_normalization": false,    # Включить нормализацию аудио перед обработкой
}
```

### Параметры

#### Обязательные параметры

- `sample_rate` (int): Частота дискретизации аудио (Hz). Должна быть > 0.
- `hop_length` (int): Размер hop для анализа (samples). Должна быть > 0.

#### Опциональные параметры

- `backend` (str): Backend для beat tracking. Должен быть "librosa" или "essentia" (no-fallback policy).
- `start_bpm` (float, optional): Начальный BPM для librosa beat tracking. Должен быть в (0, 300].
- `std_bpm` (float, optional): Стандартное отклонение BPM для librosa. Должно быть в (0, 100].
- `ac_size` (int): Размер автокорреляции для librosa. Должен быть в [1, 16].
- `max_tempo` (float, optional): Максимальный темп для librosa. Должен быть в (0, 300].

#### Feature gating flags

Все feature gating flags по умолчанию `False` (opt-in):

- `enable_basic_metrics`: Включить базовые метрики (tempo_bpm, beats_count, beat_density)
- `enable_interval_stats`: Включить статистики интервалов (avg_period, std_period, min/max/median)
- `enable_regularity_metrics`: Включить метрики регулярности (regularity, syncopation, polyrhythm, beat_strength, metrical_stability)
- `enable_beat_times`: Включить временные метки ударов (beat_times array)
- `enable_tempo_metrics`: Включить метрики темпа (median_bpm, tempo_variation, beat_consistency, tempo_mean/std/min/max)

### Параллелизм / батчинг / лимиты

- **Segment-level parallelism**: Поддерживается через `run_segments()` с параметрами `segment_parallelism` и `max_inflight`.
- **Batch processing**: Поддерживается через `extract_batch_segments()` с CPU parallelism для обработки нескольких видео одновременно.
- **CPU-only**: GPU не используется (signal processing).
- **OOM handling**: Fail-fast при ошибках обработки (no-fallback policy).

### Parallelization

**Внутренний параллелизм** (внутри компонента):
- **Segment parallelism**: обработка сегментов внутри одного файла может выполняться параллельно через `ThreadPoolExecutor` (параметр `segment_parallelism`).
- **Batch processing**: при обработке нескольких файлов одновременно, каждый файл обрабатывается в отдельном потоке (параметр `max_workers` в `extract_batch_segments()`).

**Внешний параллелизм** (выше компонента):
- Можно запускать несколько экземпляров компонента параллельно на разных видео (разные `run_id`, разные `result_store` пути).
- Требования к изоляции: каждый файл должен иметь свой `artifacts_dir` для сохранения `.npy` файлов.

**Комбинированный подход**:
- Внутренний segment parallelism (обработка сегментов внутри файла) + внешний batch processing (обработка нескольких файлов одновременно).
- Рекомендуется использовать `max_workers = os.cpu_count()` для оптимальной утилизации CPU.

**Ограничения**:
- Thread-safety: компонент thread-safe для параллельной обработки разных файлов.
- Требования к памяти: peak memory зависит от количества параллельных воркеров и размера обрабатываемых сегментов.

### Качество: sanity checks

Валидация выходных данных:
- **Диапазоны**: `tempo_bpm ∈ [40, 300]`, `regularity ∈ [0, 1]`, `beat_density ∈ [0, 10]`
- **NaN/inf**: Проверка на наличие NaN и inf во всех метриках
- **Консистентность**: Проверка согласованности `tempo_bpm` и `avg_period` (допуск 10 BPM)

### Визуализация

Рекомендации по визуализации данных для UI/сайта:

1. **Timeline визуализация**:
   - Отобразить `beat_times` как вертикальные линии на временной шкале
   - Использовать интерактивный график с возможностью zoom и pan
   - Цветом можно показать силу ударов (если доступна)

2. **Распределения**:
   - Гистограмма интервалов между ударами (`intervals`)
   - Box plot для статистики интервалов (min, max, median, quartiles)

3. **Метрики**:
   - Gauge/radial chart для `regularity` (0-1)
   - Bar chart для сравнения `tempo_bpm`, `median_bpm`, `tempo_mean`
   - Line chart для `tempo_variation` по времени (для `run_segments()`)

4. **Интерактивные элементы**:
   - Tooltips с детальной информацией о каждом ударе
   - Фильтры по диапазонам темпа
   - Сравнение с `tempo_extractor` (если доступен)

5. **HTML debug renderer**:
   - Локальный HTML renderer доступен через `render_rhythmic_extractor_html()`
   - Используется только для дебага, не попадает в production артефакты

### Performance characteristics

**Resource costs**:
- **CPU**: умеренные (beat tracking требует вычислений)
- **GPU**: не используется
- **Estimated duration**: ~1.2 секунд для типичного аудио файла

**Параметры производительности**:
- `hop_length`: меньшие значения → выше разрешение → точнее, но медленнее
- `sample_rate`: более высокие значения → точнее, но медленнее
- Essentia обычно быстрее, чем librosa для beat tracking

### Связанные компоненты

- **AudioUtils**: загрузка и предобработка аудио
- **BaseExtractor**: базовый интерфейс экстрактора
- **Essentia** (опционально): библиотека для музыкального анализа
- **librosa**: библиотека для анализа аудио

### Примечания

1. **Backend selection**: Явный выбор backend через `--rhythmic-backend` (no-fallback policy). Если выбран backend недоступен → fail-fast с error_code.
2. **Регулярность**: значение 1.0 означает идеально регулярный ритм (метроном), значения <0.5 указывают на нерегулярный ритм.
3. **Темп**: может отличаться от субъективного восприятия, особенно для сложных ритмов.
4. **Короткие файлы**: для очень коротких файлов (<1 секунды) метрики могут быть неточными.
5. **Многоканальное аудио**: рекомендуется `average_channels=True` для более устойчивой оценки.
6. **Интервалы**: вычисляются только если обнаружено более одного удара.
7. **Медианный темп**: может отличаться от основного темпа, особенно при вариациях темпа.
8. **Beat times storage**: Для больших массивов (>10000 элементов) `beat_times` сохраняется в `.npy` файл в `_artifacts/` для экономии памяти NPZ.
