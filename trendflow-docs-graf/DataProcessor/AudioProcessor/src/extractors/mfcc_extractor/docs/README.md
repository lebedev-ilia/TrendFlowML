## `mfcc_extractor` (MFCC features)

### Назначение

Извлекает **MFCC (Mel-frequency cepstral coefficients)** — кепстральные коэффициенты в мел-шкале, широко используемые для анализа речи и музыки. MFCC представляют спектральную форму сигнала в компактном виде и эффективны для распознавания речи, классификации аудио и других задач машинного обучения.

**Версия**: 2.1.1 (Audit v4.2 observability)  
**Категория**: spectral  
**GPU**: preferred (может работать на CPU, но GPU ускоряет обработку длинных файлов)

### Входы

- **`audio/audio.wav`** (Segmenter contract) — полное аудио для `run()`
- **`audio/segments.json`** (Segmenter contract, family `mfcc`) — сегменты для `run_segments()`

### Выходы

NPZ: `result_store/.../mfcc_extractor/mfcc_extractor_features.npz`, схема **`mfcc_extractor_npz_v2`** (`schemas/mfcc_extractor_npz_v2.json`, `docs/SCHEMA.md`).

#### Audit v4 — заметки по NPZ

- **Tabular:** только числа; при ошибочном включении строки (раньше — `device_used`) в tabular получался **NaN** через `as_float` — **исправлено в `npz_savers/mfcc.py`**, `device_used` читать из **`meta`**.
- **N=12** на reference run совпадает с family `mfcc`; массивы `mfcc_mean` и т.д. зависят от `meta.features_enabled`.

#### Audit v4.2 observability:
- `meta.stage_timings_ms` (dict): покомпонентные тайминги (ms), пишутся в NPZ meta
- `meta.mfcc_resource_profile` (dict|None): best-effort snapshot RSS/VMS/VRAM (если включено)
  - включение: `AP_MFCC_RESOURCE_PROFILE=1`

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Базовые фичи (feature-gated: `--mfcc-enable-basic-features`)

- **`mfcc_features`**: массив MFCC коэффициентов (numpy array, shape: `(n_mfcc, frames)`)
  - **Единицы**: безразмерные коэффициенты (кепстральные)
  - **Диапазон**: обычно [-50, 50], зависит от нормализации
  - **Форма**: `(n_mfcc, frames)`, где `frames` — количество временных кадров

- **`mfcc_statistics`**: словарь со статистиками MFCC
  - **`mfcc_mean`**: средние значения по времени для каждого MFCC коэффициента (list[float], длина `n_mfcc`)
  - **`mfcc_std`**: стандартные отклонения по времени (list[float], длина `n_mfcc`)
  - **`mfcc_min`**: минимальные значения по времени (list[float], длина `n_mfcc`)
  - **`mfcc_max`**: максимальные значения по времени (list[float], длина `n_mfcc`)
  - **`feature_shape`**: форма MFCC признаков `(n_mfcc, frames)` как tuple

#### Дельты (feature-gated: `--mfcc-enable-deltas`)

- **`delta_mean`**: средние значения первых дельт (производных по времени) (list[float], длина `n_mfcc`)
- **`delta_std`**: стандартные отклонения первых дельт (list[float], длина `n_mfcc`)
- **`delta_delta_mean`**: средние значения вторых дельт (производных от дельт) (list[float], длина `n_mfcc`)
- **`delta_delta_std`**: стандартные отклонения вторых дельт (list[float], длина `n_mfcc`)
- **`delta_shape`**: форма первых дельт `(n_mfcc, frames)` как tuple
- **`delta_delta_shape`**: форма вторых дельт `(n_mfcc, frames)` как tuple
- **`total_features`**: общее количество признаков (int, `n_mfcc * 4` для базовых + `n_mfcc * 4` для дельт)

#### Дополнительные метрики для ML/аналитики (всегда включены, если включены basic_features)

- **`mfcc_energy`**: энергия первого MFCC коэффициента (часто используется как отдельная фича)
- **`mfcc_centroid`**: центроид MFCC (взвешенное среднее по коэффициентам)
- **`mfcc_bandwidth`**: полоса пропускания MFCC (стандартное отклонение по коэффициентам)
- **`mfcc_skewness`**: асимметрия распределения MFCC
- **`mfcc_kurtosis`**: эксцесс распределения MFCC
- **`mfcc_correlation`**: корреляция между MFCC коэффициентами
- **`mfcc_stability`**: стабильность MFCC во времени (`1.0 / (1.0 + std(mfcc))`)

#### Временные серии (feature-gated: `--mfcc-enable-time-series`)

- **`mfcc_series`**: полная временная серия MFCC (float32[], shape: `(n_mfcc, frames)`)
- **`delta_series`**: полная временная серия первых дельт (float32[], shape: `(n_mfcc, frames)`)
- **`delta_delta_series`**: полная временная серия вторых дельт (float32[], shape: `(n_mfcc, frames)`)
- **`segment_centers_sec`**: центры сегментов в секундах (float32[], для `run_segments()`)
- **`segment_durations_sec`**: длительности сегментов в секундах (float32[], для `run_segments()`)

**Примечание**: Большие временные серии (>1000 элементов) сохраняются в `.npy` файлы в `_artifacts/` и регистрируются в `manifest.json`.

#### Метаданные

- `device_used`: устройство обработки (`"cpu"`, `"cuda"` или `"auto"`) — в **NPZ** в первую очередь в **`meta`**, не в `feature_values`
- `sample_rate`: частота дискретизации аудио (Hz)
- `n_mfcc`: количество MFCC коэффициентов (int, по умолчанию 13)
- `n_fft`: размер FFT окна (int, по умолчанию 2048)
- `hop_length`: размер hop для STFT (int, по умолчанию 512)
- `n_mels`: количество мел-фильтров (int, по умолчанию 128)
- `fmin`: минимальная частота (float, по умолчанию 0.0)
- `fmax`: максимальная частота (float, по умолчанию sample_rate // 2)
- `duration`: длительность аудио (секунды)
- `segments_count`: количество сегментов (для `run_segments()`)
- `mfcc_contract_version`: версия контракта (`"mfcc_contract_v1"`)
- `_features_enabled`: список включённых групп фичей (для отладки)

### Feature Dependencies

- **`mfcc_statistics`** зависит от **`mfcc_features`** (требует включения `--mfcc-enable-basic-features`)
- **`delta_mean`, `delta_std`, `delta_delta_mean`, `delta_delta_std`** зависят от **`mfcc_features`** (требуют включения `--mfcc-enable-basic-features` и `--mfcc-enable-deltas`)
- **`delta_series`, `delta_delta_series`** зависят от **`mfcc_series`** (требуют включения `--mfcc-enable-time-series` и `--mfcc-enable-deltas`)
- **Дополнительные метрики** зависят от **`mfcc_features`** (требуют включения `--mfcc-enable-basic-features`)

### Конфигурация

#### CLI аргументы

```bash
# Параметры обработки
--mfcc-sample-rate 22050              # Частота дискретизации (Hz)
--mfcc-n-mfcc 13                      # Количество MFCC коэффициентов
--mfcc-n-fft 2048                     # Размер FFT окна
--mfcc-hop-length 512                 # Размер hop для STFT
--mfcc-n-mels 128                     # Количество мел-фильтров
--mfcc-fmin 0.0                       # Минимальная частота (Hz)
--mfcc-fmax <float>                   # Максимальная частота (Hz, None = sample_rate // 2)
--mfcc-enable-audio-normalization     # Включить нормализацию аудио перед обработкой (по умолчанию включена)
--mfcc-min-gpu-duration-sec 3.0      # Минимальная длительность для использования GPU (секунды)
--mfcc-min-gpu-file-size-mb 5.0      # Минимальный размер файла для использования GPU (MB)

# Feature gating (все opt-in, по умолчанию все выключены)
--mfcc-enable-basic-features          # Включить базовые фичи (mfcc_features, mfcc_statistics: mean, std, min, max)
--mfcc-enable-deltas                  # Включить дельты (delta_mean, delta_std, delta_delta_mean, delta_delta_std)
--mfcc-enable-time-series             # Включить временные серии для всех фичей
--mfcc-enable-normalization           # Включить нормализацию MFCC по времени (z-score)
```

#### Python API

```python
from src.extractors.mfcc_extractor import MFCCExtractor

extractor = MFCCExtractor(
    device="auto",
    sample_rate=22050,
    n_mfcc=13,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    fmin=0.0,
    fmax=None,
    enable_audio_normalization=True,
    min_gpu_duration_sec=3.0,
    min_gpu_file_size_mb=5.0,
    enable_basic_features=True,
    enable_deltas=False,
    enable_time_series=False,
    enable_normalization=False,
    progress_callback=None,
    artifacts_dir=None,
)
```

### Алгоритмы

Все метрики вычисляются с использованием **torchaudio**:

1. **MFCC Extraction**: применение `torchaudio.transforms.MFCC` для получения MFCC коэффициентов
   - MFCC автоматически включает: Mel-спектрограмму → логарифм → DCT (Discrete Cosine Transform)
2. **Нормализация аудио** (опционально): приведение к диапазону [-1, 1] через `AudioUtils.normalize_audio()`
3. **Нормализация MFCC** (опционально): z-score нормализация по времени для стабильности признаков
4. **Дельты**: вычисление первых и вторых дельт через `torchaudio.functional.compute_deltas()`
5. **Статистики**: вычисление mean, std, min, max по времени для каждого MFCC коэффициента

### Обработка ошибок

Экстрактор использует **no-fallback policy** (fail-fast):

- **Ошибка загрузки аудио**: `mfcc_audio_load_failed`
- **Ошибка настройки трансформов**: `mfcc_transform_setup_failed`
- **Ошибка извлечения MFCC**: `mfcc_extraction_failed`
- **Ошибка вычисления дельт**: `mfcc_deltas_failed`
- **Ошибка вычисления статистик**: `mfcc_statistics_failed`
- **Ошибка валидации**: `mfcc_validation_failed`
- **Неизвестная ошибка**: `mfcc_unknown`

Все ошибки включают детальный `error_code` в сообщении об ошибке.

### Валидация

#### Валидация параметров (fail-fast)

- `sample_rate > 0`
- `n_mfcc > 0`
- `n_fft > 0`
- `hop_length > 0`
- `hop_length <= n_fft`
- `n_mels > 0`
- `fmin >= 0.0`
- `fmax > fmin` (если задан)
- `fmax <= sample_rate / 2` (если задан)

#### Валидация выходных данных

- Проверка диапазонов значений (NaN/inf проверки)
- Проверка консистентности (feature_shape[0] == n_mfcc, delta_shape соответствует feature_shape)
- Проверка типов и размерностей

### GPU эвристика

Улучшенная эвристика выбора CPU/GPU учитывает:

1. **Длительность аудио**: файлы короче `min_gpu_duration_sec` обрабатываются на CPU
2. **Размер файла**: файлы меньше `min_gpu_file_size_mb` обрабатываются на CPU
3. **Доступная GPU память**: проверка доступной памяти перед использованием GPU

Эвристика помогает оптимизировать производительность, уменьшая overhead инициализации GPU для коротких файлов.

### Обработка многоканального аудио

Экстрактор автоматически преобразует многоканальное аудио в моно через `AudioUtils.load_audio()`.

### Нормализация аудио

Опциональная нормализация аудио перед обработкой (включается через `--mfcc-enable-audio-normalization`, по умолчанию включена для обратной совместимости):

- Использует `AudioUtils.normalize_audio()` для нормализации амплитуды
- Может улучшить стабильность MFCC признаков
- **Внимание**: нормализация может скрыть проблемы с исходным аудио (например, низкий уровень записи)

### Нормализация MFCC

Опциональная нормализация MFCC по времени (z-score, включается через `--mfcc-enable-normalization`):

- Нормализует каждый MFCC коэффициент по времени (mean=0, std=1)
- Помогает стабилизировать признаки для ML моделей
- Применяется после извлечения MFCC, но перед вычислением статистик

### Sampling / units-of-processing requirements

**Важно**: `mfcc_extractor` **не генерирует сегменты сам** — Segmenter является единственным владельцем sampling.

**Требования к сегментам**:
- Компонент использует семейство сегментов из `audio/segments.json`:
  - **`families.mfcc.segments[]`**: окна для анализа MFCC (обязательно для `run_segments()`)
- Сегменты должны иметь обязательные поля: `start_sample`, `end_sample`, `start_sec`, `end_sec`, `center_sec`
- Отсутствие обязательного семейства → fail-fast (`raise RuntimeError`)

**Sampling policy (Segmenter contract)**:
- Segmenter строит families по **универсальной нелинейной кривой** (sampling curve):
  - Параметры в `families.mfcc.sampling_curve`: `type="ease_out_power"`, `k∈(0,1]`, `linear_until_sec`, `cap_duration_sec`
  - На коротких видео можно близко к 1:1 (секунда→окно), на длинных рост замедляется и упирается в `max_windows`
- См. `docs/contracts/SEGMENTER_CONTRACT.md` для деталей sampling policy

**Минимальные требования**:
- Минимальная длительность сегмента: **100 мс** (для точности MFCC)
- Минимальное количество сегментов: **1 сегмент** (иначе ошибка `segments_invalid`)

### Segmenter Contract

Экстрактор поддерживает работу на сегментах от Segmenter:

- **`run()`**: работает на полном аудио (`audio/audio.wav`)
- **`run_segments()`**: работает на сегментах из `audio/segments.json` (family `mfcc`)

Для `run_segments()`:
- Читает `families.mfcc.segments[]` из `audio/segments.json`
- Использует `start_sample/end_sample` для загрузки сегментов через `AudioUtils.load_audio_segment()`
- Агрегирует результаты по всем сегментам (статистики и временные серии)

### Progress Reporting

Экстрактор поддерживает progress reporting через callback:

- Для `run()`: обновление прогресса для каждого этапа (загрузка аудио, извлечение MFCC, вычисление статистик, вычисление дополнительных метрик, сохранение артефактов, валидация)
- Для `run_segments()`: обновление прогресса каждые 10% сегментов

### Per-run Storage

Большие временные серии (>1000 элементов) сохраняются в `.npy` файлы:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/mfcc_extractor/_artifacts/*.npy`
- Регистрация в `manifest.json.components[].artifacts[]` (type=`"npy"`)

### Visualization

#### Рекомендации для UI/сайта

1. **MFCC Spectrogram визуализация**:
   - Отображать `mfcc_features` как heatmap (временная ось по X, MFCC коэффициенты по Y)
   - Использовать цветовую карту (viridis, plasma, или custom) для значений MFCC
   - Добавить tooltips с точными значениями при наведении
   - Показывать временную шкалу в секундах

2. **Статистики MFCC**:
   - Отображать `mfcc_mean`, `mfcc_std`, `mfcc_min`, `mfcc_max` как line charts для каждого коэффициента
   - Использовать bar charts для сравнения статистик между разными видео
   - Показывать распределение значений через гистограммы

3. **Дельты**:
   - Отображать `delta_mean` и `delta_delta_mean` как line charts
   - Использовать разные цвета для первых и вторых дельт
   - Показывать временную динамику через timeline визуализацию

4. **Дополнительные метрики**:
   - Отображать `mfcc_energy`, `mfcc_centroid`, `mfcc_bandwidth` как отдельные карточки
   - Использовать gauge charts для `mfcc_stability`
   - Показывать корреляционную матрицу между MFCC коэффициентами

5. **Интерактивные элементы**:
   - Фильтры для выбора MFCC коэффициентов
   - Zoom для детального просмотра временных серий
   - Сравнение MFCC между разными видео
   - Экспорт MFCC данных для дальнейшего анализа

#### Локальный HTML renderer для дебага

Используйте `render_mfcc_extractor_html()` для генерации HTML страницы с результатами:

```python
from src.core.renderer import render_mfcc_extractor_html

render_mfcc_extractor_html(
    npz_path="result_store/.../mfcc_extractor/mfcc_extractor_features.npz",
    output_path="mfcc_debug.html"
)
```

HTML страница включает:
- Summary (sample_rate, n_mfcc, n_fft, hop_length, n_mels, fmin, fmax, device, duration, segments_count)
- Информацию о базовых фичах (shape, dtype, feature_shape)
- Информацию о дельтах (delta_shape, delta_delta_shape)
- Дополнительные метрики
- Временные серии (если включены)
- Raw JSON данные

### Models

**Используемые библиотеки и трансформы**:
- **torchaudio**: библиотека для обработки аудио (MFCC трансформ)
- **torch**: PyTorch для тензорных операций
- **MFCC Transform**: `torchaudio.transforms.MFCC` (CPU и GPU версии)

**Runtime**: `inprocess` (CPU/GPU)  
**Engine**: `torch` (PyTorch)  
**Precision**: `fp32`  
**Device**: `cpu` или `cuda` (автоматический выбор на основе эвристики)  
**Triton**: ❌ Нет (in-process processing)

**GPU эвристика**:
- GPU используется автоматически, если:
  - `duration_sec >= min_gpu_duration_sec` (по умолчанию 3.0 секунды)
  - ИЛИ размер файла `>= min_gpu_file_size_mb` (по умолчанию 5.0 MB)
- В противном случае используется CPU

**Загрузка**: нет моделей, только библиотеки (torchaudio, torch)

### Parallelization

**Внутренний параллелизм** (внутри компонента):
- **CPU parallelism**: компонент поддерживает batch processing через `extract_batch_segments()` с использованием `ThreadPoolExecutor`
  - Параллельная обработка сегментов из нескольких видео одновременно
  - Количество воркеров контролируется через `max_workers` (по умолчанию `os.cpu_count()`)
- **GPU processing**: при использовании GPU, обработка выполняется на GPU (torchaudio GPU transforms)
  - GPU трансформы ускоряют обработку длинных файлов

**Внешний параллелизм** (выше компонента):
- **Batch processing**: компонент batch-safe и может обрабатывать несколько файлов параллельно через `extract_batch_segments()`
  - Каждый файл обрабатывается изолированно через `run_segments()`
  - Изоляция данных: каждый файл имеет свой `tmp_path` и `artifacts_dir`
- **Video-level parallelism**: компонент может обрабатываться параллельно на разных видео (разные `run_id`)
  - Требования к изоляции: разные `run_id`, разные `result_store` пути
  - Thread-safety: компонент thread-safe (read-only shared state для трансформов)

**Ограничения**:
- GPU memory: ~0.5 GB при использовании GPU
- Параллелизм ограничен CPU (ThreadPoolExecutor) для batch processing
- Требования к памяти: O(n_mfcc * frames) для MFCC признаков

### Performance characteristics

**Resource costs**:
- **CPU**: O(N * log(N)) для FFT и DCT, где N — длина аудио
- **GPU**: ~0.5 GB (при использовании GPU)
- **Память**: O(n_mfcc * frames) для MFCC признаков
- **Estimated duration**: ~2.0 секунды для типичного аудио файла (полное аудио)
- **Estimated duration (segments)**: ~0.1-0.2 секунды на сегмент (зависит от длины сегмента)

**Параметры производительности**:
- `n_fft`: большие значения → точнее частотное разрешение, но медленнее
- `hop_length`: меньшие значения → больше временное разрешение, но медленнее
- `n_mels`: большие значения → точнее мел-шкала, но медленнее
- `enable_deltas`: `False` → меньше вычислений, быстрее
- `enable_time_series`: `False` → меньше данных в payload, быстрее передача
- `enable_normalization`: `True` → дополнительная обработка, немного медленнее
- `enable_audio_normalization`: `True` → дополнительная обработка, немного медленнее
- GPU эвристика: автоматический выбор CPU/GPU для оптимизации производительности

### Связанные компоненты

- **AudioUtils**: загрузка и предобработка аудио
- **BaseExtractor**: базовый интерфейс экстрактора
- **torchaudio**: основная библиотека для MFCC и дельт
- **torch**: для GPU-ускорения (опционально)
- **Segmenter**: источник сегментов для `run_segments()`
- **mel_extractor**: вычисляет Mel-спектрограмму, которая является промежуточным шагом для MFCC

### Примечания

1. **MFCC vs Mel-спектрограмма**: MFCC — это кепстральные коэффициенты, полученные из Mel-спектрограммы через DCT. Они более компактны и эффективны для ML.
2. **13 коэффициентов**: стандартное количество для распознавания речи (первый коэффициент часто отбрасывается как энергия)
3. **Дельты**: первые и вторые дельты учитывают временную динамику и улучшают качество распознавания
4. **GPU эвристика**: улучшенная эвристика учитывает длительность, размер файла и доступную GPU память для оптимизации производительности
5. **Нормализация**: опциональная z-score нормализация помогает стабилизировать признаки для ML моделей
6. **Применение**: MFCC широко используются в распознавании речи, классификации аудио, музыкальном анализе
7. **Feature gating**: все фичи opt-in (по умолчанию все выключены) для контроля размера NPZ и стоимости вычислений
8. **Временные серии**: большие серии (>1000 элементов) автоматически сохраняются в `.npy` файлы для экономии памяти
9. **Contract versioning**: используется `mfcc_contract_version="mfcc_contract_v1"` для валидации совместимости с downstream extractors
10. **Дополнительные метрики**: включают энергетические, статистические и корреляционные характеристики для улучшения качества ML моделей
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
