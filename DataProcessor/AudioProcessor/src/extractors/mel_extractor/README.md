## `mel_extractor` (Mel spectrogram features)

### Назначение

Извлекает **Mel-спектрограмму** — частотно-временное представление аудио сигнала в мел-шкале (mel scale), которая лучше соответствует восприятию звука человеком. Компонент вычисляет Mel-спектрограмму в децибелах, а также статистические агрегаты и спектральные характеристики (спектральный центроид и полоса пропускания).

**Версия**: 2.0.0  
**Категория**: spectral  
**GPU**: preferred (использует GPU если доступен, так как дает прирост скорости)

### Входы

- **`audio/audio.wav`** (Segmenter contract) — полное аудио для `run()`
- **`audio/segments.json`** (Segmenter contract, family `mel`) — сегменты для `run_segments()`

### Выходы

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Базовые фичи (feature-gated: `--mel-enable-basic-features`)

- **`mel_shape`**: форма Mel-спектрограммы `(n_mels, frames)` как tuple
- **`mel_elements`**: общее количество элементов в спектрограмме (int)
- **`mel_spectrogram_npy`**: путь к сохраненному .npy файлу с полной Mel-спектрограммой (shape: `(n_mels, frames)`)
  - **Единицы**: децибелы (dB)
  - **Диапазон**: [-120, 0] dB (после sanitization и clipping)
  - **Форма**: `(n_mels, frames)`, где `frames` — количество временных кадров

#### Статистики (feature-gated: `--mel-enable-statistics`)

- **`mel_mean`**: средние значения по времени для каждого mel bin (float32[n_mels], путь в `mel_mean_npy`)
- **`mel_std`**: стандартные отклонения по времени (float32[n_mels], путь в `mel_std_npy`)
- **`mel_min`**: минимальные значения по времени (float32[n_mels], путь в `mel_min_npy`)
- **`mel_max`**: максимальные значения по времени (float32[n_mels], путь в `mel_max_npy`)
- **`freq_mean`**: средние значения по частотам для каждого кадра (float32[frames], путь в `freq_mean_npy`)
- **`freq_std`**: стандартные отклонения по частотам (float32[frames], путь в `freq_std_npy`)
- **`mel_mean_shape`, `mel_std_shape`, `mel_min_shape`, `mel_max_shape`**: формы массивов статистик по частотам
- **`freq_mean_shape`, `freq_std_shape`**: формы массивов статистик по времени

#### Спектральные характеристики (feature-gated: `--mel-enable-spectral-features`)

- **`spectral_centroid`**: спектральный центроид (средняя частота энергии) по времени (float32[frames], путь в `spectral_centroid_npy`)
  - **Единицы**: Герцы (Hz)
  - **Диапазон**: [fmin, fmax]
  - **Интерпретация**: характеризует "яркость" звука (высокий центроид = яркий звук)

- **`spectral_bandwidth`**: полоса пропускания (ширина распределения энергии) по времени (float32[frames], путь в `spectral_bandwidth_npy`)
  - **Единицы**: Герцы (Hz)
  - **Диапазон**: [0, fmax]
  - **Интерпретация**: характеризует ширину спектрального распределения (узкая = тональный звук, широкая = шумный звук)

- **`spectral_centroid_shape`, `spectral_bandwidth_shape`**: формы спектральных характеристик

#### Дополнительные метрики для ML/аналитики (всегда включены, если включены basic_features)

- **`mel_energy`**: общая энергия Mel-спектрограммы (float)
- **`mel_centroid_mean`**: среднее значение spectral_centroid (float, Hz)
- **`mel_centroid_std`**: стандартное отклонение spectral_centroid (float, Hz)
- **`mel_bandwidth_mean`**: среднее значение spectral_bandwidth (float, Hz)
- **`mel_bandwidth_std`**: стандартное отклонение spectral_bandwidth (float, Hz)
- **`mel_spectrogram_entropy`**: энтропия распределения энергии (float)
- **`mel_spectrogram_contrast`**: контраст между mel bins (float)

#### Компактный вектор признаков (feature-gated: `--mel-enable-stats-vector`)

- **`mel_stats_vector`**: конкатенированный вектор статистик `[mel_mean, mel_std, mel_min, mel_max]` (float32[n_mels * 4], путь в `mel_stats_vector_npy`)
  - **Форма**: `(n_mels * 4,)`
  - **Использование**: удобный формат для ML моделей (один вектор вместо нескольких массивов)

#### Временные серии (feature-gated: `--mel-enable-time-series`)

- **`mel_series`**: полная временная серия Mel-спектрограммы (float32[n_mels, frames])
- **`segment_centers_sec`**: центры сегментов в секундах (float32[L], для `run_segments()`)
- **`segment_durations_sec`**: длительности сегментов в секундах (float32[L], для `run_segments()`)

**Примечание**: Большие временные серии (>1000 элементов) сохраняются в `.npy` файлы в `_artifacts/` и регистрируются в `manifest.json`.

#### Метаданные

- `device_used`: устройство обработки (`"cpu"`, `"cuda"` или `"auto"`)
- `sample_rate`: частота дискретизации аудио (Hz)
- `n_fft`: размер FFT окна (int, по умолчанию 2048)
- `hop_length`: размер hop для STFT (int, по умолчанию 512)
- `n_mels`: количество мел-фильтров (int, по умолчанию 128)
- `fmin`: минимальная частота (float, по умолчанию 0.0)
- `fmax`: максимальная частота (float, по умолчанию sample_rate // 2)
- `power`: степень для спектрограммы (float, по умолчанию 2.0)
- `duration`: длительность аудио (секунды)
- `segments_count`: количество сегментов (для `run_segments()`)
- `mel_contract_version`: версия контракта (`"mel_contract_v1"`)
- `_features_enabled`: список включённых групп фичей (для отладки)

### Feature Dependencies

- **`mel_statistics`** зависит от **`mel_spectrogram`** (требует включения `--mel-enable-basic-features`)
- **`spectral_centroid`, `spectral_bandwidth`** зависят от **`mel_spectrogram`** (требуют включения `--mel-enable-basic-features` и `--mel-enable-spectral-features`)
- **`mel_stats_vector`** зависит от **`mel_statistics`** (требует включения `--mel-enable-statistics` и `--mel-enable-stats-vector`)
- **Дополнительные метрики** зависят от **`mel_spectrogram`** и **`spectral_features`** (требуют включения `--mel-enable-basic-features` и `--mel-enable-spectral-features`)

### Конфигурация

#### CLI аргументы

```bash
# Параметры обработки
--mel-sample-rate 22050              # Частота дискретизации (Hz)
--mel-n-fft 2048                     # Размер FFT окна
--mel-hop-length 512                 # Размер hop для STFT
--mel-n-mels 128                     # Количество мел-фильтров
--mel-fmin 0.0                       # Минимальная частота (Hz)
--mel-fmax <float>                   # Максимальная частота (Hz, None = sample_rate // 2)
--mel-power 2.0                     # Степень для спектрограммы (1.0 = magnitude, 2.0 = power)
--mel-mix-to-mono                    # Сводить стерео в моно (по умолчанию включено, используйте --mel-no-mix-to-mono для отключения)
--mel-enable-audio-normalization     # Включить нормализацию аудио перед обработкой (по умолчанию включена, используйте --mel-disable-audio-normalization для отключения)

# Feature gating (все opt-in, по умолчанию все выключены)
--mel-enable-basic-features          # Включить базовые фичи (mel_spectrogram, mel_shape, mel_elements)
--mel-enable-statistics              # Включить статистики (mel_mean, mel_std, mel_min, mel_max, freq_mean, freq_std)
--mel-enable-spectral-features       # Включить спектральные фичи (spectral_centroid, spectral_bandwidth)
--mel-enable-time-series             # Включить временные серии для всех фичей
--mel-enable-stats-vector            # Включить компактный вектор статистик (mel_stats_vector)
```

#### Python API

```python
from src.extractors.mel_extractor import MelExtractor

extractor = MelExtractor(
    device="auto",  # Использует GPU если доступен
    sample_rate=22050,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    fmin=0.0,
    fmax=None,
    power=2.0,
    mix_to_mono=True,
    enable_audio_normalization=True,
    enable_basic_features=True,
    enable_statistics=False,
    enable_spectral_features=False,
    enable_time_series=False,
    enable_stats_vector=False,
    progress_callback=None,
    artifacts_dir=None,
)
```

### Алгоритмы

Все метрики вычисляются с использованием **torchaudio**:

1. **Mel Spectrogram Extraction**: применение `torchaudio.transforms.MelSpectrogram` для получения Mel-спектрограммы в линейной шкале
   - Mel-спектрограмма автоматически включает: STFT → Mel-фильтры → power/magnitude
2. **Amplitude to dB**: преобразование амплитуды в децибелы через `torchaudio.transforms.AmplitudeToDB`
3. **Нормализация аудио** (опционально): приведение к диапазону [-1, 1] через `AudioUtils.normalize_audio()`
4. **Sanitization**: замена NaN/inf на -120 dB и обрезка к диапазону [-120, 0] dB
5. **Статистики**: вычисление mean, std, min, max по времени для каждого mel bin и по частотам для каждого кадра
6. **Спектральные характеристики**: вычисление spectral_centroid и spectral_bandwidth на линейной шкале мощности

### Обработка ошибок

Экстрактор использует **no-fallback policy** (fail-fast):

- **Ошибка загрузки аудио**: `mel_audio_load_failed`
- **Ошибка настройки трансформов**: `mel_transform_setup_failed`
- **Ошибка извлечения Mel-спектрограммы**: `mel_spectrogram_failed`
- **Ошибка преобразования в децибелы**: `mel_amplitude_to_db_failed`
- **Ошибка вычисления статистик**: `mel_statistics_failed`
- **Ошибка вычисления спектральных характеристик**: `mel_spectral_features_failed`
- **Ошибка валидации**: `mel_validation_failed`
- **Неизвестная ошибка**: `mel_unknown`

Все ошибки включают детальный `error_code` в сообщении об ошибке.

### Валидация

#### Валидация параметров (fail-fast)

- `sample_rate > 0`
- `n_fft > 0`
- `hop_length > 0`
- `hop_length <= n_fft`
- `n_mels > 0`
- `fmin >= 0.0`
- `fmax > fmin` (если задан)
- `fmax <= sample_rate / 2` (если задан)
- `power > 0.0`

#### Валидация выходных данных

- Проверка диапазонов значений (NaN/inf проверки, диапазон [-120, 0] dB)
- Проверка консистентности (mel_shape[0] == n_mels, статистики соответствуют mel_shape)
- Проверка типов и размерностей

### GPU логика

Экстрактор использует GPU если доступен (так как дает прирост скорости):

- Если `device="auto"` и CUDA доступен → использует GPU
- Если `device="cuda"` → использует GPU
- Если `device="cpu"` → использует CPU
- Mixed precision (autocast) на CUDA для ускорения вычислений

### Обработка многоканального аудио

Экстрактор автоматически преобразует многоканальное аудио в моно:

- Если `mix_to_mono=True`: усредняет все каналы (по умолчанию)
- Если `mix_to_mono=False`: использует первый канал

### Нормализация аудио

Опциональная нормализация аудио перед обработкой (включается через `--mel-enable-audio-normalization`, по умолчанию включена для обратной совместимости):

- Использует `AudioUtils.normalize_audio()` для нормализации амплитуды
- Может улучшить стабильность Mel-спектрограммы
- **Внимание**: нормализация может скрыть проблемы с исходным аудио (например, низкий уровень записи)
- **No-fallback**: если нормализация не работает → ошибка с детальным error_code

### Sampling / units-of-processing requirements

**Важно**: `mel_extractor` **не генерирует сегменты сам** — Segmenter является единственным владельцем sampling.

**Требования к сегментам**:
- Компонент использует семейство сегментов из `audio/segments.json`:
  - **`families.mel.segments[]`**: окна для анализа Mel-спектрограммы (обязательно для `run_segments()`)
- Сегменты должны иметь обязательные поля: `start_sample`, `end_sample`, `start_sec`, `end_sec`, `center_sec`
- Отсутствие обязательного семейства → fail-fast (`raise RuntimeError`)

**Sampling policy (Segmenter contract)**:
- Segmenter строит families по **универсальной нелинейной кривой** (sampling curve):
  - Параметры в `families.mel.sampling_curve`: `type="ease_out_power"`, `k∈(0,1]`, `linear_until_sec`, `cap_duration_sec`
  - На коротких видео можно близко к 1:1 (секунда→окно), на длинных рост замедляется и упирается в `max_windows`
- См. `docs/contracts/SEGMENTER_CONTRACT.md` для деталей sampling policy

**Минимальные требования**:
- Минимальная длительность сегмента: **100 мс** (для точности Mel-спектрограммы)
- Минимальное количество сегментов: **1 сегмент** (иначе ошибка `segments_invalid`)

### Segmenter Contract

Экстрактор поддерживает работу на сегментах от Segmenter:

- **`run()`**: работает на полном аудио (`audio/audio.wav`)
- **`run_segments()`**: работает на сегментах из `audio/segments.json` (family `mel`)

Для `run_segments()`:
- Читает `families.mel.segments[]` из `audio/segments.json`
- Использует `start_sample/end_sample` для загрузки сегментов через `AudioUtils.load_audio_segment()`
- Агрегирует результаты по всем сегментам (статистики и временные серии)

### Progress Reporting

Экстрактор поддерживает progress reporting через callback:

- Для `run()`: обновление прогресса для каждого этапа (загрузка аудио, нормализация, извлечение Mel-спектрограммы, преобразование в децибелы, вычисление статистик, вычисление спектральных характеристик, вычисление дополнительных метрик, сохранение артефактов, валидация)
- Для `run_segments()`: обновление прогресса каждые 10% сегментов

### Per-run Storage

Большие массивы сохраняются в `.npy` файлы:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/mel_extractor/_artifacts/*.npy`
- Регистрация в `manifest.json.components[].artifacts[]` (type=`"npy"`)

### Visualization

#### Рекомендации для UI/сайта

1. **Mel Spectrogram визуализация**:
   - Отображать `mel_spectrogram` как heatmap (временная ось по X, mel bins по Y)
   - Использовать цветовую карту (viridis, plasma, или custom) для значений в децибелах
   - Добавить tooltips с точными значениями при наведении
   - Показывать временную шкалу в секундах
   - Добавить zoom для детального просмотра

2. **Статистики Mel**:
   - Отображать `mel_mean`, `mel_std`, `mel_min`, `mel_max` как line charts для каждого mel bin
   - Использовать bar charts для сравнения статистик между разными видео
   - Показывать распределение значений через гистограммы

3. **Спектральные характеристики**:
   - Отображать `spectral_centroid` и `spectral_bandwidth` как line charts по времени
   - Использовать разные цвета для центроида и полосы пропускания
   - Показывать временную динамику через timeline визуализацию
   - Добавить статистики (mean, std) для быстрого обзора

4. **Дополнительные метрики**:
   - Отображать `mel_energy`, `mel_centroid_mean`, `mel_bandwidth_mean` как отдельные карточки
   - Использовать gauge charts для `mel_spectrogram_entropy` и `mel_spectrogram_contrast`
   - Показывать корреляционную матрицу между mel bins

5. **Интерактивные элементы**:
   - Фильтры для выбора mel bins
   - Zoom для детального просмотра временных серий
   - Сравнение Mel-спектрограмм между разными видео
   - Экспорт Mel-спектрограммы для дальнейшего анализа

#### Локальный HTML renderer для дебага

Используйте `render_mel_extractor_html()` для генерации HTML страницы с результатами:

```python
from src.core.renderer import render_mel_extractor_html

render_mel_extractor_html(
    npz_path="result_store/.../mel_extractor/mel_extractor_features.npz",
    output_path="mel_debug.html"
)
```

HTML страница включает:
- Summary (sample_rate, n_fft, hop_length, n_mels, fmin, fmax, power, device, duration, segments_count)
- Информацию о базовых фичах (mel_shape, mel_elements)
- Информацию о статистиках (формы массивов)
- Информацию о спектральных характеристиках (формы массивов)
- Дополнительные метрики
- Временные серии (если включены)
- Raw JSON данные

### Performance characteristics

**Resource costs**:
- **CPU**: O(N * log(N)) для FFT, где N — длина аудио
- **GPU**: ~1.0 GB (при использовании GPU)
- **Память**: O(n_mels * frames) для Mel-спектрограммы
- **Estimated duration**: ~3.0 секунды для типичного аудио файла (полное аудио)
- **Estimated duration (segments)**: ~0.1-0.2 секунды на сегмент (зависит от длины сегмента)

**Параметры производительности**:
- `n_fft`: большие значения → точнее частотное разрешение, но медленнее
- `hop_length`: меньшие значения → больше временное разрешение, но медленнее
- `n_mels`: большие значения → точнее мел-шкала, но медленнее
- `power`: 1.0 (magnitude) быстрее чем 2.0 (power), но менее точный
- `enable_statistics`: `False` → меньше вычислений, быстрее
- `enable_spectral_features`: `False` → меньше вычислений, быстрее
- `enable_time_series`: `False` → меньше данных в payload, быстрее передача
- `enable_audio_normalization`: `True` → дополнительная обработка, немного медленнее
- GPU: автоматическое использование GPU если доступен (дает прирост скорости)

### Связанные компоненты

- **AudioUtils**: загрузка и предобработка аудио
- **BaseExtractor**: базовый интерфейс экстрактора
- **torchaudio**: основная библиотека для Mel-спектрограммы
- **torch**: для GPU-ускорения (опционально)
- **Segmenter**: источник сегментов для `run_segments()`
- **mfcc_extractor**: использует Mel-спектрограмму как промежуточный шаг для MFCC

### Примечания

1. **Mel scale**: Mel-шкала лучше соответствует восприятию звука человеком, чем линейная частотная шкала
2. **Децибелы**: спектрограмма преобразуется в децибелы для лучшей визуализации и анализа
3. **GPU vs CPU**: GPU значительно ускоряет обработку больших файлов, поэтому используется если доступен
4. **Сохранение в .npy**: все массивы сохраняются в .npy файлы для эффективного хранения и загрузки
5. **Спектральный центроид**: характеризует "яркость" звука (высокий центроид = яркий звук)
6. **Спектральная полоса пропускания**: характеризует ширину спектрального распределения (узкая = тональный звук, широкая = шумный звук)
7. **Применение**: Mel-спектрограмма широко используется в распознавании речи, классификации аудио, музыкальном анализе
8. **Feature gating**: все фичи opt-in (по умолчанию все выключены) для контроля размера NPZ и стоимости вычислений
9. **Временные серии**: большие серии (>1000 элементов) автоматически сохраняются в `.npy` файлы для экономии памяти
10. **Contract versioning**: используется `mel_contract_version="mel_contract_v1"` для валидации совместимости с downstream extractors
11. **Дополнительные метрики**: включают энергетические, статистические и энтропийные характеристики для улучшения качества ML моделей
12. **No-fallback policy**: все ошибки обрабатываются fail-fast с детальными error codes
