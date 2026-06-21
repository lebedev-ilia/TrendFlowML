## `source_separation_extractor` (Source separation)

### Назначение

Извлекает **доли энергии источников** (vocals, drums, bass, other) из аудио сигнала с использованием inprocess PyTorch модели разделения источников. Экстрактор работает на уровне окон сегментации и вычисляет энергетические доли для каждого сегмента, а также агрегированные статистики (transitions, distribution, stability, balance).

**Версия**: 3.0.1  
**Категория**: source_separation  
**GPU**: preferred (PyTorch модель, требует GPU memory)

### Входы

- **`audio/audio.wav`** (любой аудио файл, поддерживаемый AudioUtils)
- **`audio/segments.json`** (сегменты от Segmenter, family: `source_separation`)

**Требования**:
- Минимальная длительность аудио: **5 секунд** (иначе ошибка)
- Сегменты должны быть предоставлены через `run_segments()` (метод `run()` не поддерживается)

### Выходы

Экстрактор возвращает `ExtractorResult` с payload, который затем сохраняется AudioProcessor в NPZ (source-of-truth).

Схема (Audit v3): `source_separation_extractor_npz_v2` — см. `SCHEMA.md` + `DataProcessor/AudioProcessor/schemas/source_separation_extractor_npz_v2.json`.

#### Audit v4 — заметки по NPZ

- **Tabular** (`feature_names` / `feature_values`): **F=11**, только числа; на reference **A** NaN **0**. **`device_used`**, **`model_name`**, **`weights_digest`** — в **`meta`**, не в tabular (см. `SCHEMA.md` §5).
- **`dominant_source_id`**: в tabular — числовой индекс **0…3** в порядке `source_order` (не строка).
- Число окон **N** по оси сегментов задаётся family **`source_separation`** (на **A** может быть **N=1**, в отличие от 12 окон у других семейств на том же run).

#### Обязательные поля (всегда присутствуют)

- `segments_count`: количество обработанных сегментов (int)
- `sample_rate`: частота дискретизации аудио (int, по умолчанию 44100 Hz)
- `model_name`: имя модели из ModelManager (str)
- `weights_digest`: digest весов (str)
- `source_order`: порядок источников `["vocals", "drums", "bass", "other"]` (list)
- `device_used`: устройство обработки (str, обычно `"cuda"`)
- `segment_start_sec`: список времен начала сегментов (list[float])
- `segment_end_sec`: список времен окончания сегментов (list[float])
- `segment_center_sec`: список времен центров сегментов (list[float])
- `segment_mask`: bool[N], false для silent/zero-energy окон (Audit v3 masking policy)
- `source_separation_contract_version`: версия контракта для валидации совместимости (str, `"source_separation_contract_v1"`)
- `_features_enabled`: список включённых групп фичей (для отладки) (List[str])
- `stage_timings_ms`: тайминги этапов компонента (dict[str,float], ms)
- `source_separation_resource_profile`: снапшоты ресурсов процесса/GPU (dict; включается через `AP_SOURCE_SEPARATION_RESOURCE_PROFILE=1`)

#### Feature-gated поля (включаются через флаги)

**`--sep-enable-share-sequence`**:
- `share_sequence`: массив долей энергии по сегментам, shape `[N, 4]` float32
  - Порядок источников: `[vocals, drums, bass, other]`
  - Каждая строка соответствует одному сегменту
  - Значения нормированы (сумма = 1.0 для каждого сегмента)

**`--sep-enable-energy-sequence`**:
- `energy_sequence`: массив абсолютных энергий по сегментам, shape `[N, 4]` float32
  - Порядок источников: `[vocals, drums, bass, other]`
  - Не нормированные значения энергии

**`--sep-enable-share-mean`**:
- `share_mean`: средние доли по всем сегментам, shape `[4]` float32
  - Порядок: `[vocals_mean, drums_mean, bass_mean, other_mean]`
  - Audit v3: `share_mean` сохраняется всегда (флаг legacy/no-op)

**`--sep-enable-share-std`**:
- `share_std`: стандартные отклонения долей по всем сегментам, shape `[4]` float32
  - Порядок: `[vocals_std, drums_std, bass_std, other_std]`

**Дополнительные агрегаты** (вычисляются если включен `share_mean` или `share_sequence`):
- `dominant_source_id`: ID доминирующего источника (argmax share_mean) (int)
- `dominant_source_share`: доля доминирующего источника (float)
- `source_balance_score`: метрика баланса источников (0 = один доминирует, 1 = равномерное распределение) (float)

**Дополнительные агрегаты** (вычисляются если включен `share_sequence`):
- `source_transitions_count`: количество переходов между доминирующими источниками (int)
- `source_distribution_ratio`: распределение времени по источникам (float32[4], в canonical source_order)
- `source_segments_count`: количество сегментов для каждого источника (int32[4])
- `source_duration_sec`: длительность каждого источника в секундах (float32[4])
- `source_stability_score`: метрика стабильности источников (0 = нестабильная, 1 = стабильная) (float)

**Расширенные фичи** (автоматически вычисляются если включен `share_sequence`):
- **Transition features** (динамика изменений, для каждого источника: vocals, drums, bass, other):
  - `{source}_delta_mean`: среднее изменение доли источника между сегментами (float)
  - `{source}_delta_std`: стандартное отклонение изменений (float)
  - `{source}_delta_max`: максимальное изменение (float)
- **Stability features** (стабильность источников):
  - `{source}_stability`: стабильность источника (1 - std(P_i), clamp [0, 1]) (float)
- **Distribution features** (распределение ролей):
  - `{source}_mean_share`: средняя доля источника (float, дублирует share_mean, но доступно отдельно)
  - `{source}_dominance_ratio`: доля времени, когда источник доминирует (mean(P_i == max(P))) (float)
- **Energy balance** (музыкальный баланс):
  - `source_entropy_mean`: средняя энтропия распределения источников по сегментам (float)
  - `source_entropy_std`: стандартное отклонение энтропии (float)
  - `energy_balance_mean`: средний баланс энергии (1 - std(P over sources)) (float)
- **Musical heuristics** (музыкально-осмысленные фичи):
  - `vocals_presence_ratio`: доля времени с вокалом (mean(P_vocals > 0.35)) (float)
  - `drums_flux`: ритмичность/изменчивость барабанов (mean(|P_drums(t) - P_drums(t-1)|)) (float)
  - `bass_floor_p20`: базовая линия баса (20-й перцентиль P_bass) (float)

**`--sep-enable-quality-metrics`**:
Метрики качества разделения сохраняются как scalar keys (Audit v3, без dict object):
- `quality_share_mean_min/max/std`
- `quality_share_std_mean/max`
- `quality_share_sequence_min/max/mean`
- `quality_energy_sequence_min/max/mean`

#### Специальные случаи

**Пустое аудио** (status="empty"):
- `status`: `"empty"`
- `empty_reason`: `"audio_silent"` (если silence detection включен)
- Остальные поля присутствуют (без share_sequence и агрегатов)

### Feature Dependencies

**Зависимости между фичами**:
- `dominant_source_id/share` и `source_balance_score` зависят от `share_mean` (используют его для вычисления)
- `source_transitions_count`, `source_distribution_ratio`, `source_segments_count`, `source_duration_sec`, `source_stability_score` вычисляются по shares (и используют `segment_mask`); `share_sequence` требуется только для сохранения sequence и advanced temporal фич
- **Расширенные фичи** (transition, stability, distribution, energy balance, musical heuristics) автоматически вычисляются если включен `share_sequence` (требуют временных рядов)
- `quality_metrics` зависят от `share_mean`, `share_std`, `share_sequence`, `energy_sequence` (использует их для вычисления метрик)

**Зависимости от других extractors**:
- Нет явных зависимостей от других extractors
- Может использоваться в других компонентах как зависимость (требует `share_sequence` или `share_mean`)

**Contract version для совместимости**:
- `source_separation_contract_version="source_separation_contract_v1"` используется для валидации совместимости с downstream extractors

### Алгоритм

#### 1. Предобработка аудио

1. Загрузка сегментов из `segments.json` (family: `source_separation`)
2. Для каждого сегмента:
   - Загрузка аудио через `AudioUtils.load_audio_segment()`
   - Ресемплирование до `sample_rate` (по умолчанию 44100 Hz)
   - Преобразование в моно канал
3. Вычисление RMS и peak для каждого сегмента
4. Проверка на тишину (если `--sep-enable-silence-detection` не отключен):
   - Если все сегменты тихие (peak < `silence_peak_threshold`, RMS < `silence_rms_threshold`), возвращается пустой результат

#### 2. Преобразование в log-mel спектрограмму

Для каждого сегмента:
- **STFT**: `torchaudio.transforms.MelSpectrogram`
  - `n_fft`: 2048 (по умолчанию, из runtime_params)
  - `hop_length`: 512 (по умолчанию, из runtime_params)
  - `n_mels`: 64 (по умолчанию, из runtime_params)
  - `power`: 2.0
- **Log-scale**: `torchaudio.transforms.AmplitudeToDB(stype="power")`
- Результат: log-mel features, shape `[n_mels, T]` float32

**Валидация параметров предобработки** (информативная):
- Проверка разумных диапазонов для `sample_rate`, `n_fft`, `hop_length`, `n_mels`
- Логирование предупреждений при выходе за типичные диапазоны (не ошибки)

#### 3. Инференс через inprocess PyTorch модель

- **Вход**: log-mel features, shape `[B, n_mels, T]` float32
  - Сегменты падятся до максимальной длины по времени
  - Батчинг с размером `batch_size`
  - Автоматическое разбиение на батчи при большом количестве сегментов (>100)
- **Выход**: энергии источников, shape `[B, 4]` float32
  - Порядок: `[vocals, drums, bass, other]`
- **ModelManager spec**: `source_separation_large_inprocess` (только large)
- **Runtime**: `inprocess` (PyTorch модель загружается через ModelManager)
- **Engine**: `torch` (TorchStateDictProvider)
- **Precision**: `fp32` (полная точность)
- **Модель**: принимает log-mel spectrogram и возвращает source energies для 4 источников

#### 4. Валидация shares и energies

- Проверка dtype (float32)
- Проверка shape (2D [N, 4])
- Проверка на NaN/inf
- Проверка диапазонов [0, 1] для shares
- Проверка неотрицательности для energies
- Проверка нормализации shares (сумма по строкам ≈ 1.0)
- Проверка согласованности размеров

#### 5. Постобработка

1. Нормировка энергий: `shares = energy / (sum(energy) + eps)`
2. Вычисление базовых статистик:
   - `share_mean = mean(shares, axis=0)`
   - `share_std = std(shares, axis=0)`
3. Вычисление дополнительных агрегатов (feature-gated, требует `enable_share_sequence`):
   - `dominant_source_id`: argmax share_mean
   - `dominant_source_share`: max share_mean
   - `source_balance_score`: нормализованная энтропия share_mean
   - `source_transitions_count`: количество переходов между доминирующими источниками
   - `source_distribution_ratio`: доли времени по источникам (float32[4])
   - `source_segments_count`: количество сегментов по источникам (int32[4])
   - `source_duration_sec`: длительность по источникам (float32[4])
   - `source_stability_score`: метрика стабильности (inverse of transitions frequency)
4. Вычисление расширенных фич (автоматически, если `enable_share_sequence=True`):
   - **Transition features** (динамика изменений):
     - `{source}_delta_mean`: среднее изменение доли источника между сегментами
     - `{source}_delta_std`: стандартное отклонение изменений
     - `{source}_delta_max`: максимальное изменение
   - **Stability features** (стабильность источников):
     - `{source}_stability`: стабильность источника (1 - std(P_i))
   - **Distribution features** (распределение ролей):
     - `{source}_mean_share`: средняя доля источника
     - `{source}_dominance_ratio`: доля времени, когда источник доминирует
   - **Energy balance** (музыкальный баланс):
     - `source_entropy_mean`: средняя энтропия распределения источников
     - `source_entropy_std`: стандартное отклонение энтропии
     - `energy_balance_mean`: средний баланс энергии (1 - std(P))
   - **Musical heuristics** (музыкально-осмысленные фичи):
     - `vocals_presence_ratio`: доля времени с вокалом (P_vocals > 0.35)
     - `drums_flux`: ритмичность (mean(|P_drums(t) - P_drums(t-1)|))
     - `bass_floor_p20`: базовая линия баса (20-й перцентиль P_bass)
5. Вычисление метрик качества (feature-gated):
   - Распределение share_mean, share_std, share_sequence, energy_sequence

### Конфигурация

#### Параметры модели

```python
{
    "device": "auto",              # "auto" | "cuda" | "cpu"
    "model_size": "large",         # "large"
    "batch_size": 8,               # Размер батча для обработки окон
    "silence_peak_threshold": 1e-3,  # Порог peak для детекции тишины
    "silence_rms_threshold": 1e-4,   # Порог RMS для детекции тишины
    "enable_silence_detection": True,  # Включить проверку на тишину
    "progress_callback": None,     # Callback для прогресс-репортинга (опционально)
}
```

**Параметры модели** (из ModelManager spec `source_separation_{model_size}_inprocess`):
- `sample_rate`: 44100 (Hz)
- `n_fft`: 2048
- `hop_length`: 512
- `n_mels`: 64
- `source_order`: порядок источников `["vocals", "drums", "bass", "other"]`
- `factory`: `dp_models.factories.audio:create_source_separation_model` (factory функция для создания модели)
- `checkpoint_relpath`: путь к checkpoint файлу (например, `audio/source_separation/large.pt`)

#### Feature Gating (персональные флаги)

Audit v3 preset:
- baseline model_facing + `share_mean` + canonical axis + `segment_mask` + structured per-source stats — **всегда**
- opt-in: `share_sequence`, `energy_sequence`, `share_std`, `quality_metrics`

- `--sep-enable-share-sequence`: включить `share_sequence` (per-segment shares)
- `--sep-enable-energy-sequence`: включить `energy_sequence` (per-segment energies)
- `--sep-enable-share-mean`: (legacy/no-op) `share_mean` сохраняется всегда
- `--sep-enable-share-std`: включить `share_std` (std shares)
- `--sep-enable-quality-metrics`: включить quality scalars `quality_*`

**Рекомендации для обучения моделей**:
- Включить `enable_share_sequence` для получения расширенных фич (transition, stability, distribution, energy balance, musical heuristics)
- Включить все фичи для максимального качества и полноты данных
- Расширенные фичи автоматически вычисляются при включении `enable_share_sequence` (не требуют отдельных флагов)

### Архитектура

1. **Инициализация**:
   - Получение модели через `ModelManager.get_spec()` → `source_separation_large_inprocess`
   - Проверка runtime = "inprocess" и engine = "torch"
   - Загрузка модели через `ModelManager.get()` (PyTorch модель через TorchStateDictProvider)
   - Модель автоматически загружается на устройство
   - Извлечение параметров предобработки и source_order из runtime_params
   - Валидация параметров предобработки (информативная)
   - Валидация source_order (полная)
   - Создание MelSpectrogram и AmplitudeToDB трансформеров

2. **Обработка сегментов** (`run_segments()`):
   - Валидация входных данных (длительность ≥ 5 сек)
   - **Этап 1**: Загрузка аудио и вычисление mel features (профилирование: `load_audio_sec`)
   - **Этап 2**: Проверка на тишину (если включено) (профилирование: `silence_detection_sec`)
   - **Этап 3**: Padding для батчинга (профилирование: `padding_sec`)
   - **Этап 4**: Батчинг и инференс через inprocess PyTorch модель (с автоматическим разбиением при >100 сегментов) (профилирование: `inference_sec`)
     - Прогресс-репортинг: обновления каждые 10% батчей (если `progress_callback` установлен)
   - **Этап 5**: Нормализация энергий и валидация shares и energies (профилирование: `postprocess_sec`)
   - **Этап 6**: Вычисление агрегатов (профилирование: `aggregates_sec`)
     - Вычисление базовых статистик (share_mean, share_std)
     - Вычисление дополнительных агрегатов (feature-gated)
     - Вычисление расширенных фич (автоматически, если `enable_share_sequence=True`)
     - Вычисление метрик качества (feature-gated)
   - Формирование payload (feature-gated)
   - Детальное профилирование: логирование времени для каждого этапа

3. **Батчевая обработка** (`extract_batch_segments()`):
   - Сбор сегментов из всех видео
   - Вычисление mel features для каждого сегмента
   - Группировка в батчи по `max_segments_per_batch` (если задан) или `batch_size`
   - Обработка батчей через inprocess PyTorch модель
   - Распределение результатов обратно по видео

4. **Вспомогательные методы**:
   - `_validate_shares_and_energies()`: полная валидация shares и energies
   - `_validate_source_order()`: полная валидация source_order
   - `_validate_preprocessing_params()`: информативная валидация параметров предобработки
   - `_infer_energies_batch()`: batch inference через inprocess PyTorch модель
   - `_mel_log()`: преобразование аудио в log-mel spectrogram
   - `_rms_and_peak()`: вычисление RMS и peak значений для детекции тишины

5. **Обработка ошибок**:
   - Модель не найдена → `RuntimeError` (ModelManager raises `weights_missing`)
   - Модель не загружается → ошибка
   - Аудио < 5 сек → ошибка
   - Пустые сегменты → ошибка
   - Валидация shares/energies → ошибка

### Обработка ошибок

**Политика NO FALLBACK**:
- Отсутствие модели → `RuntimeError` (ModelManager raises `weights_missing`)
- Модель не загружается → ошибка
- Аудио < 5 секунд → ошибка
- Пустые сегменты → ошибка
- Валидация shares/energies → ошибка

**Специальные случаи**:
- **Тихое аудио**: возвращается `status="empty"`, `empty_reason="audio_silent"` (если silence detection включен)
- **Несоответствие sample rate**: ошибка с описанием
- **Неожиданная форма энергий**: ошибка с описанием

### Особенности

- **Inprocess PyTorch модель**: модель загружается локально через ModelManager (no-network policy)
- **Сегментная обработка**: работает только с сегментами от Segmenter
- **Энергетические доли**: выход - доли энергии, а не разделенные стемы (экономия места)
- **Batch processing**: эффективная обработка нескольких сегментов одновременно через inprocess PyTorch модель
- **Автоматическое разбиение**: при большом количестве сегментов (>100) автоматически разбивается на батчи
- **Нет fallback**: строгая политика - ошибка при отсутствии модели
- **Log-mel вход**: модель принимает log-mel спектрограммы, не raw audio
- **Паддинг сегментов**: сегменты разной длины падятся до максимальной для батчинга
- **Progress reporting**: обновление прогресса каждые 10% батчей (если батчей ≥10 и `progress_callback` установлен)
- **Детальное профилирование**: логирование времени выполнения для каждого этапа (load_audio, silence_detection, padding, inference, aggregates, postprocess) с детальными метриками производительности
- **Расширенные фичи**: автоматическое вычисление transition, stability, distribution, energy balance и musical heuristics фич при включении `enable_share_sequence` (не требуют отдельных флагов)
- **Feature gating**: все фичи opt-in через персональные флаги
- **Contract versioning**: версия контракта для валидации совместимости с downstream extractors
- **Полная валидация**: проверка shares и energies на всех этапах
- **Валидация параметров**: информативная валидация параметров предобработки (логирование предупреждений)
- **Batch processing для нескольких видео**: поддержка `extract_batch_segments()` для гибридного батчинга сегментов из всех видео

### Performance characteristics

**Resource costs**:
- **GPU VRAM**: зависит от размера модели (large) и batch_size
- **CPU RAM**: умеренные (предобработка mel спектрограмм через torchaudio)
- **Estimated duration**: ~12.0 секунд для типичного аудио файла
- **Batch efficiency**: обработка батчами увеличивает throughput на GPU

**Параметры производительности**:
- `model_size`: `large` (единственный поддерживаемый размер)
- `batch_size`: контроль размера батча для inference (по умолчанию 8)
- `n_mels`, `n_fft`, `hop_length`: влияют на размер входных данных

### Visualization

**Рекомендации для UI/сайта**:

1. **Timeline визуализация**:
   - Горизонтальная временная шкала с цветовой кодировкой источников
   - Каждый сегмент отображается как полоса с цветом, соответствующим доминирующему источнику
   - Интерактивные tooltips с информацией о сегменте (start, end, shares для всех источников, dominant_source_id)
   - Zoom и pan для навигации по длинным видео
   - Stacked area chart для визуализации долей всех источников одновременно

2. **Распределение источников**:
   - Pie chart или bar chart с долями времени по источникам (`source_distribution_ratio`)
   - Таблица со статистикой по каждому источнику (segments_count, duration, time_ratio)
   - Highlight доминирующего источника (`dominant_source_id`)

3. **Метрики качества**:
   - Отображение balance score и stability score
   - Визуализация transitions count
   - Индикаторы качества (good/fair/poor) на основе метрик

4. **Агрегаты**:
   - Отображение `share_mean` и `share_std` для каждого источника
   - Количество переходов между источниками (`source_transitions_count`)
   - Стабильность источников (`source_stability_score`)
   - Баланс источников (`source_balance_score`)

5. **HTML renderer для дебага**:
   - Локальный HTML renderer доступен через `render_source_separation_extractor_html()`
   - Включает timeline plot, статистику, метрики качества, распределения
   - Только для локального дебага, не в production артефактах

**Пример использования HTML renderer**:
```python
from src.core.renderer import render_source_separation_extractor_html

html_path = render_source_separation_extractor_html(
    npz_path="result_store/.../source_separation_extractor/source_separation_extractor_features.npz",
    output_path="debug_source_separation.html"
)
```

### Связанные компоненты

- **ModelManager** (`dp_models`): управление моделями и спецификациями
- **TorchStateDictProvider** (`dp_models`): провайдер для загрузки PyTorch моделей
- **AudioUtils**: загрузка и предобработка аудио
- **BaseExtractor**: базовый интерфейс экстрактора
- **Segmenter**: генерация сегментов (family: `source_separation`)
- **torchaudio**: преобразование в mel спектрограммы

### Примечания

1. **Требования**: экстрактор требует:
   - Модель разделения источников в `DP_MODELS_ROOT/audio/source_separation/{size}.pt` (checkpoint файл)
   - Спецификация модели в ModelManager (`source_separation_{size}_inprocess`)
   - PyTorch и torchaudio установлены
   - GPU рекомендуется для ускорения inference

2. **Установка моделей**: используйте скрипт `scripts/download_source_separation_models.py` для экспорта моделей в dp_models:
   ```bash
   python scripts/download_source_separation_models.py --models-root dp_models/bundled_models --sizes large
   ```
   Примечание: скрипт содержит placeholder - адаптируйте функцию `download_source_separation_model()` под конкретную модель.

3. **Сегментация**: экстрактор работает только с сегментами от Segmenter. Метод `run()` не поддерживается.

4. **Формат входных данных**: модель принимает log-mel спектрограммы, не raw audio. Предобработка выполняется локально через torchaudio.

5. **Выходные данные**: экстрактор не сохраняет разделенные стемы (слишком большие). Сохраняются только энергетические доли и статистики.

6. **Порядок источников**: всегда `[vocals, drums, bass, other]`. Указан в `source_order` в runtime_params и валидируется при инициализации.

7. **Тишина**: если все сегменты тихие и silence detection включен, возвращается пустой результат (не ошибка).

8. **Батчинг**: автоматическое разбиение на батчи при большом количестве сегментов (>100) или при заданном `batch_size`.

9. **Паддинг**: сегменты разной длины падятся до максимальной для эффективного батчинга.

10. **Feature gating**: baseline всегда сохраняется; расширенные sequences/quality — opt-in.

11. **Contract versioning**: версия контракта `source_separation_contract_v1` используется для валидации совместимости с downstream extractors.

12. **Валидация**: полная валидация shares (NaN/inf, диапазоны [0,1], нормализация) и energies (NaN/inf, неотрицательность), а также source_order (длина, дубликаты, типы).

13. **Валидация параметров предобработки**: информативная валидация параметров (sample_rate, n_fft, hop_length, n_mels) с логированием предупреждений при выходе за типичные диапазоны (не ошибки).

14. **Batch processing**: поддержка `extract_batch_segments()` для гибридного батчинга сегментов из нескольких видео одновременно.
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
