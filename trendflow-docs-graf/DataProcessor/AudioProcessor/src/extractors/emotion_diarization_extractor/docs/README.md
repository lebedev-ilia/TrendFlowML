## `emotion_diarization_extractor` (Emotion diarization)

### Назначение

Извлекает **эмоциональную диаризацию** (распознавание эмоций по временным окнам) через SpeechBrain Speech_Emotion_Diarization модель. Работает с сегментами от Segmenter (family: `emotion`) и возвращает вероятности эмоций для каждого окна, а также агрегированные метрики (доминирующая эмоция, энтропия, средние вероятности, transitions, stability, diversity).

**Версия**: 3.1.2 (Audit v4.2 observability + оптимизации)  
**Категория**: speech  
**GPU**: preferred (SpeechBrain model, requires GPU memory)

### Входы

- **`audio/audio.wav`** (любой аудио файл, поддерживаемый AudioUtils)
- **`audio/segments.json`** (сегменты от Segmenter, family: `emotion`) — **обязательно** (Audit v3)

**Требования**:
- Минимальная длительность аудио: **5 секунд** (иначе ошибка)
- **Режим работы (Audit v3)**:
  - только **сегментный режим**: `run_segments()` по Segmenter `families.emotion`

### Выходы

Экстрактор возвращает `ExtractorResult` с payload; **NPZ** пишет AudioProcessor в  
`result_store/<platform_id>/<video_id>/<run_id>/emotion_diarization_extractor/emotion_diarization_extractor_features.npz`.  
Схема: `emotion_diarization_extractor_npz_v1` (`schemas/...json`, `SCHEMA.md`).

#### Audit v4 — что попадает в **NPZ** (верхнеуровневые ключи)

| Ключ / группа | Форма | Заметка |
|---------------|-------|---------|
| `feature_names` / `feature_values` | F=7, `float32` | Порядок **фиксирован савером**: `segments_count`, `emotion_entropy`, `dominant_emotion_id`, `dominant_emotion_prob`, `emotion_transitions_count`, `emotion_stability_score`, `emotion_diversity_score`. Ид доминирующей эмоции в tabular — **float** (например `2.0`). |
| `segment_*`, `segment_mask` | `float32[N]` / `bool[N]` | Строгое N с Segmenter. |
| `emotion_id`, `emotion_confidence` | `int32[N]`, `float32[N]` | У invalid — `-1` / `NaN`. |
| `emotion_labels` | `object[C]` | Подписи классов. |
| Опционально (флаги / `features_enabled`) | | `emotion_probs` `[N,C]`, `emotion_mean_probs` `[C]`, dict-поля (`emotion_distribution`, …) как **scalar `ndarray` с `.item()` → dict**, см. `npz_savers/emotion_diarization.py`. |
| `meta` | object | `features_enabled`, `models_used`, при успешном прогоне — `model_name` и `weights_digest` в meta (из payload; в старых артефактах могли быть `null`, если не прокидывались). |
| **Не в NPZ** | | Поля **`rms`**, **`peak`** остаются в **payload** экстрактора, савер их **не** записывает. |

#### Audit v4.2 observability:
- `meta.stage_timings_ms` (dict): покомпонентные тайминги (ms), пишутся в NPZ meta
- `meta.emotion_diarization_resource_profile` (dict|None): best-effort snapshot RSS/VMS/VRAM (если включено)
  - включение: `AP_EMOTION_DIARIZATION_RESOURCE_PROFILE=1`

#### Обязательные поля payload (Audit v3, успешный `run_segments`)

- `segments_count`: количество **валидных** сегментов (int, `sum(segment_mask)`)
- `segments_total`: общее число окон от Segmenter (int)
- `sample_rate`: частота дискретизации аудио (int, по умолчанию 16000 Hz)
- `model_name`, `weights_digest`: строки из спецификации ModelManager (попадают в **meta** NPZ через payload)
- `device_used`: устройство обработки (str, обычно `"cuda"`)
- `rms`: RMS значение аудио (float)
- `peak`: пиковое значение аудио (float)
- `emotion_labels`: список названий эмоций (list[str], если указаны в runtime_params)
- `segment_start_sec`: список времен начала сегментов (list[float])
- `segment_end_sec`: список времен окончания сегментов (list[float])
- `segment_center_sec`: список времен центров сегментов (list[float])
- `segment_mask`: маска валидных сегментов (list[bool], strict-aligned)
- `emotion_contract_version`: версия контракта для валидации совместимости (str, `"emotion_contract_v1"`)

#### Feature-gated поля (включаются через флаги)

**`--emotion-enable-probs`**:
- `emotion_probs`: вероятности эмоций для каждого окна (np.ndarray [N, C] float32), где N — количество окон, C — количество классов эмоций

**(Audit v3 default: enabled)**:
- `emotion_id`: ID доминирующей эмоции для каждого окна (np.ndarray [N] int32; invalid сегменты = `-1`)

**(Audit v3 default: enabled)**:
- `emotion_confidence`: уверенность (максимальная вероятность) для каждого окна (np.ndarray [N] float32; invalid сегменты = `NaN`)

**`--emotion-enable-mean-probs`**:
- `emotion_mean_probs`: средние вероятности по всем окнам (np.ndarray [C] float32)

**(Audit v3: always-on)**:
- `emotion_entropy`: энтропия распределения эмоций (float, мера разнообразия)

**(Audit v3: always-on)**:
- `dominant_emotion_id`: ID доминирующей эмоции по всему аудио (int)
- `dominant_emotion_prob`: вероятность доминирующей эмоции (float)
- `emotion_distribution`: распределение времени по эмоциям (dict[emotion_id, time_ratio])
- `emotion_segments_per_emotion`: количество сегментов для каждой эмоции (dict[emotion_id, count])
- `emotion_duration_per_emotion`: длительность каждой эмоции в секундах (dict[emotion_id, duration_sec])
- `emotion_transitions_count`: количество переходов между эмоциями (int)
- `emotion_stability_score`: метрика стабильности эмоций (0 = нестабильная, 1 = стабильная)
- `emotion_diversity_score`: метрика разнообразия эмоций (нормализованная энтропия, 0-1)

**`--emotion-enable-quality-metrics`**:
- `emotion_quality_metrics`: метрики качества классификации:
  ```python
  {
      "confidence_mean": float,      # Среднее confidence
      "confidence_std": float,       # Стандартное отклонение confidence
      "confidence_min": float,       # Минимальное confidence
      "confidence_max": float,       # Максимальное confidence
      "confidence_median": float,    # Медианное confidence
      "mean_probs_min": float,      # Минимальная средняя вероятность
      "mean_probs_max": float,      # Максимальная средняя вероятность
      "mean_probs_std": float,      # Стандартное отклонение средних вероятностей
  }
  ```

#### Специальные случаи

**Пустое аудио** (status="empty"):
- `status`: `"empty"`
- `empty_reason`: `"audio_silent"` (если silence detection включен)
- Остальные поля присутствуют (без emotion_probs и агрегатов)

### Feature Dependencies

**Зависимости между фичами**:
- `dominant` зависит от `ids` или `confidence` (использует `emotion_id` для вычисления transitions, distribution, stability)
- `quality_metrics` зависит от `confidence` и `mean_probs` (использует их для вычисления метрик)

**Зависимости от других extractors**:
- Нет явных зависимостей от других extractors
- Может использоваться в `speech_analysis_extractor` как зависимость (требует `emotion_probs` или `emotion_id`)

**Contract version для совместимости**:
- `emotion_contract_version="emotion_contract_v1"` используется для валидации совместимости с downstream extractors (например, `speech_analysis_extractor`)

### Алгоритм

#### Режим 1: Сегментная обработка (`run_segments()`, по умолчанию)

##### 1. Предобработка аудио

1. Загрузка сегментов из `segments.json` (family: `emotion`)
2. Для каждого сегмента:
   - Загрузка аудио через `AudioUtils.load_audio_segment()`
   - Ресемплирование до `sample_rate` (по умолчанию 16000 Hz)
   - Преобразование в моно канал
3. Паддинг сегментов до максимальной длины для батчинга
4. Проверка на тишину (если `--emotion-enable-silence-detection` не отключен):
   - Если конкатенированное аудио тихое (peak < `silence_peak_threshold`, RMS < `silence_rms_threshold`), возвращается пустой результат

##### 2. Извлечение вероятностей эмоций через SpeechBrain Speech_Emotion_Diarization

- **Вход**: паддингнутые аудио сегменты, shape `[B, max_length]` float32
- **Выход**: вероятности эмоций, shape `[B, num_emotions]` float32
- **ModelManager spec**: `emotion_diarization_{size}_inprocess` (small или large)
- **Runtime**: `inprocess` (SpeechBrain модель загружается через ModelManager)
- **Engine**: `speechbrain` (SpeechBrain Speech_Emotion_Diarization)
- **Precision**: `fp32` (полная точность, SpeechBrain модели работают лучше с fp32)
- **Batching**: если сегментов >100, автоматически разбивается на батчи по 100 сегментов (или использует `batch_size`)
- **Модель**: использует WavLM-large encoder для извлечения признаков и MLP для классификации эмоций

##### 3. Валидация вероятностей

- Проверка dtype (float32)
- Проверка shape (2D [N, C])
- Проверка на NaN/inf
- Проверка диапазонов [0, 1]
- Проверка нормализации (сумма по строкам ≈ 1.0)
- Проверка количества классов (2-20)

##### 4. Валидация emotion_labels

- Проверка типа (list)
- Проверка длины (должна совпадать с количеством классов)
- Проверка на дубликаты
- Проверка типов элементов (все должны быть str)

##### 5. Нормализация вероятностей

- Защитная нормализация: сумма по строкам = 1.0 (на случай некорректных выходов модели)

##### 6. Вычисление базовых агрегатов

- `emotion_id`: argmax по каждому окну
- `emotion_confidence`: максимальная вероятность по каждому окну
- `emotion_mean_probs`: средние вероятности по всем окнам
- `emotion_entropy`: энтропия распределения эмоций
- `dominant_emotion_id`: argmax средних вероятностей
- `dominant_emotion_prob`: максимальная средняя вероятность

##### 7. Вычисление дополнительных агрегатов (если включено)

- `emotion_transitions_count`: количество переходов между эмоциями
- `emotion_distribution`: доли времени по эмоциям
- `emotion_segments_per_emotion`: количество сегментов для каждой эмоции
- `emotion_duration_per_emotion`: длительность каждой эмоции в секундах
- `emotion_stability_score`: метрика стабильности (inverse of transitions frequency)
- `emotion_diversity_score`: нормализованная энтропия (entropy / max_entropy)

##### 8. Вычисление метрик качества (если включено)

#### Режим `run()` (legacy)

`run()` отключён в audited режиме. Используйте только `run_segments()` по `families.emotion`.

1. Загрузка всего аудио файла через `AudioUtils.load_audio()`
2. Ресемплирование до `sample_rate` (по умолчанию 16000 Hz)
3. Преобразование в моно канал
4. Проверка минимальной длительности (≥ 5 секунд)

##### 2. Проверка на тишину

- Если `enable_silence_detection=True` и аудио тихое, возвращается пустой результат

##### 3. Inference для полного аудио

- Обработка всего аудио как одного сегмента через `_infer_probs_batch()`
- Получение вероятностей эмоций для всего аудио (shape: `[1, num_emotions]`)

##### 4. Постобработка и агрегация

- Нормализация вероятностей
- Валидация emotion_labels
- Вычисление агрегатов (аналогично сегментному режиму, но для одного сегмента)
- Для `run()` режима:
  - `emotion_distribution` содержит только одну эмоцию (100% времени)
  - `emotion_stability_score = 1.0` (один сегмент = стабильный)
  - `emotion_transitions_count = 0` (нет переходов)

**Прогресс-репортинг**: обновления на этапах загрузки, inference и завершения (если `progress_callback` установлен)

- Распределение confidence (mean, std, min, max, median)
- Распределение средних вероятностей (min, max, std)

### Конфигурация

#### Параметры модели

```python
{
    "device": "auto",              # "auto" | "cuda" | "cpu"
    "model_size": "small",         # "small" | "large"
    "sample_rate": 16000,          # Частота дискретизации (Hz)
    "batch_size": 16,              # Размер батча для обработки окон (для run_segments)
    "silence_peak_threshold": 1e-3,  # Порог peak для детекции тишины
    "silence_rms_threshold": 1e-4,   # Порог RMS для детекции тишины
    "enable_silence_detection": True,  # Включить проверку на тишину
    "process_full_audio": False,   # Если True, обрабатывает все аудио целиком (использует run() вместо run_segments())
    "progress_callback": None,     # Callback для прогресс-репортинга (опционально)
}
```

**Параметры модели** (из ModelManager spec `emotion_diarization_{model_size}_inprocess`):
- `model_class`: `Speech_Emotion_Diarization` (класс модели SpeechBrain)
- `local_artifacts`: `audio/emotion_diarization/wavlm_large` (путь к директории модели)
- `emotion_labels`: список названий эмоций (опционально, list[str], если не указаны, извлекаются из label_encoder модели)

#### Feature Gating (персональные флаги)

**По умолчанию в коде**: все фичи отключены (opt-in)  
**По умолчанию в конфиге** (`global_config.yaml`): `enable_ids` и `enable_confidence` включены (`true`)

- `--emotion-enable-probs` / `enable_probs`: включить `emotion_probs` (per-window probabilities)
- `--emotion-enable-ids` / `enable_ids`: включить `emotion_id` (argmax per window) - **включено по умолчанию в конфиге**
- `--emotion-enable-confidence` / `enable_confidence`: включить `emotion_confidence` (max prob per window) - **включено по умолчанию в конфиге**
- `--emotion-enable-mean-probs` / `enable_mean_probs`: включить `emotion_mean_probs` (mean probabilities)
- `--emotion-enable-entropy` / `enable_entropy`: включить `emotion_entropy`
- `--emotion-enable-dominant` / `enable_dominant`: включить `dominant_emotion_id/prob` и агрегаты (distribution, transitions, stability, diversity)
- `--emotion-enable-quality-metrics` / `enable_quality_metrics`: включить `emotion_quality_metrics` (метрики качества)

**Рекомендации для обучения моделей**:
- Включить все фичи для максимального качества и полноты данных
- Минимальный набор: `enable_ids` + `enable_confidence` (включены по умолчанию в конфиге)

### Архитектура

1. **Инициализация**:
   - Получение модели через `ModelManager.get_spec()` → `emotion_diarization_{model_size}_inprocess`
   - Проверка runtime = "inprocess" и engine = "speechbrain"
   - Загрузка модели через `ModelManager.get()` (SpeechBrain Speech_Emotion_Diarization)
   - Модель автоматически загружается на устройство через SpeechBrainProvider
   - Извлечение emotion_labels из label_encoder модели (если не указаны в runtime_params)
   - Инициализация `AudioUtils`

2. **Обработка сегментов** (`run_segments()`, по умолчанию):
   - Валидация входных данных (длительность ≥ 5 сек)
   - Загрузка и предобработка аудио сегментов
   - Паддинг до максимальной длины
   - Проверка на тишину (если включено)
   - Инференс через SpeechBrain модель (батчами):
     - `encode_batch()` для получения embeddings через WavLM encoder
     - `avg_pool()` для усреднения по времени
     - `output_mlp()` для получения logits
     - `softmax()` для преобразования в вероятности
   - Прогресс-репортинг: обновления каждые 10% батчей (если `progress_callback` установлен)
   - Валидация вероятностей
   - Валидация emotion_labels
   - Нормализация вероятностей
   - Вычисление базовых агрегатов
   - Вычисление дополнительных агрегатов (feature-gated)
   - Вычисление метрик качества (feature-gated)
   - Формирование payload (feature-gated)
   - Детальное профилирование: логирование времени для каждого этапа (load, silence, padding, inference, aggregates, postprocess)

3. **Обработка полного аудио** (`run()`, если `process_full_audio=True`):
   - Валидация входных данных (длительность ≥ 5 сек)
   - Загрузка всего аудио файла
   - Проверка на тишину (если включено)
   - Инференс для всего аудио как одного сегмента
   - Прогресс-репортинг: обновления на этапах загрузки, inference и завершения
   - Валидация и постобработка (аналогично `run_segments()`)
   - Вычисление агрегатов для одного сегмента
   - Детальное профилирование: логирование времени для каждого этапа

4. **Вспомогательные методы**:
   - `_validate_probs()`: полная валидация вероятностей
   - `_validate_emotion_labels()`: полная валидация emotion_labels
   - `_infer_probs_batch()`: batch inference через SpeechBrain модель (получение logits и преобразование в вероятности)
   - `_rms_and_peak()`: вычисление RMS и peak значений для детекции тишины

5. **Обработка ошибок**:
   - Модель не найдена → `RuntimeError` (ModelManager raises `weights_missing`)
   - Невалидный вход → `RuntimeError`
   - Ошибка inference → `RuntimeError` с описанием ошибки
   - Аудио < 5 сек → `RuntimeError`
   - Пустые сегменты → `ValueError`
   - Валидация вероятностей/labels → `ValueError`

### Обработка ошибок

**Политика NO FALLBACK**:
- Отсутствие модели → `RuntimeError` (ModelManager raises `weights_missing`)
- Модель не загружается → ошибка
- Аудио < 5 секунд → ошибка
- Пустые сегменты → ошибка
- Валидация вероятностей/labels → ошибка

**Специальные случаи**:
- **Тихое аудио**: возвращается `status="empty"`, `empty_reason="audio_silent"` (если silence detection включен)
- **Несоответствие sample rate**: ошибка с описанием
- **Неожиданная форма вероятностей**: ошибка с описанием

### Особенности

- **SpeechBrain модель**: модель Speech_Emotion_Diarization загружается локально через ModelManager (no-network policy)
- **WavLM encoder**: использует WavLM-large для извлечения аудио признаков
- **Два режима работы**:
  - **Сегментный режим** (по умолчанию): работает с сегментами от Segmenter (family: `emotion`)
  - **Полное аудио режим** (`process_full_audio=True`): обрабатывает все аудио целиком как один сегмент
- **Batch processing**: эффективная обработка окон батчами через SpeechBrain модель (для сегментного режима)
- **Автоматическое разбиение**: при большом количестве сегментов (>100) автоматически разбивается на батчи
- **Нормализация**: защитная нормализация вероятностей (сумма = 1.0)
- **Нет fallback**: строгая политика - ошибка при отсутствии модели
- **Паддинг сегментов**: сегменты разной длины падятся до максимальной для батчинга (только для сегментного режима)
- **Progress reporting**: 
  - Для `run_segments()`: обновление прогресса каждые 10% батчей (если батчей ≥10 и `progress_callback` установлен)
  - Для `run()`: обновления на этапах загрузки, inference и завершения
- **Детальное профилирование**: логирование времени выполнения для каждого этапа (load, silence, padding, inference, aggregates, postprocess)
- **Feature gating**: все фичи opt-in через персональные флаги (в конфиге по умолчанию `enable_ids` и `enable_confidence` включены)
- **Contract versioning**: версия контракта для валидации совместимости с downstream extractors
- **Полная валидация**: проверка вероятностей и emotion_labels на всех этапах
- **Автоматическое извлечение labels**: emotion_labels извлекаются из label_encoder модели, если не указаны в runtime_params

### Performance characteristics

**Resource costs**:
- **GPU VRAM**: зависит от размера модели (small/large) и batch_size
- **CPU RAM**: минимальные (только для предобработки)
- **Estimated duration**: ~6.0 секунд для типичного видео
- **Batch efficiency**: обработка батчами увеличивает throughput на GPU

**Параметры производительности**:
- `model_size`: `small` быстрее и требует меньше памяти, `large` точнее
- `batch_size`: контроль размера батча для inference (по умолчанию 16)
- Количество сегментов: влияет на количество батчей для обработки

### Visualization

**Рекомендации для UI/сайта**:

1. **Timeline визуализация**:
   - Горизонтальная временная шкала с цветовой кодировкой эмоций
   - Каждый сегмент отображается как полоса с цветом, соответствующим emotion_id
   - Интерактивные tooltips с информацией о сегменте (start, end, confidence, emotion_id)
   - Zoom и pan для навигации по длинным видео

2. **Распределение эмоций**:
   - Pie chart или bar chart с долями времени по эмоциям (`emotion_distribution`)
   - Таблица со статистикой по каждой эмоции (segments_count, duration, time_ratio)
   - Highlight доминирующей эмоции (`dominant_emotion_id`)

3. **Метрики качества**:
   - Отображение confidence distribution (mean, std, min, max, median)
   - Визуализация stability score и diversity score
   - Индикаторы качества (good/fair/poor) на основе метрик

4. **Агрегаты**:
   - Отображение `emotion_entropy` (мера разнообразия эмоций)
   - Количество переходов между эмоциями (`emotion_transitions_count`)
   - Стабильность эмоций (`emotion_stability_score`)
   - Разнообразие эмоций (`emotion_diversity_score`)

5. **HTML renderer для дебага**:
   - Локальный HTML renderer доступен через `render_emotion_diarization_extractor_html()`
   - Включает timeline plot, статистику, метрики качества, распределения
   - Только для локального дебага, не в production артефактах

**Пример использования HTML renderer**:
```python
from src.core.renderer import render_emotion_diarization_extractor_html

html_path = render_emotion_diarization_extractor_html(
    npz_path="result_store/.../emotion_diarization_extractor/emotion_diarization_extractor_features.npz",
    output_path="debug_emotion.html"
)
```

### Связанные компоненты

- **ModelManager** (`dp_models`): управление моделями и спецификациями
- **SpeechBrainProvider** (`dp_models`): провайдер для загрузки SpeechBrain моделей
- **SpeechBrain** (`speechbrain`): библиотека для загрузки и использования Speech_Emotion_Diarization
- **AudioUtils**: загрузка и предобработка аудио
- **BaseExtractor**: базовый интерфейс экстрактора
- **Segmenter**: генерация сегментов (family: `emotion`)
- **speech_analysis_extractor**: может использовать результаты emotion diarization как зависимость

### Примечания

Audit v3:

- Модель и веса должны быть доступны **строго локально** через `dp_models` (`spec_catalog/audio/emotion_diarization_*_inprocess.yaml` + `bundled_models/audio/emotion_diarization/wavlm_large/`). Любые runtime-download запрещены.
- Любые зависимости модели на HF transformers (например WavLM) должны быть уже в оффлайн-кэше, который выставляет ModelManager (иначе extractor корректно упадёт с `weights_missing/model_load_failed`).
- `run()`/`process_full_audio` отключены: только `run_segments()` по Segmenter `families.emotion`.
- Семантика short audio: <5s → `status="empty"`, `empty_reason="audio_too_short"`.
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FEATURE_DESCRIPTION (root)](../FEATURE_DESCRIPTION.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
