# Emotion Face Module

Модуль для анализа эмоций на лицах в видео с использованием модели EmoNet. Извлекает базовые эмоции Ekman, валентность/активацию, ключевые кадры, метрики качества и расширенные фичи (микроэмоции, физиологические сигналы, асимметрия лица).

## Описание

Модуль `emotion_face` обрабатывает видео и анализирует эмоциональные состояния на лицах. Использует модель EmoNet для распознавания 8 базовых эмоций Ekman (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt), а также извлекает непрерывные значения валентности (valence) и активации (arousal).

### Основные возможности

- **Распознавание эмоций**: 8 классов эмоций Ekman с вероятностями
- **Valence/Arousal**: непрерывные значения эмоциональной валентности и активации
- **Ключевые кадры**: автоматическая детекция пиков эмоций и эмоциональных переходов
- **Метрики качества**: разнообразие, переходы, монотонность, вариативность
- **Расширенные фичи**: микроэмоции, физиологические сигналы, асимметрия лица
- **Адаптивная обработка**: автоматическая адаптация параметров для разных типов видео
- **Управление памятью**: батчевая обработка с адаптивными размерами батчей

## Зависимости
### Обязательные зависимости

- **`core_face_landmarks`** (required): даёт `face_present` + `face_landmarks` (нормализованные координаты 0..1) и `frame_indices` в union-domain.

### Контракты (baseline)

- **NPZ-only**: модуль пишет только `emotion_face.npz`. UI данные — в `meta.ui_payload`. Отдельные `emotion_face.json` запрещены.
- **Time-axis**: `times_s` строго из `metadata.json["union_timestamps_sec"][frame_indices]` (no-fallback).
- **No-fallback deps**: отсутствие `core_face_landmarks/landmarks.npz` ⇒ `error` (raise).
- **Empty semantics**: если в `core_face_landmarks` нет ни одного кадра с лицом ⇒ `status="empty"`, `empty_reason="no_faces_in_video"`.

## Sampling / units-of-processing requirements

В baseline v1 sampling для `emotion_face` определяется так (decision):

1) берём **все кадры**, где `core_face_landmarks.face_present` имеет хотя бы одно лицо  
2) применяем собственную выборку **по этим кадрам**: `face_frame_stride` (по умолчанию **каждый 4-й**)  
3) применяем cap `max_frames` (по умолчанию **200**)  

Важно: модуль **не** делает fallback на `fps` и не генерирует sampling по времени самостоятельно.

## Models (ModelManager)

Модель EmoNet грузится через `dp_models.ModelManager` (локально, no-network):

- **Spec name (default)**: `emonet_8_inprocess`
- **Weights**: `DP_MODELS_ROOT/bundled_models/visual/emonet/emonet_8.pth`
- Legacy override (не рекомендуется): `emo_path` (явный путь к весам).

**Fallback механизм загрузки модели**:
Если ModelManager не может найти модель, модуль автоматически пытается найти её в следующих местах (в порядке приоритета):
1. Путь из `emo_path` (если указан в конфиге)
2. `DP_MODELS_ROOT/bundled_models/visual/emonet/emonet_8.pth` (если `DP_MODELS_ROOT` указывает на `dp_models`)
3. `DP_MODELS_ROOT/visual/emonet/emonet_8.pth` (если `DP_MODELS_ROOT` указывает на `dp_models/bundled_models`)
4. Абсолютный путь на основе корня DataProcessor: `DataProcessor/dp_models/bundled_models/visual/emonet/emonet_8.pth`
5. Путь по умолчанию в модуле: `modules/emotion_face/models/emonet/pretrained/emonet_8.pth`

## Output artifact

- **Path**: `result_store/.../emotion_face/emotion_face.npz`
- **Schema**: `emotion_face_npz_v1`

### Основные ключи (NPZ)

- **`sequence_features`** (dict):
  - `frame_indices (N,) int32`
  - `times_s (N,) float32`
  - `valence (N,) float32`
  - `arousal (N,) float32`
  - `intensity (N,) float32`
  - `emotion_confidence (N,) float32`
  - `emotion_probs (N,8) float32` (порядок классов фиксирован)
  - `dominant_emotion_id (N,) int8` (0..7)
  - `face_count (N,) int16`
  - multi-face (per-frame): `*_faces` массивы с shape `(N, max_faces_per_frame, ...)` и `NaN` для отсутствующих лиц
- **`keyframes`** (list[dict]): события `emotion_peak` / `transition` с `global_index`, `local_index`, `time_s`, `score`.
- **`summary`** (dict): содержит `stage_timings_ms`.
- **`meta.ui_payload`**: структура для UI (таймлайн + метки).

## Features

Полный список и смысл фичей: см. `FEATURES_DESCRIPTION.md`.

Noisy/expensive блоки управляются флагами (off by default):
- `enable_microexpressions`
- `enable_emotional_individuality`
- `enable_face_asymmetry`

## Progress reporting

Пишет granular progress (>=10 апдейтов) в `state_events.jsonl` на стадии `process_frames`.

## Quality validation & human-friendly inspection

### Human-friendly визуализация (Render System)

`emotion_face` генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/emotion_face/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по эмоциям (frames_count, valence_mean, arousal_mean, intensity_mean, emotion_confidence_mean, faces_found_frames)
- **Timeline**: данные по каждому кадру (frame_index, time_s, valence, arousal, intensity, emotion_confidence, face_count)
- **Distributions**: распределения valence, arousal, intensity (min, max, mean, std, median, percentiles)
- **Dominant emotion distribution**: распределение доминирующих эмоций по кадрам

Render-context может быть использован:
- **LLM** для генерации текстовых описаний эмоционального состояния в видео
- **Frontend** для построения графиков и визуализаций (timeline charts, distributions)
- **Debugging**: быстрая проверка качества эмоционального анализа без загрузки NPZ

**HTML debug страница** (опционально):
- Путь: `result_store/.../emotion_face/_render/render.html`
- Содержит интерактивные графики (Chart.js):
  - Timeline: valence, arousal и intensity по времени
  - Distributions: статистики по valence, arousal, intensity
  - Summary metrics: ключевые показатели в удобном формате

**Конфигурация** (в `global_config.yaml`):
```yaml
visual:
  modules:
    emotion_face:
      render:
        enable_render: true  # Генерировать render-context JSON
        enable_html_render: true  # Генерировать HTML debug страницу
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).

### Legacy HTML-отчёт

HTML-отчёт: `quality_report/demo_emotion_face_quality.py`

Пример:

```bash
python DataProcessor/VisualProcessor/modules/emotion_face/quality_report/demo_emotion_face_quality.py \
  --rs-path /path/to/result_store/<platform>/<video>/<run> \
  --out-html emotion_face_quality.html
```

## Batch Processing

Модуль поддерживает **batch processing** для одновременной обработки нескольких видео:

- **GPU batching**: кадры из всех видео собираются в батчи и обрабатываются через EmoNet одновременно
- **Гибридный подход**: сбор кадров из всех видео → батчинг с лимитом размера → распределение результатов обратно по видео
- **Изоляция**: каждый видео имеет свой `rs_path` для артефактов

**Конфигурация batch processing** (в `global_config.yaml`):
```yaml
visual:
  batch_processing:
    enable_gpu_batching: true
    max_frames_per_gpu_batch: null  # null = без лимита
```

**Использование**:
- Batch processing автоматически активируется при вызове `run_batch()` в `VisualProcessor/main.py`
- Модуль реализует `supports_batch = True` и метод `process_batch()`
- Batch processing utility: `utils/emotion_face_batch.py`

## Performance characteristics

Замеры `resource_costs/*` для `emotion_face` пока не добавляли (Q15: позже).

