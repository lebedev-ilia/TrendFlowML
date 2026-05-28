# Emotion Face Module

Модуль для анализа эмоций на лицах в видео с использованием модели EmoNet. Извлекает базовые эмоции Ekman, валентность/активацию, ключевые кадры, метрики качества и расширенные фичи (микроэмоции, физиологические сигналы, асимметрия лица).

## Описание

Модуль `emotion_face` обрабатывает видео и анализирует эмоциональные состояния на лицах. Использует модель EmoNet для распознавания 8 базовых эмоций Ekman (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt), а также извлекает непрерывные значения валентности (valence) и активации (arousal).

**Примечание**: Модуль по умолчанию отключён в `global_config.yaml` (`emotion_face: false`). Для использования модуля необходимо явно включить его в конфигурации.

**Версия (producer_version)**: 2.0.2  
**schema_version**: `emotion_face_npz_v3`  
**Schemas**:
- Human: `DataProcessor/VisualProcessor/modules/emotion_face/docs/SCHEMA.md`
- Machine: `DataProcessor/VisualProcessor/schemas/emotion_face_npz_v3.json`

**Проверка NPZ:** `utils/validate_emotion_face.py` — `<npz>` `--struct` / `--qa` или батч `--results-base` / `--platform-id` · краткий обзор ключей и meta: `docs/FEATURE_DESCRIPTION.md`.

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

В baseline sampling для `emotion_face` определяется так (decision):

1) **Axis alignment**: output выровнен по `metadata[emotion_face].frame_indices` (Segmenter contract; union-domain).  
   Fallback (legacy): `core_face_landmarks.frame_indices` (warning).  
2) берём **все кадры на axis**, где `face_present=true`  
3) применяем собственную выборку **по этим кадрам**: `face_frame_stride` (по умолчанию **каждый 4-й**)  
4) применяем cap `max_frames` (по умолчанию **200**)  

Важно: модуль **не** делает fallback на `fps` и не генерирует sampling по времени самостоятельно.

## Models (ModelManager)

Модель EmoNet грузится через `dp_models.ModelManager` (локально, **no-network**):

- **Spec name (default)**: `emonet_8_inprocess`
- **Weights**: `DP_MODELS_ROOT/bundled_models/visual/emonet/emonet_8.pth`
- Legacy override (debug, не рекомендуется): `emo_path` (явный путь к весам).  
  Если ModelManager не доступен/не нашёл модель, модуль **ошибается** (fail-fast), кроме случая явного `emo_path`.

## Output artifact

- **Path**: `result_store/.../emotion_face/emotion_face.npz`
- **Schema**: `emotion_face_npz_v3`

### Основные ключи (NPZ)

- **`frame_indices (N,)` / `times_s (N,)`**: time-axis (Segmenter-aligned)
- **`sequence_features`** (dict, axis-aligned):
  - `face_present (N,) bool`
  - `processed_mask (N,) bool`
  - `valence/arousal/intensity/emotion_confidence (N,) float32` (NaN если `processed_mask=false`)
  - `emotion_probs (N,8) float32`, `dominant_emotion_id (N,) int8` (`-1` если не обработано)
  - `face_count (N,) int16` (число лиц по core_face_landmarks)
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

**HTML debug страница** (опционально, offline):
- Путь: `result_store/.../emotion_face/_render/render.html`
- Содержит offline SVG графики:
  - Timeline: valence/arousal/intensity/confidence/face_count
  - Distributions: статистики по valence/arousal/intensity
  - Key facts: status/versions/stage_timings_ms

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

## Quality Status & Refinements (Audit v3)

- `schema_version`: `emotion_face_npz_v3`, `producer_version=2.0.2`
- **Axis alignment**: Segmenter axis + fallback на core_face_landmarks axis (legacy)
- **Compute gating**: inference только на face-кадрах; `processed_mask` фиксирует внутреннюю выборку
- **No-network**: EmoNet только через ModelManager (кроме explicit `emo_path` debug override)
- **Offline render**: без CDN (SVG)

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

## CLI параметры

Модуль поддерживает следующие CLI параметры (все опциональны, кроме `--frames-dir` и `--rs-path`):

### Обязательные параметры
- `--frames-dir` (required): Директория с кадрами (должна содержать metadata.json)
- `--rs-path` (required): Путь к директории ResultsStore для сохранения результатов

### Параметры валидации
- `--min-frames-ratio` (default: 0.8): Минимальное соотношение кадров
- `--min-keyframes` (default: 3): Минимальное количество ключевых кадров
- `--min-transitions` (default: 2): Минимальное количество переходов
- `--min-diversity-threshold` (default: 0.2): Минимальный порог разнообразия
- `--quality-threshold` (default: 0.4): Порог качества

### Параметры производительности (адаптивные батчи)
- `--memory-threshold-low` (default: 2000): Порог памяти (низкий уровень, MB)
- `--batch-load-low` (default: 20): Размер батча загрузки (низкий уровень)
- `--batch-process-low` (default: 8): Размер батча обработки (низкий уровень)
- `--memory-threshold-medium` (default: 4000): Порог памяти (средний уровень, MB)
- `--batch-load-medium` (default: 30): Размер батча загрузки (средний уровень)
- `--batch-process-medium` (default: 12): Размер батча обработки (средний уровень)
- `--memory-threshold-high` (default: 8000): Порог памяти (высокий уровень, MB)
- `--batch-load-high` (default: 50): Размер батча загрузки (высокий уровень)
- `--batch-process-high` (default: 15): Размер батча обработки (высокий уровень)
- `--batch-load-very-high` (default: 80): Размер батча загрузки (очень высокий уровень)
- `--batch-process-very-high` (default: 24): Размер батча обработки (очень высокий уровень)

### Параметры обработки
- `--enable-structured-metrics` (default: False): Включить структурированные метрики
- `--min-faces-threshold` (default: 1): Минимальный порог лиц
- `--target-length` (default: 256): Целевая длина последовательности
- `--max-retries` (default: 2): Максимальное количество повторных попыток
- `--transition-threshold` (default: 0.3): Порог перехода
- `--max-gap-seconds` (default: 0.5): Максимальный разрыв в секундах
- `--max-samples-per-segment` (default: 10): Максимальное количество сэмплов на сегмент

### Параметры модели
- `--emo-path` (optional): Путь к модели EmoNet (legacy, не рекомендуется)
- `--emonet-model-spec` (default: "emonet_8_inprocess"): ModelManager spec name для EmoNet (предпочтительно)
- `--device` (default: "cuda"): Устройство для обработки (cuda/cpu)

### Baseline v1 sampling / multi-face параметры
- `--face-frame-stride` (default: 4): Stride по кадрам с лицами из core_face_landmarks
- `--max-frames` (default: 200): Максимальное количество кадров для обработки после stride
- `--max-faces-per-frame` (default: 2): Максимальное количество лиц на кадр для inference
- `--face-bbox-margin` (default: 0.20): Отступ для обрезки лица (bbox margin ratio)

### Feature gating (noisy/expensive, off by default)
- `--enable-microexpressions`: Включить микроэмоции (noisy/expensive)
- `--enable-emotional-individuality`: Включить эмоциональную индивидуальность (noisy/expensive)
- `--enable-face-asymmetry`: Включить асимметрию лица (noisy/expensive)

### Логирование
- `--log-level` (default: "INFO"): Уровень логирования (DEBUG/INFO/WARN/ERROR)

## Performance characteristics

Замеры `resource_costs/*` для `emotion_face` пока не добавляли (Q15: позже).

