## Component: `core_face_landmarks` (Tier‑0 baseline)

- **Контракт NPZ, melt/QA, примеры команд:** [docs/FEATURE_DESCRIPTION.md](docs/FEATURE_DESCRIPTION.md)
- **Валидатор:** `utils/validate_core_face_landmarks_npz.py` (`--struct`, `--qa`, `--ranges` или батч `--results-base`)

### Назначение

`core_face_landmarks` извлекает landmarks лица (MediaPipe FaceMesh) по выборке кадров (union-domain).
Дополнительно (опционально) может извлекать `pose` и `hands`, но **face_mesh обязателен** для baseline, т.к. `shot_quality` зависит от face features.

### Входы

- **Кадры**: `FrameManager.get(idx)` из `frames_dir` (RGB uint8).
- **Sampling (строго)**: из `frames_dir/metadata.json`:

```json
{
  "core_face_landmarks": { "frame_indices": [0, 7, 14] }
}
```

No-fallback: отсутствие/пустота `frame_indices` ⇒ **error**.

Также (baseline policy): компонент **зависит** от `core_object_detections` и читает `result_store/.../core_object_detections/detections.npz`,
чтобы запускать анализ лица **только на кадрах**, где детектирован класс `person`.

### Выход

Путь: `result_store/<platform_id>/<video_id>/<run_id>/core_face_landmarks/landmarks.npz`

Ключи (основные):
- `ARTIFACT_FILENAME = "landmarks.npz"` — фиксированное имя артефакта (один NPZ per run)
- `frame_indices (N,) int32`
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]` (required by baseline contract)
- `face_landmarks (N, FACES, 468, 3) float32` — **filtered** landmarks (по умолчанию, для downstream)  
  (NaN если лицо не найдено)
- `face_landmarks_raw (N, FACES, 468, 3) float32` — **raw** landmarks (QA/debug source-of-truth)
- `face_present (N, FACES) bool`
- `person_present (N,) bool` — person-mask из `core_object_detections` (до window расширения)
- `face_mesh_ran (N,) bool` — на каких primary позициях **реально запускали** FaceMesh (person + window)
- `has_any_face bool`
- `empty_reason object|null`:
  - валидный empty: `"no_faces_in_video"` (FaceMesh запускался, но лиц нет)
  - валидный empty: `"skipped_due_to_person_mask_no_person"` (person-mask пуст, FaceMesh не запускался)

Опциональные (если включены флаги):
- `pose_landmarks`, `pose_landmarks_raw`, `pose_present`, `has_any_pose`
- `hands_landmarks`, `hands_landmarks_raw`, `hands_present`, `has_any_hands`

Extended empty reasons (не меняют provider-status кроме face):
- `face_empty_reason`, `pose_empty_reason`, `hands_empty_reason`

### Empty semantics

- Если `face_mesh` включён и **FaceMesh реально запускался**, но лиц нет: это **валидный empty**:
  - `status="empty"`
  - `empty_reason="no_faces_in_video"`
  - данные `face_landmarks`/`face_landmarks_raw` остаются NaN, `face_present=False`
- Если `face_mesh` включён, но **person-mask пуст** (`person_present.sum()==0` ⇒ `face_mesh_ran.sum()==0`): это тоже **валидный empty**:
  - `status="empty"`
  - `empty_reason="skipped_due_to_person_mask_no_person"`
  - важно: это **не утверждение** “лиц нет”, это утверждение “мы не запускали FaceMesh из-за gating”
- Если `pose/hands` включены и не детектируются: это **не error**,
  но записываются `pose_empty_reason="no_pose_detected"` / `hands_empty_reason="no_hands_detected"`.

### Meta / models_used

`models_used[]` содержит одну запись:
- `model_name="mediapipe"`
- `model_version=<mediapipe.__version__>`
- `weights_digest` = sha256 от (mediapipe_version + ключевые параметры конфигурации)
- `runtime="inprocess"`, `engine="mediapipe"`, `precision="fp32"`, `device="cpu"`
- `stage_timings_ms` — словарь `{stage_name: duration_ms}` с таймингами ключевых стадий (см. ниже)

Ключевые стадии (пример):
- `total_total_ms` — общее время работы компонента (от старта до записи NPZ)
- `process_video_total_ms` — время основного цикла обработки кадров (pose/hands/face + optional temporal filtering)
- Дополнительно: стадии из внутреннего `Profiler` (например, `io.frame_load_total_ms`, `inference.face_total_ms`, `postproc.temporal_filter_total_ms`)

**Профилирование**:
- Профилирование включено по умолчанию (`--enable-profiling`, можно отключить через `--disable-profiling`)
- Внутренний `Profiler` собирает детальные тайминги для каждой стадии обработки
- Тайминги логируются в консоль после завершения обработки

### Progress / state events

Компонент пишет progress‑события в `state_events.jsonl` (per‑run папка `runs/state/<platform>/<video>/<run>/state_events.jsonl`):

- Стадии:
  - `start` → `load_deps` → `process_frames` → `post_process` → `save` → `done`
- Для `process_frames` отправляется **гранулярный** прогресс по кадрам:
  - `progress ∈ [0,1]`, `done`, `total` (кол-во обработанных `frame_indices`)

---

## Models

### CPU Models

1. **MediaPipe FaceMesh** (face landmarks)
   - **Triton**: ❌ Нет (in-process)
   - **Runtime**: `inprocess`
   - **Engine**: `mediapipe`
   - **Precision**: `fp32`
   - **Device**: `cpu`
   - **Model**: Встроенная в MediaPipe (внутренние TFLite модели)
   - **Landmarks**: 468 точек для лица
   - **Weights digest**: SHA256 от (mediapipe_version + конфигурация)

2. **MediaPipe Pose** (опционально, если `--use-pose`)
   - **Runtime**: `inprocess`
   - **Engine**: `mediapipe`
   - **Precision**: `fp32`
   - **Device**: `cpu`
   - **Landmarks**: 33 точки для позы

3. **MediaPipe Hands** (опционально, если `--use-hands`)
   - **Runtime**: `inprocess`
   - **Engine**: `mediapipe`
   - **Precision**: `fp32`
   - **Device**: `cpu`
   - **Landmarks**: 21 точка на руку (до 2 рук)

**Примечание**: Все модели MediaPipe работают на CPU. Компонент использует изолированную виртуальную среду `.core_face_landmarks_venv` с зафиксированной версией MediaPipe 0.10.14 (из-за изменений API в более новых версиях).

---

## Parallelization

### Внутренний параллелизм

- **Последовательная обработка**: MediaPipe обрабатывает кадры последовательно (не поддерживает батчинг)
- **Stage-1/Stage-2 оптимизация**: 
  - Stage-1: легковесное face detection на sparse выборке (stride-based)
  - Stage-2: FaceMesh запускается только на кадрах с обнаруженными лицами (или person-mask)
- **Person-mask фильтрация**: FaceMesh запускается только на кадрах, где `core_object_detections` детектировал `person`
- **Async producer/consumer**: Опциональный режим с предзагрузкой кадров (prefetch buffer = 8) для перекрытия I/O и обработки
- **Параллельная обработка**: Опциональный режим с worker pool (каждый воркер создает свои экземпляры моделей MediaPipe для thread-safety)

### Внешний параллелизм

- **Можно запускать несколько экземпляров параллельно** на разных видео (разные `run_id`)
- **Требования к изоляции**:
  - Разные `run_id` для каждого видео
  - Разные пути `result_store` (обеспечивается через `platform_id/video_id/run_id`)
  - Изолированная виртуальная среда для каждого компонента (`.core_face_landmarks_venv`)

### Комбинированный подход

- Stage-1/Stage-2 оптимизация (внутренняя) + внешний запуск на разных видео
- Thread-safety: компонент thread-safe для параллельного запуска на разных видео

### Batch Processing (Stage 3)

Компонент поддерживает **batch processing** для одновременной обработки нескольких видео с оптимизацией:

- **CPU parallelism через subprocess**: Параллельная обработка видео через subprocess с использованием изолированной виртуальной среды `.core_face_landmarks_venv`
- **Каждое видео обрабатывается отдельным subprocess**: Гарантирует использование правильного Python из виртуальной среды и изоляцию артефактов
- **Параллельная обработка видео**: Несколько видео обрабатываются параллельно через `ThreadPoolExecutor` (контролируется через `num_workers`)

**Использование**:
- Batch processing автоматически активируется при обработке нескольких видео через `--video-input-dir` или `--video-input-list`
- Контролируется через `visual.batch_processing.enable_cpu_parallel` в `global_config.yaml`
- Количество воркеров: `visual.batch_processing.max_video_workers` или автоматически определяется на основе CPU count

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): **1.5-2x** (за счет параллельной обработки видео)
- Для single video: ускорение зависит от настроек `enable_async` и `enable_parallel`

**Важно**: 
- Batch processing использует subprocess для каждого видео, что гарантирует использование изолированной виртуальной среды
- Каждое видео обрабатывается независимо, что обеспечивает изоляцию артефактов

---

## Performance characteristics

### Resource costs (measured)

**Источник данных**: `docs/models_docs/resource_costs/core_face_landmarks_costs_v1.json` (планируется)

**Единица обработки**: `frame`

**Типичные значения (preset="default", runtime="mediapipe")**:

| Resolution | Latency per frame | CPU RAM peak | Notes |
|------------|-------------------|--------------|-------|
| 320p | TBD ms | TBD MB | measurements pending |
| 480p | TBD ms | TBD MB | measurements pending |
| 720p | TBD ms | TBD MB | measurements pending |

**Для видео с N кадрами**: Total latency ≈ N × latency_per_frame (но с оптимизацией через person-mask и stage-1/2, фактически обрабатывается меньше кадров)

**Полные данные**: см. `docs/models_docs/resource_costs/core_face_landmarks_costs_v1.json` (планируется)

### Оптимизации производительности

Компонент включает следующие оптимизации:

1. **Автоматическое определение количества воркеров**: Если `num_workers` не указан, автоматически используется 75% CPU ядер (минимум 2, максимум 8)
2. **Оптимизированный prefetch buffer**: Увеличен с 4 до 8 для лучшего перекрытия I/O и обработки в async режиме
3. **Оптимизированный chunk size**: Множитель уменьшен с 4 до 2 для более равномерного распределения нагрузки между воркерами
4. **Отдельные экземпляры моделей на воркер**: Каждый воркер создает свои экземпляры MediaPipe моделей для thread-safety (избегает SIGSEGV)
5. **Person-mask фильтрация**: FaceMesh запускается только на кадрах с детектированными людьми (значительное сокращение обрабатываемых кадров)
6. **Stage-1/Stage-2 оптимизация**: Легковесное face detection на sparse выборке перед запуском FaceMesh
7. **Temporal filtering**: OneEuro filter для сглаживания landmarks во времени (включен по умолчанию, можно отключить через `--disable-temporal-filter`)

**Рекомендации по оптимизации**:

Для максимального ускорения (2-2.5x):
```yaml
core_face_landmarks:
  pose_model_complexity: 1  # Было 2 - самое большое ускорение
  num_workers: 4  # Или null для auto-detection
  pose_min_tracking_confidence: 0.4  # Было 0.5
  hands_min_tracking_confidence: 0.4  # Было 0.5
  face_mesh_min_tracking_confidence: 0.4  # Было 0.5
```

Для баланса скорости и качества (1.8-2x):
```yaml
core_face_landmarks:
  pose_model_complexity: 1  # Было 2
  num_workers: 4  # Или null для auto
```

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): **1.5-2x** (за счет параллельной обработки)
- Для single video с оптимизациями: **1.8-2.5x** (за счет снижения сложности моделей и оптимизации параллелизма)
- Для single video без оптимизаций: **1.1-1.2x** (за счет оптимизации prefetch и chunk size)

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными, изменилась только производительность.

### Temporal Filtering

Компонент поддерживает **temporal filtering** для сглаживания landmarks во времени:

- **OneEuro filter**: адаптивный фильтр для сглаживания временных рядов landmarks
- **Параметры**: `--temporal-filter-min-cutoff` (по умолчанию 1.0), `--temporal-filter-beta` (по умолчанию 0.0)
- **Включен по умолчанию**: можно отключить через `--disable-temporal-filter`
- **Применяется к**: face, pose и hands landmarks (если включены)
- **FPS**: автоматически определяется из `metadata.json` (по умолчанию 30.0)

**Примечание (Audit v3)**:
- temporal filter **не должен “галлюцинировать”** landmarks на кадрах без детекции.
  Выходные `*_landmarks` сохраняют `NaN` на кадрах, где исходно не было landmarks (source-of-truth = `*_landmarks_raw`).
- для визуальных QA overlays в `render.html` используем `*_landmarks_raw` и рисуем bbox только при `*_present=True`.

### Качество face детекта при sparse sampling (~4 fps)

При редкой выборке кадров (например `rate_fps≈4.0`) режим трекинга FaceMesh между кадрами может давать **false negatives**.
Для повышения recall рекомендуем:

- `face_mesh_static_image_mode=true` (детект каждый кадр, без reliance на трекинг)
- слегка снизить пороги:
  - `face_mesh_min_detection_confidence=0.4`
  - `face_mesh_min_tracking_confidence=0.4`
  - (stage1) `face_detection_min_confidence=0.4`

---

## Quality validation & human-friendly inspection

### Human-friendly визуализация (Render System)

`core_face_landmarks` генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/core_face_landmarks/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по landmarks (frames_count, landmark dimensions, mean/std/min/max/median для face, pose, hands landmarks)
- **Timeline**: данные по каждому кадру (time_sec, frame_index, landmark presence, количество детектированных лиц/поз/рук)
- **Distributions**: распределения значений landmarks (min, max, mean, std, median, percentiles)

Render-context может быть использован:
- **LLM** для генерации текстовых описаний видео
- **Frontend** для построения графиков и визуализаций (timeline charts, distributions)
- **Debugging**: быстрая проверка качества landmarks без загрузки NPZ

**HTML debug страница** (опционально, offline mini-dashboard):
- Путь: `result_store/.../core_face_landmarks/_render/render.html`
- **Offline** (без CDN/интернета), содержит:
  - навигацию по секциям (Overview / QA / Previews / Table),
  - Key facts (версии, coverage, тайминги),
  - QA peaks: top/anti-top (и jump-to к строкам),
  - таблицу кадров с search/sort + фильтр по `face_area_norm`,
  - галерею примеров с overlays (если доступен `frames_dir`).

**Assets** (если доступны `frames_dir` и включён HTML render):
- `_render/assets/*.jpg` — downscaled кадры с bbox + landmarks overlay (dev-only; privacy-sensitive).

**Конфигурация** (в `global_config.yaml`):
```yaml
core_face_landmarks:
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML debug страницу
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).

### Как проверить качество выхода компонента

#### 1. Human-friendly визуализация (Legacy)

Для визуальной проверки качества landmarks можно использовать скрипт:

```bash
python3 DataProcessor/VisualProcessor/core/model_process/core_face_landmarks/quality_report/demo_core_face_landmarks_quality.py \
  --frames-dir /path/to/frames_dir \
  --rs-path /path/to/result_store \
  --out-dir /path/to/output \
  --max-frames 20
```

**Что визуализирует скрипт**:
- Кадры с нарисованными landmarks лиц (468 точек, ключевые точки выделены зеленым)
- Кадры с нарисованными landmarks позы (33 точки с соединениями, если включено `--use-pose`)
- Кадры с нарисованными landmarks рук (21 точка на руку с соединениями, если включено `--use-hands`)
- Статистика: количество кадров с лицами, среднее количество лиц на кадр, статистика по pose и hands

**Что проверять визуально**:
- ✅ Корректность landmarks (точки соответствуют лицу на изображении)
- ✅ Корректность person-mask фильтрации (FaceMesh запускается только на кадрах с person)
- ❌ False positives (ложные детекции лиц)
- ❌ False negatives (пропущенные лица)

#### 2. Статистическая валидация

**Ожидаемые диапазоны значений** (для типичных видео):

- **Количество кадров с лицами**: зависит от типа видео (0-100% кадров)
- **Количество лиц на кадр**: 0-5 (обычно 1-2)
- **Face present rate**: процент кадров с хотя бы одним лицом

**Проверка разумности**:
- Отсутствие аномальных значений (NaN где не ожидается)
- `frame_indices` отсортированы и уникальны
- `times_s` соответствует `union_timestamps_sec[frame_indices]`
- `face_present` корректно отражает наличие лиц

#### 3. Интеграция с downstream модулями

Компонент используется следующими downstream компонентами:
- `shot_quality`: использует `face_present` и `face_landmarks` для оценки качества кадра
- `core_face_identity`: использует `face_landmarks` для извлечения face crops и идентификации лиц
- `detalize_face`: использует landmarks для детального анализа лиц

**Проверка**: Убедитесь, что downstream компоненты корректно читают артефакты и `frame_indices` выровнены с shared sampling group.

### Sampling requirements (фиксируем требования компонента)

Компонент работает по baseline‑политике **person-mask**:

- Segmenter обязан передать `core_face_landmarks.frame_indices` (union-domain). Эти `frame_indices` определяют форму выходных массивов NPZ.
- `core_face_landmarks` **требует**, чтобы `core_object_detections.detections.npz.frame_indices` был **в точности равен** `core_face_landmarks.frame_indices` (no-fallback).
- FaceMesh запускается **только** на тех позициях primary списка, где `core_object_detections` имеет `valid_mask=True` и `class_id == person`.
- Опционально можно расширить выборку соседними позициями: `--person-window-radius` (по умолчанию 0, strict).

Важно: артефакт остаётся **выровненным по primary `frame_indices`**:
- для кадров, где FaceMesh не запускался, `face_landmarks` остаются NaN, `face_present=False`.

Итоговое требование к sampling:
- Primary `frame_indices` должны быть достаточно равномерными и покрывать видео (для downstream модулей вроде `shot_quality`).
- При этом стоимость FaceMesh контролируется person-mask логикой (не запускаем без `person`).

Важно:
- Segmenter — единственный владелец sampling.
- **DEFERRED** только синтез глобальной `SamplingPolicy` в Segmenter по всем требованиям компонентов.

### Требования к разрешению (фиксируем требования компонента)

Рекомендуемые границы качества (будут финализированы после аудита всех компонентов):
- **min shorter side**: 256 px
- **target shorter side**: 320–480 px
- **max useful**: ~720 px
 - **апскейл запрещён** (только downscale)
---

## Навигация

[VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
