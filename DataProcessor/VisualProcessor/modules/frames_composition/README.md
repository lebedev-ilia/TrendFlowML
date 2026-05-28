# `frames_composition`

Baseline-ready модуль VisualProcessor для извлечения **композиционных признаков** по кадрам (union-domain) и их агрегации на уровне видео.

- **Фичи и NPZ-контракт:** `docs/FEATURE_DESCRIPTION.md` (сверка с прогоном, QA, melt).
- **Валидатор артефакта:** `utils/validate_frames_composition.py` (`--struct --qa --ranges` или батч `--results-base`).

## Что делает

- Считает набор **классических** композиционных сигналов (якоря/баланс/симметрия/негативное пространство/сложность/ведущие линии) и базовые сигналы по лицам/объектам.
- Отдаёт:
  - **video-level фичи** в формате `feature_names[]` + `feature_values[]` (таблично-дружелюбно для ML / DatasetBuilder),
  - **per-frame фичи** (`frame_feature_names[]`, `frame_feature_values[N,D]`) для downstream (например, трансформер по кадрам),
  - `times_s` (строго из `union_timestamps_sec[frame_indices]`) для таймлайнов/UI.

## Контракты и политика (обязательно)

- **NPZ = source-of-truth**, JSON в `result_store` запрещён (кроме `manifest.json`).
- **No-fallback**:
  - `frame_indices` должны быть в `frames_dir/metadata.json["frames_composition"]["frame_indices"]` (Segmenter-owned).
  - Если `frame_indices` отсутствуют/пустые → **error**.
- **Time-axis**: `times_s = union_timestamps_sec[frame_indices]` (no-fallback).
- **Valid empty**:
  - Если во всём видео **нет лиц** → артефакт пишется со `status="empty"` и `empty_reason="no_faces_in_video"`.
  - При этом `frame_indices/times_s` всё равно присутствуют, а численные фичи не должны подменяться нулями (используем NaN там, где значения “нет”).

## Зависимости (core providers)

Компонент **не загружает ML-модели** напрямую и использует результаты core провайдеров:

- `core_object_detections` — **required** (может быть `status="empty"` внутри провайдера, это не ошибка для модуля).
- `core_face_landmarks` — **required**, но **валидная пустота** допустима (см. `no_faces_in_video`).
- `core_depth_midas` — **required** и должен быть `status="ok"` (по требованиям аудита).

Все зависимости должны быть выровнены по одинаковому `frame_indices` (строгая проверка, no-fallback).

## Выходной артефакт (NPZ)

Путь (per-run storage):

- `result_store/<platform_id>/<video_id>/<run_id>/frames_composition/frames_composition.npz`

`schema_version`: `frames_composition_npz_v1`

**producer_version**: `2.0.1`  
**Human schema**: `DataProcessor/VisualProcessor/modules/frames_composition/docs/SCHEMA.md`  
**Machine schema**: `DataProcessor/VisualProcessor/schemas/frames_composition_npz_v1.json`

Ключи внутри NPZ:

- `frame_indices`: `int32[N]`
- `times_s`: `float32[N]`
- `feature_names`: `object[str][F]`
- `feature_values`: `float32[F]`
- `frame_feature_names`: `object[str][D]`
- `frame_feature_values`: `float32[N,D]`
- `frame_feature_present_ratio`: `float32[D]` (доля finite по колонкам; помогает моделям интерпретировать NaN)
- `meta`: dict (object) — общий baseline meta контракт (producer/run identity/status/models_used/model_signature)

## CLI

Запускается оркестратором VisualProcessor как subprocess:

```bash
python DataProcessor/VisualProcessor/modules/frames_composition/main.py \
  --frames-dir <frames_dir> \
  --rs-path <run_rs_path> \
  --feature-set default \
  --num-workers 8
```

### Feature gating

Все фичи управляемы через аргументы:

- `--feature-set`: `default | ml | all`
- `--features`: CSV групп (перекрывает `--feature-set`)

Группы:
- `anchors`
- `balance`
- `symmetry`
- `negative_space`
- `complexity`
- `leading_lines`
- `depth`
- `objects`
- `faces`
- `style` (heuristics для UI explainability)

## Sampling / units-of-processing requirements (Visual)

**Единица обработки**: `frame` (union-domain).

Segmenter является единственным владельцем sampling; этот модуль **не генерирует** `frame_indices`.

Рекомендованная нелинейная кривая (Segmenter-owned, `type="ease_out_power"`):

- `k`: `0.6`
- `min_units`: `120`
- `max_units`: `900`
- `linear_until_sec`: `10`
- `cap_duration_sec`: `600`

Интерпретация:
- короткие видео получают близко к `min_units`,
- рост замедляется после `linear_until_sec`,
- длинные видео упираются в `max_units`.

Требования:
- `frame_indices` должны быть **sorted + unique** (int32).
- `analysis_width/analysis_height`: допускается downscale; апскейл не требуется.

## Parallelization

Модуль реализует **внутренний параллелизм** по кадрам (`--num-workers`), но чтение кадров через `FrameManager` защищено от гонок (safe access).

**Default для `--num-workers`**: автоматически определяется как `max(1, min(8, os.cpu_count() or 4))` (минимум 1, максимум 8, по умолчанию 4 если `os.cpu_count()` недоступен).

## Batch Processing

Модуль поддерживает **batch processing** для одновременной обработки нескольких видео:

- **Поддержка батчинга**: модуль реализует `supports_batch = True`
- **CPU-based**: модуль использует CPU для обработки (не требует GPU)
- **Изоляция**: каждый видео имеет свой `rs_path` для артефактов

## Progress / observability

В процессе обработки пишет progress-events в append-only файл:

- `state/<platform_id>/<video_id>/<run_id>/state_events.jsonl` (PR‑5)

Backend worker читает этот файл и пушит события в WS (`component.progress`).

## Render System

Модуль поддерживает генерацию render-context JSON и HTML debug страницы (аналогично другим модулям):

- **Render-context JSON**: `result_store/.../frames_composition/_render/render_context.json`
- **HTML debug страница**: `result_store/.../frames_composition/_render/render.html`

Render-context содержит:
- **Summary**: статистики по композиционным фичам
- **Timeline**: данные по каждому кадру (key features)
- **Distributions**: распределения композиционных метрик

HTML страница содержит **offline** графики (без CDN) для визуализации композиционных фич по времени.


