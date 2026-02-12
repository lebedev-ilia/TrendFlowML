## Component: `core_clip` (Tier‑0 baseline)

### Назначение

`core_clip` вычисляет CLIP эмбеддинги для выборки кадров (union-domain) и сохраняет их в NPZ.
Дополнительно сохраняет **text embeddings** для фиксированных prompt-наборов, чтобы downstream компоненты могли делать zero‑shot scoring **без загрузки CLIP весов** (single source-of-truth, no-network).

### Входы

- **Кадры**: через `FrameManager.get(idx)` из `frames_dir` (RGB uint8).
- **Sampling (строго)**: из `frames_dir/metadata.json`:

```json
{
  "core_clip": { "frame_indices": [0, 10, 20] }
}
```

**No-fallback**: если `core_clip.frame_indices` отсутствует или пустой — компонент **падает** (empty недопустим).

- **Time axis (строго)**: `union_timestamps_sec` в `frames_dir/metadata.json` — source-of-truth. `core_clip` сохраняет `times_s` для sampled кадров.

### Выходы

Путь: `result_store/<platform_id>/<video_id>/<run_id>/core_clip/embeddings.npz`

Ключи:
- `ARTIFACT_FILENAME = "embeddings.npz"` — фиксированное имя артефакта (source‑of‑truth = один NPZ)
- `frame_indices (N,) int32` — union-domain
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]`
- `frame_embeddings (N, D) float32`
- `shot_quality_prompts (P,) object` + `shot_quality_text_embeddings (P, D) float32`
- `scene_aesthetic_prompts (Pa,) object` + `scene_aesthetic_text_embeddings (Pa, D) float32`
- `scene_luxury_prompts (Pl,) object` + `scene_luxury_text_embeddings (Pl, D) float32`
- `scene_atmosphere_prompts (Pt,) object` + `scene_atmosphere_text_embeddings (Pt, D) float32`
- `cut_detection_transition_prompts (Pc,) object` + `cut_detection_transition_text_embeddings (Pc, D) float32`
- `popularity_topic_prompts (Pp,) object` + `popularity_topic_text_embeddings (Pp, D) float32`
- `meta` (dict, object-array) — canonical meta

### Meta (обязательное)

`meta` содержит:
- run identity: `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- `producer`, `producer_version`, `schema_version`, `created_at`
- `status="ok"`, `empty_reason=null`
- `models_used[]` + `model_signature`
- `batch_size` (контролируется верхним scheduler/DynamicBatching; auto внутри компонента запрещён)
- `prompts_version` — версия prompt‑наборов (для воспроизводимости)
- `stage_timings_ms` — словарь `{stage_name: duration_ms}` с таймингами ключевых стадий

Ключевые стадии (пример):
- `initialization` — загрузка `metadata.json`, валидация `frame_indices`, подготовка `FrameManager`
- `model_init` / `triton_init` — инициализация CLIP/Triton клиентов
- `image_embeddings_total` + подэтапы: `image_frame_loading`, `image_preprocessing`, `image_inference`
- `text_embeddings_prep`, `text_inference`, `text_embeddings_postproc`
- `saving` — формирование `meta` и атомарная запись NPZ
- `total` — общее время работы компонента

**Логирование таймингов**:
- После завершения обработки компонент логирует тайминги всех стадий в консоль:
  ```
  core_clip | stage timings (ms): image_embeddings_total=4488.0, image_frame_loading=1.4, image_inference=4101.7, image_preprocessing=383.5, initialization=0.9, text_embeddings_prep=0.5, text_inference=0.0, triton_init=5.2
  ```
- Тайминги также сохраняются в `meta.stage_timings_ms` в NPZ артефакте для последующего анализа
- Для режима `inprocess` дополнительно выводится таблица с процентным распределением времени по стадиям

### Runtime modes

#### `runtime=inprocess`

- модель и препроцессинг берутся из `openai/CLIP` (python package `clip`).

#### `runtime=triton`

В triton режиме **и image, и text эмбеддинги считаются через Triton** (no local inference).

`resolved_model_mapping` должен задать (пример):

```yaml
resolved_model_mapping:
  core_clip:
    runtime: triton
    triton_http_url: "http://triton:8000"

    triton_image_model_name: "clip_image"
    triton_image_model_version: "1"
    triton_image_input_name: "INPUT__0"
    triton_image_output_name: "OUTPUT__0"
    # Most baseline Triton deployments expose `clip_image_*` as an ensemble that expects UINT8 NHWC
    # (preprocess happens inside Triton). If you call the ONNX submodel directly, it may expect FP32 NCHW.
    triton_image_datatype: "UINT8"

    triton_text_model_name: "clip_text"
    triton_text_model_version: "1"
    triton_text_input_name: "INPUT__0"
    triton_text_output_name: "OUTPUT__0"
    triton_text_datatype: "INT64"

    # 2–3 стандартных варианта под разные input size
    triton_preprocess_preset: "openai_clip_224"  # openai_clip_224 | openai_clip_336 | openai_clip_448

    # batch_size задаётся строго (верхний scheduler; пока вручную в профиле/конфиге)
    batch_size: 16
```

Примечание:
- preprocessing для image embeddings делается локально до `(B,3,S,S) float32` (если Triton модель ожидает FP32) или `(B,S,S,3) uint8` (если Triton ensemble включает preprocess и ожидает UINT8 NHWC).
- tokenization для text embeddings делается локально (`clip.tokenize`), далее токены отправляются в Triton.

**Важно про батчинг (baseline GPU)**:
- Внутри `core_clip` батчинг всегда задаётся `--batch-size` (scheduler-controlled) и не “зажимается” автоматически.
- Для unit-cost тестов выставляем `batch_size=1`.
- Для production/DynamicBatching модели Triton должны быть batch-enabled (`max_batch_size > 0`), чтобы принимать входы вида `[B,...]`.

### Device/runtime semantics (фиксируем)

- `models_used[].runtime`:
  - `inprocess` — локальный inference внутри процесса
  - `triton-gpu` — inference через Triton (в нашем продакшен-контексте предполагается GPU)
- `models_used[].device`:
  - `inprocess`: `cpu|cuda`
  - `triton-gpu`: `cuda`

### Progress / state events

- Компонент пишет прогресс в `state_events.jsonl` (per‑run папка `runs/state/<platform>/<video>/<run>/state_events.jsonl`).
- Стадии:
  - `start` → `load_deps` → `process_frames` → `post_process` → `save` → `done`
- Для стадии `process_frames` отправляется гранулярный прогресс:
  - поле `progress` в диапазоне \([0,1]\)
  - `done` / `total` соответствуют количеству обработанных кадров (`frame_indices`)

### Фичи (выход) — группы и оценки

- **`frame_embeddings`**:
  - **алгоритм**: CLIP image encoder на sampled кадрах
  - **оценка реализации**: 9/10
  - **полезность**: 10/10 (базовый универсальный визуальный сигнал, используется многими downstream/головами)
- **`shot_quality_*` (prompts + text embeddings)**:
  - **алгоритм**: CLIP text encoder для фиксированного набора prompts
  - **оценка реализации**: 8/10
  - **полезность**: 7/10 (служебно для `shot_quality`, важно для воспроизводимости/ускорения)

### Sampling requirements (фиксируем требования компонента)

`core_clip` используется downstream несколькими компонентами (например `shot_quality`), поэтому выборка должна быть “универсальной по качеству”:
- **coverage**: обязательно покрывать начало/середину/конец и быть равномерной по времени;
- **cap**: для длинных видео иметь ограничение по числу кадров (чтобы не взрывать стоимость);
- **стабильность**: индексы должны быть отсортированы, уникальны, валидны для union-domain.

Важно:
- Segmenter — единственный владелец sampling.
- **DEFERRED** только синтез глобальной `SamplingPolicy` в Segmenter по всем требованиям компонентов.
  Но сами требования выше считаются обязательной частью контракта `core_clip`.

### Sampling policy (baseline, Segmenter-owned)

Ниже — **рекомендованная политика выборки** для primary visual sampling group (включая `core_clip`), чтобы качество было стабильным на роликах от коротких до длинных, без взрыва стоимости.

Требование: значения должны быть **непрерывными** (без скачков) и зависеть от `duration_s`.

Реализация (Segmenter):
- `target_gap_sec = f(duration_s)` — непрерывная монотонная кривая, построенная через log‑log интерполяцию по anchor‑точкам.
- `budget_n = round(duration_s / target_gap_sec)` (и затем `N = min(requested_max, budget_n)`).

Ориентиры по кривой (приблизительно):
- **≈ 5 минут**: `target_gap_sec ≈ 1s`
- **≈ 10 минут**: `target_gap_sec ≈ 2s`
- **≈ 20 минут**: `target_gap_sec ≈ 3–4s` (целимся около **3.5s**)

Примечание:
- Это **cap/budget** для primary group: итоговый `N` = `min(requested_max, budget_n)`.
- Downstream компоненты могут брать подмножество `core_clip.frame_indices`, но `core_clip` должен покрывать максимум требований по группе.

## Parallelization

- **Внутренний**: обрабатывает sampled кадры батчами размера `batch_size` (в режиме `inprocess`; в режиме `triton` зависит от batch-способностей Triton моделей).
- **Внешний**: компонент безопасно параллелить по разным видео/`run_id` (per-run storage).

### Batch Processing (Stage 2)

Компонент поддерживает **batch processing** для одновременной обработки нескольких видео с оптимизацией:

- **Гибридный батчинг**: сбор кадров из всех видео → группировка в батчи → batch inference через CLIP (inprocess или Triton) → распределение результатов обратно по видео
- **Оптимизации производительности**:
  - **Кеширование text embeddings**: text embeddings вычисляются один раз для всех видео и кешируются на диск (значительное ускорение при повторных запусках)
  - **Оптимизация разделения text embeddings**: предвычисление индексов для более эффективного slicing вместо повторных вычислений
  - **Освобождение памяти модели**: явное освобождение памяти GPU после обработки image embeddings и text embeddings
  - **Переиспользование модели**: модель загружается один раз и используется для всех батчей

**Использование**:
- Batch processing автоматически активируется при обработке нескольких видео через `--video-input-dir` или `--video-input-list`
- Контролируется через `visual.batch_processing.enable_gpu_batching` в `global_config.yaml`
- Лимит размера батча: `visual.batch_processing.max_frames_per_gpu_batch`

**Ожидаемое ускорение**:
- Для batch processing: **3-10x** (за счет кеширования text embeddings и лучшего использования GPU)
- Для single video: **1.2-1.5x** (за счет кеширования text embeddings)

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными, изменилась только производительность.

## Performance characteristics

### Оптимизации производительности

Компонент включает следующие оптимизации для batch processing:

1. **Кеширование text embeddings**: text embeddings вычисляются один раз для всех видео и кешируются на диск (значительное ускорение при повторных запусках)
2. **Оптимизация разделения text embeddings**: предвычисление индексов для более эффективного slicing вместо повторных вычислений
3. **Освобождение памяти модели**: явное освобождение памяти GPU после обработки image embeddings и text embeddings
4. **Переиспользование модели**: модель загружается один раз и используется для всех батчей

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): **3-10x** (за счет кеширования text embeddings и лучшего использования GPU)
- Для single video: **1.2-1.5x** (за счет кеширования text embeddings)

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными, изменилась только производительность.

### Измерения производительности

Format: `mean_value (outlier1, outlier2, ...)` - mean excludes outliers, outliers shown in parentheses.

---

| Model | Frame Shape | Triton model 2 |
|----------|-------------|------------|
| ViT-B/32 vunknown | 1920x1080 | clip_text |

| Triton model 1 | Triton Preprocess | Triton Batch | Frames cnt | Runs | Duration (s) | Image Inf (s) | Text Inf (s) | Peak CPU % | Peak GPU % | Triton Delta RAM (MB) | Triton Delta VRAM (MB) | Component Delta VRAM (MB) | Component Delta RAM (MB) | Summary Delta RAM | Summary Delta VRAM |
|------|-------------|------------|---------------|------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| clip_image_224 | preprocess_clip_image_224 | 1 | 1 | 3 | 47 | 0.650 | 34 | 100 | 4 | 1636 | 685 | 0 | 642 | 2278 | 685 |
| clip_image_224 | preprocess_clip_image_224 | 1 | 2 | 3 | 46 | 0.655 | 34 | 100 | 1 | 1692 | 679 | 5 | 413 | 2105 | 685 |
| clip_image_224 | preprocess_clip_image_224 | 1 | 5 | 3 | 46 | 0.813 | 34 | 100 | 4 | 1690 | 688 | 0 | 334 | 2025 | 687 |
| clip_image_224 | preprocess_clip_image_224 | 1 | 50 | 3 | 51 | 3.000 | 34 | 99 | 7 | 1713 | 678 | 22 | 162 | 1876 | 700 |
| clip_image_224 | preprocess_clip_image_224 | 1 | 131 | 3 | 57 | 7.046 | 34 | 100 | 6 | 1748 | 664 | 5 | 611 | 2359 | 670 |
| clip_image_224 | preprocess_clip_image_224 | 1 | 304 | 3 | 73 | 15.595 | 33 | 99 | 7 | 1522 | 674 | 10 | 472 | 1994 | 684 |
| clip_image_224 | preprocess_clip_image_224 | 2 | 1 | 3 | 46 | 0.627 | 34 | 100 | 1 | 1832 | 680 | 4 | 166 | 1998 | 684 |
| clip_image_224 | preprocess_clip_image_224 | 2 | 2 | 3 | 47 | 0.637 | 34 | 99 | 13 | 1547 | 680 | 8 | 446 | 1993 | 689 |
| clip_image_224 | preprocess_clip_image_224 | 2 | 131 | 3 | 57 | 6.188 | 34 | 100 | 6 | 1588 | 682 | 7 | 704 | 2292 | 689 |
| clip_image_224 | preprocess_clip_image_224 | 4 | 131 | 3 | 57 | 5.642 | 34 | 100 | 5 | 1631 | 678 | 1 | 712 | 2343 | 680 |
| clip_image_224 | preprocess_clip_image_224 | 16 | 131 | 3 | 56 | 5.189 | 34 | 100 | 5 | 1379 | 664 | 4 | 660 | 2039 | 668 |
| clip_image_224 | preprocess_clip_image_224 | 16 | 304 | 3 | 66 | 11.456 | 33 | 100 | 5 | 1674 | 678 | 7 | 547 | 2237 | 685 |
| clip_image_336 | preprocess_clip_image_336 | 1 | 131 | 3 | 64 | 12 | 35 | 100 | 7 | 1739 | 682 | 3 | 668 | 2408 | 685 |
| clip_image_336 | preprocess_clip_image_336 | 16 | 1 | 3 | 47 | 0.661 | 34 | 100 | 2 | 1599 | 685 | 3 | 741 | 2340 | 689 |
| clip_image_336 | preprocess_clip_image_336 | 16 | 131 | 3 | 66 | 10.738 | 37 | 100 | 9 | 1740 | 680 | 10 | 668 | 2408 | 690 |
| clip_image_336 | preprocess_clip_image_336 | 16 | 304 | 3 | 79 | 23.769 | 33 | 100 | 12 | 1703 | 676 | 7 | 455 | 2158 | 684 |
| clip_image_448 | preprocess_clip_image_448 | 1 | 1 | 3 | 47 | 0.681 | 35 | 99 | 3 | 1793 | 681 | 6 | 452 | 2246 | 688 |
| clip_image_448 | preprocess_clip_image_448 | 1 | 131 | 3 | 71 | 20.000 | 32 | 100 | 8 | 1646 | 678 | 6 | 440 | 2086 | 685 |
| clip_image_448 | preprocess_clip_image_448 | 1 | 304 | 3 | 106 | 47.396 | 33 | 100 | 8 | 1617 | 677 | 5 | 503 | 2121 | 682 |
| clip_image_448 | preprocess_clip_image_448 | 16 | 1 | 3 | 47 | 0.651 | 35 | 100 | 20 | 1856 | 679 | 12 | 276 | 2132 | 691 |
| clip_image_448 | preprocess_clip_image_448 | 16 | 8 | 3 | 47 | 1.674 | 33 | 100 | 8 | 1521 | 683 | 4 | 486 | 2007 | 687 |
| clip_image_448 | preprocess_clip_image_448 | 16 | 64 | 3 | 58 | 9.236 | 34 | 100 | 18 | 1600 | 681 | 516 | 491 |  2092 | 1198 |
| clip_image_448 | preprocess_clip_image_448 | 16 | 131 | 3 | 70 | 18.341 | 34 | 100 | 18 | 1726 | 679 | 516 | 444 | 2170 | 1195 |
| clip_image_448 | preprocess_clip_image_448 | 16 | 304 | 3 | 99 | 42.070 | 34 | 100 | 20 | 1739 | 683 | 530 | 445 | 2185 | 1213 |

---

Threads: 2

| Triton model 1 | Triton Preprocess | Triton Batch | Frames cnt | Runs | Duration (s) | Image Inf (s) | Text Inf (s) | Peak CPU % | Peak GPU % | Triton Delta RAM (MB) | Triton Delta VRAM (MB) | Component Delta VRAM (MB) | Component Delta RAM (MB) | Summary Delta RAM | Summary Delta VRAM |
|------|-------------|------------|---------------|------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| clip_image_224 | preprocess_clip_image_224 | 1 | 1 | 3 | 74 | 1.322 | 60 | 100 | 3 | 1706 | 678 | 9 | 1502 | 3208 | 688 |
| clip_image_224 | preprocess_clip_image_224 | 1 | 50 | 3 | 79 | 3.023 | 61 | 100 | 12 | 1572 | 680 | 7 | 1896 | 3468 | 687 |
| clip_image_224 | preprocess_clip_image_224 | 1 | 304 | 3 | 105 | 16.092 | 63 | 100 | 16 | 1521 | 687 | 17 | 1803 | 3325 | 705 |

---

| Model | Frame Shape | Triton model 2 |
|----------|-------------|------------|
| ViT-B/32 vunknown | 1024x576 | clip_text |

| Triton model 1 | Triton Preprocess | Triton Batch | Frames cnt | Runs | Duration (s) | Image Inf (s) | Text Inf (s) | Peak CPU % | Peak GPU % | Triton Delta RAM (MB) | Triton Delta VRAM (MB) | Component Delta VRAM (MB) | Component Delta RAM (MB) | Summary Delta RAM | Summary Delta VRAM |
|------|-------------|------------|---------------|------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| clip_image_224 | preprocess_clip_image_224 | 1 | 131 | 3 | 53 | 6.886 | 33 | 100 | 9 | 1618 | 677 | 10 | 368 | 1987 | 688 |

## Models

### GPU (baseline)
- CLIP image encoder (Triton): `clip_image_224|336|448`.
- CLIP text encoder (Triton): `clip_text`.

### CPU (debug / fallback dev only)
- OpenAI CLIP via python package `clip` (использовать только если компонент явно настроен на `runtime=inprocess`).

## Quality validation & human-friendly inspection

Рекомендуемые проверки:
- **Cosine similarity sanity**: эмбеддинги L2-нормированы, диагональ similarity матрицы ≈ 1.0.
- **t-SNE/UMAP** на `frame_embeddings` для нескольких видео (ручная проверка кластеризации по сценам).
- Проверка консистентности: `times_s` соответствует `union_timestamps_sec[frame_indices]`.

### Human-friendly визуализация (Render System)

`core_clip` генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/core_clip/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по эмбеддингам (frames_count, embedding_dim, norm_mean, norm_std, cosine_similarity_mean)
- **Timeline**: данные по каждому кадру (frame_index, time_sec, embedding_norm, cosine_similarity к предыдущему кадру)
- **Distributions**: распределения embedding norms и cosine similarity (min, max, mean, std, median, percentiles)
- **Text embeddings info**: информация о text embeddings для всех prompt-наборов (count, dim, norm_mean, norm_std)

Render-context может быть использован:
- **LLM** для генерации текстовых описаний видео
- **Frontend** для построения графиков и визуализаций (timeline charts, distributions)
- **Debugging**: быстрая проверка качества эмбеддингов без загрузки NPZ

**HTML debug страница** (опционально):
- Путь: `result_store/.../core_clip/_render/render.html`
- Содержит интерактивные графики (Chart.js):
  - Timeline: embedding norm и cosine similarity по времени
  - Distributions: статистики по embedding norms
  - Summary metrics: ключевые показатели в удобном формате

**Конфигурация** (в `global_config.yaml`):
```yaml
core_clip:
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML debug страницу
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).

Legacy demo (deprecated):
- `scripts/baseline/demo_core_clip_quality.py` — генерирует HTML с timeline, thumbnails, consecutive cosine similarity, PCA scatter и (опционально) prompt scoring.


