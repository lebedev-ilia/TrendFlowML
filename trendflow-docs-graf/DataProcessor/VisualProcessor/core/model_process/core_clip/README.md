## Component: `core_clip` (Tier‑0 baseline)

**Контракт NPZ, melt/QA, валидатор:** [docs/FEATURE_DESCRIPTION.md](docs/FEATURE_DESCRIPTION.md) · `utils/validate_core_clip_npz.py` (`--struct`, `--qa`, `--ranges`, батч `--results-base` / `**/core_clip/embeddings.npz`).

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
- `places365_prompts (P365,) object` + `places365_text_embeddings (P365, D) float32` — Places365 zero-shot label embeddings (365 prompts)
- **Backend-friendly proxies (Audit v3, schema v2)**:
  - `consecutive_cosine_prev (N,) float32` — cosine similarity между соседними кадрами (первый = NaN)
  - `shot_quality_scores (N,10) float32`
  - `scene_aesthetic_scores (N,6) float32`
  - `scene_luxury_scores (N,6) float32`
  - `scene_atmosphere_scores (N,6) float32`
  - `cut_detection_transition_scores (N,10) float32`
  - `popularity_topic_scores (N,10) float32` (**analytics-only**, см. ниже как подтверждаем пользу)
  - `places365_topk_indices (N,K) int32` + `places365_topk_scores (N,K) float32`
  - `places365_video_topk_indices (K,) int32` + `places365_video_topk_scores (K,) float32`
- `meta` (dict, object-array) — canonical meta

Примечание (Audit v3, breaking change):
- В `core_clip_npz_v2` **убраны legacy top-level scalar keys** (`version`, `created_at`, `model_name`, `total_frames`).
  Используем только `meta.*` для этих полей.

### Meta (обязательное)

`meta` содержит:
- run identity: `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- `producer`, `producer_version`, `schema_version`, `created_at`
- `status="ok"`, `empty_reason=null`
- `models_used[]` + `model_signature`
- `batch_size` (контролируется верхним scheduler/DynamicBatching; auto внутри компонента запрещён)
- `prompts_version` — версия prompt‑наборов (для воспроизводимости, текущая версия: `v3_2026-01-16`)
- `backend_proxy_version="core_clip_backend_proxy_v1"`
- `export_prompt_scores` (bool) + `places365_topk_k` (int)
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

**Audit v3 (FINAL)**: Triton конфигурируется **только через ModelManager specs** (no legacy triton args).

`resolved_model_mapping` должен задать (пример):

```yaml
resolved_model_mapping:
  core_clip:
    runtime: triton
    triton_image_model_spec: "clip_image_224_triton"   # dp_models spec name
    triton_text_model_spec: "clip_text_triton"         # dp_models spec name

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

Ниже — **требование** к политике выборки для primary visual sampling group (включая `core_clip`).
Segmenter будет доводиться в конце аудита, но это правило фиксируем уже сейчас.

**FINAL (Audit v3, draft for Segmenter)**:

- Если `duration_sec <= 120` (≤ 2 минуты):
  - берём **каждый 7-й кадр** на analysis timeline (т.е. stride по кадрам = 7).
- Если `duration_sec > 120`:
  - используем **непрерывную** кривую budgets вплоть до 20 минут,
  - на `duration_sec = 1200` (20 минут) **cap = 3000 кадров**.

Техническая форма (пример, можно менять в Segmenter при синтезе глобальной policy):

- `N_target(T) = round(3000 * (min(T,1200) / 1200) ^ alpha)` для `T>120`, где `alpha≈0.78`
  - это даёт около ~500 кадров на 2 минуты и ~3000 на 20 минут,
  - и уменьшает fps-давление на длинных видео.

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

### Кеширование text embeddings

Компонент поддерживает **кеширование text embeddings** для ускорения повторных запусков:

- **Автоматическое кеширование**: text embeddings вычисляются один раз и сохраняются на диск
- **Ключ кеша**: основан на prompts (в порядке), model name/version, prompts version и model size
- **Структура кеша**: `{DP_MODELS_ROOT}/cache/core_clip_text_embeddings/{model_size}/{cache_key}.npz`
- **Версионирование**: кеш организован по model size (224/336/448) для разных размеров моделей
- **Отключение**: можно отключить через `--disable-text-cache` (для benchmarks)

**Примечание**: Text embeddings одинаковы для всех размеров CLIP моделей (224/336/448), но кеш версионируется по model size для ясности и потенциальных будущих оптимизаций.

## Performance characteristics

### Оптимизации производительности

Компонент включает следующие оптимизации для batch processing:

1. **Кеширование text embeddings**: text embeddings вычисляются один раз и кешируются на диск (значительное ускорение при повторных запусках, работает и для single video)
2. **Оптимизация разделения text embeddings**: предвычисление индексов для более эффективного slicing вместо повторных вычислений
3. **Освобождение памяти модели**: явное освобождение памяти GPU после обработки image embeddings и text embeddings
4. **Переиспользование модели**: модель загружается один раз и используется для всех батчей

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): **3-10x** (за счет кеширования text embeddings и лучшего использования GPU)
- Для single video: **1.2-1.5x** (за счет кеширования text embeddings при повторных запусках)

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

### Places365 Prompts

Компонент включает **Places365 zero-shot label embeddings** (365 prompts) для downstream компонентов:

- **Источник**: `dp_models/bundled_models/visual/places365/categories_places365.txt`
  - Prod: требуется `DP_MODELS_ROOT` (строго)
  - Dev: auto-detect разрешён только при `DP_ALLOW_BUNDLE_AUTODETECT=1` или флаге `--allow-bundle-autodetect`
- **Формат**: каждый prompt преобразуется в формат `"a photo of a {category}"` (замена `_` и `/` на пробелы)
- **Выход**: `places365_prompts` (365 prompts) + `places365_text_embeddings` (365, D) в NPZ артефакте
- **Использование**: downstream компоненты могут использовать эти embeddings для zero-shot scene classification без загрузки CLIP весов

## Output classification (Audit v3)

- **model_facing**:
  - `frame_embeddings`, `frame_indices`, `times_s`
- **module-facing (service layer)**:
  - `*_prompts`, `*_text_embeddings` (чтобы модули работали “core_clip-only” без загрузки CLIP)
- **analytics / backend-proxy**:
  - `consecutive_cosine_prev`, `*_scores`, `places365_*topk*`
- **debug-only**:
  - HTML рендер (dev), любые экспериментальные поля вне схемы

## Popularity topic prompts (analytics-only) — как подтверждаем пользу

`popularity_topic_scores` сейчас считаем **analytics-only**. Подтверждаем пользу так:

- **Ablation на baseline**: обучаем baseline (CatBoost/LightGBM) с/без `popularity_topic_*` агрегатов → смотрим прирост Spearman на holdout/regression_mini.
- **Stability check**: корреляции по разным под-доменам (gaming/travel/food/…) и по age-buckets; фича не должна “ломаться” на отдельных типах контента.
- **Human sanity**: топ-темы по видео должны быть семантически правдоподобны (через рендер/сайт интерпретации).

## Backend contract (без передачи raw embeddings)

На сайт **не отправляем** `frame_embeddings` и `*_text_embeddings`. Вместо этого backend/сайт использует NPZ и отдаёт пользователю:

- динамику `consecutive_cosine_prev` (насколько видео “меняется” по времени),
- интерпретируемые скора `shot_quality_scores / scene_*_scores` (как прокси),
- `places365_video_topk_*` и per-frame `places365_topk_*` (топ сцен/мест),
- сами `*_prompts` (как справочник интерпретации).

## Quality validation & human-friendly inspection

Рекомендуемые проверки:
- **Cosine similarity sanity**: эмбеддинги L2-нормированы, диагональ similarity матрицы ≈ 1.0.
- **t-SNE/UMAP** на `frame_embeddings` для нескольких видео (ручная проверка кластеризации по сценам).
- Проверка консистентности: `times_s` соответствует `union_timestamps_sec[frame_indices]`.

### Render (dev-only): как читать результаты `core_clip` человеку

Цель рендера `core_clip`: дать человеку (не ML-инженеру) быстрый ответ на вопросы:

- **Видео “разнообразное” или “однообразное” по картинке?**
- **Где по таймлайну происходят сильные изменения сцены/плана?**
- **Есть ли явные проблемы качества данных** (битые кадры, странные значения, рассинхрон sampling)?

#### Где лежат файлы рендера

- **Render-context JSON** (машиночитаемый, source для HTML/фронта/LLM):  
  `result_store/<platform_id>/<video_id>/<run_id>/core_clip/_render/render_context.json`
- **HTML** (человекочитаемый, dev-only):  
  `result_store/<platform_id>/<video_id>/<run_id>/core_clip/_render/render.html`

#### Что показывает текущий HTML (MVP сейчас)

- **Timeline: Embedding Norm** — “масштаб” эмбеддинга кадра (скаляр).
  - Это **не** “качество кадра”, а скорее sanity check: значения должны быть стабильными, без “иголок”.
- **Timeline: Cosine similarity (prev)** — насколько соседние sampled кадры похожи визуально.
  - **Ближе к 1.0** → кадры почти одинаковые (статичная сцена / мало движения / малый gap).
  - **Ниже** → кадры различаются (смена сцены, резкий монтаж, крупное движение).
- **Summary + Distributions** — базовые статистики по этим двум кривым.

#### Как интерпретировать типовые ситуации

- **cosine_similarity почти всегда ~1.0**:
  - видео статичное, или sampling слишком “частый”, или кадры почти одинаковые (например, подкаст).
- **частые провалы cosine_similarity**:
  - много монтажных склеек/смен сцен; нормальная ситуация для динамичных роликов.
- **embedding_norm с редкими огромными выбросами**:
  - подозрение на битые кадры/экстремальные артефакты/неверный цветовой порядок (RGB↔BGR),
    или ошибка preprocessing/модели (в triton режиме).

#### Что ДОЛЖНО появиться в персонализированном рендере (target для доработки `render.py`)

Чтобы человек понимал 90% логики без Python/ML, рендер `core_clip` должен включать:

- **Thumbnails sampled кадров** (минимум K=12 равномерно по видео) + их `time_s` и `frame_index`.
- **Топ “самых разных” кадров**:
  - кадры с минимальной cosine_similarity к предыдущему (показывают “переходы”).
- **Places365 интерпретация**:
  - top-K сцен по видео (`places365_video_topk_*`) + top-K по кадрам (`places365_topk_*`),
    чтобы человек видел “что это за видео” словами.
- **Prompt-score панели** (если `*_scores` включены):
  - например, `shot_quality_scores`: топ-3 “лучших” и “худших” кадров по score + мини-превью кадров.
- **PCA/UMAP scatter (debug-only)**:
  - 2D проекция `frame_embeddings` с подсветкой времени — помогает увидеть “кластеры сцен”.

#### Время выполнения (что смотреть)

- В `render.html`/`render_context.json` отображать `meta.stage_timings_ms`:
  - `image_inference` / `image_embeddings_total` — основной cost
  - `saving` — должно быть маленьким (обычно миллисекунды/десятки мс)

#### Какие параметры конфига сильнее всего меняют картину

- **Sampling**: количество и частота sampled кадров (`frame_indices`) — определяет детализацию таймлайна.
- **`core_clip.runtime`**: `triton` vs `inprocess` (качество одинаковое по контракту, но latency/infra разные).
- **`core_clip.triton_preprocess_preset`**: `openai_clip_224/336/448` — меняет входное разрешение модели и стоимость.
- **`core_clip.batch_size`**: влияет на throughput/VRAM, но **не** на значения.

#### Конфигурация render

```yaml
core_clip:
  render:
    enable_render: true       # render_context.json
    enable_html_render: true  # render.html (dev-only)
```

**Примечание**: render — best-effort и не должен ломать основной pipeline.

Legacy demo (deprecated, но полезно как “эталон чего хотим”):
- `scripts/baseline/demo_core_clip_quality.py` — HTML с timeline, thumbnails, consecutive cosine similarity, PCA scatter и (опционально) prompt scoring.
---

## Навигация

[VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
