## Component: `core_depth_midas` (Tier‑0 baseline)

- **Контракт NPZ, melt/QA, примеры команд:** [docs/FEATURE_DESCRIPTION.md](docs/FEATURE_DESCRIPTION.md)
- **Валидатор:** `utils/validate_core_depth_midas_npz.py` (`--struct`, `--qa`, `--ranges` или батч `--results-base`)

### Назначение

`core_depth_midas` вычисляет depth maps на primary выборке кадров (union-domain) и сохраняет их в `depth.npz`.

Политика: **Triton-only** (локальные `torch.hub` / `engine=torch` запрещены).

### Входы

- **Кадры**: `FrameManager.get(idx)` из `frames_dir` (RGB uint8 по умолчанию).
  - Если `FrameManager` возвращает BGR изображения, используйте `--frames-bgr` для автоматического преобразования в RGB.
- **Sampling (строго)**: из `frames_dir/metadata.json`:

```json
{
  "core_depth_midas": { "frame_indices": [0, 10, 20] }
}
```

No-fallback:
- отсутствие/пустота `frame_indices` ⇒ **error**

### Runtime (Triton)

`core_depth_midas` вызывает Triton HTTP v2 и ожидает, что preprocessing живёт на стороне Triton (ensemble/модельный граф).

Контракт (baseline, рекомендуемый):
- **input**: `(B,S,S,3) uint8` — RGB, NHWC (клиент делает только resize до пресета S×S)
- **output**: `(B,h,w) float32` или `(B,1,h,w) float32` — depth logits
- далее клиент ресайзит depth до `out_height/out_width` (по умолчанию 384×384) и сохраняет в NPZ

Поддерживаем 2–3 пресета размера входа:
- `midas_256`
- `midas_384` (default)
- `midas_512`

### Batch size (scheduler-controlled)

`--batch-size` обязателен и задаётся верхним scheduler/DynamicBatching (пока вручную в конфиге/профиле).
Auto-batching внутри компонента запрещён.

### Models

GPU (Triton):
- **MiDaS** (depth)
  - **Triton**: ✅ да (`DataProcessor/triton/models/midas_{256,384,512}/`)
  - **Runtime**: `triton`
  - **Engine**: `onnxruntime_onnx` (через ensemble с Python-preprocess)
  - **Input contract**: `UINT8 NHWC RGB` на входе ensemble (`midas_*`)
  - **Presets / branches**: `256`, `384`, `512` (square input)

### Parallelization

- **Internal batching**: кадры отправляются в Triton батчами размера `--batch-size` (scheduler-controlled).
- **External parallelism**: допускается параллельный запуск на разных видео (разные `run_id` / разные `result_store` пути). Ограничение — VRAM.

### Batch Processing (Stage 3)

Компонент поддерживает **batch processing** для одновременной обработки нескольких видео с оптимизацией:

- **Гибридный батчинг**: сбор кадров из всех видео → группировка в батчи → batch inference через Triton → распределение результатов обратно по видео
- **Оптимизации производительности**:
  - **Векторизованное вычисление статистик**: mean и std вычисляются для всего батча одновременно через `np.mean()` и `np.std()` с `axis=1`
  - **Оптимизированная проверка isfinite**: предвычисление маски `np.isfinite()` для всего батча перед вычислением percentiles
  - **Векторизованное сохранение результатов**: использование индексации массивов вместо поэлементного присваивания
  - **Оптимизация post-processing**: уменьшение количества циклов и использование batch операций numpy

**Использование**:
- Batch processing автоматически активируется при обработке нескольких видео через `--video-input-dir` или `--video-input-list`
- Контролируется через `visual.batch_processing.enable_gpu_batching` в `global_config.yaml`
- Лимит размера батча: `visual.batch_processing.max_frames_per_gpu_batch`

**Ожидаемое ускорение**:
- Для batch processing: **2-5x** (за счет векторизации post-processing и лучшего использования GPU)
- Для single video: **1.2-1.5x** (за счет оптимизации вычислений статистик)

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными, изменилась только производительность.

### Выход

Путь: `result_store/<platform_id>/<video_id>/<run_id>/core_depth_midas/depth.npz`

Ключи:
- `ARTIFACT_FILENAME = "depth.npz"` — фиксированное имя артефакта (один NPZ per run)
- `frame_indices (N,) int32`
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]`
- `depth_maps (N, out_h, out_w) float32`
- `depth_maps_norm (N, out_h, out_w) float32` — robust per-frame normalization to `[0,1]` using `depth_p05/depth_p95`
- `depth_mean (N,) float32` — mean(depth_maps[i])
- `depth_std (N,) float32` — std(depth_maps[i])
- `depth_p05 (N,) float32` — 5-й перцентиль depth_maps[i] (по finite значениям)
- `depth_p95 (N,) float32` — 95-й перцентиль depth_maps[i] (по finite значениям)
- `depth_range_robust (N,) float32` — `depth_p95 - depth_p05`
- `depth_complexity_score (N,) float32` — proxy сложности сцены по depth (mean |grad| на `depth_maps_norm`)
- `foreground_background_separation_proxy (N,) float32` — proxy разделения FG/BG: `depth_range_robust / (depth_std + eps)`
- **Backend preview maps (Audit v3)**:
  - `preview_frame_indices (K,) int32` — subset из `frame_indices` (равномерно по времени)
  - `preview_times_s (K,) float32`
  - `preview_depth_maps (K, out_h, out_w) float32`
  - `preview_depth_maps_norm (K, out_h, out_w) float32`
- `meta` (dict, object-array)

Примечание (Audit v3, breaking change):
- В `core_depth_midas_npz_v2+` **убраны legacy top-level scalar keys** (`version`, `created_at`, `model_name`, `total_frames`).
  Эти поля допускаются только внутри `meta.*`.

### Backend contract (что отдаём наружу)

Audit v3 goal: backend может **рисовать несколько depth-карт** (например `K=10`) равномерно по видео + показывать timeline и простые прокси-скоры.

- Для карт используем `preview_*` (и предпочтительно `preview_depth_maps_norm` для визуализации).
- Для timeline используем `depth_mean/std/p05/p95` + `depth_complexity_score` + `foreground_background_separation_proxy`.

### Empty/error semantics

- Empty недопустим: если отсутствует depth-map хотя бы для одного кадра ⇒ **error**.

### Meta / models_used

- Если задан `--triton-model-spec`: используем `models_used[]` из ModelManager (это фиксирует identity spec’а + weights_digest).
- Иначе (legacy): `models_used[]` формируется из CLI аргументов компонента.
- `models_used[].device="cuda"`
- `weights_digest="unknown"` (baseline)
- `stage_timings_ms` — словарь `{stage_name: duration_ms}` с таймингами ключевых стадий (initialization, depth_inference_total, saving, total)

Стадии (пример):
- `initialization` — чтение `metadata.json`, валидация `frame_indices`, инициализация `FrameManager` и Triton‑клиента
- `depth_inference_total` — основной цикл по кадрам (инференс глубины + ресайз + агрегация статистик)
- `saving` — подготовка `meta` и атомарная запись NPZ
- `total` — общее время работы компонента

**Логирование таймингов**:
- После завершения обработки компонент логирует тайминги всех стадий в консоль:
  ```
  core_depth_midas | stage timings (ms): depth_inference_total=280.1, initialization=26.8, total=10507.4
  ```
- Тайминги также сохраняются в `meta.stage_timings_ms` в NPZ артефакте для последующего анализа

### Progress / state events

Компонент пишет progress‑события в `state_events.jsonl` (per‑run папка `runs/state/<platform>/<video>/<run>/state_events.jsonl`):

- Стадии:
  - `start` → `load_deps` → `process_frames` → `post_process` → `save` → `done`
- Для `process_frames` отправляется **гранулярный** прогресс по кадрам:
  - `progress ∈ [0,1]`, `done`, `total` (кол-во обработанных `frame_indices`)

### Sampling requirements (фиксируем требования компонента)

`core_depth_midas` входит в shared sampling group с `shot_quality` и другими core providers:
`core_clip`, `core_depth_midas`, `core_object_detections`, `core_face_landmarks` должны работать на **одном и том же** primary `frame_indices` (иначе downstream падает из-за mismatch).

### Quality validation & human-friendly inspection

Рекомендуемые проверки:
- **Depth range sanity**: depth maps должны быть finite, диапазон значений разумен для относительной глубины.
- **Temporal consistency**: depth_mean должен изменяться плавно между соседними кадрами (без резких скачков).
- Проверка консистентности: `times_s` соответствует `union_timestamps_sec[frame_indices]`.

### Render (dev-only): как читать глубину человеку

Цель рендера `core_depth_midas`: объяснить человеку (без ML) “что такое depth” и как по выходу понять,
есть ли ошибки/аномалии, и что означает динамика по времени.

#### Важная оговорка (что такое depth MiDaS)

MiDaS даёт **относительную** глубину (это **не метры**):

- “ближе”/“дальше” внутри одного кадра — осмысленно
- сравнивать абсолютные значения между разными видео нельзя

Поэтому в контракте есть нормализованные карты `depth_maps_norm` и robust шкала через `depth_p05/depth_p95`.

#### Где лежат файлы рендера

- `.../core_depth_midas/_render/render_context.json`
- `.../core_depth_midas/_render/render.html`

#### Что показывает текущий HTML (MVP сейчас)

- **Timeline**:
  - `depth_mean` — средняя “глубина” кадра (относительная)
  - `depth_std` / `depth_range_robust` — насколько глубина “разнообразная” (сложная сцена vs плоская)
  - `depth_complexity_score` — proxy сложности (градиенты на `depth_maps_norm`)
  - `foreground_background_separation_proxy` — proxy “есть ли сильное разделение FG/BG”
- **Distributions** по ключевым скалярам.

Это помогает понять “есть ли жизнь” в depth, и нет ли NaN/inf/скачков.

#### Что ДОЛЖНО появиться в персонализированном рендере (target)

Чтобы человек понял результат “глазами”, рендер обязан включать **картинки**:

- **Preview depth maps**: минимум K=10 кадров равномерно по видео:
  - исходный кадр (thumbnail)
  - depth overlay (например colormap “magma/viridis”) + легенда (0..1 для `depth_maps_norm`)
- **Топ-кадры по сложности**:
  - топ-3 по `depth_complexity_score` (и анти-топ-3) с превью
- **Гистограммы depth**:
  - per-frame histogram для 1–2 выбранных кадров (показывает, есть ли “две моды” FG/BG)

Важно: показывать именно `preview_depth_maps_norm` (0..1), чтобы визуализация была стабильной.

#### Как интерпретировать типовые ситуации

- **Depth карты “шумные”, пятна, рябь**:
  - часто означает плохой контраст/ночь/сильный compression или неподходящий preprocessing.
- **Depth почти константа на всех кадрах**:
  - возможно, ошибка модели/подачи (не тот preset, неверные каналы, проблемы Triton).
- **Резкие скачки `depth_mean` при плавном видео**:
  - подозрение на рассинхрон кадров, проблемы sampling или артефакты в кадрах.

#### Время выполнения (что смотреть)

- В `meta.stage_timings_ms`:
  - `depth_inference_total` (основная часть)
  - `saving` (должно быть небольшим)

#### Параметры конфига, которые меняют результат и стоимость

- **`triton_model_spec` / preset `midas_{256,384,512}`**:
  - качество ↑ с размером, стоимость ↑ сильно.
- **`out_width/out_height`**:
  - влияет на размер сохранённых карт и скорость post-processing.
- **`batch_size`**:
  - влияет на throughput/VRAM, но не на значения.

#### Конфигурация render

```yaml
core_depth_midas:
  render:
    enable_render: true
    enable_html_render: true
```

Legacy demo (deprecated, но полезно как референс визуализации карт):
- `scripts/baseline/demo_core_depth_midas_quality.py`

### Требования к разрешению (фиксируем требования компонента)

- input frames (analysis timeline): min shorter side **320**, target **640**, max useful **1080**, апскейл запрещён
- output depth map: default **384×384** (допускаются пресеты по бюджету)

## Performance

### Архитектура Triton ensemble

Triton ensemble `midas_256` (и аналогичные `midas_384`, `midas_512`) состоит из двух шагов:
1. **`preprocess_midas_256`** — работает на **CPU** (`instance_group: KIND_CPU`)
   - Преобразует `(B,256,256,3) UINT8 NHWC` → `(B,3,256,256) FP32 NCHW`
   - Нормализация и подготовка данных для модели
2. **`midas_256_onnx`** — работает на **GPU** (`instance_group: KIND_GPU`)
   - ONNX inference для depth estimation
   - Выдает `(B,h,w) FP32` depth maps

### Оптимизации производительности

Компонент включает следующие оптимизации для batch processing и single video режима:

1. **Векторизованное вычисление mean и std**: статистики вычисляются для всего батча одновременно через `np.mean()` и `np.std()` с `axis=1` вместо циклов
2. **Оптимизированная проверка isfinite**: предвычисление маски `np.isfinite()` для всего батча перед вычислением percentiles
3. **Векторизованное сохранение результатов**: использование индексации массивов (`depth_maps_out[global_indices] = resized_batch`) вместо поэлементного присваивания
4. **Оптимизация post-processing**: уменьшение количества циклов и использование batch операций numpy для вычисления статистик

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): **2-5x** (за счет векторизации post-processing и лучшего использования GPU)
- Для single video: **1.2-1.5x** (за счет оптимизации вычислений статистик)

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными, изменилась только производительность.

### Характеристики производительности

**Источник данных**: измерения на реальных видео (1920x1080, 30 fps)

| Model | Frame Shape |
|----------|-------------|
| midas_256 vunknown | 1920x1080 |

| Triton model | Triton Preprocess | Triton Batch | Frames cnt | Duration (s) | Peak CPU % | Peak GPU % | Triton Delta RAM (MB) | Triton Delta VRAM (MB) | Component Delta VRAM (MB) | Component Delta RAM (MB) | Summary Delta RAM | Summary Delta VRAM |
|------|-------------|------------|---------------|------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| midas_256 | midas_256 | 1 | 1 | 24.131 | 88.468 | 30.333 | 667 | 288 | 9 | 441 | 1109 | 298 |
| midas_256 | midas_256 | 1 | 10 | 25.445 | 88.395 | 33.667 | 728 | 291 | 6 | 387 | 1116 | 297 |
| midas_256 | midas_256 | 1 | 100 | 41.601 | 96.553 | 31.000 | 571 | 292 | 11 | 454 | 1025 | 303 |
| midas_256 | midas_256 | 1 | 304 | 77.305 | 98.580 | 31.667 | 626 | 289 | 7 | 452 | 1079 | 296 |
| midas_256 | midas_256 | 16 | 1 | 23.923 | 97.709 | 27.667 | 1039 | 295 | 15 | 86 | 1125 | 311 |
| midas_256 | midas_256 | 16 | 10 | 7.165 | 76.863 | 62.000 | 847 | 273 | 413 | 321 | 1169 | 686 |
| midas_256 | midas_256 | 16 | 100 | 24.880 | 89.676 | 68.667 | 848 | 290 | 904 | 363 | 1211 | 1195 |
| midas_256 | midas_256 | 16 | 304 | 51.591 | 76.727 | 69.000 | 551 | 301 | 900 | 537 | 1088 | 1201 |
| midas_384 | midas_384 | 1 | 1 | 24.738 | 93.777 | 41.333 | 536 | 291 | 6 | 365 | 901 | 297 |
| midas_384 | midas_384 | 1 | 10 | 27.322 | 92.174 | 43.000 | 905 | 300 | 130 | 229 | 1134 | 430 |
| midas_384 | midas_384 | 1 | 100 | 53.471 | 98.261 | 44.000 | 636 | 310 | 126 | 637 | 1273 | 436 |
| midas_384 | midas_384 | 1 | 304 | 111.655 | 95.254 | 41.000 | 569 | 285 | 135 | 207 | 777 | 420 |
| midas_384 | midas_384 | 16 | 1 | 24.907 | 95.575 | 38.000 | 548 | 305 | 5 | 570 | 1118 | 310 |
| midas_384 | midas_384 | 16 | 10 | 9.461 | 69.000 | 79.000 | 644 | 298 | 901 | 480 | 1124 | 1199 |
| midas_384 | midas_384 | 16 | 100 | 38.769 | 95.370 | 82.000 | 658 | 293 | 1930 | 777 | 1435 | 2223 |
| midas_384 | midas_384 | 16 | 304 | 88.539 | 100.0 | 84.0 | 636 | 298 | 1931 | 665 | 1301 | 2229 |
| midas_512 | midas_512 | 1 | 1 | 28.952 | 96.330 | 65.000 | 704 | 296 | 133 | 341 | 1045 | 429 |
| midas_512 | midas_512 | 1 | 10 | 33.472 | 99.180 | 53.000 | 1231 | 296 | 137 | 232 | 1463 | 433 |
| midas_512 | midas_512 | 1 | 100 | 72.709 | 91.453 | 55.000 | 715 | 288 | 133 | 680 | 1395 | 421 |
| midas_512 | midas_512 | 1 | 304 | 161.313 | 98.305 | 63.000 | 589 | 298 | 134 | 530 | 1119 | 432 |
| midas_512 | midas_512 | 16 | 1 | 29.005 | 100.000 | 54.000 | 651 | 229 | 204 | 655 | 1306 | 433 |
| midas_512 | midas_512 | 16 | 10 | 14.312 | 71.296 | 85.000 | 554 | 285 | 1932 | 490 | 1044 | 2217 |
| midas_512 | midas_512 | 12 | 100 | 59.997 | 99.130 | 82.000 | 722 | 286 | 1938 | 635 | 1357 | 2224 |
| midas_512 | midas_512 | 14 | 304 | 147.853 | 100.000 | 89.000 | 706 | 295 | 3968 | 551 | 1257 | 4263 |

### Влияние `batch_size` на производительность

**Важное наблюдение**: Для малого количества кадров (`frames_cnt ≤ 10`) `batch_size=1` может быть **быстрее**, чем `batch_size=8` или `batch_size=16`.

**Причины**:

1. **CPU preprocessing bottleneck**:
   - Preprocessing на CPU (`preprocess_midas_256`) становится узким местом при больших батчах
   - При `batch_size=1` каждый запрос быстрее обрабатывается на CPU
   - При `batch_size=8-16` CPU preprocessing занимает больше времени, чем GPU inference

2. **Ensemble coordination overhead**:
   - Координация между CPU и GPU шагами добавляет overhead
   - При больших батчах этот overhead может перевешивать выигрыш от батчинга

3. **HTTP overhead**:
   - Каждый HTTP запрос к Triton имеет фиксированный overhead
   - При `batch_size=1` больше запросов, но каждый быстрее
   - При `batch_size=8-16` один большой запрос может быть медленнее из-за размера данных и CPU preprocessing

4. **Post-processing не оптимизирован**:
   - После каждого батча идет последовательный post-processing каждого кадра (resize, mean, std, percentiles)
   - Это не распараллелено и может быть узким местом

**Рекомендации по выбору `batch_size`**:

- **Для unit-cost тестов**: используйте `batch_size=1` (быстрее для малого количества кадров)
- **Для production throughput** (большое количество кадров, `frames_cnt ≥ 100`):
  - `midas_256`: `batch_size=16` дает ускорение ~2-3x для больших батчей
  - `midas_384`: `batch_size=16` дает ускорение ~1.4-2x, но требует больше VRAM (~2GB)
  - `midas_512`: `batch_size=12-14` оптимально, `batch_size=16` может вызвать VRAM overflow
- **Для малого количества кадров** (`frames_cnt ≤ 10`): используйте `batch_size=1`

**Детальное логирование**:
- Компонент логирует детальные тайминги для каждого батча на уровне DEBUG:
  - `load`: время загрузки кадров из FrameManager
  - `prep`: время preprocessing (resize до input_size)
  - `infer`: время Triton inference (CPU preprocessing + GPU inference)
  - `post`: время post-processing (resize depth maps, вычисление статистик)
  - `total`: общее время батча и `ms/frame`

Для просмотра детальных таймингов запустите с уровнем логирования `DEBUG`.

---
---

## Навигация

[VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
