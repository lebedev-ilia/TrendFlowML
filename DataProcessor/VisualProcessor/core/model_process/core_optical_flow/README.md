## Component: `core_optical_flow` (Tier‑0 baseline)

### Назначение

`core_optical_flow` вычисляет **покадровую кривую движения** (optical flow) на primary выборке кадров и сохраняет её в NPZ для downstream модулей (например, `video_pacing`).

Политика: **Triton-only** (локальный `torch/torchvision` режим запрещён).

### Sampling / units-of-processing requirements

- Источник `frame_indices` — **только Segmenter**: `metadata["core_optical_flow"]["frame_indices"]` (union-domain индексы для `FrameManager.get()`).
- Единица обработки (unit) для cost/benchmark: **`frame_pair`** (одна пара соседних sampled кадров).
- В baseline политика sampling находится в Segmenter (shared primary sampling group). `core_optical_flow` сам sampling не генерирует и fallback не имеет.

### Входы

- **Кадры**: `FrameManager.get(idx)` из `frames_dir` (RGB uint8).
- **Sampling (строго)**: Segmenter обязан положить `metadata["core_optical_flow"]["frame_indices"]` (union-domain).

No-fallback:
- отсутствие/пустота `frame_indices` ⇒ **error**
- `len(frame_indices) < 2` ⇒ **error**

### Runtime (Triton)

Компонент вызывает Triton HTTP v2 и ожидает, что preprocessing (normalization/padding policy) находится в Triton.

Контракт (baseline, рекомендуемый):
- **input0**: `(B,S,S,3) uint8` — предыдущий кадр (RGB, NHWC)
- **input1**: `(B,S,S,3) uint8` — текущий кадр (RGB, NHWC)
- **output**: `(B,2,h,w) float32` — flow (dx, dy)

Примечание:
- В baseline GPU предполагается, что **полный preprocessing находится в Triton ensemble** (resize/normalize/layout).
- Если вы зовёте ONNX submodel напрямую, контракт может отличаться (обычно FP32 NCHW).

Поддерживаем 2–3 пресета размера входа:
- `raft_256` (default, быстрее)
- `raft_384`
- `raft_512`

### Models

GPU (Triton) модели:

- **RAFT ensemble**:
  - Triton repo: `DataProcessor/triton/models_raft/raft_{256,384,512}/`
  - Triton model names: `raft_256`, `raft_384`, `raft_512`
  - Input: `UINT8 NHWC` (см. contract выше)
  - Output: `FP32` flow (dx,dy)

- **RAFT ONNX submodels** (часть ensemble):
  - Triton repo: `DataProcessor/triton/models_raft/raft_{256,384,512}_onnx/`
  - Triton model names: `raft_256_onnx`, `raft_384_onnx`, `raft_512_onnx`

Spec names (ModelManager, рекомендуется использовать):
- `raft_256_triton`, `raft_384_triton`, `raft_512_triton` (`dp_models/spec_catalog/vision/`)

### Параметры конфигурации компонента

Все параметры принимаются через аргументы командной строки:

| Параметр | Тип | По умолчанию | Описание | Влияние на скорость/стоимость |
|----------|-----|--------------|----------|------------------------------|
| `--triton-preprocess-preset` | str | `raft_256` | Пресет размера входа модели (raft_256/384/512) | **Критично**: влияет на latency и VRAM |
| `--batch-size` | int | `16` | Количество пар кадров в одном Triton запросе (scheduler-controlled) | Влияет на throughput и VRAM |
| `--triton-model-spec` | str | `None` | Spec name из ModelManager (рекомендуется, переопределяет явные triton_* параметры) | Нет прямого влияния |
| `--triton-http-url` | str | `None` | URL Triton HTTP сервера (требуется, если не используется --triton-model-spec) | Нет прямого влияния |
| `--triton-model-name` | str | `None` | Имя модели в Triton (требуется, если не используется --triton-model-spec) | Нет прямого влияния |
| `--triton-model-version` | str | `None` | Версия модели в Triton | Нет прямого влияния |
| `--triton-input0-name` | str | `INPUT0__0` | Имя первого входа модели (предыдущий кадр) | Нет влияния (технический параметр) |
| `--triton-input1-name` | str | `INPUT1__0` | Имя второго входа модели (текущий кадр) | Нет влияния (технический параметр) |
| `--triton-output-name` | str | `OUTPUT__0` | Имя выхода модели | Нет влияния (технический параметр) |
| `--triton-datatype` | str | `UINT8` | Тип данных входов (baseline: UINT8 NHWC) | Нет влияния (фиксировано для baseline) |
| `--model-version` | str | `unknown` | Версия модели для meta (информационный) | Нет влияния (только метаданные) |
| `--weights-digest` | str | `unknown` | Digest весов для meta (информационный) | Нет влияния (только метаданные) |
| `--precision` | str | `fp32` | Precision модели для meta (информационный) | Нет влияния (только метаданные) |
| `--runtime` | str | `triton` | Runtime режим (фиксировано: только triton) | Нет влияния (фиксировано) |

**Влияние `--triton-preprocess-preset` на скорость и стоимость**:

| Preset | Latency per frame_pair (B=1) | Latency per frame_pair (B=8) | CPU RAM peak | GPU VRAM peak (Triton) | Δ latency vs raft_256 | Δ cost vs raft_256 | Notes |
|--------|------------------------------|------------------------------|--------------|------------------------|----------------------|-------------------|-------|
| `raft_256` | ~213 ms | ~182 ms | ~75 MB | ~1012 MB | baseline | baseline | Рекомендуется для baseline, стабильно |
| `raft_384` | ~440 ms | ~424 ms | ~101 MB | ~1168 MB | +227 ms (+107%) | ~2.1x | Баланс качества/скорости |
| `raft_512` | ~743 ms | ~761 ms | ~135 MB | ~3642 MB | +530 ms (+249%) | ~3.5x | Высокое качество, но spikes=true, VRAM drift на 6GB GPU |

**Влияние `--batch-size` на скорость и стоимость**:

| Batch size | Latency per frame_pair (raft_256) | Latency per frame_pair (raft_384) | VRAM delta | Notes |
|------------|-----------------------------------|-----------------------------------|------------|-------|
| `1` | ~213 ms | ~440 ms | baseline | Unit-cost для scheduler |
| `8` | ~182 ms (-15%) | ~424 ms (-4%) | +260-1098 MB drift | Улучшение throughput, но VRAM drift |
| `16` | ~N/A | ~N/A | +518-4108 MB | Может требовать restart Triton между группами |

**Рекомендации по выбору параметров**:

1. **Для baseline/production**: используйте `raft_256` (быстро, стабильно, низкое потребление VRAM)
2. **Для высокого качества**: используйте `raft_384` (баланс качества/скорости)
3. **Для максимального качества** (только при достаточном VRAM бюджете): `raft_512` (может требовать restart Triton между группами)
4. **Batch size**: для unit-cost используйте `batch_size=1`, для production `batch_size=8-16` (зависит от VRAM бюджета и необходимости перезапуска Triton)
5. **ModelManager**: рекомендуется использовать `--triton-model-spec` вместо явных `--triton-*` параметров (единый source-of-truth)

**Примеры конфигурации**:

Минимальная конфигурация (baseline):
```yaml
core_optical_flow:
  runtime: triton
  triton_model_spec: raft_256_triton
  batch_size: 8
```

Расширенная конфигурация (высокое качество):
```yaml
core_optical_flow:
  runtime: triton
  triton_model_spec: raft_384_triton
  batch_size: 16
  model_version: "1.0"
  weights_digest: "abc123..."
  precision: fp32
```

Примечание (Docker / Triton):
- При больших batch/размерах (особенно `raft_512`) Triton python backend может требовать увеличенного shared memory.
- Если видите ошибки вида `Failed to increase the shared memory pool size ... No space left on device`, запускайте Triton с `--shm-size` (например `--shm-size=1g`).

### Parallelization

- **Внутренний**: batching по парам (`--batch-size`) + один Triton HTTP запрос на batch.
- **Внешний**: допускается параллельный запуск на разных видео/run_id (per-run storage не конфликтует). Ограничение — общий GPU/Triton (scheduler должен контролировать concurrency/VRAM).

### Batch Processing (Stage 3)

Компонент поддерживает **batch processing** для одновременной обработки нескольких видео с оптимизацией:

- **Гибридный батчинг**: сбор пар кадров из всех видео → группировка в батчи → batch inference через Triton → распределение результатов обратно по видео
- **Оптимизации производительности**:
  - Предвычисление маппинга `frame_idx → position` (O(1) lookup вместо O(n) `.index()`)
  - Предвычисление `dt` для всех пар заранее
  - Векторизованное вычисление magnitude для всего батча (`np.hypot()`)
  - Детальный тайминг для диагностики (load/prep/infer/post стадии)

**Использование**:
- Batch processing автоматически активируется при обработке нескольких видео через `--video-input-dir` или `--video-input-list`
- Контролируется через `visual.batch_processing.enable_gpu_batching` в `global_config.yaml`
- Лимит размера батча: `visual.batch_processing.max_frames_per_gpu_batch`

**Ожидаемое ускорение**:
- Для batch processing: **30-1000x** (зависит от размера видео, особенно заметно для больших видео)
- Для single video: **1.1-1.2x** (за счёт оптимизации вычислений)

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными, изменилась только производительность.

### Выход (артефакт)

Путь: `result_store/<platform_id>/<video_id>/<run_id>/core_optical_flow/flow.npz`

Ключи:
- `frame_indices (N,) int32`
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]` (source-of-truth)
- `motion_norm_per_sec_mean (N,) float32`:
  - `0` для первого кадра
  - для остальных: \( \mathrm{mean}(\sqrt{dx^2+dy^2}) / dt / \max(h,w) \)
- `dt_seconds (N,) float32` (`NaN` для первого кадра)
- `meta` (dict, object-array)

### Empty/error semantics

`empty` недопустим. Любая невозможность посчитать кривую движения по контракту ⇒ **error**.

### Artifact save / validation (baseline contract)

- NPZ сохраняется **атомарно** (tmp → `os.replace`).
- После записи артефакт проходит `artifact_validator.validate_npz()` (fail-fast).

### Meta / models_used

`models_used[]` обязателен:
- `runtime="triton-gpu"`
- `engine="onnx"`
- `device="cuda"`
- `weights_digest="unknown"` (baseline)

Также обязательно:
- `dataprocessor_version` (baseline допускает `"unknown"`, в проде — версия релиза)
- `stage_timings_ms` — словарь `{stage_name: duration_ms}` с таймингами ключевых стадий

**Логирование таймингов**:
- После завершения обработки компонент логирует тайминги всех стадий в консоль:
  ```
  core_optical_flow | stage timings (ms): flow_inference_total=363.5, initialization=24.4, saving=2.7, total=20793.7
  ```
- Тайминги также сохраняются в `meta.stage_timings_ms` в NPZ артефакте для последующего анализа

### Sampling requirements (shared group)

Компонент должен быть в **shared sampling group** с потребителями, которые требуют выравнивания индексов:
- `video_pacing` (и любой другой модуль, который использует `core_optical_flow`)

Практическое правило: Segmenter должен выдавать одинаковые `frame_indices` для группы (иначе downstream получит mismatch и упадёт).

### Требования к разрешению

Для качества motion-curve достаточно умеренного разрешения, но:
- input frames (analysis timeline): min shorter side **320**, target **640**, max useful **1080**, апскейл запрещён
- внутри модели используются пресеты `raft_256/384/512` (выбор по бюджету)

### Quality validation & human-friendly inspection

**Render System** (автоматическая генерация):

Компонент автоматически генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/core_optical_flow/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по motion curve (frames_count, motion_mean, motion_std, motion_min, motion_max, dt_mean, dt_std)
- **Timeline**: данные по каждому кадру (frame_index, time_sec, motion_norm_per_sec, dt_seconds)
- **Distributions**: распределения motion_norm_per_sec и dt_seconds (min, max, mean, std, median, percentiles)

**HTML debug страница** (опционально):
- Путь: `result_store/.../core_optical_flow/_render/render.html`
- Содержит интерактивные графики (Chart.js):
  - Timeline: motion norm per second и dt seconds по времени
  - Distributions: статистики по motion и dt
  - Summary metrics: ключевые показатели в удобном формате

**Конфигурация** (в `global_config.yaml`):
```yaml
core_optical_flow:
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML debug страницу
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).

**Legacy demo** (deprecated):
- `scripts/baseline/demo_core_optical_flow_quality.py` — генерирует HTML с timeline, thumbnails, motion curve

### Оптимизации производительности

Компонент включает следующие оптимизации:

1. **Векторизация вычислений**: вычисление magnitude векторизовано для всего батча (`np.hypot()` вместо цикла)
2. **Оптимизация поиска**: использование словарей для O(1) lookup вместо O(n) `.index()`
3. **Предвычисление**: `dt` вычисляется заранее для всех пар
4. **Детальный тайминг**: логирование времени для каждой стадии (load/prep/infer/post)

**Детальное логирование**:
- Компонент логирует детальные тайминги для каждого батча на уровне DEBUG:
  - `load`: время загрузки кадров из FrameManager
  - `prep`: время preprocessing (resize до input_size)
  - `infer`: время Triton inference (два входа: prev_frame, cur_frame)
  - `post`: время post-processing (вычисление motion norm)
  - `total`: общее время батча и `ms/pair`

Для просмотра детальных таймингов запустите с уровнем логирования `DEBUG`.

## Performance

| Model Version | Triton Batch | Frames cnt | Duration (s) | Peak CPU % | Peak GPU % | Triton Delta RAM (MB) | Triton Delta VRAM (MB) | Component Delta VRAM (MB) | Component Delta RAM (MB) | Summary Delta RAM | Summary Delta VRAM |
|------|------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| x256 | 1 | 2 | 2.869 | 55.193 | 59.75 |  |  | 27.75 | 324.5 |  |  |
| x256 | 1 | 4 | 3.393 | 54.347 | 66.25 |  |  | 32.0 | 347.5 |  |  |
| x256 | 1 | 8 | 4.37 | 71.348 | 80.5 |  |  | 33.25 | 297.25 |  |  |
| x256 | 1 | 32 | 9.964 | 59.174 | 75.0 |  |  | 30.75 | 361.5 |  |  |
| x256 | 1 | 64 | 17.691 | 88.334 | 75.0 |  |  | 31.0 | 291.0 |  |  |
| x256 | 1 | 100 | 26.233 | 78.11 | 77.25 |  |  | 31.25 | 338.75 |  |  |
| x256 | 1 | 304 | 74.052 | 96.462 | 72.25 |  |  | 31.0 | 293.25 |  |  |
| x256 | 8 | 2 | 2.939 | 68.676 | 57.25 |  |  | 32.5 | 308.5 |  |  |
| x256 | 8 | 4 | 3.795 | 55.98 | 73.5 |  |  | 64.0 | 348.25 |  |  |
| x256 | 8 | 8 | 4.818 | 56.859 | 81.75 |  |  | 127.0 | 488.75 |  |  |
| x256 | 8 | 32 | 11.666 | 66.927 | 82.25 |  |  | 255.0 | 269.75 |  |  |
| x256 | 8 | 64 | 18.102 | 67.657 | 81.5 |  |  | 255.0 | 413.5 |  |  |
| x256 | 8 | 100 | 25.084 | 83.655 | 81.0 |  |  | 255.0 | 488.0 |  |  |
| x256 | 8 | 304 | 68.291 | 96.516 | 82.5 |  |  | 319.0 | 532.5 |  |  |
| x256 | 16 | 2 | 2.921 | 56.453 | 73.75 |  |  | 33.0 | 297.0 |  |  |
| x256 | 16 | 4 | 3.772 | 73.091 | 77.25 |  |  | 65.0 | 363.75 |  |  |
| x256 | 16 | 8 | 4.882 | 65.087 | 78.75 |  |  | 126.75 | 402.25 |  |  |
| x256 | 16 | 16 |  |  |  |  |  |  |  |  |  |
| x256 | 16 | 32 | 12.578 | 62.621 | 83.0 |  |  | 518.0 | 624.25 |  |  |
| x256 | 16 | 64 | 19.217 | 83.569 | 83.75 |  |  | 519.0 | 402.75 |  |  |
| x256 | 16 | 100 | 25.726 | 96.949 | 84.0 |  |  | 519.0 | 405.0 |  |  |
| x256 | 16 | 304 | 67.897 | 98.269 | 83.75 |  |  | 518.75 | 482.75 |  |  |

| x384 | 1 | 2 | 0.814 | 54.409 | 0.5 |  |  | 0.25 | -23.0 |  |  |
| x384 | 1 | 4 | 0.83 | 68.097 | 0.0 |  |  | 1.0 | 25.75 |  |  |
| x384 | 1 | 8 | 0.807 | 54.359 | 0.25 |  |  | 0.25 | -39.75 |  |  |
| x384 | 1 | 32 | 0.843 | 64.609 | 0.0 |  |  | 0.5 | 4.75 |  |  |
| x384 | 1 | 64 | 0.816 | 63.798 | 0.25 |  |  | 1.0 | 20.75 |  |  |
| x384 | 1 | 100 | 0.806 | 55.634 | 0.5 |  |  | 1.0 | -6.75 |  |  |
| x384 | 1 | 304 | 0.846 | 64.602 | 0.25 |  |  | 0.75 | 28.75 |  |  |
| x384 | 8 | 2 | 0.803 | 52.631 | 0.0 |  |  | 0.5 | 24.0 |  |  |
| x384 | 8 | 4 | 1.061 | 54.884 | 0.0 |  |  | 0.0 | 54.0 |  |  |
| x384 | 8 | 8 | 1.558 | 57.183 | 0.75 |  |  | -0.25 | 117.5 |  |  |
| x384 | 8 | 32 | 1.683 | 61.272 | 0.75 |  |  | -0.75 | 42.75 |  |  |
| x384 | 8 | 64 | 1.691 | 59.142 | 0.75 |  |  | -1.0 | -87.75 |  |  |
| x384 | 8 | 100 | 1.731 | 55.616 | 0.75 |  |  | -1.0 | 129.75 |  |  |
| x384 | 8 | 304 | 1.706 | 72.563 | 0.75 |  |  | -1.0 | 26.75 |  |  |
| x384 | 16 | 2 | 0.799 | 53.149 | 0.0 |  |  | 0.75 | 13.75 |  |  |
| x384 | 16 | 4 | 1.555 | 70.783 | 0.5 |  |  | 0.25 | 47.5 |  |  |
| x384 | 16 | 8 | 1.569 | 56.889 | 0.25 |  |  | 0.75 | 55.0 |  |  |
| x384 | 16 | 32 | 2.676 | 58.37 | 1.0 |  |  | -0.25 | 47.25 |  |  |
| x384 | 16 | 64 | 2.745 | 65.398 | 1.0 |  |  | -1.5 | 109.0 |  |  |
| x384 | 16 | 100 | 2.695 | 66.555 | 1.0 |  |  | -1.5 | -16.75 |  |  |
| x384 | 16 | 304 | 2.7 | 57.241 | 1.0 |  |  | -0.5 | 90.75 |  |  |

| x512 | 1 | 1 |  |  |  |  |  |  |  |  |  |
| x512 | 1 | 2 | 0.875 | 58.66 | 0.0 |  |  | -0.5 | 17.25 |  |  |
| x512 | 1 | 4 | 0.889 | 53.094 | 0.75 |  |  | -0.5 | -35.5 |  |  |
| x512 | 1 | 8 | 0.902 | 63.032 | 0.5 |  |  | -0.5 | 60.0 |  |  |
| x512 | 1 | 32 | 0.884 | 54.088 | 1.0 |  |  | -0.25 | 27.0 |  |  |
| x512 | 1 | 64 | 0.886 | 61.476 | 0.25 |  |  | -0.25 | -8.5 |  |  |
| x512 | 1 | 100 | 0.902 | 58.235 | 0.5 |  |  | -0.5 | 24.5 |  |  |
| x512 | 1 | 304 | 1.572 | 63.599 | 0.5 |  |  | -0.25 | 27.75 |  |  |
| x512 | 8 | 2 | 0.897 | 54.372 | 0.25 |  |  | 0.25 | -63.0 |  |  |
| x512 | 8 | 4 | 1.294 | 64.955 | 0.5 |  |  | 0.0 | 18.75 |  |  |
| x512 | 8 | 8 | 2.124 | 61.796 | 0.75 |  |  | -0.25 | 135.25 |  |  |
| x512 | 8 | 32 | 2.341 | 65.702 | 1.0 |  |  | -0.25 | 33.25 |  |  |
| x512 | 8 | 64 | 2.298 | 58.843 | 1.0 |  |  | -0.25 | 63.25 |  |  |
| x512 | 8 | 100 | 2.426 | 77.685 | 0.75 |  |  | -0.5 | -12.75 |  |  |
| x512 | 8 | 304 | 2.305 | 58.792 | 1.0 |  |  | -1.0 | 24.25 |  |  |
| x512 | 16 | 2 | 0.907 | 64.631 | 0.25 |  |  | 0.25 | 27.5 |  |  |
| x512 | 16 | 4 | 1.275 | 55.089 | 0.25 |  |  | -0.75 | 32.75 |  |  |
| x512 | 16 | 8 | 2.145 | 61.665 | 1.0 |  |  | 0.0 | -1.75 |  |  |
| x512 | 16 | 32 | 3.939 | 73.419 | 1.0 |  |  | -1.5 | 85.0 |  |  |
| x512 | 16 | 64 | 3.966 | 73.74 | 1.0 |  |  | -1.5 | 122.25 |  |  |
| x512 | 16 | 100 | 3.911 | 68.856 | 1.0 |  |  | -1.25 | 106.75 |  |  |
