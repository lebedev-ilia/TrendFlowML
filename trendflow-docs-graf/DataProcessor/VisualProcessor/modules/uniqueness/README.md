# `uniqueness` (Visual module, Tier‑0 baseline)

Baseline‑компонент “уникальности” в MVP: считает **intra‑video** метрики повторяемости/разнообразия по sampled кадрам, используя **только `core_clip` embeddings**.

> Важно: это не “novelty vs reference videos”. Референс‑логика (топ‑видео и т.п.) не является частью Tier‑0 baseline и здесь не реализуется.

## Входы

### Основной вход
- **`frames_dir`**: директория Segmenter с `metadata.json` и батчами кадров.
- **`metadata["uniqueness"]["frame_indices"]`**: индексы кадров в **union-domain** (0..N-1), которые обрабатывает модуль.

### Time-axis (обязательно)
- **`metadata["union_timestamps_sec"]`**: timestamp’ы (сек) для каждого union-кадра — **source-of-truth** времени. Используется, чтобы считать temporal‑метрики **per‑second**, а не в “пер‑кадр” масштабе.

### Зависимости (hard deps, no-fallback)
- **`core_clip`**: `result_store/<platform>/<video>/<run>/core_clip/embeddings.npz`
  - `frame_indices (N,) int32`
  - `frame_embeddings (N, D) float32`

Контракт: `core_clip.frame_indices` обязан **полностью покрывать** `metadata["uniqueness"]["frame_indices"]`. Иначе — error (no-fallback).

## Sampling requirements (Visual)

Компонент строит pairwise similarity \(N\times N\) (сложность \(O(N^2)\)), поэтому Sampling должен быть ограничен.

- **min frames**: рекомендуется 60 (но не проверяется в коде, только fail-fast при пустом frame_indices)  
- **target frames**: 120  
- **max frames**: 200 (проверяется в коде, fail-fast при превышении)

Если Segmenter выдаст больше `max_frames` — компонент должен **fail-fast** (это ошибка sampling policy).

## Выход (артефакт)

Пишется через `BaseModule.save_results()` в:
- `result_store/<platform_id>/<video_id>/<run_id>/uniqueness/uniqueness.npz` (**фиксированное имя**)

- **Сводка полей, meta → CSV, melt/QA:** `docs/FEATURE_DESCRIPTION.md`

### Ключи NPZ
- **`frame_indices`**: `(N,) int32` — union-domain кадры модуля.
- **`times_s`**: `(N,) float32` — `union_timestamps_sec[frame_indices]` (source-of-truth).
- **`max_sim_to_other`**: `(N,) float32` — для каждого кадра максимальная cosine similarity к любому *другому* кадру (diag исключена).
- **`cos_dist_next`**: `(N-1,) float32` — cosine distance между соседними кадрами (по времени/порядку sampling).
- **`feature_names`**: `(F,) object` — имена агрегированных model-facing scalar фич (фиксированный порядок).
- **`feature_values`**: `(F,) float32` — значения scalar фич (bool как 0/1).
- **`meta`**: `object(dict)` — canonical meta (run identity keys, schema/producer versions, models_used/model_signature, status/empty_reason и т.д.).

## Метрики (model-facing scalars, `feature_names/feature_values`)

### Repetition / similarity
- **`repeat_threshold_mode`**: `otsu|fixed` (строка хранится в `meta.ui_payload`, а в model-facing — флаг `repeat_threshold_is_otsu`).
- **`repeat_threshold_used`**: итоговый порог (cosine similarity), выше которого кадр считается “повтором”.
- **`repeat_threshold_raw`**: сырое значение порога из Otsu (до clamp), если `mode=otsu`.
- **`repeat_threshold_quality`**: качество auto-порога (0..1 approx) для Otsu (NaN для fixed).
- **`repeat_threshold_min/max`**: clamp-границы для auto режима.
- **`repetition_ratio`**: доля кадров, у которых `max_sim_to_other >= repeat_threshold_used`.
- **`max_sim_to_other_mean/p95`**: агрегаты по `max_sim_to_other`.
- **`pairwise_sim_mean`**: средняя попарная cosine similarity по верхнему треугольнику.
- **`pairwise_sim_p95`**: 95‑й перцентиль попарной similarity.

### Temporal change (per-second)
Считаем cosine distance между соседними кадрами и нормируем на \(dt\) из `union_timestamps_sec`.
- **`temporal_change_mean`**: средняя скорость изменения семантики (per-second).
- **`cos_dist_next_mean/p95`**: агрегаты по `cos_dist_next`.

### Diversity proxy
- **`diversity_score`**: `clip(1 - pairwise_sim_mean, 0..1)` (чем меньше средняя similarity, тем выше diversity).
- **`effective_unique_frames/ratio`**: эффективное число/доля “уникальных” кадров по порогу `repeat_threshold_used`.
- **`n_frames`**: число sampled кадров \(N\).

## No-fallback / empty semantics

- **No-fallback**:
  - отсутствует `frame_indices`;
  - отсутствует/битый `union_timestamps_sec` или не покрывает `frame_indices`;
  - отсутствует `core_clip/embeddings.npz` или он не покрывает `frame_indices`;
  - `N > max_frames`.
- **Empty outputs**: для baseline не предусмотрены; пустой `frame_indices` → error.

## Параметры (config)

CLI: `VisualProcessor/modules/uniqueness/main.py`
- **`repeat_threshold_mode`** (`auto|fixed`, default `auto`, auto=Otsu)
- **`repeat_threshold`** (`float`, default `0.97`) — используется если `mode=fixed`
- **`repeat_threshold_min/max`** (`float`, default `0.90/0.99`) — clamp для auto режима
- **`repeat_threshold_bins`** (`int`, default `128`) — количество бинов для Otsu threshold (если `mode=auto`)
- **`ui_topk`** (`int`, default `8`) — top-K повторов для `meta.ui_payload`
- **`max_frames`** (`int`, default `200`) — safety‑лимит на \(N\) (дублирует sampling contract).

## Observability / UI

- Прогресс пишется в `state_events.jsonl` (stage-based): `start → load_deps → compute → save → done`
- `meta.stage_timings_ms` содержит timing.
- `meta.ui_payload` содержит:
  - pointers на `max_sim_to_other` и `cos_dist_next`
  - `top_repeats` (time + score + frame_index)
  - `top_unique` (anti-top: самые “уникальные” кадры по минимальному `max_sim_to_other`)

## Производительность

Компонент CPU‑heavy по времени из‑за \(N \times N\) similarity:
- Время: \(O(N^2)\)
- Память: \(O(N^2)\) (матрица similarity)

Для baseline обязательно держать `N <= 200` (см. sampling contract).

## Models

`uniqueness` **не запускает ML‑модели напрямую**. Модельная часть приходит через hard dep:
- `core_clip` (CLIP image encoder, обычно через Triton).

## Parallelization

- Внутренний параллелизм: нет.
- Внешний параллелизм: допускается параллельный запуск на разных видео/run_id (разные директории `result_store`).

## Batch Processing (Stage 3)

Компонент поддерживает **batch processing** для одновременной обработки нескольких видео:

- **Batch-safe**: использует per-video rs_path (нет shared mutable state между видео).
- **Дефолтный process_batch()**: последовательная обработка каждого видео через BaseModule.
- **GPU batching**: не требуется (CPU-only модуль).

**Использование**:
- Batch processing автоматически активируется при обработке нескольких видео через `--video-input-dir` или `--video-input-list`
- Контролируется через `visual.batch_processing.enable_gpu_batching` в `global_config.yaml`

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): без изменений (компонент работает через subprocess)
- Для single video: без изменений

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными.

## Human-friendly визуализация (Render System)

`uniqueness` генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/uniqueness/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по uniqueness метрикам (frames_count, repetition_ratio, diversity_score, pairwise_sim_mean, temporal_change_mean, repeat_threshold_used)
- **Timeline**: данные по каждому кадру (frame_index, time_sec, max_sim_to_other, cos_dist_next)
- **Distributions**: распределения метрик (max_sim_to_other, cos_dist_next) с min, max, mean, std, median, percentiles

Render-context может быть использован:
- **LLM** для генерации текстовых описаний уникальности и повторяемости видео
- **Frontend** для построения графиков и визуализаций (timeline charts с max_sim_to_other и cos_dist_next, distributions метрик)
- **Debugging**: быстрая проверка качества uniqueness метрик без загрузки NPZ

**HTML debug страница** (опционально):
- Путь: `result_store/.../uniqueness/_render/render.html`
- Содержит offline SVG графики (без CDN):
  - Timeline: max_sim_to_other и cos_dist_next по времени
  - Distributions: статистики по метрикам
  - Summary metrics: ключевые показатели в удобном формате

**Конфигурация** (в `global_config.yaml`):
```yaml
uniqueness:
  repeat_threshold: 0.97
  repeat_threshold_mode: "auto"  # auto|fixed
  repeat_threshold_min: 0.90
  repeat_threshold_max: 0.99
  ui_topk: 8
  max_frames: 200
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML debug страницу
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).

## Quality validation & human-friendly inspection

Demo‑скрипт (HTML + sanity checks):
- `scripts/baseline/demo_uniqueness_quality.py`

Он:
- валидирует NPZ (`validate_npz`)
- показывает распределение `max_sim_to_other`, `cos_dist_next`, ключевые агрегаты
- генерирует HTML отчёт (thumbnails по наиболее “повторяющимся” кадрам).

Пример запуска:

```bash
PY="/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/.data_venv/bin/python"
$PY scripts/baseline/demo_uniqueness_quality.py --frames-dir "<frames_dir>" --rs-path "<result_store_run>" --out-dir "<out_dir>"
```
---

## Навигация

[VisualProcessor](../../docs/MAIN_INDEX.md) · [DataProcessor](../../../docs/MAIN_INDEX.md) · [Vault](../../../../docs/MAIN_INDEX.md)
