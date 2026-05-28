# `cut_detection` (Visual module, Tier‑0 baseline)

**Контракт NPZ, melt/QA, валидатор:** [docs/FEATURE_DESCRIPTION.md](docs/FEATURE_DESCRIPTION.md) · `utils/validate_cut_detection_npz.py` (`--struct`, `--qa`, `--ranges`, батч `--results-base`).

## Purpose

Detects **hard cuts** and **soft transitions** (fade/dissolve + motion transitions) on a sampled frame sequence and produces:
- a **shot boundary** timeline used by downstream modules (notably `shot_quality`)
- a rich set of **editing / pacing** features.

This module is part of the **baseline** stage and is **required**.

## Inputs

- **Primary input (frames)**: `frames_dir/metadata.json` + RGB frames via `FrameManager`.
  - **Sampling**: uses only `metadata["cut_detection"]["frame_indices"]` provided by Segmenter (no internal resampling).
  - **Time axis**: uses `union_timestamps_sec` from `frames_dir/metadata.json` as the **source‑of‑truth** timeline.
- **Optional input (audio)**: `audio/audio.wav` produced by Segmenter (auto‑resolved).

## Dependencies (inputs from other components)

**Required (baseline)**:
- `core_optical_flow` (`rs_path/core_optical_flow/flow.npz`) — **REQUIRED** (no-fallback). Module reuses `core_optical_flow/flow.npz` and forbids local flow computation.

**Quality deps (soft)**:
- `core_face_landmarks` (`rs_path/core_face_landmarks/landmarks.npz`) — если нет/битый → jump-cut эвристика отключается, качество хуже (warning).
- `core_object_detections` (`rs_path/core_object_detections/detections.npz`) — если нет/битый → jump-cut эвристика отключается, качество хуже (warning).

**Note**: The `--prefer-core-optical-flow` and `--require-core-optical-flow` CLI flags are deprecated. Baseline policy: `core_optical_flow` is always required. Use `--no-require-core-optical-flow` only for debugging (not for production/baseline).

The DAG stage must ensure these cores run **before** `cut_detection`.

## Outputs (artifact contract)

Artifact path (timestamped, via `BaseModule.save_results`):
- `result_store/<run>/<video_id>/cut_detection/cut_detection_features_<ts>_<uid>.npz`

Additional (recommended) model-facing artifact (schema v1):
- `result_store/<platform_id>/<video_id>/<run_id>/cut_detection/cut_detection_model_facing_<ts>_<uid>.npz`
  - schema: `VisualProcessor/modules/cut_detection/SCHEMA_MODEL_FACING.md`

### Schema system (Audit v3)

- **Main artifact**:
  - `schema_version=cut_detection_npz_v1`
  - Human schema: [SCHEMA.md](./SCHEMA.md)
  - Machine schema: `DataProcessor/VisualProcessor/schemas/cut_detection_npz_v1.json`
- **Model-facing artifact**:
  - `schema_version=cut_detection_model_facing_npz_v1`
  - Human schema: [SCHEMA_MODEL_FACING.md](./SCHEMA_MODEL_FACING.md)
  - Machine schema: `DataProcessor/VisualProcessor/schemas/cut_detection_model_facing_npz_v1.json`

Writing policy (MVP):
- Baseline (Audit v3): CLI writes the model-facing NPZ **required** (fail-fast on write/validation errors).
- Disable (rare): CLI `--no-write-model-facing-npz`
- Debug override: disable requirement with CLI `--no-require-model-facing-npz`

Hard cuts performance knobs:
- Default mode computes SSIM + optical flow for **all** frame pairs (higher quality / stable thresholds).
- Optional speed mode (OFF by default): histogram-gated cascade for hard cuts:
  - CLI: `--hard-cuts-cascade --hard-cuts-cascade-keep-top-p 0.25 --hard-cuts-cascade-hist-margin 0.0`
  - Tradeoff: may miss rare cuts where histogram change is extremely low; use only for `fast` experiments.
  - Model-facing NPZ note: in cascade mode `ssim_drop/flow_mag/deep_cosine_dist` may contain `NaN` for pairs where the signal was not computed; use `*_valid_mask` keys.
  - **Implementation (2026):** pass 1 still evaluates cheap `hist_diff` on every consecutive pair. Pass 2 **does not** walk the full frame list linearly: it loads from `FrameManager` only for candidate pairs (`cand[j]` is true), via `get(frame_indices[j])` and `get(frame_indices[j+1])`. SSIM / optional local Farneback / deep features for a candidate pair are the same as in non-cascade mode—only **frame I/O** is skipped for non-candidates. With required `core_optical_flow`, per-pair flow still comes from **`external_flow_mags`** for all pairs; the main saving is fewer RGB decodes/loads for SSIM (and deep, if enabled).

Hard cuts presets (CLI convenience):
- `--hard-cuts-preset quality|default|fast`
  - `quality`: ssim=640, flow=384, cascade=off
  - `default`: ssim=512, flow=320, cascade=off
  - `fast`: ssim=384, flow=256, cascade=on (keep_top_p=0.25, margin=0.0)
  - explicit flags `--ssim-max-side/--flow-max-side/--hard-cuts-cascade*` override preset values.

Soft/motion optimizations:
- Baseline: `core_optical_flow` is always reused (required dependency). Soft cuts and motion-based cuts use the core motion curve for `*_flow_mag`.
- Motion detection uses a cascade when core flow is available: it computes expensive Farneback direction/variance only for motion spike candidates (best-effort).

### Advanced detection parameters

The module supports several advanced parameters for fine-tuning cut detection:

- **`--use-adaptive-thresholds`** (default: `True`): Use adaptive thresholds for cut detection. Can be disabled with `--no-use-adaptive-thresholds`.
- **`--use-semantic-clustering`** (default: `False`): Use semantic clustering for scene grouping. Requires CLIP embeddings (`--use-clip`).
- **`--fade-threshold`** (default: `0.02`): Threshold for fade detection in HSV/LAB color space.
- **`--min-duration-frames`** (default: `4`): Minimum duration in frames for soft transitions (fade/dissolve).
- **`--use-flow-consistency`** (default: `True`): Use flow consistency check for soft cuts. Can be disabled with `--no-use-flow-consistency`.

NPZ keys (high level):
- `meta`: dict (required baseline keys + `models_used[]` / `model_signature` when model‑based)
- `frame_indices`: `int32 [N]` (union‑domain indices used by this module)
- `times_s`: `float32 [N]` = `union_timestamps_sec[frame_indices]`
- `features`: dict of scalar features (counts/ratios/statistics)
- `detections`: dict with intermediate events:
  - `hard_cuts`: list[int] (positions in sampled sequence)
  - `soft_events`: list[dict] with `{type, start, end, duration_s}`
  - `motion_cuts`: list[int]
  - `jump_cuts`: list[int] (subset of hard cuts)

### Model-facing output (recommended for Transformers)

For `baseline/v1/v2` models (especially transformers), **events-only** output is not enough.
We recommend exposing (and saving) **raw per-step signals** so a downstream FeatureEncoder can learn
robust pooling/attention over time:

- **Dense curves (per frame_pair / per sampled step)**:
  - `hist_diff[t]` — cheap content change proxy
  - `ssim_drop[t]` — structural similarity drop
  - `flow_mag[t]` — motion magnitude (preferably from `core_optical_flow`)
  - `hard_score[t]` — combined hard-cut score before postprocessing
- **Sparse events**:
  - hard/soft/motion events with `{time_s, type, strength, contributors}`

Rationale:
- Keeps the pipeline reproducible while letting the model learn thresholds and interactions.
- Avoids overfitting to brittle postprocessing heuristics.
- Works for both short and very long videos via fixed-budget encoding (see `docs/models_docs/FEATURE_ENCODER_CONTRACT.md`).

### Models metadata

- If `use_clip=true`, module records `models_used[]` with:
  - `model_name="openai_clip_vit_b32"`, `runtime="inprocess"`, `engine="clip"`, `device`, and best‑effort `weights_digest` (sha256 of local weight file).

## Sampling / units‑of‑processing requirements (Visual)

- **Coverage goal**: uniform coverage over the entire video to reliably detect cuts and compute pacing statistics.
- **Min/target/max** (start values, will be refined after full audit):
  - `min_frames`: **400**
  - `target_frames`: **800**
  - `max_frames`: **1500**
- **Time axis**: `union_timestamps_sec` is required and must be monotonic.
- **Max sampling gap (quality gate)**:
  - if `max(diff(times_s)) > 6.0s` → **error** (sampling too sparse for reliable cut detection).
- **Resolution**:
  - This module is robust to moderate downscaling; it does not require per‑component high‑res.
  - It operates on RGB frames from Segmenter; upstream should avoid upscaling (no‑upscale policy).

## Empty / error semantics (no‑fallback)

- `frame_indices` missing/empty → **error**
- `union_timestamps_sec` missing/invalid/non‑monotonic → **error**
- `core_optical_flow` artifact missing (baseline) → **error**
- `core_face_landmarks` / `core_object_detections` missing/invalid → **warning** + jump-cut detection disabled (quality degraded)
- `len(frame_indices) < 2` → **error**

Valid "empty" is generally **not expected** for this baseline module (it must produce a timeline).

## Performance characteristics

### Resource costs (measured)

**Источник данных**: `docs/models_docs/resource_costs/cut_detection_*.json`

Компонент разделён на три части для измерений:
1. **Hard cuts detection** (`detect_hard_cuts`)
2. **Soft cuts detection** (`detect_soft_cuts`)
3. **Motion-based cuts detection** (`detect_motion_based_cuts`)

#### Hard cuts (типичные значения для preset="default", 16:9, short_side=320-640)

**Единица обработки**: `frame_pair` (N-1 пар для N кадров)

| Resolution | Preset | Latency per pair | CPU RAM peak | GPU VRAM peak | Notes |
|------------|--------|------------------|--------------|---------------|-------|
| 284×160 | default | ~17 ms | ~178 MB | ~653 MB | baseline low-res |
| 398×224 | default | ~26 ms | ~191 MB | ~633 MB | typical |
| 568×320 | default | ~35 ms | ~200 MB | ~634 MB | typical |
| 1024×576 | default | ~60 ms | ~220 MB | ~650 MB | high-res |

**Preset "fast"** (cascade enabled): ~4-6x быстрее, но может пропускать редкие cuts.

**Preset "quality"**: ~1.2-1.5x медленнее чем default, более стабильные пороги.

#### Soft cuts (типичные значения)

**Единица обработки**: `frame_pair`

| Resolution | Preset | Latency per pair | CPU RAM peak | GPU VRAM peak |
|------------|--------|------------------|--------------|---------------|
| 284×160 | default | ~8 ms | ~178 MB | ~650 MB |
| 568×320 | default | ~15 ms | ~200 MB | ~635 MB |

#### Motion-based cuts (типичные значения)

**Единица обработки**: `frame_pair`

| Resolution | Preset | Latency per pair | CPU RAM peak | GPU VRAM peak |
|------------|--------|------------------|--------------|---------------|
| 284×160 | default | ~12 ms | ~178 MB | ~650 MB |
| 568×320 | default | ~22 ms | ~200 MB | ~635 MB |

**Примечания**:
- Pure CPU heuristics (hist/SSIM/Farneback) + optional audio features
- CLIP (if enabled) добавляет ~40-100ms per candidate window (если используется)
- GPU VRAM используется только если включён CLIP через Triton
- Все измерения на CPU (device='cpu'), без GPU acceleration для heuristics

**Полные данные**: см. `docs/models_docs/resource_costs/cut_detection_costs_v1.json`, `cut_detection_soft_costs_v1.json`, `cut_detection_motion_costs_v1.json`

### Performance knobs (quality-preserving)

This module has explicit, deterministic knobs to keep quality while reducing CPU cost on high-res inputs:

- **`flow_max_side`** (default **320**): caps resolution used for Farneback optical flow magnitude.
- **`ssim_max_side`** (default **512**): caps resolution used for SSIM (grayscale) drop metric.

Rules:
- If Segmenter already outputs frames with max-side <= these values, behavior is unchanged.
- If frames are large (e.g., 720p+), SSIM/flow are computed on downscaled grayscale images (aspect ratio preserved).
- This is not an adaptive heuristic based on content; it is a deterministic performance policy.

### Reuse of `core_optical_flow` (baseline policy)

**Baseline**: `core_optical_flow` is **required** (no-fallback). The module always reuses `core_optical_flow/flow.npz` and forbids local Farneback computation.

**Deprecated CLI flags** (kept for backward compatibility, but ignored in baseline):
- `--prefer-core-optical-flow` — deprecated (baseline always prefers core flow)
- `--require-core-optical-flow` — deprecated (baseline always requires core flow)

**Debug-only flag**:
- `--no-require-core-optical-flow` — disables baseline requirement (debug only, not for production)

This policy avoids duplicate CPU Farneback computation and is consistent with the project plan to share heavy signals via core providers.

### Batch Processing (Stage 3)

Компонент поддерживает **batch processing** для одновременной обработки нескольких видео с оптимизацией:

- **Гибридный батчинг**: обработка каждого видео через subprocess с оптимизацией конфигурации
- **Оптимизации производительности**:
  - Переиспользование конфигурации для всех видео в батче
  - Параллельная обработка видео (если включено в конфигурации)
  - Оптимизация передачи параметров через CLI

**Использование**:
- Batch processing автоматически активируется при обработке нескольких видео через `--video-input-dir` или `--video-input-list`
- Контролируется через `visual.batch_processing.enable_gpu_batching` в `global_config.yaml`
- Лимит размера батча: `visual.batch_processing.max_frames_per_gpu_batch`

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): **1.5-3x** (за счет оптимизации конфигурации и параллельной обработки)
- Для single video: без изменений (компонент работает через subprocess)

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными, изменилась только производительность.

---

## Quality & optimization notes (implementation guidance)

### 1) Prefer shared motion from `core_optical_flow`

If `core_optical_flow/flow.npz` is available **and** aligned with `cut_detection.frame_indices`,
reuse it to avoid redundant optical-flow work and to keep motion semantics consistent across modules.
CPU Farneback should remain only as a fallback.

### 2) Cascade / early-exit (cheap → expensive)

To preserve quality but reduce cost on “easy” segments:
- compute `hist_diff` first (cheap)
- only compute `ssim_drop` / `flow_mag` for candidates (or at a lower rate) when `hist_diff` suggests a possible transition

This reduces average runtime without changing deterministic behavior (policy can be parameterized explicitly).

### 3) Stability / determinism

- All performance knobs (`ssim_max_side`, `flow_max_side`, reuse flags) must be **explicit parameters** (no hidden heuristics).
- Always use `union_timestamps_sec` as the source-of-truth timeline.
- If sampling is too sparse (`max gap > 6s`), fail-fast: cut statistics become unreliable.

### 4) Quality risks to watch

- Over-aggressive temporal smoothing can suppress isolated true cuts. Prefer "keep strong spikes" logic.
- `soft_cuts` and `motion_based_cuts` are inherently noisier; treat them as **probabilistic cues** for the model,
  not as ground-truth boundaries.

---

## Quality validation & human-friendly inspection

### Как проверить качество детекции cuts

Компонент детектирует переходы между кадрами. Для проверки качества выходных данных рекомендуется:

#### 1. Демонстрация качества с визуализацией (рекомендуется для первого знакомства)

**Скрипт**: `scripts/baseline/demo_cut_detection_quality.py`

**Использование**:
```bash
python scripts/baseline/demo_cut_detection_quality.py \
  --video-path /path/to/test_video.mp4 \
  --out-dir /path/to/output \
  --preset default
```

**Выход**:
- HTML файл с интерактивной визуализацией:
  - Timeline с отмеченными cuts (hard/soft/motion/jump)
  - Thumbnails кадров до/после каждого hard cut (первые 20)
  - Графики сигналов детекции (hist_diff, ssim_drop, flow_mag, hard_score)
  - Статистика по всем типам cuts и shots
- Статистика в консоли

**Для просмотра**: Откройте HTML файл в браузере.

**Что проверять**:
- ✅ Timeline: cuts отмечены в правильных позициях?
- ✅ Thumbnails: hard cuts показывают реальные переходы между сценами?
- ✅ Статистика: значения разумны для данного типа видео?
- ✅ Графики сигналов: пики соответствуют найденным cuts?
- ✅ Результат утверждён: демо-прогон подтверждён визуально; HTML страница подходит для ручного ревью
- ⚠️ Jump cuts: в демо без `core_face_landmarks/core_object_detections` — повторить с подключёнными core для финальной проверки jump cuts

#### 2. Автоматическая оценка качества (скрипт для метрик)

**Скрипт**: `scripts/baseline/eval_cut_detection_quality.py`

```bash
python scripts/baseline/eval_cut_detection_quality.py \
  --videos /path/to/video1.mp4,/path/to/video2.mp4 \
  --out-dir /path/to/quality_eval \
  --task hard \
  --ref quality
```

**Выход**: 
- `quality_report.json` с метриками precision/recall/F1 для разных presets
- Сравнение с reference preset (quality/default)
- Для stitched videos: сравнение с ground-truth cuts

**Метрики**:
- Precision: доля найденных cuts, которые действительно являются cuts
- Recall: доля реальных cuts, которые были найдены
- F1: гармоническое среднее precision и recall
- Tolerance: временная толерантность для матчинга (обычно 1.5× median frame interval)

#### 2. Human-friendly визуализация (рекомендуется для финальной проверки)

**Подход**: Создать визуализацию, показывающую:
- Временную шкалу видео с отмеченными cuts
- Кадры до и после каждого cut
- Сигналы детекции (hist_diff, ssim_drop, flow_mag) как графики
- Shot boundaries как вертикальные линии на timeline

**Пример структуры для визуализации**:

```python
# Псевдокод для создания human-friendly отчёта
def create_cut_visualization(npz_path, video_path, out_html):
    """
    Создаёт HTML страницу с визуализацией cuts:
    - Timeline с отмеченными cuts (hard/soft/motion/jump)
    - Thumbnails кадров до/после каждого cut
    - Графики сигналов детекции (hist_diff, ssim_drop, flow_mag)
    - Статистика: количество cuts, средняя длина shot, etc.
    """
    # Загрузить NPZ
    # Извлечь cuts и сигналы
    # Создать HTML с timeline + thumbnails + графики
    # Сохранить в out_html
```

**Рекомендуемый формат вывода**:
- HTML страница с интерактивным timeline
- Или JSON с метаданными для внешней визуализации
- Или набор изображений (thumbnails + графики) для ручного просмотра

**Что проверять визуально**:
1. **Hard cuts**: действительно ли это резкие переходы между сценами?
2. **Jump cuts**: правильно ли детектируются "прыжки" в одной сцене?
3. **Soft cuts**: fade/dissolve переходы найдены корректно?
4. **Motion cuts**: whip pan/zoom переходы детектируются?
5. **False positives**: нет ли ложных срабатываний на плавных движениях камеры?
6. **False negatives**: не пропущены ли явные cuts?

#### 3. Статистическая валидация

Проверить разумность статистических фичей:
- `hard_cuts_per_minute`: типично 2-10 для обычных видео, 10-30 для быстрого монтажа
- `avg_shot_length`: типично 2-8 секунд для обычных видео
- `jump_cut_ratio_per_minute`: должно быть ≤ `hard_cuts_per_minute`

**Скрипт для статистической проверки**: можно использовать `VisualProcessor/utils/quality_validator.py`

#### 4. Интеграция с downstream модулями

Проверить, что выходы `cut_detection` корректно используются downstream:
- `shot_quality` должен получать `shot_boundaries` из `cut_detection`
- Shot boundaries должны быть в правильном формате (union frame indices)

**Рекомендация**: Для финальной проверки качества перед baseline запуском создать визуализацию на 5-10 репрезентативных видео и провести ручной review.

### Human-friendly визуализация (Render System)

`cut_detection` генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/cut_detection/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по детекции cuts (frames_count, hard_cuts_count, soft_cuts_count, motion_cuts_count, jump_cuts_count, shots_count, avg_shot_length, cuts_per_minute)
- **Timeline**: данные по каждому кадру/паре (frame_index, time_sec, hist_diff, ssim_drop, flow_mag, hard_score, is_hard_cut, is_soft_cut, is_motion_cut, is_jump_cut)
- **Distributions**: распределения сигналов детекции (hist_diff, ssim_drop, flow_mag, hard_score) с min, max, mean, std, median, percentiles
- **Cuts events**: список всех обнаруженных cuts с временными метками, типами и силой сигнала
- **Shots**: информация о shot boundaries и длительности shots

Render-context может быть использован:
- **LLM** для генерации текстовых описаний структуры видео и монтажа
- **Frontend** для построения графиков и визуализаций (timeline charts с cuts, distributions сигналов, shot boundaries)
- **Debugging**: быстрая проверка качества детекции cuts без загрузки NPZ

**HTML debug страница** (опционально):
- Путь: `result_store/.../cut_detection/_render/render.html`
- Содержит offline SVG графики (без CDN):
  - Timeline: сигналы детекции (hist_diff, ssim_drop, flow_mag, hard_score) по времени с отмеченными cuts
  - Cuts events: таблица со всеми обнаруженными cuts (hard/soft/motion/jump) с временными метками
  - Distributions: статистики по сигналам детекции
  - Shots: информация о shot boundaries и длительности shots
  - Summary metrics: ключевые показатели в удобном формате

**Конфигурация** (в `global_config.yaml`):
```yaml
cut_detection:
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML debug страницу
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).


