# `high_level_semantic` (Visual module, baseline-ready)

## Purpose

`high_level_semantic` produces **high-level semantic signals aligned to the Visual time axis** for:
- **ML schema / encoder input** (dense per-frame vectors + scene embeddings),
- **analytics** (simple scalar stats),
- **UI explainability** (sparse events stream + scene timeline).

Key design:
- **Unit-of-processing**: `frame` (union-domain indices).
- **Time axis source-of-truth**: `frames_dir/metadata.json.union_timestamps_sec`.
- **No-fallback**: Segmenter must provide `metadata["high_level_semantic"]["frame_indices"]`.
- **No-network / deterministic**: does **not** load CLIP weights; consumes `core_clip/embeddings.npz`.
- **Scene source**: consumes `cut_detection` artifacts (no internal cut detection).

## Dependencies (required inputs)

VisualProcessor / per-run storage:
- `core_clip/embeddings.npz` (**required**): source-of-truth frame embeddings.
- `cut_detection/*.npz` (**required**): shot boundaries + scene grouping.
- `emotion_face/emotion_face.npz` (**required**): emotion timeline (mapped to the union time axis by time interpolation).

External processors (also in per-run storage under the same `rs_path`):
- `text_processor/text_features.npz` (**optional by default**, privacy-safe; enabled via config/CLI).
- `loudness_extractor/*.npz` (**optional by default**).
- `tempo_extractor/*.npz` (**optional by default**).
- `clap_extractor/*.npz` (optional by default).

Important: this module requires **aligned sampling**:
- `cut_detection.frame_indices` must **exactly match** `high_level_semantic.frame_indices`.
- `core_clip` must cover all `high_level_semantic.frame_indices`.

## Output (NPZ source-of-truth)

Fixed filename (per-run unique by path):
- `result_store/<platform_id>/<video_id>/<run_id>/high_level_semantic/high_level_semantic.npz`

Schema:
- `schema_version="high_level_semantic_npz_v2"` (registered in `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`)
- **Audit v3**: `producer_version=2.0.2`, strict schema JSON + `SCHEMA.md`, offline render (без CDN), `meta.stage_timings_ms` + config highlights.
- **Feature catalog (wide CSV / melt / QA):** `docs/FEATURE_DESCRIPTION.md`

High-signal NPZ keys:
- **Time axis**: `frame_indices (N,) int32`, `times_s (N,) float32`
- **Scenes**:
  - `scene_id (N,) int32`
  - `scene_embeddings (S, D) float32` (mean of core_clip embeddings inside each scene; L2-normalized)
  - `scene_start_frame_idx/scene_end_frame_idx (S,) int32` (union-domain)
  - `scene_start_time_s/scene_end_time_s/scene_duration_s (S,) float32`
  - `scene_representative_frame_idx (S,) int32`
- **Dense per-frame vector**:
  - `frame_feature_names (F,) object[str]`
  - `frame_features (N, F) float32` (NaN for missing optional signals)
  - `frame_feature_present_ratio (F,) float32` (доля finite по каждой колонке; помогает моделям интерпретировать NaN)
- **Sparse events stream**:
  - `event_times_s (E,) float32`
  - `event_type_id (E,) int16`
  - `event_strength (E,) float32`
  - `event_frame_pos (E,) int32` (0..N-1)
- **Text snapshot (optional copy)**:
  - `text_feature_names`, `text_feature_values`

## CLI

```bash
python3 DataProcessor/VisualProcessor/modules/high_level_semantic/main.py \
  --frames-dir /path/to/frames_dir \
  --rs-path /path/to/result_store/<platform>/<video>/<run> \
  --feature-groups core,scenes,events,audio,emotion,text \
  --require-cut-detection-model-facing \
  --progress-every-frames 50
```

## Sampling / units-of-processing requirements (Visual)

This module **does not choose sampling**. Segmenter is the single owner of sampling policy.

- **Unit**: `frame` in union-domain.
- **Required inputs from Segmenter**:
  - `metadata["high_level_semantic"]["frame_indices"]`
  - `union_timestamps_sec`
- **Alignment requirement**: Segmenter must ensure `high_level_semantic.frame_indices` equals `cut_detection.frame_indices` and is covered by `core_clip.frame_indices`.

Source of truth: `docs/contracts/SEGMENTER_CONTRACT.md`.

## Models

This module does not load ML models directly.

It consumes upstream model outputs:
- `core_clip` (Triton-backed in baseline): provides `frame_embeddings`.
- `emotion_face` (EmoNet in-process): provides emotion sequences.
- Audio/Text processors: provide modality-specific signals/features.

## Parallelization

Internal parallelism:
- parallel artifact loading (core_clip/cut_detection/emotion/text/audio) using a thread pool.

External parallelism:
- safe across runs due to per-run storage by `run_id`.
- **Batch processing**: Модуль поддерживает batch processing (`supports_batch = True`) и может обрабатывать несколько видео последовательно. Использует дефолтную реализацию `BaseModule.process_batch()` (цикл по видео).

## Progress reporting

Writes append-only events to:
- `state/<platform_id>/<video_id>/<run_id>/state_events.jsonl`

The backend tails this file and pushes `component.progress` to the website.

## Configuration parameters

All parameters are passed from the website profile via `DataProcessor/configs/*.yaml` and forwarded to CLI as flags.

| Param | Type | Default | Meaning |
|---|---:|---:|---|
| `feature_groups` | str | `"core,scenes,events,audio,emotion,text"` | Output feature groups (CSV) |
| `require_cut_detection_model_facing` | bool | `false` | If true, requires cut_detection model-facing NPZ |
| `require_text_processor` | bool | `true` (CLI) / `false` (API) | If true, requires `text_processor/text_features.npz` (otherwise текстовые фичи best-effort). **Note**: CLI defaults to `true`, but can be disabled with `--no-require-text-processor`. API (HighLevelSemanticModule) defaults to `false`. |
| `require_audio_loudness` | bool | `true` (CLI) / `false` (API) | If true, requires `loudness_extractor` (иначе аудио‑громкость best-effort). **Note**: CLI defaults to `true`, but can be disabled with `--no-require-audio-loudness`. API defaults to `false`. |
| `require_audio_tempo` | bool | `true` (CLI) / `false` (API) | If true, requires `tempo_extractor` (иначе аудио‑темп best-effort). **Note**: CLI defaults to `true`, but can be disabled with `--no-require-audio-tempo`. API defaults to `false`. |
| `require_audio_clap` | bool | `false` | If true, requires `clap_extractor` (иначе clap‑фичи best-effort) |
| `progress_every_frames` | int | `50` | Progress granularity (unit=frame) |
| `semantic_jump_topk_events` | int | `256` | Max semantic-jump events |
| `semantic_jump_min_strength` | float | `0.25` | Min jump strength for event candidates (1-cosine) |
| `semantic_jump_min_distance_frames` | int | `6` | Min distance (frames) between semantic-jump peaks |

## Алгоритмы

### Детекция событий

1. Нормализация и сглаживание мультимодальных кривых (face, audio, text, OCR, pose)
2. Комбинирование с learnable attention weights
3. Поиск пиков через `scipy.signal.find_peaks` с порогом `mean + k*std`
4. Классификация типа события по доминантному каналу

### Выравнивание эмоций

1. Нормализация кривых эмоций (face и text)
2. Кросс-корреляция для поиска оптимального лага
3. Вычисление корреляции Пирсона
4. Опционально: DTW для нелинейного выравнивания

### Topic Detection

1. Вычисление cosine similarity между scene embeddings и topic vectors
2. Softmax с температурой для калибровки вероятностей
3. Агрегация по сценам и вычисление метрик разнообразия

### Multimodal Attention Fusion

1. Нормализация сигналов различных модальностей
2. Вычисление корреляций между модальностями
3. Взвешивание через learnable weights или корреляционную матрицу
4. Объединение в единый attention score

## Performance characteristics

This module is lightweight (no heavy model inference inside):
- dominated by reading `core_clip` embeddings and simple vectorized ops,
- optional interpolation of audio/emotion time-series.

## Quality validation & human-friendly inspection

Use the demo script:
- `VisualProcessor/modules/high_level_semantic/quality_report/demo_high_level_semantic_quality.py`

It renders:
- timeline of `clip_novelty_prev` + hard cuts + semantic-jump events,
- per-scene embeddings norms/distribution,
- optional audio/emotion curves (if present).

## Troubleshooting

### Проблема: Out of Memory

**Решение**: 
- Уменьшите `clip_batch_size` (например, до 32)
- Используйте меньшую модель CLIP (`ViT-B/32` вместо `ViT-L/14`)
- Уменьшите `max_scenes`

### Проблема: Медленная обработка

**Решение**:
- Используйте режим `"fast"` вместо `"full"`
- Увеличьте `clip_batch_size` (если позволяет память)
- Используйте GPU (`device="cuda"`)

### Проблема: Неточная детекция событий

**Решение**:
- Настройте параметры `k_std` и `min_distance_frames` в `detect_events()`
- Убедитесь, что мультимодальные кривые корректно нормализованы
- Проверьте `reliability_flags` для оценки качества входных данных

## Лицензия

См. основную лицензию проекта TrendFlowML.

## Контакты и поддержка

Для вопросов и предложений создайте issue в репозитории проекта.
---

## Навигация

[VisualProcessor](../../docs/MAIN_INDEX.md) · [DataProcessor](../../../docs/MAIN_INDEX.md) · [Vault](../../../../docs/MAIN_INDEX.md)
