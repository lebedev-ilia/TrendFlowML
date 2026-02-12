# ✅ Baseline Audit — `video_pacing`

Компонент: `DataProcessor/VisualProcessor/modules/video_pacing/`  
Тип: Visual module (Tier‑0 baseline)  
Статус: **✅ CLOSED (baseline)** (2026‑01‑16)  

---

## Резюме

`video_pacing` вычисляет признаки **темпа/монтажа** (shot pacing) и связанные метрики движения/семантических/цветовых изменений **строго на sampled кадрах** от Segmenter.

Hard deps (no‑fallback):
- `cut_detection` — shot boundaries (source-of-truth)
- `core_optical_flow` (`flow.npz`) — кривая движения `motion_norm_per_sec_mean`
- `core_clip` (`embeddings.npz`) — эмбеддинги `frame_embeddings` для semantic change rate

---

## Соответствие `BASELINE_COMPONENT_AUDIT_CRITERIA.md`

### 1) Наследование / интерфейсы

- `VideoPacingModule` наследуется от `BaseModule`
- реализует `process(frame_manager, frame_indices, config)`
- `required_dependencies()` → `["core_optical_flow", "core_clip"]`

### 2) Контракты входа/выхода

- `frame_indices` строго из `frames_dir/metadata.json["video_pacing"]["frame_indices"]` (Segmenter‑owned)
- `times_s` строго из `union_timestamps_sec[frame_indices]` (no‑fallback)
- отсутствие `union_timestamps_sec` / не монотонна → error
- отсутствие `cut_detection` или отсутствие `detections.shot_boundaries_frame_indices` → error
- отсутствие core providers или непокрытие `frame_indices` → error

### 3) Per‑run storage + atomic save + validation

- Артефакт: `result_store/<platform>/<video>/<run_id>/video_pacing/video_pacing_features.npz` (**фиксированное имя**)
- Сохранение атомарное + `validate_npz()` в `BaseModule.save_results()` (fail‑fast, удаление файла при ошибке)

---

## Артефакт (NPZ)

Путь: `.../video_pacing/video_pacing_features.npz`

Ключи:
- `frame_indices (N,) int32`
- `times_s (N,) float32`
- `shot_boundary_frame_indices (S,) int32` (union-domain)
- `motion_norm_per_sec_mean (N,) float32` (aligned; from core_optical_flow)
- `semantic_change_rate_per_sec (N,) float32` (aligned; from core_clip)
- `color_change_rate_per_sec (N,) float32` (aligned; cheap LAB proxy)
- `features` (dict, object-array) — агрегированные pacing признаки
- `meta` (dict, object-array) — canonical meta (run identity + models_used + model_signature)
  - `meta.ui_payload` (dict) — UI графики/маркеры (curves + shot boundaries)
  - `summary.stage_timings_ms` — тайминги стадий выполнения

Schema:
- `video_pacing_npz_v2`

---

## Производительность (resource costs)

Источник правды:
- `docs/models_docs/resource_costs/video_pacing_costs_v1.json`

Evidence:
- `storage/reports/out/checklist-video-pacing/checklist_components_micro_results.json`

Unit:
- `frame` (per sampled frame, CPU module)

Примечание:
- `gpu_vram_peak_mb` в component‑micro является best‑effort и может включать фоновые процессы (не является строгой метрикой для CPU‑модуля).

---

## Проверка качества (human‑friendly)

Скрипт:
- `scripts/baseline/demo_video_pacing_quality.py`

Evidence (пример HTML):
- `storage/reports/out/checklist-video-pacing/demo_video_pacing_quality_20260116-031902-335155.html`
- `storage/reports/out/video_pacing_real/demo_video_pacing_quality_20260116-033332-643946.html` (реальное видео `NSumhkOwSg`, run_id=`60a6781270dd`)

Краткий sanity‑итог для `NSumhkOwSg` (32.6s, sampled_frames=131):
- `shots_count=6`, границы: `[0, 38, 46, 50, 57, 83]`
- длительности шотов (sec): min ~0.76, median ~4.02, mean ~5.23, max ~11.8
- `cuts_per_10s ~ 1.53`, без burst’ов (`quick_cut_burst_count=0`)

Сanity checks (в demo):
- `times_s` монотонен
- `shot_boundary_frame_indices` в диапазоне `frame_indices`
- `validate_npz()` проходит без ошибок

---

## Известные ограничения / next steps

- Качество shot boundary detection зависит от sampling density и разрешения кадров (SSIM/гист/edges работают на downscale).
- Для production‑quality можно добавить отдельный модуль “shot boundaries” и отдавать `video_pacing` готовые границы (но baseline‑MVP OK).


