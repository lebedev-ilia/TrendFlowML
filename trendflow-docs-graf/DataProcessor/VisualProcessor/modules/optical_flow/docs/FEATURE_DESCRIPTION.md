# `optical_flow` (модуль) — описание фич (Audit v3)

**Компонент:** `optical_flow` (VisualProcessor `BaseModule`, не core)  
**producer** в `meta`: `optical_flow`  
**producer_version (код `utils/optical_flow.py`):** **2.0.2**  
**schema_version NPZ:** **`optical_flow_npz_v3`**  
**Артефакт:** `result_store/<platform>/<video>/<run>/optical_flow/optical_flow.npz`

## Назначение

**Потребитель** `core_optical_flow/flow.npz`: кривая `motion_norm_per_sec_mean`, per-frame таблица `frame_feature_*` и video-level агрегаты в `feature_names` / `feature_values`. RAFT / Triton здесь **не** вызывается.

## Ключи NPZ

| Группа | Ключи |
|--------|--------|
| Ось | `frame_indices (N,)`, `times_s (N,)`, `motion_norm_per_sec_mean (N,)` — из core, та же ось |
| Per-frame | `frame_feature_names (D,)`, `frame_feature_values (N, D)` |
| Агрегаты | `feature_names` / `feature_values` — фиксированный набор (см. валидатор `_FIXED_FEATURE_NAMES`, `SCHEMA.md`) |

## `meta` и тайминги

`stage_timings_ms` (мс): **`frame_manager_ms`**, **`process_ms`**, **`total_ms`**. В плоском CSV / melt: `meta_timing_frame_manager_ms`, `meta_timing_process_ms`, `meta_timing_total_ms` (см. `component_feature_qa.flatten_meta`).

Дополнительно: `analysis_fps`, `analysis_width` / `analysis_height`, `processed_frames`, `total_frames`, run identity, `ui_payload` (опц.), `models_used` (provenance из core npz).

## Нормальные диапазоны (`--ranges`)

| Проверка | Ожидание |
|----------|----------|
| `times_s` | Неубывающий ряд |
| `motion_norm_per_sec_mean` | **≥ 0** для finite (как в core) |
| `missing_frame_ratio`, `flow_consistency_mean` (tabular) | **[0, 1]** |
| `motion_curve_*`, `cam_*` (tabular) | **≥ 0** (finite) |
| `meta.processed_frames` / `meta.total_frames` | `processed_frames` ≤ `total_frames` (когда оба int ≥ 0) |

Полные правила wide/melt: `storage/result_store/view_csv_feature_qa.json` → **`optical_flow`**.

## CSV / melt / RU

- Melt: `view_csv_melt_interesting.json` → **`optical_flow`**
- RU: `view_csv_feature_descriptions_ru.json`

## Валидатор

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/optical_flow/utils/validate_optical_flow_npz.py \
  <path/to/optical_flow.npz> --struct --qa --ranges
```

Батч (схема + struct, без `--qa`):

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/optical_flow/utils/validate_optical_flow_npz.py \
  --results-base /path/to/storage/result_store --platform-id youtube
```

## Сверка с прогоном (пример)

`storage/result_store/youtube/-15jH8mtfJw/25506df0-a75a-4c26-a3f1-79d07c4cb810/optical_flow/optical_flow.npz`  
— ключи v3, `meta.stage_timings_ms`: `frame_manager_ms`, `process_ms`, `total_ms`.

## См. также

- [README.md](../README.md)  
- [SCHEMA.md](SCHEMA.md), [FEATURES_DESCRIPTION.md](FEATURES_DESCRIPTION.md)  
- Machine schema: `DataProcessor/VisualProcessor/schemas/optical_flow_npz_v3.json` (в репозитории)
---

## Навигация

[FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
