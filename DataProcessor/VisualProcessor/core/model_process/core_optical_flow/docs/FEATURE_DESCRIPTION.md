# `core_optical_flow` — описание фич (Audit v3)

**Компонент:** `core_optical_flow` (VisualProcessor **core** provider)  
**producer** в `meta`: `core_optical_flow`  
**producer_version (код `main.py`):** `2.2`  
**schema_version NPZ:** `core_optical_flow_npz_v3`  
**Артефакт:** `result_store/<platform>/<video>/<run>/core_optical_flow/flow.npz` — [README.md](../README.md)

## Назначение

**RAFT/ONNX** через **Triton** (runtime только `triton`, без fallback): плотный оптический поток по семплированным кадрам (`metadata["core_optical_flow"]["frame_indices"]`, union timestamps). Пишется кривая `motion_norm_per_sec_mean`, компактные per-frame статистики flow/камеры (audit v3) и **preview** теплокарт для отладки.

## Ось **N** (семпл кадров)

Одинаковая длина: `frame_indices`, `times_s`, `dt_seconds`, `motion_norm_per_sec_mean`, все `flow_*` / `cam_*` / `bg_ratio` — см. `docs/SCHEMA.md` (per-frame, float32, NaN по политике первого кадра/ошибок).

## Preview (отладка)

`preview_k` пар, массивы `preview_pair_pos`, `preview_prev/cur_frame_indices`, `preview_prev/cur_times_s`, `preview_flow_mag_map_norm` (K, H, W) — K и размеры в `meta.preview_map_size`.

## Meta (сводка)

- `backend_proxy_version` (`core_optical_flow_backend_proxy_v1`), `preview_k`, `preview_map_size`  
- Triton: `triton_model_spec`, `triton_model_name` (при ModelManager)  
- `stage_timings_ms`: **обычно** `initialization`, `flow_inference_total`, `saving`, `total` (сек → мс; ключи в плоском CSV: `meta_timing_initialization`, `meta_timing_flow_inference_total`, …)  

## Полный перечень ключей NPZ (v3)

См. [SCHEMA.md](SCHEMA.md); кратко по осям:

| Ключ | Shape | Смысл |
|------|-------|--------|
| `frame_indices`, `times_s` | (N) | Семпл кадров и время по `union_timestamps_sec` |
| `dt_seconds` | (N) | **dt[0]=NaN**; с индекса 1 — `max(diff(times_s), ε)` |
| `motion_norm_per_sec_mean` | (N) | Средняя ‖flow‖ / dt / max(H,W); **индекс 0 = 0**, далее по парам |
| `flow_mag_*_per_sec_norm`, `flow_dx/dy_mean_per_sec_norm` | (N) | Компактные статистики потока (норм./с); **индекс 0 = NaN** |
| `flow_dir_*`, `flow_dir_dispersion` | (N) | Направление (взвеш. sin/cos), рассеивание **0…1** |
| `flow_div_abs_mean`, `flow_consistency` | (N) | Прокси дивергенции **≥0** и **1/(1+div)** |
| `cam_*` | (N) | Affine-прокси камеры (scale **≥0**, ty/tx норм./с, shake, …) |
| `bg_ratio` | (N) | Доля низкодвижущихся пикселей **0…1** |
| `preview_*`, `preview_flow_mag_map_norm` | (K), (K,H,W) | Отладочные пары и теплокарты **0…1** |
| `meta` | dict | `producer`, `producer_version`, `backend_proxy_version`, `preview_k`, `preview_map_size`, `triton_*`, `stage_timings_ms`, … |

## Нормальные диапазоны (QA / флаг `--ranges`)

Валидатор проверяет (только **finite**, кроме явных исключений):

| Поле / группа | Ожидание |
|---------------|----------|
| `bg_ratio`, `flow_dir_dispersion`, `flow_consistency` | **∈ [0, 1]** |
| `flow_dir_sin_mean`, `flow_dir_cos_mean` | **∈ [−1, 1]** |
| `flow_div_abs_mean` | **≥ 0** |
| `flow_consistency` vs `flow_div_abs_mean` | **consistency ≈ 1/(1+div)** (допуск по float) |
| `cam_affine_scale` | **≥ 0** (finite) |
| `times_s` | Неубывающий ряд |
| `len(frame_indices)` vs `meta.total_frames` | **N ≤ total_frames** (семпл ⊆ union) |
| `dt_seconds` | `[0]` = **NaN**; на `1..N-1` при **finite** — **> 0** |
| `preview_flow_mag_map_norm` | **∈ [0, 1]** |
| `motion_norm_per_sec_mean`, `flow_mag_*_per_sec_norm` | **≥ 0** там, где finite |
| `meta.preview_k` | Совпадает с `len(preview_pair_pos)` |

- QA / melt-HTML: `storage/result_store/view_csv_feature_qa.json` → **`core_optical_flow`**.
- Melt: `view_csv_melt_interesting.json` → **`core_optical_flow`** (`add_all_meta_timing: true`).
- RU: `view_csv_feature_descriptions_ru.json` (общие `meta_triton_*`, `meta_preview_k`, …).

## Валидатор

Из корня репозитория (нужен `numpy` — удобно venv VisualProcessor):

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/core/model_process/core_optical_flow/utils/validate_core_optical_flow_npz.py \
  <path/to/flow.npz> --struct --qa --ranges
```

Батч (схема + struct):

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/core/model_process/core_optical_flow/utils/validate_core_optical_flow_npz.py \
  --results-base storage/result_store --platform-id youtube
```

## Сверка с прогоном (пример)

Проверено:  
`storage/result_store/youtube/-15jH8mtfJw/25506df0-a75a-4c26-a3f1-79d07c4cb810/core_optical_flow/flow.npz`  
— ключи v3 и `meta.stage_timings_ms`: `initialization`, `flow_inference_total`, `saving`, `total` (мс).

## Потребители

`optical_flow` (модуль) и другие компоненты читают этот NPZ; контракт по ключам v3 — жёсткий.

## Схема

[SCHEMA.md](SCHEMA.md), [README.md](../README.md), machine schema: `DataProcessor/VisualProcessor/schemas/core_optical_flow_npz_v3.json` (в репозитории).
