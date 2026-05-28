# `core_depth_midas` — что в NPZ и CSV

**Компонент:** `core_depth_midas` (VisualProcessor **core** provider)  
**producer** в `meta` NPZ: `core_depth_midas`  
**producer_version (код `main.py`):** `2.2`  
**schema_version NPZ:** `core_depth_midas_npz_v3`  
**Артефакт:** `result_store/<platform>/<video>/<run>/core_depth_midas/depth.npz` (`ARTIFACT_FILENAME`)

## Роль

**Tier-0** карты глубины (семейство MiDaS) по union-domain кадрам: Triton-инференс, относительная глубина, агрегаты per-frame и нормализованные карты для бэкенда.

## Схема

- **schema_version**: `core_depth_midas_npz_v3` (см. [SCHEMA.md](SCHEMA.md), система `vp_schema_v1`; machine schema: `DataProcessor/VisualProcessor/schemas/core_depth_midas_npz_v3.json`).
- **producer**: `core_depth_midas`; **backend_proxy_version**: `core_depth_midas_backend_proxy_v1`.

## Ключи NPZ (кратко)

| Группа | Ключи |
|--------|--------|
| Ось | `frame_indices` (N), `times_s` (N) из `union_timestamps_sec` |
| Карты | `depth_maps` (N,H,W), `depth_maps_norm` (N,H,W) в [0,1] |
| Per-frame | `depth_mean`, `depth_std`, `depth_p05`, `depth_p95`, `depth_range_robust`, `depth_complexity_score`, `foreground_background_separation_proxy` — всё (N,) float32 |
| Preview | `preview_frame_indices` (K), `preview_times_s` (K), `preview_depth_maps` (K,H,W), `preview_depth_maps_norm` (K,H,W); **K** = `meta.preview_k` (до 10) |
| Meta | `meta` (dict): run identity, `models_used` / `model_signature`, `stage_timings_ms`, Triton-поля (`triton_model_spec`, `triton_model_name`, `triton_preprocess_preset`), `out_width` / `out_height`, `batch_size`, `runtime` (`triton-gpu`), `device`, и т.д. |

## CSV / melt

Плоский `meta_*` и `meta_timing_*` приходят из `flatten_meta` (`stage_timings_ms` → `meta_timing_<ключ>`). В `main.py` длительности копятся в **секундах**, в `meta.stage_timings_ms` пишутся **мс**. Типичные ключи: `initialization` (метаданные, `FrameManager`, клиент Triton), `depth_inference_total` (основной цикл батчей), `total` (весь прогон до сохранения).

- Конфиг melt: `storage/result_store/view_csv_melt_interesting.json` → секция `core_depth_midas`.
- QA-диапазоны: `storage/result_store/view_csv_feature_qa.json` → `core_depth_midas`.
- RU-подсказки к колонкам: `storage/result_store/view_csv_feature_descriptions_ru.json` (при необходимости).

## Нормальные диапазоны (флаг `--ranges`)

| Поле / группа | Ожидание (finite) |
|---------------|-------------------|
| `depth_maps_norm`, `preview_depth_maps_norm` | **∈ [0, 1]** (после `clip` в пайпе) |
| `depth_std`, `depth_range_robust` | **≥ 0** |
| `depth_p05`, `depth_p95` | **p05 ≤ p95** |
| `depth_complexity_score` | **∈ [0, 1]** (средние градиенты по норм. карте) |
| `times_s` | **Неубывающий** ряд (union) |
| `len(frame_indices)` vs `meta.total_frames` | **N ≤ total_frames** |
| `meta.preview_k` | **=** `len(preview_frame_indices)` |

`foreground_background_separation_proxy` = `depth_range_robust / (depth_std+ε)` — **без** жёсткого верхнего предела в QA (может быть >1).

## Валидатор

Из корня репозитория (нужен `numpy` — удобно через venv VisualProcessor):

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/core/model_process/core_depth_midas/utils/validate_core_depth_midas_npz.py \
  <path/to/depth.npz> --struct --qa --ranges
```

Батч по дереву `result_store` (только **схема + struct**):

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/core/model_process/core_depth_midas/utils/validate_core_depth_midas_npz.py \
  --results-base storage/result_store --platform-id youtube
```

## Сверка с прогоном (пример)

Проверено: `storage/result_store/youtube/-15jH8mtfJw/25506df0-a75a-4c26-a3f1-79d07c4cb810/core_depth_midas/depth.npz` — ключи и `meta.stage_timings_ms` (`initialization`, `depth_inference_total`, `total`) совпадают с `main.py`.

## См. также

- [README.md](../README.md) — входы, Triton, кадры.
- [SCHEMA.md](SCHEMA.md) — полный перечень полей и meta.
