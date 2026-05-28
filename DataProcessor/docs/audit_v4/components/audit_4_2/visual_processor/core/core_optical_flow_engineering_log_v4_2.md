# Audit v4.2 — engineering log: `core_optical_flow`

**Дата:** 2026-04-13  
**Компонент:** `core_optical_flow` (VisualProcessor core)  
**Цель:** довести отчёт Audit v4 по компоненту до **L2 (A+B)** и добавить наблюдаемость ресурсов без изменения контракта.

## Изменения кода (после L1)

### 1) Env-gated resource profiling (RSS + CUDA)

Добавлено best-effort поле `meta.resource_profile_before`, которое записывается **только** при включении переменной окружения:

- `VP_RESOURCE_PROFILE=1|true|yes|y|on` → в meta появится `resource_profile_before`
- иначе поле отсутствует

Содержимое (best-effort):

- `rss_bytes`, `rss_mib` (через `psutil`)
- `cuda_max_memory_allocated_bytes`, `cuda_max_memory_reserved_bytes` (через `torch.cuda`, если доступно)

Файл:

- `DataProcessor/VisualProcessor/core/model_process/core_optical_flow/main.py`

## L2 статистика (A+B, 5 прогонов)

JSON:

- `storage/audit_v4/core_optical_flow_l2/core_optical_flow_audit_v4_stats.json`

Ключевые итоги (по агрегатам JSON):

- `N_total=543`, `K_set=[10]`, preview map size `[[64,64]]`
- `frame_indices_strict_inc_all=true`, `times_s_monotonic_all=true`
- NaN‑политика для flow‑зависимых рядов подтверждена: **NaN только на idx 0** (`flow_dep_nan_at_0_only_all=true`)
- `motion_norm_per_sec_mean`: `motion0_is_zero_all=true`, NaN отсутствуют (`motion_finite_all=true`)
- `preview_flow_mag_map_norm`: NaN отсутствуют (`preview_nan_total=0`), значения строго в \([0,1]\) (`preview_in_01_all=true`)

## Наблюдения/риски для downstream

- В NPZ есть два «семантических режима» для первого кадра:
  - `motion_norm_per_sec_mean[0]=0.0` (не NaN)
  - `dt_seconds[0]` и все flow/camera/bg ряды на **idx 0** — NaN (нет пары prev→cur)

## Что осталось (DoD)

- Набор **C** (edge) + **§4.8 golden**: TODO (зафиксировать «golden signature» по A и минимальный набор инвариантов).

