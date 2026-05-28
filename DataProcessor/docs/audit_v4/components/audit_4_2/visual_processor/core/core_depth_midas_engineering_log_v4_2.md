# Audit v4.2 — engineering log: `core_depth_midas`

**Дата:** 2026-04-13  
**Компонент:** `core_depth_midas` (VisualProcessor core)  
**Цель:** закрыть Audit v4 **L2 (A+B)** и добавить наблюдаемость ресурсов без изменения контракта.

## Изменения кода (после L1)

### 1) Env-gated resource profiling (RSS + CUDA)

Добавлено best-effort поле `meta.resource_profile_before`, которое записывается **только** при включении переменной окружения:

- `VP_RESOURCE_PROFILE=1|true|yes|y|on` → в meta появится `resource_profile_before`
- иначе поле отсутствует

Содержимое (best-effort):

- `rss_bytes`, `rss_mib` (через `psutil`)
- `cuda_max_memory_allocated_bytes`, `cuda_max_memory_reserved_bytes` (через `torch.cuda`, если доступно)

Файл:

- `DataProcessor/VisualProcessor/core/model_process/core_depth_midas/main.py`

## L2 статистика (A+B, 5 прогонов)

JSON:

- `storage/audit_v4/core_depth_midas_l2/core_depth_midas_audit_v4_stats.json`

Ключевые итоги:

- `N_total=543`, `H=W=256`, `K=10`
- `depth_maps_norm` строго в диапазоне **[0, 1]** (min=0, max=1), NaN/Inf **0**
- `preview_frame_indices ⊆ frame_indices` на всех 5 run

## Что осталось (DoD)

- Набор **C** (edge) + **§4.8 golden**: TODO (зафиксировать «golden signature» по A и минимальный набор инвариантов).

