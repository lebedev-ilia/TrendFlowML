# Аудит соответствия optical_flow требованиям baseline

**Дата проверки**: 2026-01-28  
**Компонент**: `optical_flow` (Visual module, consumer)  
**Расположение**: `VisualProcessor/modules/optical_flow/`

## Резюме

`optical_flow` — baseline-ready consumer, который:
- **не считает RAFT** сам (policy: consumer-only),
- читает единственный источник движения из `core_optical_flow/flow.npz`,
- строго использует `frame_indices` из Segmenter (no-fallback) и строит `times_s` из `union_timestamps_sec`,
- пишет один фиксированный артефакт `optical_flow.npz` со `schema_version="optical_flow_npz_v1"`,
- репортит прогресс в `state_events.jsonl` (unit=frame),
- пишет `summary.stage_timings_ms` (profiling by stages),
- корректно проксирует upstream empty от `core_optical_flow` как `status="empty"` (с тем же `empty_reason`).

## ✅ Соответствие требованиям

### 1) Интерфейсы и per-run storage
- ✅ Наследуется от `BaseModule`
- ✅ `required_dependencies() -> ["core_optical_flow"]`
- ✅ NPZ per-run: `result_store/<platform>/<video>/<run>/optical_flow/optical_flow.npz`
- ✅ Фиксированное имя артефакта (`ARTIFACT_FILENAME`)

### 2) Контракты входа/выхода (no-fallback)
- ✅ `frame_indices` только из `metadata.json[optical_flow.frame_indices]`
- ✅ `times_s = union_timestamps_sec[frame_indices]`
- ✅ если `core_optical_flow` не покрывает indices → `RuntimeError` (no partial)

### 3) Empty semantics
- ✅ Если upstream `core_optical_flow` пустой → `status="empty"`, `empty_reason` проксируется.

### 4) Прогресс + profiling
- ✅ пишет progress-events в `state_events.jsonl`
- ✅ пишет `summary.stage_timings_ms`

## ⚠️ Требует evidence / измерений

### Производительность (measured resource_costs)
- ⚠️ Нужен файл `docs/models_docs/resource_costs/optical_flow_costs_v1.json`:
  - unit: `frame`
  - latency/cpu_rss/gpu_vram (для consumer ожидается near-zero, но фиксируем)

## Ссылки

- `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
- `docs/contracts/SEGMENTER_CONTRACT.md`
- `VisualProcessor/modules/optical_flow/README.md`
- `VisualProcessor/core/model_process/core_optical_flow/README.md`


