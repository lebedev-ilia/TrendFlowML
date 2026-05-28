# Audit 4.2 — `scene_classification` engineering log

**Компонент:** `VisualProcessor/modules/scene_classification`  
**Контракт NPZ:** `scene_classification_npz_v2`  
**Связанный L2 отчёт:** `docs/audit_v4/components/visual_processor/modules/scene_classification_audit_v4.md`  
**Статистика (A+B, 5 run):** `storage/audit_v4/scene_classification_l2/scene_classification_audit_v4_stats.json`

## Изменения (после L2 статистики)

### 1) Env-gated resource snapshot (observability)

Добавлен best‑effort снимок ресурсов **до** основного `process()`:

- **Гейт**: `VP_RESOURCE_PROFILE=1|true|yes|on`
- **Что пишем** (если доступно):
  - RSS процесса (`rss_bytes`, `rss_mib`) через `psutil`
  - CUDA max memory (`cuda_max_memory_allocated_bytes`, `cuda_max_memory_reserved_bytes`) через `torch.cuda`
- **Куда пишем**: в `meta` как `resource_profile_before` (через `save_metadata`).

Цель: иметь одинаковый след observability по VisualProcessor модулям (для Audit 4.2) без изменения базового контракта полей.

## Примечания / follow-ups

- `stage_timings_ms.total_ms` уже вычислялся в `run()`; resource snapshot добавлен отдельно и не влияет на расчёты.
- Golden (§4.8) для A: зафиксировать сигнатуру по `N/S/topk`, диапазону сумм top‑5 (`frame_topk_probs`), списку ключей и NaN/Inf счётчикам.

