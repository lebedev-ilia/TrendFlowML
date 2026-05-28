# Audit 4.2 — `shot_quality` engineering log

**Компонент:** `VisualProcessor/modules/shot_quality`  
**Контракт NPZ:** `shot_quality_npz_v3`  
**Связанный L2 отчёт:** `docs/audit_v4/components/visual_processor/modules/shot_quality_audit_v4.md`  
**Статистика (A+B, 5 run):** `storage/audit_v4/shot_quality_l2/shot_quality_audit_v4_stats.json`

## Изменения (после L2 статистики)

### 1) Env-gated resource snapshot (observability)

Добавлен best‑effort снимок ресурсов **до** основного `process()`:

- **Гейт**: `VP_RESOURCE_PROFILE=1|true|yes|on`
- **Что пишем** (если доступно):
  - RSS процесса (`rss_bytes`, `rss_mib`) через `psutil`
  - CUDA max memory (`cuda_max_memory_allocated_bytes`, `cuda_max_memory_reserved_bytes`) через `torch.cuda`
- **Куда пишем**: `meta.resource_profile_before` (через `save_metadata`).

Цель: стандартизировать observability по VisualProcessor модулям (Audit 4.2) без изменения базовой математики/выходов NPZ.

## Примечания / follow-ups

- На A+B стабильно повторяются 4 фичи, которые **полностью NaN** (`vignetting_level`, `chromatic_aberration_level`, `lens_sharpness_drop_off`, `rolling_shutter_artifacts_score`) — это выглядит как «выключенные оценщики/заглушки». На L3 стоит либо:
  - явно закрепить это как политику (ожидаемо всегда NaN), либо
  - включить расчёт, если планировалось.
- Golden (§4.8) для A: зафиксировать сигнатуру по `N/S/F/P/K`, диапазону сумм `quality_probs` (≈1), диапазону сумм `shot_quality_topk_probs` (не 1), и списку fully‑NaN фич.

