# Audit 4.2 — `similarity_metrics` engineering log

**Компонент:** `VisualProcessor/modules/similarity_metrics`  
**Контракт NPZ:** `similarity_metrics_npz_v3`  
**Связанный L2 отчёт:** `docs/audit_v4/components/visual_processor/modules/similarity_metrics_audit_v4.md`  
**Статистика (A+B, 5 run):** `storage/audit_v4/similarity_metrics_l2/similarity_metrics_audit_v4_stats.json`

## Изменения (после L2 статистики)

### 1) Env-gated resource snapshot (observability)

Добавлен best‑effort снимок ресурсов **до** основного `process()`:

- **Гейт**: `VP_RESOURCE_PROFILE=1|true|yes|on`
- **Что пишем** (если доступно):
  - RSS процесса (`rss_bytes`, `rss_mib`) через `psutil`
  - CUDA max memory (`cuda_max_memory_allocated_bytes`, `cuda_max_memory_reserved_bytes`) через `torch.cuda`
- **Куда пишем**: `meta.resource_profile_before` (через `save_metadata`).

Цель: стандартизировать observability по VisualProcessor модулям (Audit 4.2) без изменения контрактов и математики.

## Примечания / follow-ups

- На A+B `reference_present=False` на всех 5 run ⇒ reference‑агрегаты в `feature_values` ожидаемо **NaN**. Для закрытия L3/§4.8 нужен edge‑кейс C с включённым reference set (`reference_present=True`).
---

## Навигация

[Audit v4 hub](../../README.md) · [DataProcessor](../../../../../MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
