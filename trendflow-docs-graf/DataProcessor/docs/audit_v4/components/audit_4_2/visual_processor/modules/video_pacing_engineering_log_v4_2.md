# Audit 4.2 — `video_pacing` engineering log

**Компонент:** `VisualProcessor/modules/video_pacing`  
**Контракт NPZ:** `video_pacing_npz_v3`  
**Связанный L2 отчёт:** `docs/audit_v4/components/visual_processor/modules/video_pacing_audit_v4.md`  
**Статистика (A+B, 5 run):** `storage/audit_v4/video_pacing_l2/video_pacing_audit_v4_stats.json`

## Изменения (после L2 статистики)

### 1) Env-gated resource snapshot (observability)

Добавлен best‑effort снимок ресурсов **до** основного `process()`:

- **Гейт**: `VP_RESOURCE_PROFILE=1|true|yes|on`
- **Что пишем** (если доступно):
  - RSS процесса (`rss_bytes`, `rss_mib`) через `psutil`
  - CUDA max memory (`cuda_max_memory_allocated_bytes`, `cuda_max_memory_reserved_bytes`) через `torch.cuda`
- **Куда пишем**: `meta.resource_profile_before` (через `save_metadata`).

Цель: стандартизировать observability по VisualProcessor модулям (Audit 4.2) без изменения контрактов/выходов NPZ.

## Примечания / follow-ups

- На A+B many tabular поля стабильно NaN из-за выключенных флагов (`enable_entropy_features`, `enable_histograms`, `enable_pace_curve_peaks`, `enable_periodicity`, `enable_bursts`). Для edge‑набора C нужен кейс(ы) с включёнными фичефлагами, чтобы проверить заполнение соответствующих полей.
---

## Навигация

[Audit v4 hub](../../README.md) · [DataProcessor](../../../../../MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
