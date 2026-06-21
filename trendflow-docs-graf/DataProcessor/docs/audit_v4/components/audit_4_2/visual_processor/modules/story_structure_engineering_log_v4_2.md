# Audit 4.2 — `story_structure` engineering log

**Компонент:** `VisualProcessor/modules/story_structure`  
**Контракт NPZ:** `story_structure_npz_v3`  
**Связанный L2 отчёт:** `docs/audit_v4/components/visual_processor/modules/story_structure_audit_v4.md`  
**Статистика (A+B, 5 run):** `storage/audit_v4/story_structure_l2/story_structure_audit_v4_stats.json`

## Изменения (после L2 статистики)

### 1) Env-gated resource snapshot (observability)

Добавлен best‑effort снимок ресурсов **до** основного `process()`:

- **Гейт**: `VP_RESOURCE_PROFILE=1|true|yes|on`
- **Что пишем** (если доступно):
  - RSS процесса (`rss_bytes`, `rss_mib`) через `psutil`
  - CUDA max memory (`cuda_max_memory_allocated_bytes`, `cuda_max_memory_reserved_bytes`) через `torch.cuda`
- **Куда пишем**: `meta.resource_profile_before` (через `save_metadata`).

### 2) NPZ IO hygiene

В `_load_npz_dict()` добавлено корректное закрытие `np.load(...).close()` через `try/finally` (best‑effort), чтобы не держать открытые file handles при чтении optional deps (OCR).

## Примечания / follow-ups

- На A+B `topic_shift_curve_present=False` на всех 5 run ⇒ текстовая ветка не верифицирована. Для L3 нужен C‑кейс с `topic_shift_curve_present=True`.
- `hook_to_avg_energy_ratio` в табличных фичах может быть экстремальным по модулю; downstream лучше использовать робастные трансформации/клиппинг.
---

## Навигация

[Audit v4 hub](../../README.md) · [DataProcessor](../../../../../MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
