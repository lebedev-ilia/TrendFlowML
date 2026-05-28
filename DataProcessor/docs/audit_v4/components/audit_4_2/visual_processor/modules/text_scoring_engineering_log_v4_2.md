# Audit 4.2 — `text_scoring` engineering log

**Компонент:** `VisualProcessor/modules/text_scoring`  
**Контракт NPZ:** `text_scoring_npz_v2`  
**Связанный L2 отчёт:** `docs/audit_v4/components/visual_processor/modules/text_scoring_audit_v4.md`  
**Статистика (A+B, 5 run):** `storage/audit_v4/text_scoring_l2/text_scoring_audit_v4_stats.json`

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

- На A+B `ocr_raw`/`ocr_unique_elements` пустые на всех 5 run (privacy defaults). Для edge‑набора C нужно проверить режим `store_debug_objects=true` (и отдельно `retain_raw_ocr_text=true` в безопасной среде).

