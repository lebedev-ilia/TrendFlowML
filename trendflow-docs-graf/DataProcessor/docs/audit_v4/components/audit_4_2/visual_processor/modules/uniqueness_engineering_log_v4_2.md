# Audit 4.2 — `uniqueness` engineering log

**Компонент:** `VisualProcessor/modules/uniqueness`  
**Контракт NPZ:** `uniqueness_npz_v4`  
**Связанный L2 отчёт:** `docs/audit_v4/components/visual_processor/modules/uniqueness_audit_v4.md`  
**Статистика (A+B, 5 run):** `storage/audit_v4/uniqueness_l2/uniqueness_audit_v4_stats.json`

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

- На A+B `repeat_threshold_is_otsu=1` и `repeat_threshold_used` держится в ~[0.90, 0.99]; вариативность `repetition_ratio` ожидаема (контент-зависимая).
- Для edge‑набора C полезны кейсы с низкой повторяемостью (ожидаемо низкий `repetition_ratio` и высокий `effective_unique_ratio`) и кейс около/выше `max_frames` (fail-fast политика).
---

## Навигация

[Audit v4 hub](../../README.md) · [DataProcessor](../../../../../MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
