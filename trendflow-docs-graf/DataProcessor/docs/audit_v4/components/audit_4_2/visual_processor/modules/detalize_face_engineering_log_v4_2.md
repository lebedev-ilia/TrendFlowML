# Audit 4.2 — engineering log: `detalize_face` (VisualProcessor)

Дата: 2026-04-13  
Компонент: `DataProcessor/VisualProcessor/modules/detalize_face`  
Статус отчёта Audit v4: **L2 (A+B, 5 run)** — см. канонический отчёт ниже.

## Канонический Audit v4 отчёт (эмпирика/контракт)

- Отчёт: [`../../visual_processor/modules/detalize_face_audit_v4.md`](../../visual_processor/modules/detalize_face_audit_v4.md)
- Machine schema: [`DataProcessor/VisualProcessor/schemas/detalize_face_npz_v3.json`](../../../../../VisualProcessor/schemas/detalize_face_npz_v3.json)
- Human schema: [`DataProcessor/VisualProcessor/modules/detalize_face/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/detalize_face/docs/SCHEMA.md)

## Якорь данных (набор A)

- `platform_id/video_id/run_id`: `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Артефакт: `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/detalize_face/detalize_face.npz`
- `manifest.json`: строка компонента `name=detalize_face`: `status=ok`, `schema_version=detalize_face_npz_v3`, `producer_version=2.0.2`

## Статистика (L2, A+B)

- JSON stats (5 run): `storage/audit_v4/detalize_face_l2/detalize_face_audit_v4_stats.json`
- Итог (A+B): суммарно **N_total=1250**, `primary_valid=True` **73** (**~5.84%**).
- Семантика масок/0-fill подтверждается на A+B:
  - `processed_mask_true_total == primary_valid_true_total == face_present_true_total`
  - `primary_compact_features` имеет **~94.16%** нулевых строк (ожидаемо при редких лицах).
- Опциональные `primary_*` curves на всех 5 run отсутствуют (`write_primary_curves=false`).

## Что поменялось / инженерные заметки (4.2)

### 1) Профилирование (env-gated)

- Добавлен best-effort snapshot ресурсов в `meta.resource_profile_before` при `VP_RESOURCE_PROFILE=1` (RSS через `psutil`).

## Что осталось сделать (следующий шаг)

1. **Набор C (edge)**: кейс `status=empty`, `empty_reason=no_faces_in_video`; а также кейсы частичного покрытия оси (axis содержит кадры вне `core_face_landmarks.frame_indices`).
2. **Golden (§4.8)**: сигнатуры по A (доля `primary_valid`, агрегаты `compact_l2_*`, список ключей NPZ).
3. Зафиксировать/документировать для downstream: **0-fill** в `primary_compact_features` требует строгого использования `primary_valid`/`processed_mask`.
---

## Навигация

[Audit v4 hub](../../README.md) · [DataProcessor](../../../../../MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
