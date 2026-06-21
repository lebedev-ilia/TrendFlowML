# Audit 4.2 — engineering log: `optical_flow` (VisualProcessor)

Дата: 2026-04-13  
Компонент: `DataProcessor/VisualProcessor/modules/optical_flow`  
Статус отчёта Audit v4: **L2 (A+B, 5 run)** — см. канонический отчёт ниже.

## Канонический Audit v4 отчёт (эмпирика/контракт)

- Отчёт: [`../../visual_processor/modules/optical_flow_audit_v4.md`](../../visual_processor/modules/optical_flow_audit_v4.md)
- Machine schema: [`DataProcessor/VisualProcessor/schemas/optical_flow_npz_v3.json`](../../../../../VisualProcessor/schemas/optical_flow_npz_v3.json)
- Human schema: [`DataProcessor/VisualProcessor/modules/optical_flow/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/optical_flow/docs/SCHEMA.md)

## Якорь данных (набор A)

- `platform_id/video_id/run_id`: `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Артефакт: `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/optical_flow/optical_flow.npz`
- `manifest.json`: строка компонента `name=optical_flow`: `status=ok`, `schema_version=optical_flow_npz_v3`, `producer_version=2.0.2`

## Статистика (L2, A+B)

- JSON stats (5 run): `storage/audit_v4/optical_flow_l2/optical_flow_audit_v4_stats.json`
- Итог (A+B): суммарно **N_total=1250**, стабильные **D=16**, **F=9**.
- Missing/NaN:
  - `missing_ratio_curve_mean≈0.886`
  - `missing_ratio_matrix_mean≈0.890`
  - разница долей NaN между кривой и матрицей стабильно мала (~**0.00375**) и объяснима (матрица усредняет по 16 колонкам).

## Что поменялось / инженерные заметки (4.2)

### 1) Профилирование (env-gated)

- Добавлен best-effort snapshot ресурсов в `meta.resource_profile_before` при `VP_RESOURCE_PROFILE=1` (RSS через `psutil`, CUDA max allocated/reserved через `torch.cuda`).

## Что осталось сделать (следующий шаг)

1. **Набор C (edge)**: случаи с `core_optical_flow.status=empty` и/или с существенно меньшим missing_ratio (плотный flow).
2. **Golden (§4.8)**: сигнатуры по A (NaN ratio по кривой/матрице + `feature_values` + sanity для `missing_frame_ratio`).
---

## Навигация

[Audit v4 hub](../../README.md) · [DataProcessor](../../../../../MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
