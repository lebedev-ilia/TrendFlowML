# Audit 4.2 — engineering log: `frames_composition` (VisualProcessor)

Дата: 2026-04-13  
Компонент: `DataProcessor/VisualProcessor/modules/frames_composition`  
Статус отчёта Audit v4: **L2 (A+B, 5 run)** — см. канонический отчёт ниже.

## Канонический Audit v4 отчёт (эмпирика/контракт)

- Отчёт: [`../../visual_processor/modules/frames_composition_audit_v4.md`](../../visual_processor/modules/frames_composition_audit_v4.md)
- Machine schema: [`DataProcessor/VisualProcessor/schemas/frames_composition_npz_v1.json`](../../../../../VisualProcessor/schemas/frames_composition_npz_v1.json)
- Human schema: [`DataProcessor/VisualProcessor/modules/frames_composition/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/frames_composition/docs/SCHEMA.md)

## Якорь данных (набор A)

- `platform_id/video_id/run_id`: `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Артефакт: `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/frames_composition/frames_composition.npz`
- `manifest.json`: строка компонента `name=frames_composition`: `status=ok`, `schema_version=frames_composition_npz_v1`, `producer_version=2.0.1`

## Статистика (L2, A+B)

- JSON stats (5 run): `storage/audit_v4/frames_composition_l2/frames_composition_audit_v4_stats.json`
- Итог (A+B): суммарно **N_total=543**, стабильные размерности **D=32**, **F=217**, `axis_ok_all=true`.
- `frame_feature_present_ratio`: совпадает с долей finite по столбцам (max_abs_diff ~**2e-8**).
- Video-level избыточность: в top корреляциях ожидаемые пары, напр. `negative_space_ratio__*` vs `object_bbox_coverage_ratio__*` (детерминированная связь по определению метрики).

## Что поменялось / инженерные заметки (4.2)

### 1) Профилирование (env-gated)

- Добавлен best-effort snapshot ресурсов в `meta.resource_profile_before` при `VP_RESOURCE_PROFILE=1`:
  - RSS через `psutil`
  - CUDA max allocated/reserved через `torch.cuda` (если доступно)

### 2) Исправление `stage_timings_ms`

- В `FramesCompositionModule.process()` исправлен баг с переопределением `t0`, из-за которого `stage_timings_ms["total"]` не включал `axis/load_deps` и был неконсистентен.
- Теперь тайминги — **per-stage durations**, а `total` покрывает весь пайплайн.
- Это влияет только на `meta.stage_timings_ms` (наблюдаемость/профилирование) и не меняет численные массивы NPZ.

## Что осталось сделать (следующий шаг)

1. **Набор C (edge)**: `status=empty` (`no_faces_in_video`), кейсы с letterbox/чёрными полосами и экстремальными объектными сценами (много bbox).
2. **Golden (§4.8)**: сигнатура по A (ключи NPZ + N/D/F + NaN ratio по `frame_feature_values` + sanity по `present_ratio`).
---

## Навигация

[Audit v4 hub](../../README.md) · [DataProcessor](../../../../../MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
