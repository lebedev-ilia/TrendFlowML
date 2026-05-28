# Audit 4.2 — engineering log: `color_light` (VisualProcessor)

Дата: 2026-04-13  
Компонент: `DataProcessor/VisualProcessor/modules/color_light`  
Статус отчёта Audit v4: **L2 (A+B, 5 run)** — см. канонический отчёт ниже.

## Канонический Audit v4 отчёт (эмпирика/контракт)

- Отчёт: [`../../visual_processor/modules/color_light_audit_v4.md`](../../visual_processor/modules/color_light_audit_v4.md)
- Machine schema: [`DataProcessor/VisualProcessor/schemas/color_light_npz_v2.json`](../../../../../VisualProcessor/schemas/color_light_npz_v2.json)
- Human schema: [`DataProcessor/VisualProcessor/modules/color_light/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/color_light/docs/SCHEMA.md)

## Якорь данных (набор A)

- `platform_id/video_id/run_id`: `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Артефакт: `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/color_light/color_light_features.npz`
- `manifest.json`: строка компонента `name=color_light`: `status=ok`, `schema_version=color_light_npz_v2`, `producer_version=2.0.2`

## Статистика (L2, A+B)

- JSON stats (5 run): `storage/audit_v4/color_light_l2/color_light_audit_v4_stats.json`
- Итог (A+B): суммарно **M_total=142**, диапазон **M=18…36** (пересечение Segmenter-оси со сценами).
- `video_features`: стабильно **543** ключа; NaN-ключи стабильны (**7**): `color_distribution_gini`, `nima_mean`, `nima_std`, `laion_mean`, `laion_std`, `cinematic_lighting_score`, `professional_look_score`.

## Что поменялось / инженерные заметки (4.2)

### 1) Профилирование (env-gated)

- Добавлен best-effort snapshot ресурсов в `meta.resource_profile_before` при `VP_RESOURCE_PROFILE=1` (RSS через `psutil`).

### 2) Прогресс-логика (state_events)

- Исправлен расчёт `total`/`progress` для `state_events.jsonl`: ранее знаменатель ошибочно зависел от длины текущей сцены; теперь `total` вычисляется как сумма кадров по всем сценам (после пересечения с Segmenter-осью).
- Это влияет только на наблюдаемость/UX и не меняет численные выходы NPZ.

## Что осталось сделать (следующий шаг)

1. **Набор C (edge)**: `after_filt_empty`, отсутствует `scene_classification`, отсутствуют timestamps.
2. **Golden (§4.8)**: сигнатуры по A (минимум: ключи NPZ + агрегаты по `frame_compact_features` + список NaN-ключей в `video_features`).
3. Уточнить/документировать **missing policy** для NaN в `video_features` (опциональные внешние скореры): либо `*_present` поля как каноническая маска, либо вынос в `meta`/`empty_reason` для “скореры не запускались”.

