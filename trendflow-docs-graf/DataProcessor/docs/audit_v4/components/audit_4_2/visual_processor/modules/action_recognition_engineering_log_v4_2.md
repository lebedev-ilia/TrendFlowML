# Audit 4.2 — engineering log: `action_recognition` (VisualProcessor)

Дата: 2026-04-13  
Компонент: `DataProcessor/VisualProcessor/modules/action_recognition`  
Статус отчёта Audit v4: **L2 (A+B, 5 run)** — см. канонический отчёт ниже.

## Канонический Audit v4 отчёт (эмпирика/контракт)

- Отчёт: [`../../visual_processor/modules/action_recognition_audit_v4.md`](../../visual_processor/modules/action_recognition_audit_v4.md)
- Machine schema: [`DataProcessor/VisualProcessor/schemas/action_recognition_npz_v2.json`](../../../../../VisualProcessor/schemas/action_recognition_npz_v2.json)
- Human schema: [`DataProcessor/VisualProcessor/modules/action_recognition/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/action_recognition/docs/SCHEMA.md)

## Якорь данных (набор A)

- `platform_id/video_id/run_id`: `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Артефакт: `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/action_recognition/action_recognition_features.npz`
- `manifest.json`: `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/manifest.json` (строка компонента `action_recognition`: `status=ok`, `schema_version=action_recognition_npz_v2`, `producer_version=2.0`)

## Статистика (L2, A+B)

- JSON stats (5 run): `storage/audit_v4/action_recognition_l2/action_recognition_audit_v4_stats.json`
- Наблюдение: на текущем наборе A+B **все треки имеют `num_clips=1`** → метрики динамики (`temporal_jumps`, `stability`, `num_switches`) остаются вырожденными; нужен B‑поднабор с `num_clips>1`.

## Что поменялось / инженерные заметки (4.2)

### 1) Валидация артефакта и `metric__*`

- **Проблема (исторически)**: лишние/не-скалярные поля попадали в плоские `metric__*` и ломали схему; списочные поля могли превращаться в «-1».
- **Текущее состояние**:
  - `ResultsStore.store_compressed()` экспортирует в `metric__*` **только скалярные** per-track значения.
  - Списки/вложенные структуры живут только в `results_json`.
  - Схема `action_recognition_npz_v2` разрешает extra-ключи по префиксу `metric__`.

### 2) Batch-ветка (GPU batching)

- `VisualProcessor/utils/action_recognition_batch.py` и `VisualProcessor/utils/emotion_face_batch.py` используют `ResultsStore.get_component_path()`.
- **Важно**: API `ResultsStore` должен содержать этот метод, иначе batch-ветка нерабочая (вызов упадёт до обработки).

### 3) Профилирование (env-gated)

- В `action_recognition` добавлен лёгкий snapshot ресурсов в `meta.resource_profile_before` при `VP_RESOURCE_PROFILE=1` (best-effort: RSS, CUDA max allocated/reserved).

## Что осталось сделать (без запусков — только план)

1. **Набор B (L2)**: подобрать ≥5 видео, где внутри треков есть `num_clips > 1`, чтобы оценить `temporal_jumps`, `stability`, `num_switches` на невырожденной временной оси.
2. **Набор C (edge)**: `no_person_detections`, низкий confidence, короткие треки/клипы.
3. **Golden (§4.8)**: определить формат «golden stats» для A (минимум: список ключей NPZ + агрегаты по `metric__*` и проверки форм/норм эмбеддингов).
---

## Навигация

[Audit v4 hub](../../README.md) · [DataProcessor](../../../../../MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
