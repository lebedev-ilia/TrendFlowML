# Audit 4.2 — engineering log: `emotion_face` (VisualProcessor)

Дата: 2026-04-13  
Компонент: `DataProcessor/VisualProcessor/modules/emotion_face`  
Статус отчёта Audit v4: **L2 (A+B, 5 run)** — см. канонический отчёт ниже.

## Канонический Audit v4 отчёт (эмпирика/контракт)

- Отчёт: [`../../visual_processor/modules/emotion_face_audit_v4.md`](../../visual_processor/modules/emotion_face_audit_v4.md)
- Machine schema: [`DataProcessor/VisualProcessor/schemas/emotion_face_npz_v3.json`](../../../../../VisualProcessor/schemas/emotion_face_npz_v3.json)
- Human schema: [`DataProcessor/VisualProcessor/modules/emotion_face/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/emotion_face/docs/SCHEMA.md)

## Якорь данных (набор A)

- `platform_id/video_id/run_id`: `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Артефакт: `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/emotion_face/emotion_face.npz`
- `manifest.json`: строка компонента `name=emotion_face`: `status=ok`, `schema_version=emotion_face_npz_v3`, `producer_version=2.0.2`

## Статистика (L2, A+B)

- JSON stats (5 run): `storage/audit_v4/emotion_face_l2/emotion_face_audit_v4_stats.json`
- Итог (A+B): **N_total=1000**, `face_present=True` **42** (**4.2%**), `processed_mask=True` **12** (**1.2%**), `keyframes_total=0` (все 5 run).
- Семантика масок подтверждается на A+B: вне `processed_mask` сохраняются **NaN** в VA/probs и `dominant_emotion_id=-1`.

## Что поменялось / инженерные заметки (4.2)

### 1) Профилирование (env-gated)

- Добавлен best-effort snapshot ресурсов в `meta.resource_profile_before` при `VP_RESOURCE_PROFILE=1`:
  - RSS через `psutil`
  - CUDA max allocated/reserved (best-effort) при `device=cuda`

### 2) Микро-оптимизация CPU-path

- Ускорен расчёт `processed_mask` на оси: set `selected_fi` теперь строится один раз (раньше создавался внутри list-comprehension).

## Что осталось сделать (следующий шаг)

1. **Набор C (edge)**: `status=empty`, `empty_reason=no_faces_in_video`; а также кейсы low-quality gating/слишком мало обработанных кадров.
2. **Golden (§4.8)**: сигнатуры по A (доля `processed_mask`, sanity по sum(prob)=~1 на processed, распределения valence/arousal/intensity).
3. Проверить, что `keyframes` не всегда пустой на “богатом” B (с более частыми лицами и изменениями эмоций).
---

## Навигация

[Audit v4 hub](../../README.md) · [DataProcessor](../../../../../MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
