# Audit 4.2 — engineering log: `high_level_semantic` (VisualProcessor)

Дата: 2026-04-13  
Компонент: `DataProcessor/VisualProcessor/modules/high_level_semantic`  
Статус отчёта Audit v4: **L2 (A+B, 5 run)** — см. канонический отчёт ниже.

## Канонический Audit v4 отчёт (эмпирика/контракт)

- Отчёт: [`../../visual_processor/modules/high_level_semantic_audit_v4.md`](../../visual_processor/modules/high_level_semantic_audit_v4.md)
- Machine schema: [`DataProcessor/VisualProcessor/schemas/high_level_semantic_npz_v2.json`](../../../../../VisualProcessor/schemas/high_level_semantic_npz_v2.json)
- Human schema: [`DataProcessor/VisualProcessor/modules/high_level_semantic/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/high_level_semantic/docs/SCHEMA.md)

## Якорь данных (набор A)

- `platform_id/video_id/run_id`: `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Артефакт: `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/high_level_semantic/high_level_semantic.npz`
- `manifest.json`: строка компонента `name=high_level_semantic`: `status=ok`, `schema_version=high_level_semantic_npz_v2`, `producer_version=2.0.2`

## Статистика (L2, A+B)

- JSON stats (5 run): `storage/audit_v4/high_level_semantic_l2/high_level_semantic_audit_v4_stats.json`
- Итог (A+B): суммарно **N_total=543**, стабильные **D=512**, **F=8**; количество сцен **S** варьирует (**2…8**) по видео.
- `frame_feature_present_ratio` совпадает с долей finite (max_abs_diff ~**1.7e‑8**).
- Опциональные модальности:
  - `loudness_dbfs`, `tempo_bpm` — **100% NaN** на всех 5 run (ожидаемо при отключённых require‑флагах / отсутствии аудио‑артефактов).
  - `emo_*` может быть **100% NaN** на части видео (зависит от присутствия лиц / `emotion_face`).
  - `text_feature_*`: на части run **T=0** (нет `text_processor/text_features.npz` → модуль корректно пишет пустые массивы).

## Что поменялось / инженерные заметки (4.2)

### 1) Профилирование (env-gated)

- Добавлен best-effort snapshot ресурсов в `meta.resource_profile_before` при `VP_RESOURCE_PROFILE=1`:
  - RSS через `psutil`
  - CUDA max allocated/reserved через `torch.cuda` (если доступно)

### 2) Закрытие файловых хендлов при чтении NPZ

- В `_load_npz()` добавлено корректное `close()` для `np.load(...)` (избегаем утечек дескрипторов при множественных загрузках upstream артефактов).

## Что осталось сделать (следующий шаг)

1. **Набор C (edge)**: video без `cut_detection_model_facing`, кейсы с большим числом сцен/ивентов, и кейсы с включёнными require‑флагами для аудио/текста (чтобы проверить заполнение `loudness_dbfs/tempo_bpm` и `text_feature_*`).
2. **Golden (§4.8)**: сигнатуры по A (N/S/F/T + NaN ratios + event_type_counts + sanity для ‖scene_embeddings‖₂≈1).

