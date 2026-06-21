# Audit 4.2 — engineering log: `micro_emotion` (VisualProcessor)

Дата: 2026-04-13  
Компонент: `DataProcessor/VisualProcessor/modules/micro_emotion`  
Статус отчёта Audit v4: **L2 (A+B)** — частично **blocked** (в B один run `status=error`, нет NPZ).

## Канонический Audit v4 отчёт (эмпирика/контракт)

- Отчёт: [`../../visual_processor/modules/micro_emotion_audit_v4.md`](../../visual_processor/modules/micro_emotion_audit_v4.md)
- Machine schema: [`DataProcessor/VisualProcessor/schemas/micro_emotion_npz_v3.json`](../../../../../VisualProcessor/schemas/micro_emotion_npz_v3.json)
- Human schema: [`DataProcessor/VisualProcessor/modules/micro_emotion/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/micro_emotion/docs/SCHEMA.md)

## Якорь данных (набор A)

- `platform_id/video_id/run_id`: `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Артефакт: `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/micro_emotion/micro_emotion.npz`
- `manifest.json`: строка компонента `name=micro_emotion`: `status=ok`, `schema_version=micro_emotion_npz_v3`, `producer_version=2.0.2`

## Статистика (L2, A+B)

- JSON stats (OK артефакты): `storage/audit_v4/micro_emotion_l2/micro_emotion_audit_v4_stats.json`
- Итог по OK NPZ: `n_runs=4`, **N_total=1000**, `face_present_any` True **70** (**7%**), `K_total=2`, `video_feature_values_nan_total=16`.
- `feature_values` (V=75): собраны top‑корреляции (по 4 run) как навигация по избыточности.

## Что случилось с 5‑м run (B)

- `youtube / -Ga4edhrfog / e2dc8851-6c51-43c0-9757-3c0fed803348`: `micro_emotion` в `manifest.json` имеет `status=error` (NPZ не записан).
- Причина (из `manifest.error`): PCA `n_components=3` при `min(n_samples,n_features)=2` (малый объём валидных OpenFace строк после best‑effort фильтров).

## Что поменялось / инженерные заметки (4.2)

### 1) Профилирование (env-gated)

- Добавлен best-effort snapshot ресурсов в `meta.resource_profile_before` при `VP_RESOURCE_PROFILE=1` (RSS через `psutil`, CUDA max allocated/reserved через `torch.cuda`).

### 2) Как лечить падение PCA (план)

- Правильная политика: `n_components = min(pca_components, n_samples, n_features)` и pad до фиксированного размера векторов.
- После восстановления B (≥5 OK) — пересобрать L2 stats и закрыть пункт в `RUN_LOG`.

## Что осталось сделать (следующий шаг)

1. **Добить B**: получить ≥5 OK NPZ (сейчас один run из B error → нет NPZ).
2. **Golden (§4.8)**: сигнатура по A (ключи NPZ + N/F/V + face_present_ratio + NaN counts + `frames_processed_openface`).
3. **Набор C (edge)**: случаи с крайне малым количеством face‑кадров / частичным OpenFace coverage.
---

## Навигация

[Audit v4 hub](../../README.md) · [DataProcessor](../../../../../MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
