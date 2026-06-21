# Audit 4.2 — engineering log: `cut_detection` (VisualProcessor)

Дата: 2026-04-13  
Компонент: `DataProcessor/VisualProcessor/modules/cut_detection`  
Статус отчёта Audit v4: **L2 (A+B, 5 run)** — см. канонический отчёт ниже.

## Канонический Audit v4 отчёт (эмпирика/контракт)

- Отчёт: [`../../visual_processor/modules/cut_detection_audit_v4.md`](../../visual_processor/modules/cut_detection_audit_v4.md)
- Machine schemas:
  - `cut_detection_npz_v1`: [`DataProcessor/VisualProcessor/schemas/cut_detection_npz_v1.json`](../../../../../VisualProcessor/schemas/cut_detection_npz_v1.json)
  - `cut_detection_model_facing_npz_v1`: [`DataProcessor/VisualProcessor/schemas/cut_detection_model_facing_npz_v1.json`](../../../../../VisualProcessor/schemas/cut_detection_model_facing_npz_v1.json)
- Human schemas:
  - [`DataProcessor/VisualProcessor/modules/cut_detection/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/cut_detection/docs/SCHEMA.md)
  - [`DataProcessor/VisualProcessor/modules/cut_detection/docs/SCHEMA_MODEL_FACING.md`](../../../../../VisualProcessor/modules/cut_detection/docs/SCHEMA_MODEL_FACING.md)

## Якорь данных (набор A)

- `platform_id/video_id/run_id`: `youtube / -Q6fnPIybEI / e2bc964f-1983-4075-a523-1a6cd0cf0759`
- Артефакты (динамические имена; см. `manifest.json`):
  - `cut_detection/cut_detection_features_*.npz` (`cut_detection_npz_v1`)
  - `cut_detection/cut_detection_model_facing_*.npz` (`cut_detection_model_facing_npz_v1`)
- В `features` NPZ присутствует `model_facing_npz_path` (фактически абсолютный путь) → связывает два файла.

## Статистика (L2, A+B)

- JSON stats (5 run): `storage/audit_v4/cut_detection_l2/cut_detection_audit_v4_stats.json`
- Итог (A+B): **N_total=543**, **pairs_total=538**, **E_total=53**
  - `deep_valid_ratio_mean=0.0` (ветка deep не активна)
  - `ssim_valid_ratio_mean≈0.254`, `flow_valid_ratio_mean=1.0`

## Что поменялось / инженерные заметки (4.2)

### 1) Профилирование (env-gated)

- Добавлен best-effort snapshot ресурсов в `meta.resource_profile_before` при `VP_RESOURCE_PROFILE=1`:
  - RSS через `psutil`
  - CUDA max allocated/reserved при `device=cuda` (best-effort)
- Snapshot прокидывается в meta **обоих** NPZ (features + model-facing).

### 2) Микро-оптимизации inference-path

- `torch.no_grad()` заменён на `torch.inference_mode()` (если доступен) для deep/semantic веток.

## Что осталось сделать (следующий шаг)

1. **Набор C (edge)**: ошибки deps (нет `core_optical_flow` при baseline-require), невалидные timestamps, gap-check (`max_sampling_gap_sec`) триггерится.
2. **Golden (§4.8)**: выбрать формат фиксации “динамических имён” (рекомендуется: брать пути из `manifest.json` и хранить в golden JSON).
3. Опционально: стабилизировать имена артефактов (или стандартизировать `manifest`/reference lookup), чтобы regression не зависел от timestamp-хэша.
---

## Навигация

[Audit v4 hub](../../README.md) · [DataProcessor](../../../../../MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
