# `core_face_landmarks` — что в NPZ и CSV

**Компонент:** `core_face_landmarks`  
**Артефакт:** `core_face_landmarks/landmarks.npz` (`ARTIFACT_FILENAME`)  
**producer** в `meta` NPZ: `core_face_landmarks`  
**producer_version:** `main.VERSION` (сейчас **2.1**)  
**schema_version NPZ:** **`core_face_landmarks_npz_v2`**

## Роль

**MediaPipe** landmarks по выборке кадров: FaceMesh (baseline), опционально pose/hands. Лицо запускается по **person-mask** из `core_object_detections` (окно `person_window_radius`). См. [README.md](../README.md).

## Схема

- Human: [SCHEMA.md](SCHEMA.md), `vp_schema_v1`; machine: `DataProcessor/VisualProcessor/schemas/core_face_landmarks_npz_v2.json` (если есть в репо).

## Ключи NPZ (кратко)

| Группа | Ключи |
|--------|--------|
| Ось | `frame_indices` (N), `times_s` (N) из `union_timestamps_sec` |
| Лицо | `face_landmarks` / `face_landmarks_raw` **(N, F, 468, 3)** float32, `face_present` **(N, F)** bool, `face_mesh_ran` **(N,)** bool |
| Person | `person_present` **(N,)** — кадр в маске person из детекций |
| Флаги | `has_any_face`, `has_any_pose`, `has_any_hands` — **0-d** bool в NPZ |
| Строки причин | `empty_reason`, `face_empty_reason`, `pose_empty_reason`, `hands_empty_reason` (object/str; в т.ч. `skipped_due_to_person_mask_no_person`, `no_faces_in_video`, …) |
| Опц. pose/hands | `pose_*`, `hands_*` — если в прогоне были `--use-pose` / `--use-hands` |
| Legacy top-level | `version`, `created_at`, `model_name`, `total_frames` (дубли/совместимость) |
| Meta | `meta` (dict): `producer` / `producer_version`, `schema_version`, `status`, run identity, `face_mesh_frames_count`, person/temporal-фильтр, `stage_timings_ms`, `model_signature` / `models_used`, … |

**F** — число «слотов» лиц (обычно 1); валидатор проверяет согласованность `face_present` с осями `face_landmarks`.

## `stage_timings_ms` → CSV

`flatten_meta` маппит `stage_timings_ms` в `meta_timing_<ключ>`; ключи с точками (профайлер) сохраняют точку, напр. `inference.face_total` → **`meta_timing_inference.face_total`**.

Типичные верхние этапы (мс): `process_video_total`, `total_total`; плюс этапы вроде `inference.face_total`, `io.frame_load_total`, `postproc.temporal_filter_total`, … (см. прогон в `meta`).

- Melt: `storage/result_store/view_csv_melt_interesting.json` → **`core_face_landmarks`**, `add_all_meta_timing: true`.
- QA: `storage/result_store/view_csv_feature_qa.json` → **`core_face_landmarks`**.
- RU: `storage/result_store/view_csv_feature_descriptions_ru.json` (доп. по колонкам).

## Типичные диапазоны (`--ranges`)

| Проверка | Ожидание |
|----------|----------|
| `has_any_face` | Совпадает с `np.any(face_present)` (при N>0) |
| `face_landmarks` (filtered) | При `face_present=0` — NaN; при `1` — конечные координаты |
| `has_any_pose` / `has_any_hands` | Если в NPZ есть `pose_present` / `hands_present`, совпадает с `np.any(...)` |
| `meta.face_mesh_frames_count` | **∈ [0, N]** (N = `len(frame_indices)`) |
| `len(frame_indices)` vs `meta.total_frames` | **N ≤ total_frames** (семпл — подмножество кадров источника) |
| `times_s` | Неубывающий ряд (union) |

## Валидатор

Из корня репозитория (нужен `numpy` — удобно venv VisualProcessor):

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/core/model_process/core_face_landmarks/utils/validate_core_face_landmarks_npz.py \
  <path/to/landmarks.npz> --struct --qa --ranges
```

Батч по дереву `result_store` (схема + struct):

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/core/model_process/core_face_landmarks/utils/validate_core_face_landmarks_npz.py \
  --results-base storage/result_store --platform-id youtube
```

## Сверка с прогоном (пример)

Проверено:  
`storage/result_store/youtube/-15jH8mtfJw/25506df0-a75a-4c26-a3f1-79d07c4cb810/core_face_landmarks/landmarks.npz`  
— ключи совпадают с контрактом; в `meta.stage_timings_ms` присутствуют, например, `process_video_total`, `total_total`, `inference.face_total`, `io.frame_load_total`, …

## См. также

- [README.md](../README.md) — зависимости и CLI.
- [SCHEMA.md](SCHEMA.md) — обязательные/опциональные поля.
---

## Навигация

[SCHEMA](SCHEMA.md) · [Module README](../README.md) · [VisualProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
