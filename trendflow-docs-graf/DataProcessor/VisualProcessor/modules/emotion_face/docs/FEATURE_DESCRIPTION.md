# `emotion_face` — описание фич и артефакта

**Компонент:** `emotion_face` (EmoNet, валентность/активация, keyframes)  
**NPZ:** `emotion_face/emotion_face.npz`  
**Schema:** `emotion_face_npz_v3` — `docs/SCHEMA.md`, `DataProcessor/VisualProcessor/schemas/emotion_face_npz_v3.json`  
**Код:** `core/video_processor.py` (`VERSION` / `SCHEMA_VERSION` / `ARTIFACT_FILENAME`)

Семантика сигналов и сэмплинга — в **`FEATURES_DESCRIPTION.md`**. Ниже — ключи NPZ, meta → CSV / melt / QA.

---

## 1. Зависимости и ось

- **Hard dep:** `core_face_landmarks/landmarks.npz` (no-fallback).  
- **Время:** `times_s` = `union_timestamps_sec[frame_indices]`.  
- **Ось:** `metadata[emotion_face].frame_indices` (при отсутствии — legacy fallback на ось `core_face_landmarks`, см. `meta.module_sampling_policy_version`).

---

## 2. Ключи NPZ (сводно)

| Группа | Ключи |
|--------|--------|
| Ось N | `frame_indices`, `times_s`, `face_present`, `processed_mask`, `face_count`, `valence`, `arousal`, `intensity`, `emotion_confidence`, `dominant_emotion_id` — длина **N**; `emotion_probs` **(N, 8)** |
| Debug / UI | `sequence_features` (object), `keyframes`, `summary`, `features`, `advanced_features` |
| Опц. | `axis_source` (object) — источник оси |
| Контракт | `meta` |

Подробно и про multi-face `*_faces` в **`SCHEMA.md`**.

---

## 3. Meta → wide CSV

`flatten_meta` → `meta_*`; `stage_timings_ms` → `meta_timing_*` (мс): `load_deps`, `select_frames`, `process_frames`, `save`, `total`.

Highlights: `face_frame_stride`, `max_frames`, `max_faces_per_frame`, пороги keyframes, `emonet_model_spec`, `device`, флаги microexpressions / individuality / asymmetry, `module_sampling_policy_version`, `face_frames_sampling_policy_version`.

---

## 4. Melt / QA / RU

- `view_csv_melt_interesting.json` → `emotion_face` (`add_all_meta_timing`)
- `view_csv_feature_qa.json` → `emotion_face`
- `view_csv_feature_descriptions_ru.json` — пояснения к `meta_*`

---

## 5. Проверка

```bash
python3 DataProcessor/VisualProcessor/modules/emotion_face/utils/validate_emotion_face.py \
  /path/to/emotion_face.npz --struct
python3 DataProcessor/VisualProcessor/modules/emotion_face/utils/validate_emotion_face.py \
  /path/to/emotion_face.npz --qa
```

Батч (каталог прогонов, прежнее поведение):

```bash
python3 DataProcessor/VisualProcessor/modules/emotion_face/utils/validate_emotion_face.py \
  --results-base /path/to/dp_results --platform-id youtube
```

Нужны `numpy` и `DataProcessor/qa` на `PYTHONPATH` для `--qa`.
---

## Навигация

[FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
