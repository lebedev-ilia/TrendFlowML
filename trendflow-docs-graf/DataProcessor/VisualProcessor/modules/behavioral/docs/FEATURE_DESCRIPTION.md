# `behavioral` — описание фич и артефакта

**Компонент:** `behavioral` (анализ поведения по `core_face_landmarks` + жесты / язык тела / прокси стресса)  
**NPZ:** `behavioral/behavioral_features.npz`  
**Schema:** `behavioral_npz_v1` — `docs/SCHEMA.md`, `DataProcessor/VisualProcessor/schemas/behavioral_npz_v1.json`  
**Код:** `utils/behavior_analyzer.py` (класс модуля), `main.py`

Семантика `frame_results`, смысл `seq_*` и агрегатов — в **`FEATURES_DESCRIPTION.md`**. Ниже — структура выхода и meta → wide CSV / melt / QA.

---

## 1. Назначение

- Ось: union-domain `frame_indices (N,)` + `times_s (N,)` = `union_timestamps_sec[frame_indices]`.
- Вход: landmarks из **`core_face_landmarks`**; при пустом upstream (`status=empty` / нет лиц) — **валидный** NPZ с `meta.status=empty`, `landmarks_present=false`, NaN в sequence-рядах (см. `SCHEMA.md`).

---

## 2. Ключи NPZ (сводка)

| Группа | Ключи |
|--------|--------|
| Ось + маски | `frame_indices`, `times_s`, `landmarks_present (N,) bool` |
| Debug / UI | `hand_gestures (N,) object`, `frame_results (N,) object` (при `store_debug_objects=false` — пустые dict/списки, ключи сохраняются) |
| Агрегаты | `aggregated` (object / 0d object array) — video-level dict |
| Sequence | `seq_*` — все поля `float32 (N,)` из `_pack_npz_results`: движения рук/головы/рта, `seq_speech_activity_proxy`, blink/self_touch/fidget, `seq_timestamp_norm` |
| Жесты (soft) | `seq_gesture_prob_<name>` — по одному ряду на тип из `GestureClassifier.gesture_types` (12 ключей) |
| Контракт | `meta` |

Полный перечень соответствует **machine schema** и списку файлов в прогоне (≈ 42 тензора + `meta`).

---

## 3. Meta → wide CSV

`batch_runs_feature_report` — только `flatten_meta(meta)` → `meta_*`.

`stage_timings_ms` (мс) в коде: **`process`**, **`pack`**, **`total_without_save`** → `meta_timing_process`, `meta_timing_pack`, `meta_timing_total_without_save` (сохранение NPZ в BaseModule идёт отдельно; итоговый wall логируется в лог, не обязан дублировать все поля в meta).

Дополнительно в meta: `status`, `empty_reason`, `ui_payload`, `total_frames` / `processed_frames`, `analysis_*`, `dataprocessor_version`, run identity.

---

## 4. Melt / QA / RU

- `view_csv_melt_interesting.json` → `behavioral` (+ `add_all_meta_timing`)
- `view_csv_feature_qa.json` → `behavioral`
- `view_csv_feature_descriptions_ru.json` — пояснения к `meta_*` (часть имён `meta_timing_*` общие с другими модулями; в таблице различает колонка **component**)

---

## 5. Проверка артефакта

```bash
python3 DataProcessor/VisualProcessor/modules/behavioral/utils/validate_behavioral.py \
  /path/to/behavioral_features.npz --struct
python3 DataProcessor/VisualProcessor/modules/behavioral/utils/validate_behavioral.py \
  /path/to/behavioral_features.npz --qa
```

Нужны `numpy` и `DataProcessor/qa` на `PYTHONPATH`.
---

## Навигация

[FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
