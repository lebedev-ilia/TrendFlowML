# `action_recognition` — описание фич и артефактов

**Компонент:** `action_recognition` (SlowFast, per-track эмбеддинги и метрики динамики)  
**NPZ (baseline):** `action_recognition/action_recognition_features.npz` (в старых прогонах встречается `action_recognition_emb.npz`)  
**Schema:** `action_recognition_npz_v2` — `docs/SCHEMA.md`  
**Код:** `utils/action_recognition_slowfast.py` (`SlowFastActionRecognizer`, `SCHEMA_VERSION`, `ARTIFACT_FILENAME`)

Семантика фич и треков — в **`FEATURES_DESCRIPTION.md`**. Ниже — структура артефакта, meta → CSV / melt / QA.

---

## 1. Назначение и зависимости

- Сегментация треков person из **`core_object_detections`**, кадры — строго по `frame_indices` сегментера.
- **Пустой результат** валиден: `status="empty"`, `empty_reason` (часто `no_person_detections`), пустые `tracks` / `embeddings` / `results_json`.

---

## 2. Ключи NPZ

| Группа | Ключи |
|--------|--------|
| Per-track | `tracks (T,) int32` — ID треков |
| Per-track | `embeddings (T,) object` — на трек: `[num_clips, 256]` float32, L2 по строкам |
| Per-track | `results_json (T,) object` — словари метрик/отладки (см. `SCHEMA.md`) |
| Контракт | `meta` (run identity, `clip_len` / `stride` / `batch_size`, `stage_timings_ms`, `ui_payload`, …) |

---

## 3. `results_json` (кратко)

Обязательные аналитические поля включают: `max_temporal_jump`, `mean_temporal_jump`, `stability`, `stability_centroid_dist`, `num_switches`, `num_clips`, `track_frame_count`, вложенный `embedding_normed_256d`, а также debug-поля (`clip_center_frame_indices`, `temporal_jumps`, …) — полная таблица в **`SCHEMA.md`**.

---

## 4. Meta → wide CSV

`flatten_meta` → `meta_*`; стадии `stage_timings_ms` → `meta_timing_*` (имена стадий: `initialization`, `load_deps`, `process`, `post_process`, `save`, `total` — значения в **мс**).

---

## 5. Melt / QA / RU

- `view_csv_melt_interesting.json` → `action_recognition` (+ `add_all_meta_timing`)
- `view_csv_feature_qa.json` → `action_recognition`
- `view_csv_feature_descriptions_ru.json` — пояснения к `meta_*`

---

## 6. Проверка артефакта

```bash
python3 DataProcessor/VisualProcessor/modules/action_recognition/utils/validate_action_recognition.py \
  /path/to/action_recognition_features.npz --struct
python3 DataProcessor/VisualProcessor/modules/action_recognition/utils/validate_action_recognition.py \
  /path/to/action_recognition_features.npz --qa
```

Полный отчёт (как в старом CLI, JSON/метрики): добавьте **`--legacy`** (или **`--verbose`** вместе с legacy — см. `--help`).

Нужны `numpy` и `DataProcessor/qa` на `PYTHONPATH` для `--qa`.
---

## Навигация

[FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
