# `car_semantics` — описание фич и трассировка артефактов (Audit v3 / v4)

**Компонент:** `car_semantics` (semantic head: детекции → кропы → Embedding Service **`car`**)  
**NPZ:** `car_semantics/car_semantics.npz`  
**Schema:** `car_semantics_npz_v2` — `SCHEMA.md`, JSON-схема в `VisualProcessor/schemas/`  
**Код:** `main.py` · **render:** `utils/render.py`

Сверка с обзором: `README.md`.

---

## 1. Назначение

По **`core_object_detections`** (bbox, class, tracks, `frame_indices`) кропы машин, поиск top‑K в ES категории **`car`**, дублирование на per-det, агрегация на per-frame. Дополнительно в NPZ: **`semantic_label_make`**, **`semantic_label_model`** — best-effort разбор имени (make/model) для дебага/витрины.

- **K = 5** (контракт `TOP_K`); в meta поле **`top_k`** (→ `meta_top_k` в плоском CSV).
- **Категория** в meta: `category` = `"car"` → **`meta_category`**.
- **Уверенность:** `confidence_threshold_top1` → `track_is_confident_top1` / `frame_is_confident_top1` / `det_is_confident_top1` (ES возвращает similarity **0…1** в коде; пороги не режут K).

**Пустой прогон:** `status=empty` с `empty_reason` в духе `no_car_proposals` / `no_valid_crops` (см. `main.py` ветки до `output_meta`).

---

## 2. Оси и хранилище (NPZ)

| Группа | Ключи (см. `SCHEMA.md`) |
|--------|-------------------------|
| Ось кадра | `frame_indices`, `times_s`, `frame_topk_*`, `frame_is_confident_top1` |
| Ось трека | `track_ids`, `track_present_mask`, `track_topk_*`, `track_is_confident_top1`, `track_best_*` |
| Ось N×M дет | `det_present_mask`, `det_topk_*`, `det_is_confident_top1` |
| Label space | `semantic_label_names`, `semantic_object_ids`, `threshold_per_label_arr`, `semantic_label_make`, `semantic_label_model` |
| Provenance | `meta`, `meta_json` |

Формы: **N** кадров, **M** слотов детекций, **T** треков, **A** меток, **K=5**.

---

## 3. Meta → wide CSV

`stage_timings_ms` (мс) → `meta_timing_initialization|load_deps|process_frames|saving|total`.

Доп. поля: `labels_count` → **`meta_labels_count`**, `tracks_total` / `tracks_present` / `dets_present`, `pad_ratio`, `use_sharpness`, `proposal_class_ids` (длинные списки могут **не** войти в `flatten_meta`).

---

## 4. Melt / QA

- `view_csv_melt_interesting.json` → `car_semantics`
- `view_csv_feature_qa.json` → `car_semantics`
- `view_csv_feature_descriptions_ru.json` — пояснения к колонкам

---

## 5. Проверка артефакта

```bash
python3 .../car_semantics/utils/validate_car_semantics.py /path/to/car_semantics.npz --struct
python3 .../validate_car_semantics.py /path/to/car_semantics.npz --qa
```

---

## 6. Согласование с README

Источник истины: **`main.py`**, **`SCHEMA.md`**; `README` описывает базу `known_cars` и sync — не заменяет схему NPZ.
