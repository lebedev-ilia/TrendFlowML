# `brand_semantics` — описание фич и трассировка артефактов (Audit v3 / v4)

**Компонент:** `brand_semantics` (semantic head: детекции → кропы → Embedding Service `brand`)  
**NPZ:** `brand_semantics/brand_semantics.npz`  
**Schema:** `brand_semantics_npz_v2` — `SCHEMA.md`, JSON-схема в `VisualProcessor/schemas/`  
**Код:** `main.py` · **render:** `utils/render.py`

Сверка с обзором: `README.md`.

---

## 1. Назначение

По осям **`core_object_detections`** (bbox, class, tracks, `frame_indices` = shared sampling) строятся кропы логотипов/брендов, для **каждого трека** делается поиск в ES категории **`brand`**, результаты размножаются на per-detection и агрегируются на уровень кадра.

- **K = 5** (контракт), в поиск передаётся `similarity_threshold=0.0` (без отсечения top‑K в ответе ES); **уверенность** — через `confidence_threshold_top1` → `track_is_confident_top1` / `frame_is_confident_top1` / `det_is_confident_top1`.
- Статусы **`empty`**: `no_logo_proposals` (нет подходящих детекций по `proposal_classes`) или `no_valid_crops` (нет валидных кропов при выставленных треках).

---

## 2. Оси и формы

Обозначения: **N** — число кадров в семпле (= `len(frame_indices)`), **M** — `MAX` детекций на кадр из `detections.npz`, **T** — число треков после группировки (и опционально `max_tracks`), **A** — размер канон. label space ES, **K=5**.

| Ключ | Форма | Смысл |
|------|--------|--------|
| `frame_indices` | `(N,)` int32 | = `core_object_detections` sampling |
| `times_s` | `(N,)` float32 | `union_timestamps_sec[...]` |
| `semantic_label_names` | `(A,)` str | `"id:name"` |
| `semantic_object_ids` | `(A,)` str | UUID в сторе |
| `threshold_per_label_arr` | `(A,)` float32 | обычно NaN |
| `track_ids` | `(T,)` int32 | отсортированные id треков |
| `track_present_mask` | `(T,)` bool | трек обработан ES (есть кроп) |
| `track_topk_ids` / `track_topk_scores` | `(T, K)` | top‑K брендов по **лучшему кропу** трека; similarity **0…1** |
| `track_is_confident_top1` | `(T,)` bool | top1 ≥ `confidence_threshold_top1` |
| `track_best_*` | позиция лучшего кропа, bbox, score, class | для render/QA |
| `det_present_mask` | `(N, M)` bool | детекция получила результат трека |
| `det_topk_ids` / `det_topk_scores` | `(N, M, K)` | копия с трека на детекции трека |
| `det_is_confident_top1` | `(N, M)` bool | |
| `frame_topk_ids` / `frame_topk_scores` | `(N, K)` | агрегация по кадру: лучший score на label |
| `frame_is_confident_top1` | `(N,)` bool | |
| `meta` / `meta_json` | | DB provenance, `brand_category`, `tracks_total` / `tracks_present` / `dets_present`, тайминги, `proposal_classes`, и т.д. |

---

## 3. Meta → wide CSV (`meta_*`)

`stage_timings_ms` в NPZ в **мс** → после `flatten_meta`: `meta_timing_initialization`, `meta_timing_load_deps`, `meta_timing_process_frames`, `meta_timing_saving`, `meta_timing_total`.

Важные поля: `brand_category` → **`meta_brand_category`** (`brand`), `topk` → **`meta_topk`** (=5), `confidence_threshold_top1`, `proposal_classes` (может не плоско уйти), `tracks_total`, `tracks_present`, `dets_present`, `pad_ratio`, `use_sharpness`, `max_tracks`, `max_dets_per_track` (optional), `embedding_service_url`, `db_*`, `embedding_model`.

---

## 4. Melt / QA

- **Интересные колонки:** `view_csv_melt_interesting.json` → `brand_semantics`.
- **Диапазоны:** `view_csv_feature_qa.json` → `brand_semantics`.
- **Подписи RU:** `view_csv_feature_descriptions_ru.json`.

---

## 5. Проверка артефакта

```bash
python3 .../brand_semantics/utils/validate_brand_semantics.py /path/to/brand_semantics.npz --struct
python3 .../validate_brand_semantics.py /path/to/brand_semantics.npz --qa
```

---

## 6. Согласование с README

Источник истины для полей и статусов — **`main.py`** + **`SCHEMA.md`**; `README` — сценарии и ES, при расхождении правит код/схема.
---

## Навигация

[SCHEMA](SCHEMA.md) · [Module README](../README.md) · [VisualProcessor](../../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
