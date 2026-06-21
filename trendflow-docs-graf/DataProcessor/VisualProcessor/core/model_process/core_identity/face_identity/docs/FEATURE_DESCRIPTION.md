# `core_face_identity` — что в NPZ и CSV

**Компонент:** `core_face_identity`  
**producer** в `meta` NPZ: `core_face_identity`  
**producer_version:** см. `main.VERSION` (например `0.2`).

## Роль

Идентификация лиц по **Embedding Service**: кропы из `core_face_landmarks`, поиск похожих в коллекции `face`, top-K имён и сходств на кадр, детерминированное label-space (`semantic_label_names` / UUID). Артефакт: `result_store/.../core_face_identity/face_identity.npz`.

## Схема

- **schema_version**: `core_face_identity_npz_v2` (см. [SCHEMA.md](SCHEMA.md), `vp_schema_v1`).
- **K**: задаётся `--topk` (по умолчанию 5), фиксируется в `meta.top_k` и вторая ось `face_*`.

## Ключи NPZ (кратко)

| Группа | Ключи |
|--------|--------|
| Ось | `frame_indices` (N), `times_s` (N) — только кадры с лицами; N может быть 0 (valid empty) |
| Label space | `semantic_label_names` (A), `semantic_object_ids` (A) — канон. строки `"int:name"` и UUID |
| Per-frame | `face_ids` (N,K) int32 (−1 если нет), `face_names` (N,K), `face_similarities` (N,K) в [0,1] |
| Рендер | `face_bbox_xyxy` (N,4) float32, top-1 бокс; NaN если лица нет |
| Meta | `meta` + **`meta_json`** (строка JSON, тот же dict) |
| Provenance | `meta`: `producer` / `producer_version`, `db_name`, `db_version`, `db_digest`, `embedding_service_url`, `category`, `embedding_model`, `top_k`, `similarity_threshold`, `n_frames`, `total_faces_processed` |

## Типичные диапазоны (`--ranges`)

| Объект | Ожидание |
|--------|----------|
| `face_similarities` | Finite значения в [0, 1] |
| `face_ids` | −1 или индекс в [0, A), A = len(`semantic_label_names`) |
| `meta.top_k` | Совпадает с K (вторая ось `face_ids` / `face_similarities` / `face_names`) |
| `times_s` | **Неубывающий** ряд (кадры с лицами по `union`) |
| `face_bbox_xyxy` | Строка целиком NaN, либо четыре конечных числа с x2≥x1, y2≥y1 |

## CSV / melt

- Плоский `meta_*` и `meta_timing_*` из `flatten_meta` (`stage_timings_ms` → `meta_timing_<этап>`). Типичные этапы: `initialization`, `load_deps`, `process_frames`, `saving`, `total`.
- Melt: `storage/result_store/view_csv_melt_interesting.json` → `core_face_identity`.
- QA: `storage/result_store/view_csv_feature_qa.json` → `core_face_identity`.

## Валидатор

Из каталога модуля:

```bash
python utils/validate_core_face_identity_npz.py <path/to/face_identity.npz> [--struct] [--qa] [--ranges]
```

## См. также

- [README.md](../README.md) — входы, Embedding Service, зависимость от `core_face_landmarks`.
- [SCHEMA.md](SCHEMA.md) — поля и NaN-политика.
---

## Навигация

[SCHEMA](SCHEMA.md) · [Module README](../README.md) · [VisualProcessor](../../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
