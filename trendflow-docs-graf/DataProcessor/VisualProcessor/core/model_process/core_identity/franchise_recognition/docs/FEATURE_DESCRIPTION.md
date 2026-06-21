# `franchise_recognition` — описание фич и трассировка артефактов (Audit v3 / v4)

**Компонент:** `franchise_recognition` (semantic head: Embedding Service + `core_clip`)  
**NPZ:** `franchise_recognition/franchise_recognition.npz`  
**Schema:** `franchise_recognition_npz_v2` — `SCHEMA.md`, `VisualProcessor/schemas/franchise_recognition_npz_v2.json`  
**Код:** `main.py` · **render:** `utils/render.py`

Согласованность с обзором: `README.md`.

---

## 1. Назначение

По сэмплированным кадрам (`core_clip.frame_indices`) и эмбеддингам `core_clip/embeddings.npz` выполняется поиск **франшиз/тайтлов** (категория ES **`franchise`**) через **Embedding Service** (HTTP). Результаты маппятся в **канонический label space** (`get_labels()`): пары `semantic_object_ids` (UUID) ↔ int id в `semantic_label_names` (`"id:name"`).

- **K = 5** (контракт semantic head v1), top‑K **не** режется порогами; пороги дают только `frame_is_confident_top1` / `track_is_confident_top1` vs `threshold_global` (и optional per-label в схеме; в v0.2 `threshold_per_label_arr` заполняется **NaN**).
- **OCR** опционален: подсказки/фильтр кандидатов, статистика в `meta` (`ocr_*`).

---

## 2. Тензоры NPZ

| Ключ | dtype / shape | Смысл |
|------|----------------|-------|
| `frame_indices` | int32 `(N)` | = `core_clip.frame_indices` |
| `times_s` | float32 `(N)` | `union_timestamps_sec[frame_indices]` |
| `semantic_label_names` | str `(A)` | `"id:name"`, id стабилен при том же `db_digest` |
| `semantic_object_ids` | str `(A)` | UUID в сторе, строго aligned с именами |
| `threshold_per_label_arr` | float32 `(A)` | Пороги (v0.2: обычно **NaN** везде) |
| `frame_topk_ids` | int32 `(N, 5)` | id из label space; **-1** — нет результата в слоте |
| `frame_topk_scores` | float32 `(N, 5)` | Similarity **0…1** от ES; **NaN** если нет результата |
| `frame_is_confident_top1` | bool `(N)` | top1 ≥ `threshold_global` (и top1 валиден) |
| `track_ids` | int32 `(1)` | `[0]` — псевдотрек «весь ролик» |
| `track_present_mask` | bool `(1)` | `[True]` |
| `track_topk_ids` | int32 `(1, 5)` | Видео-агрегат: top‑K франшиз по **max** similarity по времени |
| `track_topk_scores` | float32 `(1, 5)` | Соответствующие max |
| `track_is_confident_top1` | bool `(1)` | Уверенность по треку (top1 трека) |
| `track_topk_evidence_frame_indices` | int32 `(1, 5)` | Union `frame_index` кадра, где для данной top‑K франшизы максимум similarity (или -1) |
| `meta` / `meta_json` | object / str | Провенанс, тайминги, ES URL, OCR, `models_used` |

**Диапазон scores:** в коде пишутся как `similarity` из ответа ES; для QA в melt считаем **0…1** (см. `validate_franchise_recognition.py`).

---

## 3. Meta (wide CSV: префикс `meta_`)

`stage_timings_ms` в NPZ в **мс**; после `flatten_meta` → `meta_timing_initialization`, `meta_timing_load_deps`, `meta_timing_process_frames`, `meta_timing_saving`, `meta_timing_total`.

Характерные поля: `embedding_service_url`, `franchise_category`, `topk` (=5), `similarity_threshold`, `threshold_global`, `num_franchises`, `num_frames`, `franchises_found_count`, `db_name` / `db_version` / `db_digest`, `embedding_model`, `core_clip_model_signature`, опционально `ocr_*`.

Вложенные dict/list в meta могут **не** попасть в плоский CSV (ограничения `flatten_meta`).

---

## 4. Melt / QA

- **Интересные колонки:** `view_csv_melt_interesting.json` → `franchise_recognition`.
- **Диапазоны и подсветка:** `storage/result_store/view_csv_feature_qa.json` → `franchise_recognition`.
- **Русские подписи:** `view_csv_feature_descriptions_ru.json`.

---

## 5. Проверка артефакта

```bash
python3 .../franchise_recognition/utils/validate_franchise_recognition.py /path/to/franchise_recognition.npz --struct
python3 .../validate_franchise_recognition.py /path/to/franchise_recognition.npz --qa
```

---

## 6. Сверка с README

Перечисление полей и контракт K=5 / ES required должны совпадать с `README.md`; при расхождении приоритет **код** (`main.py`) и **`SCHEMA.md`**.
---

## Навигация

[SCHEMA](SCHEMA.md) · [Module README](../README.md) · [VisualProcessor](../../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
