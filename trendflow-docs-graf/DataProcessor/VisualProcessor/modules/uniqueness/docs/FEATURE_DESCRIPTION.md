# `uniqueness` — описание фич и артефактов

**Компонент:** `uniqueness` (intra-video повторяемость / разнообразие по `core_clip`)  
**NPZ:** `uniqueness/uniqueness.npz`  
**Schema:** `uniqueness_npz_v4` — `docs/SCHEMA.md`, `DataProcessor/VisualProcessor/schemas/uniqueness_npz_v4.json`  
**Код:** `utils/uniqueness.py` (`UniquenessModule`, `_FEATURE_NAMES_V1`)

Детали метрик и табличных имён см. **`FEATURES_DESCRIPTION.md`** и `README.md`. Здесь — структура NPZ, meta → CSV, melt / QA.

---

## 1. Назначение

- По сэмплам кадров строится матрица попарных cosine similarity \(O(N^2)\); при **`N > max_frames`** (по умолчанию 200) — **fail-fast**.
- Жёсткая зависимость: **`core_clip/embeddings.npz`** полностью покрывает `uniqueness.frame_indices` (union-domain).

---

## 2. Ключи NPZ

| Группа | Ключи |
|--------|--------|
| Ось | `frame_indices (N,) int32`, `times_s (N,) float32` |
| Последовательности | `max_sim_to_other (N,) float32`, `cos_dist_next (N-1,) float32` (при **N=1** — пустой вектор) |
| Таблица | `feature_names (F,) object`, `feature_values (F,) float32` — фиксированный порядок **`_FEATURE_NAMES_V1`** |
| Контракт | `meta` (в т.ч. `ui_payload` с top repeats / curves) |

---

## 3. Имена табличных фич (_FEATURE_NAMES_V1)

`repeat_threshold_is_otsu`, `repeat_threshold_used`, `repeat_threshold_raw`, `repeat_threshold_quality`, `repeat_threshold_min`, `repeat_threshold_max`, `repeat_threshold_bins`, `max_frames`, `repetition_ratio`, `max_sim_to_other_mean`, `max_sim_to_other_p95`, `pairwise_sim_mean`, `pairwise_sim_p95`, `cos_dist_next_mean`, `cos_dist_next_p95`, `temporal_change_mean`, `diversity_score`, `effective_unique_frames`, `effective_unique_ratio`, `n_frames`.

---

## 4. Meta → wide CSV

`flatten_meta` → `meta_*`:

- Стадии **`stage_timings_ms`**: `frame_manager_ms`, `process_ms`, `save_ms`, `total_ms` → `meta_timing_frame_manager_ms`, `meta_timing_process_ms`, `meta_timing_save_ms`, `meta_timing_total_ms`.
- Конфиг в meta: `repeat_threshold_mode`, `repeat_threshold`, `repeat_threshold_min`, `repeat_threshold_max`, `repeat_threshold_bins`, `ui_topk`, `max_frames`, плюс run identity / `analysis_*` / `dataprocessor_version`.

---

## 5. Melt / QA / RU

- `view_csv_melt_interesting.json` → `uniqueness` (+ `add_all_meta_timing`)
- `view_csv_feature_qa.json` → `uniqueness`
- `view_csv_feature_descriptions_ru.json` — пояснения к `meta_*`

---

## 6. Нормальные диапазоны (`--ranges`)

| Проверка | Ожидание |
|----------|-----------|
| `max_sim_to_other` | Косинусное сходство к другим кадрам, finite **∈ [0, 1]** |
| `cos_dist_next` | `1 − cos(соседи)` = дистанция, finite **∈ [0, 2]** |
| `meta.processed_frames` / `meta.total_frames` (если оба int) | `processed ≤ total` |
| `meta.processed_frames` | **N = len(frame_indices)** (в коде `len(frame_indices)`) |
| `meta.stage_timings_ms` | **≥ 0** |

`meta.status=error`: в `--struct` / батче — краткое сообщение; в `--ranges` — проверки пропускаются.

## 7. Проверка артефакта (single-file)

```bash
cd <repo>
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/uniqueness/utils/validate_uniqueness.py \
  <path/to/uniqueness.npz> --struct --qa --ranges
```

Батч: `--results-base storage/result_store --platform-id youtube` (обход `**/uniqueness/uniqueness.npz`).

Нужны `numpy` и `DataProcessor/qa` на `PYTHONPATH` (как у других `validate_*.py`).
---

## Навигация

[FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
