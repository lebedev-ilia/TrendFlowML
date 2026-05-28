# Audit v4 — `uniqueness` (VisualProcessor)

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**, **5** прогонов из `result_store`).  
**Статистика (A+B):** `storage/audit_v4/uniqueness_l2/uniqueness_audit_v4_stats.json`  
**Артефакты (5 run):** см. `RUN_LOG.md` запись `uniqueness` (A+B)  
**Контракт:** [`VisualProcessor/schemas/uniqueness_npz_v4.json`](../../../../../VisualProcessor/schemas/uniqueness_npz_v4.json) · [`modules/uniqueness/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/uniqueness/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ось и deps | ✓ | Segmenter **`frame_indices`**, **`core_clip`** строго по индексам ([`SCHEMA.md`](../../../../../VisualProcessor/modules/uniqueness/docs/SCHEMA.md)) |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `uniqueness_npz_v4` | ✓ | Полное совпадение; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** на **A** |
| **N**, **N−1** | ✓ | **N=48**; **`cos_dist_next`**: **(47,)** |
| Табличные **F** | ✓ | **`feature_names`** / **`feature_values`**: **F=20** |

#### §4.2 — NaN / Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Последовательности | ✓ | **`max_sim_to_other`**, **`cos_dist_next`**, **`times_s`**: **0%** NaN на **A** |
| **`feature_values`** | ✓ | **0%** NaN на **A** |
| Inf | ✓ | **0** в float-массивах на **A** |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices` | ✓ | Строго возрастают (**0…337**, **48** точек) |

#### §4.1a — Порог и «повторность» (интерпретация на **A**)

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Otsu | ✓ | **`repeat_threshold_is_otsu=1`**, **`repeat_threshold_used` ≈ 0.973** в окне **[0.9, 0.99]** |
| Содержательно | ◐ | Высокие **`max_sim_to_other`** (**~0.93…0.99**) и **`repetition_ratio` ≈ 0.79** на **A** — ролик визуально однороден в CLIP-пространстве (не баг контракта) |
| Вспомогательный порог | ◐ | **`repeat_threshold_quality` ≠ `repeat_threshold_used`** на **A** — второй скаляр отражает иной критерий/качество биннинга; downstream читать имена полей, не смешивать |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| `meta.models_used` | На **A**: **2** записи |

#### §6 — Verdict

**Итог L1:** схема и NPZ **совпадают**, manifest **чистый**, массивы **плотные**; метрики на **A** согласованы с сильной кросс-похожестью кадров в **CLIP**. **~8.5 / 10** на L1.

#### §8 — DoD

**Не закрыт:** C, §4.6, §4.8.

---

## 1. L2 summary (A+B, 5 run)

По агрегатам JSON:

- **N_total**: **467**
- **F**: **20** на всех
- `repeat_threshold_is_otsu=1` на всех
- `repeat_threshold_used` диапазон: **~0.90…0.973**
- `repetition_ratio` диапазон: **~0.792…0.967**
- `effective_unique_frames` диапазон: **4…22**
- `diversity_score` диапазон: **~0.185…0.422**

## 2. Снимок **A** (исторический, L1)

| Величина | Значение |
|----------|----------|
| N | 48 |
| F | 20 |
| `max_sim_to_other` (min, max) | ~0.935, ~0.997 |
| `cos_dist_next` (min, max) | ~0.0029, ~0.349 |
| `repetition_ratio` | ~0.792 |
| `effective_unique_frames` | 10 |
| `diversity_score` | ~0.239 |
