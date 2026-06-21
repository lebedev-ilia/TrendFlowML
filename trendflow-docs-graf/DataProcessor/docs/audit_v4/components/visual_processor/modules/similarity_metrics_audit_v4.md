# Audit v4 — `similarity_metrics` (VisualProcessor)

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**, **5** прогонов из `result_store`).  
**Статистика (A+B):** `storage/audit_v4/similarity_metrics_l2/similarity_metrics_audit_v4_stats.json`  
**Артефакты (5 run):** см. `RUN_LOG.md` запись `similarity_metrics` (A+B)  
**Контракт:** [`VisualProcessor/schemas/similarity_metrics_npz_v3.json`](../../../../../VisualProcessor/schemas/similarity_metrics_npz_v3.json) · [`modules/similarity_metrics/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/similarity_metrics/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Hard deps | ✓ | `core_clip` / ось Segmenter ([`SCHEMA.md`](../../../../../VisualProcessor/modules/similarity_metrics/docs/SCHEMA.md)) |
| Optional modalities | ✓ | Reference-агрегаты допускают **NaN** при отсутствии reference / модальности |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `similarity_metrics_npz_v3` | ✓ | Полное совпадение; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** на **A** |
| **N**, оси | ✓ | **N=48**; **`temporal_sim_next`** длина **N−1=47** |
| **`reference_present`** | ✓ | Скаляр **`bool`**, на **A** = **`False`** |
| **F** | ✓ | **`feature_names`** / **`feature_values`**: **F=39** |

#### §4.2 — NaN / Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Покадровые кривые | ✓ | **`centroid_sims`**, **`temporal_sim_next`**, **`times_s`**: **0%** NaN на **A** |
| **`feature_values`** | ◐ | **~61.5%** ячеек NaN (**24/39**): при **`reference_present=False`** заполняются только счётчики/статы по кадрам + флаги **`modality_*_present`**; все **`reference_similarity_*`** и **`uniqueness_*`** — **NaN** (согласуется с [`SCHEMA.md`](../../../../../VisualProcessor/modules/similarity_metrics/docs/SCHEMA.md)) |
| Inf | ✓ | **0** в float-массивах на **A** |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices` | ✓ | Строго возрастают |
| `times_s` | ✓ | Выровнены по **N** |

#### §4.12 — Anti-leakage

| Вопрос | Ответ |
|--------|-------|
| Базовая когерентность из `core_clip` | Да; reference — отдельный контур |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| `meta.models_used` | На **A**: **2** записи (**CLIP** image + text **triton**), не пусто |
| `ui_payload` | Присутствует (**`similarity_metrics_ui_v1`**), полезно для UI-трассировки |

#### §6 — Verdict

**Итог L2:** по 5 run (A+B) схема и NPZ **совпадают**, `manifest.status=ok`, `schema_version=similarity_metrics_npz_v3`, `producer_version=2.0.2`.  
На всех 5 run `reference_present=False`, поэтому reference‑агрегаты в `feature_values` ожидаемо **NaN**. При этом `centroid_sims`/`temporal_sim_next` на всех 5 плотные (см. JSON stats) — downstream обязан маскировать по `reference_present` и NaN.

**Оценка:** **~8.6 / 10** на L2.

#### §8 — DoD

**Не закрыт:** C, §4.6, §4.8; сценарий с **`reference_present=True`** (заполненные reference-поля).

---

## 1. L2 summary (A+B, 5 run)

По агрегатам JSON:

- **N_total**: **543**
- **F**: **39** на всех
- **reference_present**: `False` на всех 5
- **feature_values finite/total**: **75 / 195** (то есть **15/39** на каждый run)

## 2. Снимок **A** (исторический, L1)

| Величина | Значение |
|----------|----------|
| N | 48 |
| F (агрегаты) | 39 |
| `centroid_sims` range (min, max) | ~0.786, ~0.918 |
| `temporal_sim_next` range (min, max) | ~0.651, ~0.997 |
| `feature_values` finite / total | 15 / 39 |
| `meta.models_used` | 2 |
---

## Навигация

[Audit v4 hub](../../audit_4_2/README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
