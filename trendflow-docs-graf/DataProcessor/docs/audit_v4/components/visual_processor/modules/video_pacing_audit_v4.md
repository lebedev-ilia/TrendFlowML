# Audit v4 — `video_pacing` (VisualProcessor)

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**, **5** прогонов из `result_store`).  
**Статистика (A+B):** `storage/audit_v4/video_pacing_l2/video_pacing_audit_v4_stats.json`  
**Артефакты (5 run):** см. `RUN_LOG.md` запись `video_pacing` (A+B)  
**Контракт:** [`VisualProcessor/schemas/video_pacing_npz_v3.json`](../../../../../VisualProcessor/schemas/video_pacing_npz_v3.json) · [`modules/video_pacing/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/video_pacing/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Hard deps | ✓ | `cut_detection`, `core_optical_flow`, `core_clip` ([`SCHEMA.md`](../../../../../VisualProcessor/modules/video_pacing/docs/SCHEMA.md)) |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `video_pacing_npz_v3` | ✓ | Полное совпадение; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** на **A** |
| **N**, **S** | ✓ | **N=48**; **`shot_boundary_frame_indices`**: **S=5**, индексы **неубывают** (union **0…337**) |
| **F** | ✓ | **`feature_names`** / **`feature_values`**: **F=57** |

#### §4.2 — NaN / Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Временные ряды | ✓ | **`motion_norm_per_sec_mean`**, **`semantic_change_rate_per_sec`**, **`color_change_rate_per_sec`**, **`times_s`**: **0%** NaN на **A** |
| **`feature_values`** | ◐ | **13/57 ≈ 22.8%** NaN на **A** |
| Отключённые блоки | ✓ | В **`meta`** все **`enable_entropy_features`**, **`enable_histograms`**, **`enable_pace_curve_peaks`**, **`enable_periodicity`**, **`enable_bursts`** = **False** → NaN у **`shot_duration_entropy`**, **`shot_length_gini`**, **`tempo_entropy`**, **`shot_length_histogram_5bins_*`**, **`pace_curve_*`** (периодика), **`semantic_change_burst_count`**, **`color_change_bursts`** |
| Inf | ✓ | **0** в float-массивах на **A** |

#### §4.1a — Табличные аномалии

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **`shot_duration_min`** | ◐ | На **A** = **0** при **`shots_count=5`** — возможен краевой нулевой интервал у границы; иметь в виду при интерпретации длительностей |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices` | ✓ | Строго возрастают |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| `meta.models_used` | На **A**: **пустой список** (при наличии тяжёлых зависимостей CLIP/optical flow трассировка могла бы дублировать upstream) |

#### §6 — Verdict

**Итог L2:** по 5 run (A+B) схема и NPZ **совпадают**, `manifest.status=ok`, `schema_version=video_pacing_npz_v3`, `producer_version=2.0.1`.  
Покадровые кривые плотные; NaN в `feature_values` на A+B согласуются с выключенными флагами optional блоков (entropy/histograms/peaks/periodicity/bursts).

#### §8 — DoD

**Не закрыт:** C, §4.6, §4.8; кейсы с включёнными **entropy / histograms / peaks / periodicity / bursts**.

---

## 1. L2 summary (A+B, 5 run)

По агрегатам JSON:

- **N_total**: **467**
- **F**: **57** на всех
- **S_set**: `[3,5,7,9]`
- `feature_values` finite/total: **218 / 285** (≈ **44/57** на run)

## 2. Снимок **A** (исторический, L1)

| Величина | Значение |
|----------|----------|
| N | 48 |
| S (shot boundaries) | 5 |
| F | 57 |
| `feature_values` finite / total | 44 / 57 |
| Optional flags (все false) | entropy, histograms, pace peaks, periodicity, bursts |
---

## Навигация

[Audit v4 hub](../../audit_4_2/README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
