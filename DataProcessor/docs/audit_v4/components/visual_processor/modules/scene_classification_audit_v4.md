# Audit v4 — `scene_classification` (VisualProcessor)

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**, **5** прогонов из `result_store`).  
**Статистика (A+B):** `storage/audit_v4/scene_classification_l2/scene_classification_audit_v4_stats.json`  
**Артефакты (5 run):** см. `RUN_LOG.md` запись `scene_classification` (A+B)  
**Контракт:** [`VisualProcessor/schemas/scene_classification_npz_v2.json`](../../../../../VisualProcessor/schemas/scene_classification_npz_v2.json) · [`modules/scene_classification/docs/SCHEMA.md`](../../../../../VisualProcessor/modules/scene_classification/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Hard deps | ✓ | `core_clip`, `cut_detection`, ось Segmenter ([`SCHEMA.md`](../../../../../VisualProcessor/modules/scene_classification/docs/SCHEMA.md)) |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `scene_classification_npz_v2` | ✓ | Полное совпадение; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** на **A** |
| **N, S** | ✓ | **N=48** кадров, **S=4** сцен |
| Top-k | ✓ | **`frame_topk_ids/probs (48,5)`** |

#### §4.1a — Вероятности top-5

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Сумма по строке | ◐ | На **A**: **~0.41 … 0.96** — это **срез top-5** полного распределения, **не** обязан суммироваться в 1; encoder/interpreter не должен предполагать нормировку без явного контракта |
| NaN | ✓ | **0%** в `frame_topk_probs` |

#### §4.1a — `frame_scene_id`

| Критерий | Статус | Заметка |
|----------|--------|---------|
| SCHEMA | ✓ | **0 … S−1**, без **−1** |
| На **A** | ✓ | **min=0, max=3**, **4** уникальных id |

#### §4.2 — NaN / Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Выборочно per-frame / per-scene | ✓ | `frame_entropy`, `frame_top1_prob`, `mean_score`, `class_entropy_mean` — **0%** NaN на **A** |
| Inf | ✓ | **0** в float-массивах на **A** |

#### §4.3 — Сцены (**A**)

| Поле | Наблюдение |
|------|------------|
| `scene_label` | `beauty_salon`, `beauty_salon`, `office`, `music_studio` |
| `label_fusion` | **`places`** (boxed в NPZ) |
| `min_scene_seconds` | **2.0** |

#### §4.4 — Prompt lists (debug)

| Массив | Длина на **A** |
|--------|----------------|
| `scene_aesthetic_prompts` | 6 |
| `scene_luxury_prompts` | 6 |
| `scene_atmosphere_prompts` | 6 |
| `places365_prompts` | 365 |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices`, `times_s` | ✓ | Монотонны |

#### §4.12 — Anti-leakage

| Вопрос | Ответ |
|--------|--------|
| CLIP embeddings локально | Да, через **`core_clip`** |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|--------|
| Encoder | Per-frame: **top-k + entropy + gaps**; per-scene: многие **(S,)** ряды + словарь **`scenes`** |
| Tabular | Агрегаты по сценам + при необходимости flatten |

#### §6 — Verdict

**Итог L2:** по 5 run (A+B) схема и NPZ **совпадают**, все `manifest.status=ok`, `schema_version=scene_classification_npz_v2`, `producer_version=2.0.1`; `label_fusion=places` стабилен. **Top‑5 probs не суммируются в 1** (это expected «срез» распределения) — диапазон по A+B: min≈**0.186**, max≈**0.999996** (см. JSON stats).

**Оценка:** **~8.7 / 10** на L2.

#### §8 — DoD

**Не закрыт:** C, §4.6, §4.8.

---

## 1. L2 summary (A+B, 5 run)

По агрегатам JSON:

- **N_total**: **543**
- **S**: **2…7** (S_set: `[2,3,4,7]`)
- **top-k**: **5** на всех
- **label_fusion**: `places` на всех
- **`places365_prompts`**: **365** на всех

## 2. Снимок **A** (исторический, L1)

| Величина | Значение |
|----------|----------|
| N | 48 |
| S | 4 |
| top-k | 5 |
| `frame_topk_probs` ∑ (min, max) | ~0.41, ~0.96 |
| `meta.models_used` | 3 записи |
| `places365_prompts` | 365 строк |
