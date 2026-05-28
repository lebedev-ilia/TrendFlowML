# Audit v4 — `core_clip` (VisualProcessor core)

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A + B**, 5 run).  
**Артефакты (A+B):** см. `storage/audit_v4/core_clip_l2/core_clip_audit_v4_stats.json` (пути `npz_path` внутри).  
**Контракт:** [`VisualProcessor/schemas/core_clip_npz_v2.json`](../../../../../VisualProcessor/schemas/core_clip_npz_v2.json) · [`core/model_process/core_clip/docs/SCHEMA.md`](../../../../../VisualProcessor/core/model_process/core_clip/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Роль | ✓ | Общий **CLIP** контур: кадровые эмбеддинги, текстовые эмбеддинги промптов, матрицы скоров по семействам промптов, **Places365** top-**K** |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `core_clip_npz_v2` | ✓ | Полное совпадение; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** на **A** |
| **N**, **D** | ✓ | **N=48**, **D=512**; все `*_text_embeddings (P, D)` согласованы с соответствующими `*_prompts (P,)` |
| **K_places365** | ✓ | **K=5**; **`places365_topk_* (N, 5)`**; **`meta.places365_topk_k`**: **5** |
| **P-размеры** | ✓ | **shot_quality 10**; **scene aesthetic/luxury/atmosphere по 6**; **cut_transition 10**; **pop_topic 10**; **places365 365** |

#### §4.1a — Семантика скоров

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Prompt score matrices | ◐ | На **A** строка **`shot_quality_scores`** **не** softmax по **P** (пример: сумма **~2.13** на первом кадре) — это **не** вероятностная нормировка; downstream трактовать как задумано продюсером (например масштабированные логиты/сходства) |
| **`places365_topk_scores`** | ◐ | Сумма по **K** на кадре **~1.05…1.24** (top-**K** срез полного **365**-классового слоя) — **не** обязана быть **1** |
| **`places365_video_topk_scores`** | ◐ | Сумма **~1.09** на **A** — аналогично |

#### §4.2 — NaN / Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **`consecutive_cosine_prev`** | ✓ | **1/48** NaN (**индекс 0**) — нет предыдущего кадра для косинуса |
| Остальные проверенные float-массивы | ✓ | **0%** NaN на **A** (включая **`frame_embeddings`**, score-матрицы, **`times_s`**) |
| Inf | ✓ | **0** |

#### §4.2a — Нормы эмбеддингов

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **`frame_embeddings`** L2 | ✓ | На **A** **≈1** по строкам (нормализация соблюдена с численной погрешностью) |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices` | ✓ | Строго возрастают |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| `meta.models_used` | На **A**: **2** записи |

#### §6 — Verdict (L2)

**Итог L2:** на 5 run (A+B) контракт и shape‑инварианты **стабильны**: **D=512**, **K=5**, размеры семейств промптов неизменны (shot_quality=10; scene aesthetic/luxury/atmosphere=6; cut_transition=10; pop_topic=10; places365=365). `frame_embeddings` L2‑нормированы (mean≈1). `consecutive_cosine_prev` имеет ровно **1 NaN** на run (idx 0) — ожидаемо. `*_scores` и `places365_*_topk_scores` остаются similarity/logit‑подобными величинами (не обязаны суммироваться в 1). **~8.8 / 10** на L2 (до L3/§8 не `passed`).

#### §8 — DoD

**Не закрыт:** C, §4.6, §4.8 (golden).

---

## 2. L2 stats (A+B)

JSON: `storage/audit_v4/core_clip_l2/core_clip_audit_v4_stats.json`

Коротко по агрегатам:

- `N_total=543`
- `D_set=[512]`
- `K_places365_set=[5]`
- `consecutive_cosine_prev_nan_total=5` (по 1 на run)

---

## 1. Снимок **A**

| Величина | Значение |
|----------|----------|
| N | 48 |
| D | 512 |
| K (Places365 per frame) | 5 |
| `consecutive_cosine_prev` NaN count | 1 (idx 0) |
| `places365_topk_scores` ∑ per row (typical) | ~1.05 … 1.24 |
