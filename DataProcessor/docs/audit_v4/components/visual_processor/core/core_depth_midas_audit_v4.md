# Audit v4 — `core_depth_midas` (VisualProcessor core)

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A + B**, 5 run).  
**Артефакты (A+B):** см. `storage/audit_v4/core_depth_midas_l2/core_depth_midas_audit_v4_stats.json` (пути `npz_path` внутри).  
**Контракт:** [`VisualProcessor/schemas/core_depth_midas_npz_v3.json`](../../../../../VisualProcessor/schemas/core_depth_midas_npz_v3.json) · [`core/model_process/core_depth_midas/docs/SCHEMA.md`](../../../../../VisualProcessor/core/model_process/core_depth_midas/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Роль | ✓ | Покадровые карты глубины **Midas** + аналитика по кадру + **preview** подмножество **K** кадров |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `core_depth_midas_npz_v3` | ✓ | Полное совпадение; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** на **A** |
| **N**, **H**, **W** | ✓ | **N=48**; **`depth_maps`** / **`depth_maps_norm`**: **(48, 256, 256)** |
| **K (preview)** | ✓ | **K=10**; **`preview_*`**: **(10, …)**; **`meta.preview_k`**: **10** |
| Согласованность preview | ✓ | **`preview_frame_indices`** ⊆ **`frame_indices`** (на **A** — 10 union-индексов **0…337**) |

#### §4.2 — NaN / Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Все float-массивы (включая **(N,H,W)**) | ✓ | **0%** NaN на **A** |
| Inf | ✓ | **0** |

#### §4.1a — Диапазоны

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **`depth_maps_norm`** | ✓ | На **A**: **min=0**, **max=1** (per-schema ожидание нормализованной карты) |
| **`depth_maps`** (сырые) | ✓ | На **A**: в разумном диапазоне **~0…976** (относительные единицы Midas, не метры) |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices` | ✓ | Строго возрастают |

#### §5.3 — Models / meta

| Вопрос | Ответ |
|--------|-------|
| `meta.models_used` | На **A**: **1** запись |
| `backend_proxy_version` | Присутствует (required в схеме **meta**) |

#### §6 — Verdict (L2)

**Итог L2:** на 5 run (A+B) контракт и shape‑инварианты **стабильны**: `depth_maps`/`depth_maps_norm` всегда **(N, 256, 256)**, preview всегда **K=10**, `preview_frame_indices ⊆ frame_indices` на всех run, NaN/Inf отсутствуют. `depth_maps_norm` строго **[0,1]**. **~8.9 / 10** на L2 (до L3/§8 не `passed`).

#### §8 — DoD

**Не закрыт:** C, §4.6, §4.8 (golden).

---

## 2. L2 stats (A+B)

JSON: `storage/audit_v4/core_depth_midas_l2/core_depth_midas_audit_v4_stats.json`

Коротко по агрегатам:

- `N_total=543`, `H_set=[256]`, `W_set=[256]`, `K_set=[10]`
- `depth_maps_norm`: min=0, max=1; NaN/Inf = 0
- `preview_subset_ok_all=true`, `frame_indices_strict_inc_all=true`

---

## 1. Снимок **A**

| Величина | Значение |
|----------|----------|
| N | 48 |
| H × W | 256 × 256 |
| K | 10 |
| `depth_maps_norm` range | [0, 1] |
| `preview_frame_indices` | 0, 37, 72, 116, 152, 185, 221, 265, 300, 337 |
