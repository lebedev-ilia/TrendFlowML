# Audit v4 — `core_optical_flow` (VisualProcessor core)

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A + B**; **5** прогонов).  
**Артефакты:** см. `storage/audit_v4/core_optical_flow_l2/core_optical_flow_audit_v4_stats.json` (полный список путей)  
**Контракт:** [`VisualProcessor/schemas/core_optical_flow_npz_v3.json`](../../../../../VisualProcessor/schemas/core_optical_flow_npz_v3.json) · [`core/model_process/core_optical_flow/docs/SCHEMA.md`](../../../../../VisualProcessor/core/model_process/core_optical_flow/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Роль | ✓ | Покадровые метрики optical flow (нормы, направление, affine/camera proxy) + **preview** карт магнитуды для **K** пар кадров |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `core_optical_flow_npz_v3` | ✓ | Полное совпадение; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** на **A+B** (5/5) |
| **N** | ✓ | На L2: **N_set=[48,59,133,147,156]**; `frame_indices` строго возрастают (5/5) |
| **Preview** | ✓ | **K=10**, **`preview_flow_mag_map_norm`**: **(10, 64, 64)** на всех 5; `meta.preview_k=10`, `preview_map_size=[64,64]` |
| Пары preview | ✓ | **`preview_prev_frame_indices`** / **`preview_cur_frame_indices`** задают пары (на **A** шаг **+8** между соседями в union) |

#### §4.2 — NaN / Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Первый кадр | ✓ | На L2 подтверждено: `dt_seconds` и все flow/cam/bg ряды — **NaN только на idx 0** (5/5; см. `flow_dep_nan_at_0_only_all=true`) |
| **`motion_norm_per_sec_mean`** | ✓ | На L2: NaN нет (5/5), и `motion_norm_per_sec_mean[0]=0` (`motion0_is_zero_all=true`) |
| **`times_s`** | ✓ | **0%** NaN |
| **`preview_flow_mag_map_norm`** | ✓ | На L2: NaN **0**, значения в \([0,1]\) (`preview_in_01_all=true`) |
| Inf | ✓ | **0** |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| `meta.models_used` | На **A**: **1** запись |
| `backend_proxy_version` | В **`meta`** (required) |

#### §6 — Verdict

**Итог L2:** по 5 run (A+B) схема и NPZ **совпадают**, manifest **чистый** (status=ok), NaN‑политика по idx 0 **согласована** по всем flow‑зависимым рядам, preview‑карты нормализованы в \([0,1]\), `times_s` монотонен. **~8.8 / 10** на L2.

#### §8 — DoD

**Не закрыт:** C, §4.6, §4.8.

---

## 1. Снимок **A+B** (L2)

| Величина | Значение |
|----------|----------|
| N_total | 543 |
| N_set | 48, 59, 133, 147, 156 |
| K | 10 |
| H_preview × W_preview | 64 × 64 |
| NaN по flow‑зависимым рядам | только idx 0 (5/5) |

Статистика/воспроизводимость:

- JSON stats: `storage/audit_v4/core_optical_flow_l2/core_optical_flow_audit_v4_stats.json`
