# Audit v4 — `core_object_detections` (VisualProcessor core)

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A + B**; **5** прогонов).  
**Артефакты:** см. `storage/audit_v4/core_object_detections_l2/core_object_detections_audit_v4_stats.json` (полный список путей)  
**Контракт:** [`VisualProcessor/schemas/core_object_detections_npz_v2.json`](../../../../../VisualProcessor/schemas/core_object_detections_npz_v2.json) · [`core/model_process/core_object_detections/docs/SCHEMA.md`](../../../../../VisualProcessor/core/model_process/core_object_detections/docs/SCHEMA.md)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../../AUDIT_4_CRITERIA_AND_PLAN.md)

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Роль | ✓ | YOLO-подобный детектор: до **M** боксов на кадр, **`valid_mask`**, агрегаты person/text/logo |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи vs `core_object_detections_npz_v2` | ✓ | Полное совпадение; **`allow_extra_keys: false`** |
| `manifest.notes` | ✓ | **`null`** на **A+B** (5/5) |
| **N**, **M** | ✓ | На L2: **N_set=[48,59,133,147,156]**, **M=100**; формы **`boxes`**, **`valid_mask`**, … согласованы |
| **`class_names`** | ✓ | **`(41,)`**, строки вида **`id:name`** для **0…40** |
| **`det_count`** | ✓ | На L2: `det_count_matches_mask_all=true` (5/5) |
| **`meta_json`** | ✓ | Непустая строка (дубль **`meta`** для cross-venv) |

#### §4.2 — NaN / маскирование слотов

| Критерий | Статус | Заметка |
|----------|--------|---------|
| NaN в float полях | ✓ | На L2: NaN в `scores` **0** (5/5) |
| Inf | ✓ | **0** |
| Паддинг | ◐ | Слоты с **`valid_mask=False`** **не** обнулены по смыслу: **`scores`** у «невалидных» до **~0.59**, **`class_ids`** произвольные — **единственный источник истины — `valid_mask`**, не порог по score |
| Валидные скоры | ✓ | На L2: `score_valid_min_min≈0.60001`, `score_valid_max_max≈0.97930`; при `valid_mask=False`: `score_invalid_max_max≈0.59886` |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `frame_indices` | ✓ | Строго возрастают |

#### §4.12 — Anti-leakage / downstream

| Вопрос | Ответ |
|--------|-------|
| Треки | В этом NPZ **нет** полей **`tracks`**; зависимые модули (см. логи **`brand_semantics`**) могут синтетически назначать track id — это **вне контракта v2**, фиксировать при интеграции |

#### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| `meta.models_used` | На **A**: **1** запись |

#### §6 — Verdict

**Итог L2:** по 5 run (A+B) схема и NPZ **совпадают**, manifest **чистый** (status=ok), **`det_count` синхронизирован с `valid_mask`**; важно **не читать** `boxes`/`scores`/`class_ids` без `valid_mask` (padding не обнуляется). **~8.8 / 10** на L2.

#### §8 — DoD

**Не закрыт:** C, §4.6, §4.8.

---

## 1. Снимок **A+B** (L2)

| Величина | Значение |
|----------|----------|
| N_total | 543 |
| N_set | 48, 59, 133, 147, 156 |
| M | 100 |
| Всего валидных детекций (sum `det_count`) | 2299 |
| `valid_mask` true ratio (по ячейкам), min..max | ~2.375% … 5.184% |
| `det_count` per frame, глобально (min, max) | 0, 19 |

Статистика/воспроизводимость:

- JSON stats: `storage/audit_v4/core_object_detections_l2/core_object_detections_audit_v4_stats.json`
---

## Навигация

[Audit v4 hub](../../audit_4_2/README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
