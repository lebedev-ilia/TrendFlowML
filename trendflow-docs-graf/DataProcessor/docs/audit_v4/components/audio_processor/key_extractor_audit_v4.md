# Audit v4 — `key_extractor`

**Дата:** 2026-04-06  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**; **C** и §8 DoD — не закрыты).  
**Reference A (reproducible):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/key_extractor/key_extractor_features.npz`  
**Статистика L2 (A+B):** `storage/audit_v4/key_extractor_l2/key_extractor_audit_v4_stats.json` (+ `figures/`)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6), `--seed 0`

**Engineering log 4.2 (после L2):** `DataProcessor/docs/audit_v4/components/audit_4_2/audio_processor/key_extractor_engineering_log_v4_2.md`

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** · **◐** · **✗** · **N/A** — как в других компонентных отчётах v4.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика + вердикт | ✓ | Отчёт + README/SCHEMA |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Схема / код | ◐ | `key_extractor_npz_v1.json`; optional meta keys дополнены |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Компонент + run **A** | ✓ | [`RUN_LOG.md`](../../RUN_LOG.md) |

#### §3 — Validation set

| Набор | Статус |
|-------|--------|
| **A** | ✓ |
| **B** | ✓ |
| **C** | ✗ |

#### §3.1 — Уровень отчёта

| Критерий | Статус |
|----------|--------|
| L2 product stats | ✓ |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Оси сегментов **N=12** | ✓ | `key_id_by_segment` все **5** (D minor), confidence ≈ **0.356** |
| `meta.key_id` | ✓ | **5**, согласовано с табличным смыслом (после фикса — и `key_id` в tabular) |

#### §4.1a — Строки, NaN, tabular

| Критерий | Статус | Заметка |
|----------|--------|---------|
| NaN в `feature_values` на **A** (до фикса) | ✗→✓ | **5/10** имён: `key_name`, `key_mode`, `method`, `key_confidence_category`, `key_confidence_reason` — строки через **`as_float` → NaN**; **`key_method`** уже был корректно в **meta** |
| Исправление | ✓ | [`npz_savers/key.py`](../../../../AudioProcessor/src/core/npz_savers/key.py): строки/категории в **meta**; в tabular **`key_id` + числовые конфиги + confidence** |

#### §4.2 — Прочее

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `key_scores` | ✓ | Все нули (detailed_scores выключен) |

#### §4.4 — meta / chroma

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `chroma_reused` | ✓ | **True** на **A** |

#### §4.5 — Временная ось

| Критерий | Статус |
|----------|--------|
| `segment_*` | ✓ |

#### §4.6–§4.12, §7–§8

| Критерий | Статус |
|----------|--------|
| Корреляции (tabular A+B) | ✓ | `storage/audit_v4/key_extractor_l2/figures/tabular_corr_heatmap.png` |
| Golden, полный DoD, межкомпонентные корреляции | ✗ / ◐ | L3 / C |

#### §5 — Документация

| Подпункт | Статус |
|----------|--------|
| README ссылка на схему | ◐ | Исправлено: было `audio_npz_v1` |

##### §5.3 — Baseline / Models

| Вопрос | Ответ |
|--------|-------|
| В [`BASELINE_MODEL.md`](../../../../../Models/docs/contracts/BASELINE_MODEL.md) audio trio? | **Нет** |

#### §6 — Verdict

Сильная сторона: строгие **per-segment** id/confidence, reuse chroma, осмысленный глобальный ключ. Слабое место на историческом артефакте — **загрязнённые NaN** в tabular из-за строк; устранено в савере.

#### §6.1 — Оценка 0–10

| Критерий | Балл |
|----------|------|
| Контракт / стабильность | **8** (до фикса tabular) / **9** (после) |
| Tabular для моделей | **7**→**8** (`key_id` + confidence явно) |
| Encoder | **6** (без time_series на **A**) |
| Аналитика | **8** |

**Итог:** ~**8/10** после повторного прогона A.

---

## 1. Мета (**A**, до фикса, фрагмент)

- `key_id`: 5, `key_method`: `librosa`, `chroma_reused`: True  
- `key_name` / `key_mode` **не** были в meta до фикса савера (только через интерпретацию id)

## 2. Сверка с кодом

- Доминанта по числу сегментов с одинаковой парой name+mode; `_add_confidence_metadata` задаёт строковые поля confidence — они не должны попадать во **float**-вектор.

## 3. Изменения (код)

- `npz_savers/key.py`: tabular без строк; расширен `meta.extra` (`key_name`, `key_mode`, confidence metadata).  
- `schemas/key_extractor_npz_v1.json`: optional meta keys для confidence metadata.

Повторный прогон **A** для golden §4.8.
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
