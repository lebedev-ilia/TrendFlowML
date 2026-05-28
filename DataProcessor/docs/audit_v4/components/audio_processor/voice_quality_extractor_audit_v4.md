# Audit v4 — `voice_quality_extractor`

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (набор **A+B**; **C** и §8 — не закрыты).  
**Stats JSON:** `storage/audit_v4/voice_quality_extractor_l2/voice_quality_extractor_audit_v4_stats.json`  
**Фигуры:** `storage/audit_v4/voice_quality_extractor_l2/figures/`  
**Анализ / tooling:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/voice_quality_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** · **◐** · **✗** · **N/A**

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + вердикт | ✓ | Отчёт, `SCHEMA.md` / `README.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Схема / код | ◐ | `voice_quality_extractor_npz_v1`, `npz_savers/voice_quality.py` |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Путь + `run_id` | ✓ | [`RUN_LOG.md`](../../RUN_LOG.md) |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | `youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759` (старый L1‑путь `4c3bf25b-…` может отсутствовать в текущем `result_store`) |
| **B** | ✓ | `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…` |
| **C** | ✗ | TODO |

#### §4.1a — Строки, NaN

| Критерий | Статус | Заметка |
|----------|--------|---------|
| До фикса | ◐ | **1 NaN** — `f0_method` (на **A**: **`torchcrepe`** в **meta**) |
| После фикса | ✓ | **`f0_method`** только **meta**; tabular **F=29** |
| Прочие tabular на **A** | ✓ | Конечные значения |

#### §4.2 — Ось / серии

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **N=12** | ✓ | `jitter_by_segment`, `shimmer_by_segment`, `hnr_by_segment` |
| `f0`/`amps`/`hnr_vals` в NPZ | ◐ | На **A** пустые массивы (вероятно вынесены в `_artifacts/` при `time_series`) |

#### §4.8 — Golden

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Hash | ✗ | После повторного **A** |

##### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| torchcrepe / GPU | Опционально; **`f0_method`** в meta |

#### §8 — DoD

**Не закрыт:** C, golden.

---

## 2.5. Статистика (набор **A+B**)

- JSON: `storage/audit_v4/voice_quality_extractor_l2/voice_quality_extractor_audit_v4_stats.json`
- Figures:
  - `storage/audit_v4/voice_quality_extractor_l2/figures/hist_tabular_*.png`
  - `storage/audit_v4/voice_quality_extractor_l2/figures/tabular_corr_heatmap.png`

---

## 1. Мета (набор **A**)

| Поле | Значение |
|------|----------|
| `schema_version` | `voice_quality_extractor_npz_v1` |
| `voice_quality_contract_version` | `voice_quality_contract_v1` |
| `device_used` | `cuda` |
| `f0_method` | `torchcrepe` |
| `features_enabled` | `jitter`, `shimmer`, `hnr`, `f0_stats`, `time_series` |

---

## 2. Код

**`npz_savers/voice_quality.py`:** убран `add("f0_method", …)`; **`f0_method`** уже пишется в **`meta`** (`extra` в `build_meta`).

---

## 3. Вердикт

**Плюсы:** богатый voice-quality слой; явный контракт; meta уже содержала корректный **`f0_method`**.

**Минусы:** до фикса — молчаливый NaN в tabular (как `backend` / `device_used` в других компонентах).

---

## 3.5. Audit 4.2 — engineering log (после L2)

- Engineering log: [`../audit_4_2/audio_processor/voice_quality_extractor_engineering_log_v4_2.md`](../audit_4_2/audio_processor/voice_quality_extractor_engineering_log_v4_2.md)

---

## 4. Оценка 0–10

| Критерий | Балл |
|----------|------|
| Контракт / полезность | **8** |

**Итог: ~8/10** после свежего NPZ и §4.8.
