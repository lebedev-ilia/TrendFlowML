# Audit v4 — `speaker_diarization_extractor`

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (набор **A+B**; **C** и §8 — не закрыты).  
**Stats JSON:** `storage/audit_v4/speaker_diarization_extractor_l2/speaker_diarization_extractor_audit_v4_stats.json`  
**Фигуры:** `storage/audit_v4/speaker_diarization_extractor_l2/figures/`  
**Анализ / tooling:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/speaker_diarization_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** · **◐** · **✗** · **N/A**

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика + вердикт | ✓ | Отчёт + `docs/README.md` / `SCHEMA.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Схема / код | ◐ | `speaker_diarization_extractor_npz_v2`; савер — эталон порядка tabular |

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

#### §4.1 / §4.1a — Типы, NaN

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Tabular **F=10** на **A** | ✓ | Согласовано с `save_speaker_diarization_npz` |
| NaN в tabular на **A** | ✓ | **0** |
| Строки в tabular | ✓ | Нет; **`device_used`** / модель в **`meta`** |

#### §4.4 — Категориальные / object

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Токены / IDs | ✓ | `turn_speaker_id`, `speaker_ids` — int-массивы; **`meta`** для строк |

#### §4.5 — Ось сегментов и тёрны

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **N=1** по `segment_*` | ✓ | Ожидание контракта family `diarization` |
| Тёрны **K=3** на **A** | ✓ | `turn_*` выровнены; **`speaker_count=2`** |

#### §4.8 — Golden

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Hash | ✗ | TODO |

##### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| Загрузка | Через ModelManager (`pyannote` и т.д.); **`model_name`**, **`weights_digest`** в **`meta`** |

#### §8 — DoD

**Не закрыт:** C, golden.

---

## 2.5. Статистика (набор **A+B**)

- JSON: `storage/audit_v4/speaker_diarization_extractor_l2/speaker_diarization_extractor_audit_v4_stats.json`
- Figures:
  - `storage/audit_v4/speaker_diarization_extractor_l2/figures/hist_tabular_*.png`
  - `storage/audit_v4/speaker_diarization_extractor_l2/figures/tabular_corr_heatmap.png`

---

## 1. Мета (набор **A**)

| Поле | Значение |
|------|----------|
| `schema_version` | `speaker_diarization_extractor_npz_v2` |
| `diarization_contract_version` | `diarization_contract_v1` |
| `device_used` | `cpu` |
| `features_enabled` | `turns` |

---

## 2. Tabular (на **A**)

Пример: `speaker_count` 2; `duration_sec` ~12.03; `sample_rate` 16000; `rms` / `peak` конечны; `dominant_speaker_id` 1; `speaker_turns_count` 3; плотность и `speaker_transitions_count` согласованы с **K=3** и **2** сменами спикера.

---

## 3. Документация vs код

Ранее в `SCHEMA.md` / `README.md` фигурировал «минимальный» список **7** скаляров **без** `sample_rate` / `rms` / `peak`. Фактический NPZ и савер — **F=10**. Документация **выровнена** под код (Audit v4).

Код савера **не менялся**.

---

## 4. Вердикт

**Плюсы:** чистый tabular; плоские turn-массивы; per-speaker статистика; дисциплина meta для модели и устройства.

**Минусы:** L1 без B/C, golden, edge empty/silent.

---

## 4.5. Audit 4.2 — engineering log (после L2)

[`../audit_4_2/audio_processor/speaker_diarization_extractor_engineering_log_v4_2.md`](../audit_4_2/audio_processor/speaker_diarization_extractor_engineering_log_v4_2.md)

---

## 5. Оценка 0–10

| Критерий | Балл |
|----------|------|
| Контракт / приватность / структура | **9** |
| Полезность для downstream (кто когда говорил) | **9** |

**Итог: ~9/10** при закрытии §4.8 и L2.
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
