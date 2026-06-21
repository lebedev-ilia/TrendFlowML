# Audit v4 — `speech_analysis_extractor`

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (набор **A+B**; **C** и §8 — не закрыты).  
**Stats JSON:** `storage/audit_v4/speech_analysis_extractor_l2/speech_analysis_extractor_audit_v4_stats.json`  
**Фигуры:** `storage/audit_v4/speech_analysis_extractor_l2/figures/`  
**Анализ / tooling:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/speech_analysis_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** · **◐** · **✗** · **N/A**

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + вердикт | ✓ | Отчёт, `SCHEMA.md` / `README.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Схема / код | ◐ | `speech_analysis_extractor_npz_v1`, `main.py` bundle, `npz_savers/speech_analysis.py` |

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

#### §4.1a — NaN, согласованность флагов

| Критерий | Статус | Заметка |
|----------|--------|---------|
| До фикса на **A** | ◐ | **6 NaN**: `pitch_f0_*`, `pitch_stability`; **`pitch_enabled`** в tabular **0**; **`meta.features_enabled`** содержал **`pitch_metrics`** |
| Причина | ✓ | `_features_enabled` включал **`pitch_metrics`** при **`enable_pitch_metrics`**, даже если **`pitch_payload`** не мержился (`pitch_enabled=false`) → савер вызывал `add(pitch_*)` с **`None`** → **NaN** |
| После фикса | ✓ | **`pitch_metrics`** в **`_features_enabled`** только при **`pitch_payload is not None`**; иначе pitch-колонки в NPZ **не пишутся** |

#### §4.4 — Строки / meta

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `device_used` | ✓ | В **meta** / payload, не в tabular савера |

#### §4.8 — Golden

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Hash | ✗ | После повторного **A** |

##### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| Bundle | Агрегирует **ASR** / **diar** / **pitch** из `extractor_results`; тяжёлые модели в зависимостях |

#### §8 — DoD

**Не закрыт:** C, golden.

---

## 2.5. Статистика (набор **A+B**)

- JSON: `storage/audit_v4/speech_analysis_extractor_l2/speech_analysis_extractor_audit_v4_stats.json`
- Figures:
  - `storage/audit_v4/speech_analysis_extractor_l2/figures/hist_tabular_*.png`
  - `storage/audit_v4/speech_analysis_extractor_l2/figures/tabular_corr_heatmap.png`

---

## 1. Мета (набор **A**, фрагмент)

| Поле | Значение |
|------|----------|
| `schema_version` | `speech_analysis_extractor_npz_v1` |
| `speech_analysis_contract_version` | `speech_analysis_contract_v1` |
| `device_used` | `cuda` |
| `features_enabled` (на старом **A**) | `asr_metrics`, `pitch_metrics` |

---

## 2. Tabular (на **A**, до исправления кода)

**Без NaN:** `duration_sec`, `sample_rate`, блок ASR (7 скаляров), **`pitch_enabled`** = 0.  
**С NaN:** семь pitch-метрик (несогласованно с флагом в meta).

---

## 3. Код

**`main.py`:** условие включения **`pitch_metrics`** в **`enabled_features`** — `enable_pitch_metrics and pitch_payload is not None`.

**`npz_savers/speech_analysis.py`:** без изменений.

---

## 4. Вердикт

**Плюсы:** компактный bundle-выход; feature-gates в савере; после фикса meta NPZ согласован с фактическим наличием pitch-данных.

**Минусы:** старый **A** остаётся с NaN до перезапуска; L1 без B/C.

---

## 4.5. Audit 4.2 — engineering log (после L2)

- Engineering log: [`../audit_4_2/audio_processor/speech_analysis_extractor_engineering_log_v4_2.md`](../audit_4_2/audio_processor/speech_analysis_extractor_engineering_log_v4_2.md)

---

## 5. Оценка 0–10

| Критерий | Балл |
|----------|------|
| Контракт / согласованность | **8** (до повторного A) |

**Итог: ~8/10** после свежего артефакта и §4.8.
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
