# Audit v4 — `spectral_extractor`

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (набор **A+B**; **C** и §8 — не закрыты).  
**Stats JSON:** `storage/audit_v4/spectral_extractor_l2/spectral_extractor_audit_v4_stats.json`  
**Фигуры:** `storage/audit_v4/spectral_extractor_l2/figures/`  
**Анализ / tooling:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/spectral_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** · **◐** · **✗** · **N/A**

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика + вердикт | ✓ | Отчёт, `SCHEMA.md` / `README.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Схема / код | ◐ | `spectral_extractor_npz_v2`, `npz_savers/spectral.py`, `main.py` |

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

#### §4.1a — NaN, строки

| Критерий | Статус | Заметка |
|----------|--------|---------|
| До фикса на **A** | ◐ | **4 NaN**: `hop_length`, `n_fft`, `duration`, `device_used` |
| Причины | ✓ | `run_segments` не клал STFT-параметры и duration в payload; `device_used` — строка в float tabular |
| После фикса | ✓ | `device_used` только **meta**; числовой контекст из **`self`** и охвата оси; tabular **F=46** |

#### §4.2 — Ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **N=12** на **A** | ✓ | Маска везде true; per-segment mean ряды |

#### §4.8 — Golden

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Hash | ✗ | После повторного **A** |

##### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| Тяжёлый DL | **Нет** — librosa / классика |

#### §8 — DoD

**Не закрыт:** C, golden.

---

## 1.5. Статистика (набор **A+B**)

- JSON: `storage/audit_v4/spectral_extractor_l2/spectral_extractor_audit_v4_stats.json`
- Figures:
  - `storage/audit_v4/spectral_extractor_l2/figures/hist_tabular_*.png`
  - `storage/audit_v4/spectral_extractor_l2/figures/tabular_corr_heatmap.png`

---

## 1. Мета (набор **A**, фрагмент)

| Поле | Значение |
|------|----------|
| `schema_version` | `spectral_extractor_npz_v2` |
| `spectral_contract_version` | `spectral_contract_v1` |
| `device_used` | `cpu` |
| `features_enabled` | `basic_features`, `contrast`, `advanced_features`, `time_series` |
| `hop_length` / `n_fft` / `duration` в meta (до фикса) | **None** (после фикса — из payload) |

---

## 2. Код

1. **`npz_savers/spectral.py`:** убран `add("device_used", …)`.
2. **`main.py` `run_segments`:** в успешный и empty payload добавлены **`hop_length`**, **`n_fft`**, **`duration`** (\(\max(end)-\min(start)\) по канонической оси).
3. **`utils/render.py`:** в `summary` добавлен **`device_used`** из **`meta`**.

---

## 3. Вердикт

**Плюсы:** богатый спектральный блок; строгая ось; после фикса tabular без «тихих» строк и без пропажи STFT-контекста в сегментном режиме.

**Минусы:** исторический артефакт **A** остаётся с NaN до перезапуска; **duration** в `run_segments` — охват окон, не обязательно длина всего клипа.

---

## 3.5. Audit 4.2 — engineering log (после L2)

[`../audit_4_2/audio_processor/spectral_extractor_engineering_log_v4_2.md`](../audit_4_2/audio_processor/spectral_extractor_engineering_log_v4_2.md)

---

## 4. Оценка 0–10

| Критерий | Балл |
|----------|------|
| Контракт / полезность | **8** (−1 до повторного A) |

**Итог: ~8/10** после свежего NPZ и §4.8.
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
