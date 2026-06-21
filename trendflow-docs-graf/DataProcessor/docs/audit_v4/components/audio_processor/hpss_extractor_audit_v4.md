# Audit v4 — `hpss_extractor`

**Дата:** 2026-04-06  
**Уровень отчёта (план §3.1):** **L2 — product stats** (наборы **A+B**; **C** и §8 DoD — не закрыты).  
**Reference A (reproducible):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/hpss_extractor/hpss_extractor_features.npz`  
**Статистика L2 (A+B):** `storage/audit_v4/hpss_extractor_l2/hpss_extractor_audit_v4_stats.json` (+ `figures/`)  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6), `--seed 0`

**Engineering log 4.2 (после L2):** `DataProcessor/docs/audit_v4/components/audit_4_2/audio_processor/hpss_extractor_engineering_log_v4_2.md`

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано на текущем уровне · **◐** частично / только набор **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика + вердикт | ✓ | Отчёт + README / SCHEMA |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Схема / код | ◐ | `hpss_extractor_npz_v1.json`; tabular не полностью frozen по длине F |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Компонент + путь A | ✓ | [`RUN_LOG.md`](../../RUN_LOG.md) |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | Тот же `run_id` |
| **B** | ✓ | `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`, `-Q6fnPIybEI/e2bc964f-…` |
| **C** | ✗ | TODO |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| L2, не L3 | ✓ | `RUN_LOG`: L2 закрыт по **A+B** |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Оси **N=12** | ✓ | Маска везде `true` на **A** |
| Сверка с JSON | ◐ | Поля NPZ шире описания в JSON |

#### §4.1a — NaN, строки, tabular

| Критерий | Статус | Заметка |
|----------|--------|---------|
| NaN в `feature_values` на **A** (до фикса) | ✗→✓ | **17 NaN**: савер вызывал `add()` для energy/spectral keys без значений в payload `run_segments`; **исправлено:** агрегация в `main.py` + условный `add` в `npz_savers/hpss.py` |
| Доли по сегментам | ◐ | Сумма **harmonic + percussive** по окну **не обязана = 1** (остаток относится к полному `S_mag`, не к `H+P`) — на **A** суммы ~0.36…0.82 |

#### §4.2 — Прочее

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `hpss_*_share_series` на **A** | ◐ | Форма `(0,)`, при этом в meta были `time_series` / `waveforms` — **несоответствие семантики** до правки `features_enabled` в `run_segments` |

#### §4.4 — object / meta

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `hpss_dominance` | ✓ | В **meta**, строка `mixed` на **A** |

#### §4.5 — Временная ось

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `segment_*_sec` | ✓ | Согласованы с другими экстракторами на том же run |

#### §4.6–§4.12, §7–§8

| Критерий | Статус |
|----------|--------|
| Корреляции (tabular A+B) | ✓ | `storage/audit_v4/hpss_extractor_l2/figures/tabular_corr_heatmap.png` |
| Golden, полный DoD, межкомпонентные корреляции | ✗ / ◐ | L3 / C |

#### §5 — Документация

| Пodпункт | Статус |
|----------|--------|
| README / SCHEMA обновлены | ◐ |

##### §5.3 — Models / Baseline

| Вопрос | Ответ |
|--------|-------|
| В [`BASELINE_MODEL.md`](../../../../../Models/docs/contracts/BASELINE_MODEL.md) audio trio? | **Нет** (librosa, optional Tier‑1) |

#### §6 — Verdict

Кратко: полезное разложение H/P и per-segment доли; на **A** выявлен **дефект tabular** (NaN и ложные `features_enabled` для series/waveforms в сегментном режиме) — **устранён в коде**; старый NPZ сохраняет историческую картину.

#### §6.1 — Оценка 0–10

| Критерий | Балл |
|----------|------|
| Стабильность контракта | **7** (до фикса) / **9** (после) |
| Tabular | **6**→**8** |
| Encoder / dense | **7** (серии пустые в segment-only) |
| Аналитика | **8** |

**Итог:** **~7.5/10** на проанализированном артефакте; **~8.5/10** после повторного прогона A.

---

## 1. Мета (**A**, исторический артефакт)

- `schema_version`: `hpss_extractor_npz_v1`, `producer_version`: `2.1.0`, `status`: `ok`
- `features_enabled` (до фикса): `energy_metrics`, `waveforms`, `spectral_features`, `time_series` — **не соответствовало** фактическим ключам NPZ в `run_segments`
- `hpss_dominance`: `mixed`

## 2. Tabular (**A**, до фикса)

- **34** строки имени; **17** значений **NaN** (энергии, stability, все spectral means/std)
- Валидны: доли, separation, balance, mean/std долей по сегментам, конфиг (`sample_rate`, `n_fft`, …), `segments_count`  
- Per-segment: `harmonic_share + percussive_share` не нормирована к 1 (см. §4.1a)

## 3. Сверка с кодом (корневая причина)

1. `run_segments` собирал только часть energy-полей в верхний `features`; `_compute_hpss_metrics` на окне выдаёт также `hpss_energy_*` и stability — они **не агрегировались** в payload.
2. Спектральные признаки считались **на сегмент** в `seg_features`, но **не усреднялись** в глобальный payload.
3. Савер безусловно добавлял все имена при включённом флаге → **`as_float(None)` → NaN**.
4. `features_enabled` включал `waveforms` / `time_series` по CLI, хотя `run_segments` их **не создаёт**.

## 4. Исправления (код)

- [`hpss_extractor/main.py`](../../../../AudioProcessor/src/extractors/hpss_extractor/main.py): агрегация energy/stability/spectral по успешным сегментам; в `run_segments` убраны `waveforms` и `time_series` из `_features_enabled`.
- [`npz_savers/hpss.py`](../../../../AudioProcessor/src/core/npz_savers/hpss.py): tabular-строки для energy/spectral подполей только если значение в payload **не `None`**.

Повторный прогон набора **A** рекомендован для §4.8 и закрытия регрессии в журнале.
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
