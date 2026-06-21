# Audit v4 — `rhythmic_extractor`

**Дата:** 2026-04-13  
**Уровень отчёта (план §3.1):** **L2 — product stats** (набор **A+B**; **C** и §8 — не закрыты).  
**Stats JSON:** `storage/audit_v4/rhythmic_extractor_l2/rhythmic_extractor_audit_v4_stats.json`  
**Фигуры:** `storage/audit_v4/rhythmic_extractor_l2/figures/`  
**Анализ / tooling:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт: `AudioProcessor/src/extractors/rhythmic_extractor/scripts/audit_v4_npz_stats.py` **`--seed 0`**

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** · **◐** · **✗** · **N/A** — как в плане.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика NPZ + семантика + вердикт | ✓ | Отчёт + `docs/README.md` / `SCHEMA.md` |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Схема / код | ◐ | `rhythmic_extractor_npz_v2.json`, `npz_savers/rhythmic.py` |

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
| Tabular **F=9** на **A+B** | ✓ | Совпадает с замороженным подмножеством в `SCHEMA.md` |
| NaN в tabular на **A+B** | ✓ | **0** на всех 5 NPZ (см. stats JSON) |
| Строки в tabular | ✓ | Нет; `backend` и др. в **`meta`** |

#### §4.2 — Семантика длительности

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `duration_sec` vs клип | ◐ | На **A** ≈ **68.07** с = \(\sum_i (segment\_end - segment\_start)\); ось клипа до **~12** с по `segment_end_sec[-1]` — **не противоречие**, см. `SCHEMA.md` |

#### §4.4 — Категориальные

| Критерий | Статус | Заметка |
|----------|--------|---------|
| `meta.backend` и т.д. | ✓ | `librosa`, `sampling_family_used`: `tempo` на **A** |

#### §4.8 — Golden

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Hash на **A** | ✗ | TODO |

#### §4.11 — >24 scalar

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Порог | N/A | F=9 в tabular; доп. скаляры — отдельные ключи NPZ |

##### §5.3 — Models

| Вопрос | Ответ |
|--------|-------|
| ML в [`BASELINE_MODEL.md`](../../../../../Models/docs/contracts/BASELINE_MODEL.md) | **N/A** — beat tracking / сигнал (librosa/essentia), не нейросеть в классическом baseline |

#### §8 — DoD

**Не закрыт:** C, golden.

---

## 1. Мета (набор **A**)

| Поле | Значение |
|------|----------|
| `schema_version` | `rhythmic_extractor_npz_v2` |
| `rhythmic_contract_version` | `rhythmic_contract_v1` |
| `device_used` | `cpu` |
| `backend` | `librosa` |
| `sampling_family_used` | `tempo` |
| `features_enabled` | `basic_metrics`, `interval_stats`, `regularity_metrics`, `tempo_metrics` |

---

## 2. Tabular (на **A**)

| Имя | Значение (пример) |
|-----|-------------------|
| `rhythm_tempo_bpm` | ~107.67 |
| `rhythm_beats_count` | 117 |
| `rhythm_beat_density` | ~1.72 |
| `rhythm_regularity` | ~0.064 |
| `rhythm_tempo_variation` | ~14.57 |
| `rhythm_beat_consistency` | ~0.064 |
| `duration_sec` | ~68.07 |
| `sample_rate` | 22050 |
| `segments_count` | 12 |

**Доп. скаляры в NPZ** (не в tabular): `rhythm_avg_period_sec`, …, `rhythm_tempo_max` и т.д. — см. ключи артефакта.

**Сегменты:** N=12, маска все true на **A**.

---

## 3. Статистика (набор **A+B**)

- JSON: `storage/audit_v4/rhythmic_extractor_l2/rhythmic_extractor_audit_v4_stats.json`
- Figures:
  - `storage/audit_v4/rhythmic_extractor_l2/figures/hist_tabular_*.png`
  - `storage/audit_v4/rhythmic_extractor_l2/figures/tabular_corr_heatmap.png`

Коротко по результатам:
- **Tabular**: 9 фичей, NaN‑фракция по всем фичам = **0.0** на этих 5 прогонах.
- **Диапазоны** и **корреляции** — см. figures (heatmap + hist).

---

## 4. Код

Исправлений класса «строка в tabular» **не потребовалось**: `npz_savers/rhythmic.py` не вызывает `add` для `device_used` / `backend`. Уточнена документация **`duration_sec`** в `SCHEMA.md` и краткая заметка Audit v4 в `README.md`.

---

## 5. Audit 4.2 — engineering log (после L2)

[`../audit_4_2/audio_processor/rhythmic_extractor_engineering_log_v4_2.md`](../audit_4_2/audio_processor/rhythmic_extractor_engineering_log_v4_2.md)

---

## 6. Вердикт

**Плюсы:** чистый tabular; категориальные поля в meta; богатый analytics-слой (отдельные scalar keys + опционально beat массивы / `.npy`).

**Минусы / наблюдения:** без явной документации легко перепутать **`duration_sec`** с длительностью целого файла — **исправлено в документации**; L1 без B/C и golden.

---

## 7. Оценка 0–10

| Критерий | Балл |
|----------|------|
| Контракт / дисциплина типов | **9** |
| Полезность для rhythm / downstream | **8** |

**Итог: ~8.5/10** при закрытии §4.8 и L2.
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
