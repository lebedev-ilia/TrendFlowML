# Audit v4 — `cosine_metrics_extractor` (TextProcessor)

**Дата отчёта:** 2026-04-14 (дополнение L2 + L1)  
**Уровень отчёта (план §3.1):** **L1 — draft** (исторический набор **A**) + попытка **L2** (A+B) — **заблокирована** на текущем `result_store`.  
**Артефакт (канон L1):** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/text_processor/text_features.npz`  
**Артефакт (воспроизводимый A для L2-скрипта):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/text_processor/text_features.npz`  
**Срез компонента:** **39** ключей `tp_cos_*` — [`cosine_metrics_extractor_output_v1`](../../../../TextProcessor/schemas/cosine_metrics_extractor_output_v1.json). Входы — пути к эмбеддингам в `doc.tp_artifacts` (title/description/transcript agg/comments agg).  
**Статистика L2 (инструмент):** `storage/audit_v4/cosine_metrics_extractor_l2/cosine_metrics_extractor_audit_v4_stats.json`  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт L2: `TextProcessor/src/extractors/cosine_metrics_extractor/scripts/audit_v4_npz_stats.py` (`--seed 0`)  
**Engineering log 4.2:** [`../audit_4_2/text_processor/cosine_metrics_extractor_engineering_log_v4_2.md`](../audit_4_2/text_processor/cosine_metrics_extractor_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/cosine_metrics_extractor/SCHEMA.md); upstream: embedders + transcript aggregator + comments pipeline |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` |
| Путь под `run_id` | ✓ | `text_processor/text_features.npz` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | Все четыре источника present; **aggregates** mode для комментариев |
| **B** | ✗ | Другой `comments_mode=matrix`, другой источник транскрипта |
| **C** | ✗ | Пустые embedding’и, require_* fail-fast, dim mismatch |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема (`allow_extra_keys: false`) | ✓ | **39** имён, совпадение множеств |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| NaN по контракту | ✓ | При **`emit_extra_metrics=False`**: `tp_cos_load_ms`, `tp_cos_compute_ms`; в режиме **aggregates** — также **`tp_cos_tc_*`** (matrix-only extras) — все **NaN** на **A** |
| Cosine скаляры | ✓ | Пять пар **конечны** (∈ **~0.84–0.88**) |

#### §4.7 — Трактовка

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Наблюдения → выводы | ✓ | §5–6 |

#### §4.8 — Golden на **A**

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Сигнатура | ✗ | TODO |

#### §5.3 — Сверка с Models

| Вопрос | Ответ |
|--------|--------|
| Собственная модель | **Нет** — только скалярное произведение L2-нормированных векторов из артефактов |
| Baseline tabular | Нет явного перечисления в [`BASELINE_MODEL.md`](../../../../../Models/docs/contracts/BASELINE_MODEL.md) |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация: **1.3.0** ([`main.py`](../../../../TextProcessor/src/extractors/cosine_metrics_extractor/main.py)). `meta` агрегированного NPZ — как у других шагов TextProcessor.

---

## 2.1. L2: `result_store` и блокировка

Скрипт `TextProcessor/src/extractors/cosine_metrics_extractor/scripts/audit_v4_npz_stats.py` (выход: `storage/audit_v4/cosine_metrics_extractor_l2/cosine_metrics_extractor_audit_v4_stats.json`) берёт 5 путей A+B (как у Visual L2) и выделяет `tp_cos_*`.

**Факт по `storage/result_store/youtube` (2026-04-14):** из 5 путей **2** файла `meta.status=ok` и содержат полный срез (**39** ключей); **3** файла `meta.status=error` и не содержат табличного слоя (пустой `feature_names`).

Причина блокировки — сбой всего `text_processor` на части mock-run (часто OOM в эмбеддерах до выполнения downstream шагов), а не логика `cosine_metrics_extractor`.

---

## 2. Наблюдения на наборе **A**

### 2.1 Присутствие источников и режим

| Поле | Значение |
|------|----------|
| `tp_cos_title_present` … `tp_cos_comments_present` | все **1** |
| `tp_cos_empty_no_*` | все **0** |
| `tp_cos_comments_mode_aggregates` | **1** |
| `tp_cos_comments_mode_matrix` | **0** |
| Источник агрегата транскрипта | **`whisper`** (`tp_cos_transcript_agg_source_whisper=1`, остальные **0**) |

### 2.2 Косинусы (model_facing)

| Пара | Значение |
|------|----------|
| `tp_cos_title_desc` | **0.856** |
| `tp_cos_title_transcript` | **0.853** |
| `tp_cos_desc_transcript` | **0.880** |
| `tp_cos_transcript_comments_mean` | **0.847** |
| `tp_cos_transcript_comments_median` | **0.839** |

Значения согласуются с высокой схожестью нормированных текстовых эмбеддингов одного ролика.

### 2.3 Флаги ошибок

`tp_cos_zero_norm_flag`, `tp_cos_dim_mismatch_flag`, `tp_cos_pair_dim_mismatch_flag`, `tp_cos_tc_dim_mismatch_flag`, `tp_cos_unsafe_relpath_flag` — все **0**.

### 2.4 HTML

`text_processor/_render/cosine_metrics_extractor_report.html`.

---

## 3. Вердикт

**Плюсы**

- Полное совпадение **39** ключей с machine schema.
- Явные **presence / empty / mismatch** флаги и one-hot источника транскрипта.
- Численно осмысленные косинусы, без NaN в основных метриках на **A**.

**Минусы / внимание**

- Потребители должны отличать **`comments_mode`** (aggregates vs matrix) и трактовку **NaN** в `tp_cos_tc_*` / таймингах при выключенных extras.
- L1 не покрывает **matrix**-режим, **require_*** fail-fast и смену приоритета `transcript_source_priority`.

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | Множества имён совпали |
| Семантика пар и флагов | **8.5** | Источник whisper явно отмечен |
| Downstream clarity | **8** | Режим aggregates vs matrix |
| Edge coverage | **6** | Один «счастливый» путь на **A** |

**Итог L1: ~8.3 / 10** (до **B/C**, `matrix`, §4.8).
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
