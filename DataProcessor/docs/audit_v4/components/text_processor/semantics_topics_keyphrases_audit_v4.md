# Audit v4 — `semantics_topics_keyphrases` (TextProcessor)

**Дата отчёта:** 2026-04-14 (дополнение L2 + L1)  
**Уровень отчёта (план §3.1):** **L1 — draft** (исторический набор **A**) + попытка **L2** (A+B) — **заблокирована** на текущем `result_store`.  
**Артефакт (канон L1):** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/text_processor/text_features.npz`  
**Артефакт (воспроизводимый A для L2-скрипта):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/text_processor/text_features.npz`  
**Срез компонента:** **116** ключей `tp_topics_*` — [`semantics_topics_keyphrases_output_v1`](../../../../TextProcessor/schemas/semantics_topics_keyphrases_output_v1.json). Сырые ключевые фразы — в **`result`** вне таблицы (**`tp_topics_keyphrases_raw`**); эмбеддинги фраз — **`_artifacts/tp_topics_keyphrase_embeddings.npy`**.  
**Статистика L2 (инструмент):** `storage/audit_v4/semantics_topics_keyphrases_l2/semantics_topics_keyphrases_audit_v4_stats.json`  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт L2: `TextProcessor/src/extractors/semantics_topics_keyphrases/scripts/audit_v4_npz_stats.py` (`--seed 0`)  
**Engineering log 4.2:** [`../audit_4_2/text_processor/semantics_topics_keyphrases_engineering_log_v4_2.md`](../audit_4_2/text_processor/semantics_topics_keyphrases_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применimo.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/semantics_topics_keyphrases/SCHEMA.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` + dense `.npy` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | **`tp_topics_present=1`**, топ-темы и **10** keyphrases; режим экспорта слотов **`none`** |
| **B** | ✗ | **`export_keyphrases_mode=hashed`** (заполнение **`tp_topics_kp_top*_*`**), **`emit_extra_metrics=true`**, disabled/enabled ветки |
| **C** | ✗ | Пустой текст, **`export_keyphrases_mode=raw`**, выключенные **keyphrases**/embeddings |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема | ✓ | **116** имён, `allow_extra_keys: false` |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ожидаемые NaN на **A** | ✓ | **`emit_extra_metrics=false`** → **`_fill_extra_metrics`** не заполняет **`tp_topics_extra_*`** ([`main.py`](../../../../TextProcessor/src/extractors/semantics_topics_keyphrases/main.py) ~L387–393) |
| Слоты тем | ✓ | **`top_k_slots=5`** (на **A**) → **`tp_topics_topic_top6…8_*`** остаются **NaN** |
| Слоты keyphrases в таблице | ✓ | **`export_keyphrases_mode=none`** → блок **`hashed`** не выполняется (~L717–723), после **`_nan_kp_slots`** все **`tp_topics_kp_top*_{hash01,len}`** — **NaN**, **`present`** — **0** |

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
| Модели | Retrieval/embeddings через **`dp_models`** + bundled taxonomy ([`main.py`](../../../../TextProcessor/src/extractors/semantics_topics_keyphrases/main.py)); на **A** эмбеддинги фраз **(10, 1024)** |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация: **2.1.0** ([`main.py`](../../../../TextProcessor/src/extractors/semantics_topics_keyphrases/main.py)) — README может отставать по номеру версии.

---

## 2.1. L2: `result_store` и блокировка

Скрипт `TextProcessor/src/extractors/semantics_topics_keyphrases/scripts/audit_v4_npz_stats.py` (выход: `storage/audit_v4/semantics_topics_keyphrases_l2/semantics_topics_keyphrases_audit_v4_stats.json`) берёт 5 путей A+B и проверяет:

- табличный срез `tp_topics_*` (**116** ключей),
- артефакт `text_processor/_artifacts/tp_topics_keyphrase_embeddings.npy` (shape/dtype/finite + согласование с `tp_topics_keyphrases_count/dim`).

**Факт по `storage/result_store/youtube` (2026-04-14):** из 5 путей **2** файла `meta.status=ok` и содержат полный `tp_topics_*` + валидный `tp_topics_keyphrase_embeddings.npy` (**(10, 1024)**).

Ещё **3** пути имеют `meta.status=error` и не содержат табличного слоя (пустой `feature_names`). При этом `tp_topics_keyphrase_embeddings.npy` **может присутствовать** (частичный выход до падения пайплайна) — для успеха ориентироваться на `meta.status` и наличие tabular slice.

Причина блокировки — сбой всего `text_processor` на части mock-run (часто OOM в эмбеддерах до выполнения downstream шагов), а не логика `semantics_topics_keyphrases`.

---

## 2. Наблюдения на наборе **A**

### 2.1 Сводка

| Группа | Значение |
|--------|----------|
| Статус | **`tp_topics_present` 1** |
| Текст | **`tp_topics_text_chars` 491**; **ASR/title/description** флаги **1** |
| Распределение тем | **`tp_topics_enable_topic_distribution` 1**; **`entropy_topk` ≈ 1.588**, **`top_k_topics` / `top_k_slots` = 5** |
| Топ-1 | **`topic_top1_id` 4**, **`prob` ≈ 0.238** |
| Keyphrases | **`count` 10**, **`dim` 1024**, **`score_top1` ≈ 1.2** |
| Экспорт в таблицу | **`export_keyphrases_mode_none` 1** — слоты **`kp_top*`** пустые в **features_flat** |

### 2.2 Артефакт плотных эмбеддингов

`…/text_processor/_artifacts/tp_topics_keyphrase_embeddings.npy` — **(10, 1024)**, **`float32`**.

### 2.3 HTML

`text_processor/_render/semantics_topics_keyphrases_report.html`.

---

## 3. Вердикт

**Плюсы**

- Полное совпадение **116** ключей со схемой; на **A** основной сценарий (**topics + keyphrase matrix**) живой.
- Политика privacy/экспорта (**`hashed`/`none`/`raw`**) отражена one-hot в таблице.

**Минусы / внимание**

- При **`mode=none`** счётчик **`tp_topics_keyphrases_count`** и **`.npy`** есть, но **плоские** **`kp_top*`** намеренно **пустые** — потребители NPZ должны читать **payload** / файлы, а не ждать hash в таблице.
- Много **NaN** при **`emit_extra_metrics=false`** и неиспользуемых тем-слотах — норма, но документировать в downstream.

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | Множества имён совпали |
| Полнота эмпирики на **A** | **8** | Topics+KPE; без hashed-слотов и extra |
| Документированность ветвлений | **8** | Режимы экспорта и extra читаются из кода |
| Готовность к модели / продукту | **8** | Явный **`present`**, топ-темы и плотный KPE |

**Итог L1: ~8.2 / 10** (условно: **B** для **`hashed` + `emit_extra_metrics=true`**).
