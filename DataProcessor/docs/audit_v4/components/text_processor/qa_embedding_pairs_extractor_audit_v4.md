# Audit v4 — `qa_embedding_pairs_extractor` (TextProcessor)

**Дата отчёта:** 2026-04-14 (дополнение L2 + L1)  
**Уровень отчёта (план §3.1):** **L1 — draft** (исторический набор **A**) + попытка **L2** (A+B) — **заблокирована** на текущем `result_store`.  
**Артефакт (канон L1):** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/text_processor/text_features.npz`  
**Артефакт (воспроизводимый A для L2-скрипта):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/text_processor/text_features.npz`  
**Срез компонента:** **34** ключа `tp_qa_*` — [`qa_embedding_pairs_extractor_output_v1`](../../../../TextProcessor/schemas/qa_embedding_pairs_extractor_output_v1.json). При **`num_questions>0`**: матрица **`qa_question_embeddings.npy`** (не в таблице).  
**Статистика L2 (инструмент):** `storage/audit_v4/qa_embedding_pairs_extractor_l2/qa_embedding_pairs_extractor_audit_v4_stats.json`  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт L2: `TextProcessor/src/extractors/qa_embedding_pairs_extractor/scripts/audit_v4_npz_stats.py` (`--seed 0`)  
**Engineering log 4.2:** [`../audit_4_2/text_processor/qa_embedding_pairs_extractor_engineering_log_v4_2.md`](../audit_4_2/text_processor/qa_embedding_pairs_extractor_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/qa_embedding_pairs_extractor/SCHEMA.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | **Валидный пустой исход:** после эвристики извлечения вопросов **`num_questions=0`** → **`present=0`**, `.npy` **не** пишется |
| **B** | ✗ | Контент с сегментами `…?` **и** вопросительным словом из **`question_langs`** ([`_extract_questions_from_text`](../../../../TextProcessor/src/extractors/qa_embedding_pairs_extractor/main.py) ~L283–312) |
| **C** | ✗ | **`require_min_questions>0`** при нуле вопросов (fail-fast), **`emit_extra_metrics=true`**, опциональные **`qa_question_hashes.npy`** |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема | ✓ | **34** имени, `allow_extra_keys: false` |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ожидаемые NaN на **A** | ✓ | Шаблон [`_stable_template`](../../../../TextProcessor/src/extractors/qa_embedding_pairs_extractor/main.py): **`tp_qa_embedding_dim`**, **`tp_qa_questions_per_min`**, **`tp_qa_questions_per_1k_chars`**, **`tp_qa_mean_cosine_to_centroid`** — **NaN** при **0** вопросов и **`emit_extra_metrics=false`** (ветка early return ~L490–515 не меняет эти поля, кроме случая **`emit_extra_metrics=true`** для **`questions_per_min`** = **0.0** при валидной длительности) |

На фактическом **A** конфиг **`emit_extra_metrics=false`**, поэтому все четыре поля остаются **NaN** из шаблона.

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
| Модель | Sentence-transformers через **`get_model_with_meta`** — на **A** кодирование **не вызывалось** (**0** вопросов) |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация: **1.3.0** ([`main.py`](../../../../TextProcessor/src/extractors/qa_embedding_pairs_extractor/main.py)).

---

## 2.1. L2: `result_store` и блокировка

Скрипт `TextProcessor/src/extractors/qa_embedding_pairs_extractor/scripts/audit_v4_npz_stats.py` (выход: `storage/audit_v4/qa_embedding_pairs_extractor_l2/qa_embedding_pairs_extractor_audit_v4_stats.json`) берёт 5 путей A+B и проверяет:

- табличный срез `tp_qa_*` (**34** ключа),
- артефакт `text_processor/_artifacts/qa_question_embeddings.npy` **только** когда `tp_qa_present=1` / `tp_qa_num_questions>0` (для `present=0` это валидный пустой исход).

**Факт по `storage/result_store/youtube` (2026-04-14):** из 5 путей **2** файла `meta.status=ok` и содержат полный `tp_qa_*`:

- на 1 OK-run `tp_qa_present=1`, `tp_qa_num_questions=2` и присутствует `qa_question_embeddings.npy` формы **(2, 1024)**;
- на 1 OK-run `tp_qa_present=0`, `tp_qa_num_questions=0` — валидный пустой исход, артефакт отсутствует (ожидаемо).

Ещё **3** пути имеют `meta.status=error` и не содержат табличного слоя (пустой `feature_names`).

Причина блокировки — сбой всего `text_processor` на части mock-run (часто OOM в эмбеддерах до выполнения downstream шагов), а не логика `qa_embedding_pairs_extractor`.

---

## 2. Наблюдения на наборе **A**

### 2.1 Итог извлечения

| Поле | Значение |
|------|----------|
| **`tp_qa_enabled`** | **1** |
| **`tp_qa_present`** | **0** |
| **`tp_qa_num_questions`** | **0** |
| **`tp_qa_q_title` / `description` / `transcript` / `comments`** | **0** |
| **`tp_qa_require_min_questions`** | **0** (пустой исход **не** является ошибкой) |

Источники включены (**`use_*` = 1**), транскрипт **ASR-only** (**`tp_qa_transcript_source_policy_asr_only` = 1**), **`allow_legacy_transcripts` = 0**.

### 2.2 Артефакты

Файла **`qa_question_embeddings.npy`** в `…/text_processor/_artifacts/` **нет** — согласовано с кодом: при **`num_q<=0`** артефакты не пишутся (~L491–515).

### 2.3 Дополнительные метрики

| Поле | Значение |
|------|----------|
| **`tp_qa_mean_cosine_to_centroid_present`** | **0** |
| **`tp_qa_hashes_written` / `source_ids_written`** | **0** (флаги записи optional-артефактов **выключены**) |

### 2.4 Почему **0** вопросов (семантика)

Кандидаты — только фрагменты, которые **заканчиваются на `?` / `？`**, длина ≥ **`min_chars_per_question`** (**8** на артефакте), и содержат **целое слово** из RU/EN списка вопросительных слов (**regex** ~L268–281, фильтр ~L308–310). На этом видео ни заголовок, ни описание, ни ASR/комментарии не дали такого сегмента.

### 2.5 HTML

`text_processor/_render/qa_embedding_pairs_extractor_report.html`.

---

## 3. Вердикт

**Плюсы**

- Полное совпадение **34** ключей со схемой; «пустой успех» чётко кодируется **`present=0`** и отсутствием файла эмбеддингов.
- Политика транскрипта и лимиты (**`max_*`**, **`dedup`**) отражены в табличных полях.

**Минусы / внимание**

- На reference **A** не демонстрируются **ни эмбеддинги**, ни **`emit_extra_metrics`** (косинус к центроиду, плотность по времени/символам) — для L1 по «содержательному» выходу нужен run **B**.
- Потребители должны не путать имя экстрактора с парами Q–A (см. README / schema **description**).

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | Множества имён совпали |
| Полнота эмпирики на **A** | **6** | Только валидный нулевой путь |
| Документированность ветвлений | **8** | Пустой return и NaN в шаблоне читаются из кода |
| Готовность к модели / продукту | **8** | **`tp_qa_present`** однозначен; нужны данные с вопросительными конструкциями |

**Итог L1: ~7.8 / 10** (условно: контракт **9/10**, «счастливый» путь — **B**).
