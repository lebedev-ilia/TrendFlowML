# Audit v4 — `lexico_static_features` (TextProcessor)

**Дата отчёта:** 2026-04-14 (дополнение L2 + L1)  
**Уровень отчёта (план §3.1):** **L1 — draft** (исторический набор **A**) + попытка **L2** (A+B) — **заблокирована** на текущем `result_store`.  
**Артефакт (канон L1):** `storage/result_store/youtube/-Q6fnPIybEI/4c3bf25b-e300-47b3-915e-4699c72ab190/text_processor/text_features.npz`  
**Артефакт (воспроизводимый A для L2-скрипта):** `storage/result_store/youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759/text_processor/text_features.npz`  
**Срез компонента:** **67** ключей `tp_lex_*` — [`lexico_static_features_output_v1`](../../../../TextProcessor/schemas/lexico_static_features_output_v1.json). Транскрипт: **`doc.asr`** при **`transcript_source_policy=asr_only`** ([`main.py`](../../../../TextProcessor/src/extractors/lexico_static_features/main.py)).  
**Статистика L2 (инструмент):** `storage/audit_v4/lexico_static_features_l2/lexico_static_features_audit_v4_stats.json`  
**Анализ:** `DataProcessor/.data_venv/bin/python` (numpy 2.2.6); скрипт L2: `TextProcessor/src/extractors/lexico_static_features/scripts/audit_v4_npz_stats.py` (`--seed 0`)  
**Engineering log 4.2:** [`../audit_4_2/text_processor/lexico_static_features_engineering_log_v4_2.md`](../audit_4_2/text_processor/lexico_static_features_engineering_log_v4_2.md)

### Соответствие [`AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md)

**Условные обозначения:** **✓** сделано · **◐** частично / только **A** · **✗** не делалось · **N/A** не применимо.

#### §0 — Зачем v4

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Эмпирика + семантика + вердикт | ✓ | Этот документ |

#### §1 — Source-of-truth

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ссылки на контракты | ✓ | Machine JSON + [`SCHEMA.md`](../../../../TextProcessor/src/extractors/lexico_static_features/SCHEMA.md) |

#### §2 — Scope

| Критерий | Статус | Заметка |
|----------|--------|---------|
| TextProcessor, e2e | ✓ | `text_npz_v1` |

#### §3 — Validation set

| Набор | Статус | Заметка |
|-------|--------|---------|
| **A** | ✓ | Title + description + transcript непустые; транскрипт с **ASR**; без эмодзи в тексте → **`tp_lex_emoji_diversity`** = **NaN** |
| **B** | ✗ | Legacy transcript, **`require_transcript`**, выключенные группы (**`enable_title`** и т.д.) |
| **C** | ✗ | Пустые поля, **`emoji_policy=required`** без библиотеки, усечение **`max_*_chars`** |

#### §3.1 — Уровень отчёта

| Критерий | Статус | Заметка |
|----------|--------|---------|
| **L1**, не `passed` | ✓ | `RUN_LOG.md` |

#### §4.1 — Целостность и типы

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ключи ↔ схема | ✓ | **67** имён, `allow_extra_keys: false` |

#### §4.2 — NaN, Inf

| Критерий | Статус | Заметка |
|----------|--------|---------|
| Ожидаемые NaN на **A** | ✓ | **`tp_lex_emoji_diversity`**: нет символов из **`emoji.EMOJI_DATA`** в объединённом тексте → по коду **NaN** ([`main.py`](../../../../TextProcessor/src/extractors/lexico_static_features/main.py) ~L321) |
| NER отключён | ✓ | **`tp_lex_named_entity_density`** = **NaN**, **`tp_lex_named_entity_density_enabled`** = **0** (заглушка после удаления spaCy из экстрактора, ~L399–401) |

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
| Обучаемая модель | **Нет** — эвристики и счётчики; схема помечает часть полей как proxy / analytics |

#### §6–§8 (кратко)

L3 DoD — **✗**.

---

## 1. Мета

Реализация: **1.2.0**.

---

## 2.1. L2: `result_store` и блокировка

Скрипт `TextProcessor/src/extractors/lexico_static_features/scripts/audit_v4_npz_stats.py` (выход: `storage/audit_v4/lexico_static_features_l2/lexico_static_features_audit_v4_stats.json`) берёт 5 путей A+B и выделяет `tp_lex_*`.

**Факт по `storage/result_store/youtube` (2026-04-14):** из 5 путей **2** файла `meta.status=ok` и содержат полный срез (**67** ключей); **3** файла `meta.status=error` и не содержат табличного слоя (пустой `feature_names`).

Причина блокировки — сбой всего `text_processor` на части mock-run (часто OOM в эмбеддерах до выполнения downstream шагов), а не логика `lexico_static_features`.

---

## 2. Наблюдения на наборе **A**

### 2.1 Наличие текстов и политика транскрипта

| Поле | Значение |
|------|----------|
| **`tp_lex_enabled`** | **1** |
| **`tp_lex_present_title` / `_description` / `_transcript` / `_any`** | **1** |
| **`tp_lex_transcript_source_policy_asr_only`** | **1** |
| **`tp_lex_transcript_source_used_asr`** | **1** |
| **`tp_lex_transcript_source_used_legacy` / `_none`** | **0** |
| Группы | title/description/transcript/emoji/clickbait **включены** (**1**) |

### 2.2 Объёмы (сжато)

| Поле | Значение |
|------|----------|
| Title | **`chars_used`/`kept` 56**, **`tp_lex_title_len_chars` 56**, усечения **0** |
| Description | **`chars` 123**, усечения **0** |
| Transcript | **`chars` 331**, усечения **0** |

### 2.3 Эмодзи и NER

| Поле | Значение |
|------|----------|
| **`tp_lex_has_emoji_lib`** | **1** |
| **`tp_lex_title_emoji_count` / `tp_lex_description_emoji_count`** | **0** |
| **`tp_lex_emoji_diversity`** | **NaN** (пустой список эмодзи после фильтра библиотекой) |
| **`tp_lex_named_entity_density_enabled`** | **0**, плотность **NaN** |

### 2.4 Тайминги

**`tp_lex_load_ms`** = **0**, **`tp_lex_compute_ms`** ≈ **2.15** — в коде **load** не измеряется отдельно (всегда 0 в успешном пути).

### 2.5 HTML

`text_processor/_render/lexico_static_features_report.html`.

---

## 3. Вердикт

**Плюсы**

- Полное совпадение **67** ключей со схемой на **A**; три текстовых поля заполнены, ASR-транскрипт отражён флагами источника.
- Контракт и реализация явно разделяют **эвристики** (stopword, clickbait proxy, readability) и **отключённый NER**.

**Минусы / внимание**

- **`tp_lex_emoji_diversity`** = **NaN** при отсутствии «канонических» эмодзи — потребителям моделей лучше трактовать совместно с **`tp_lex_title_emoji_count`** + **`tp_lex_description_emoji_count`** или вводить **0.0** как «нет разнообразия» (сейчас семантика «не определено»).
- **`tp_lex_load_ms`** всегда **0**: слабая диагностика IO, если позже добавят тяжёлые входы.

---

## 4. Оценка 0–10 (L1)

| Критерий | Балл | Комментарий |
|----------|------|-------------|
| Схема ↔ NPZ ↔ код | **9** | Множества имён совпали |
| Полнота эмпирики на **A** | **8** | Основной путь + ожидаемые **NaN** |
| Документированность ветвлений | **8** | NER/emoji поведение читается из кода и schema description |
| Готовность к модели / продукту | **8** | Явные **`present_*`**, внимание к proxy-полям |

**Итог L1: ~8.2 / 10** (условно: **B/C** для пустых полей и legacy transcript).
---

## Навигация

[Audit v4 hub](../audit_4_2/README.md) · [DataProcessor](../../../MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
