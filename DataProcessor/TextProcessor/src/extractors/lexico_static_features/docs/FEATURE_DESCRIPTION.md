# `lexico_static_features` — описание фич и артефактов

**Компонент:** `LexicalStatsExtractor` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **67** скаляров `tp_lex_*` в `text_processor/text_features.npz`.  
**Контракт:** [`../../../../schemas/lexico_static_features_output_v1.json`](../../../../schemas/lexico_static_features_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

**Версия:** 1.2.0 (`LexicalStatsExtractor.VERSION`). Многие метрики — **эвристики / proxy**, не «истина» NLP (см. поля `tier` в JSON и комментарии в схеме).

---

## 1. Назначение

- Статические признаки по **title**, **description**, **транскрипт** (ASR `doc.asr.segments` и/или legacy `doc.transcripts` по политике).  
- Без spaCy/langdetect/сети; тяжёлые модели — в отдельных экстракторах.  
- При **`enabled=False`**: `tp_lex_disabled_by_policy=1`, числовые поля групп — **NaN**, тайминги: `load_ms=0`, `compute_ms` — время early-return.

---

## 2. Группы (кратко)

| Группа | Ключи (логика) |
|--------|----------------|
| Включение | `tp_lex_enabled`, `tp_lex_disabled_by_policy` |
| Наличие текста | `tp_lex_present_*`, `tp_lex_present_any` |
| Gating групп | `tp_lex_group_*_enabled`, `tp_lex_require_transcript_enabled` |
| Emoji | `tp_lex_has_emoji_lib`, `tp_lex_emoji_dependency_missing_flag` |
| Политика транскрипта | one-hot: `tp_lex_transcript_source_policy_*` (ровно один **1**) |
| Источник фактически | `tp_lex_transcript_source_used_asr` / `legacy` / `none` (в валидном прогоне сумма **1**) |
| Обрезка | `*_chars_used/kept`, `*_truncated_flag` |
| Title / Description / Transcript | длины, счётчики, доли, флаги (см. схему) |
| Комбинированные | `tp_lex_emoji_diversity`, `tp_lex_punctuation_entropy` (+ `*_present`), `tp_lex_special_character_ratio`, `tp_lex_upper_lower_ratio_title` |
| NER | `tp_lex_named_entity_density` (**NaN**), `tp_lex_named_entity_density_enabled` (**0** в текущей версии) |
| Тайминги | `tp_lex_load_ms` (в коде **0**), `tp_lex_compute_ms` (мс полного прохода) |

---

## 3. Нормальные диапазоны (`--ranges`)

| Категория | Ожидание |
|-----------|----------|
| Бинарные 0/1 (при finite) | `enabled`, `disabled`, `present*`, `group*`, флаги политики/источника/обрезки/префиксов и т.д. (**30** ключей, см. валидатор) |
| Доли / «ratio» / score 0..1 | Типично **∈ [0, 1]** (в т.ч. `clickbait_score`, `transcript_lexical_diversity`, стоп-слова, …) |
| `tp_lex_transcript_readability_score` | **≥ 0** (отношение длины предложения к длине слова; может быть **> 1**) |
| `tp_lex_upper_lower_ratio_title` | **≥ 0** |
| `tp_lex_punctuation_entropy` (finite) | **≥ 0** |
| Счётчики / длины (finite) | **≥ 0** |
| `tp_lex_load_ms`, `tp_lex_compute_ms` (finite) | **≥ 0**, разумно **< 1e7** мс |

---

## 4. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_lexico_static_features_text_npz.py`](../utils/validate_lexico_static_features_text_npz.py)

---

## 5. Чеклист

1. Срез **67** имён = `lexico_static_features_output_v1`.  
2. One-hot политики транскрипта и triplet «used» согласованы (суммы **1**).  
3. HTML/CSV каталоги признаков могут жить отдельно (`view_csv_*.json`) — сверять ключи `tp_lex_*` при обновлении таблиц.
