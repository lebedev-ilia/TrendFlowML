# `lexico_static_features` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `lexico_static_features` |
| Класс | `LexicalStatsExtractor` |
| Machine schema | `DataProcessor/TextProcessor/schemas/lexico_static_features_output_v1.json` |
| `schema_version` | `lexico_static_features_output_v1` |
| Версия реализации | `1.2.0` |

Выход попадает в агрегат `text_features.npz` как плоские ключи `tp_lex_*`.

## Upstream и порядок раннера

В стандартном `MainProcessor` **`TagsExtractor` идёт раньше**. Метрики по **title/description** считаются по **уже очищенным** полям документа (хэштеги вынесены), если tags включён с `mutate_doc_clean_texts`. Транскрипт берётся из **`doc.asr`** / политики — не из title.

## Транскрипт и Audit v3

- **Канон полного Audit v3** (preflight): **`transcript_source_policy="asr_only"`** — без fallback на legacy `doc.transcripts`. Деградированные прогоны (`asr_then_legacy`) помечайте в `RUN_LOG.md`.
- **`require_transcript=false`** (default): пустой транскрипт после политики — валидный empty (`tp_lex_present_transcript=0`, метрики транскрипта `NaN`).
- **`require_transcript=true`**: непустой транскрипт обязателен — иначе **`RuntimeError`** (строгий CI / smoke с обязательным ASR).

## Эмодзи (default Audit v3 baseline)

- **`enable_emoji=true`**, **`emoji_policy="optional"`** (default): если пакет `emoji` не установлен, счётчики эмодзи и `tp_lex_emoji_diversity` → **`NaN`**, `tp_lex_emoji_dependency_missing_flag=1`.
- **`emoji_policy="required"`** + `enable_emoji`: при отсутствии пакета — fail-fast при **инициализации**.

## Лимиты длины

По умолчанию **`max_*_chars=None`** — усечения нет (кроме того, что уже применил upstream к title/description). Явные лимиты задаются конфигом.

## Tier: model_facing vs analytics

- **model_facing** в machine JSON: в основном счётчики длины, presence, простые отношения (например type-token ratio, lexical diversity).
- **analytics** (эвристики, **не** ground truth NLP):
  - `tp_lex_title_clickbait_score` — словарь + пунктуация
  - `tp_lex_title_stopword_ratio`, `tp_lex_transcript_stopword_ratio` — короткий bilingual список
  - `tp_lex_transcript_readability_score`, `tp_lex_transcript_orthographic_error_rate`, `tp_lex_transcript_avg_token_frequency_percentile`, `tp_lex_transcript_rare_word_ratio` — прокси
  - `tp_lex_punctuation_entropy`, URL/@/timestamps и т.д.

Язык / POS / NER **не** в этом экстракторе — отдельные компоненты через `dp_models`.

## Полный перечень ключей

См. `main.py` → `features_flat` (source-of-truth) и `lexico_static_features_output_v1.json`.

### Новый ключ v1.2.0

- `tp_lex_require_transcript_enabled` — отражает параметр `require_transcript`

## Версионирование

Изменение смысла или набора ключей → bump **`lexico_static_features_output_v2`** + отчёт + `RUN_LOG.md`.
