# CRITERIA.md — lexico_static_features

**Версия компонента:** 1.2.0  
**Дата согласования:** 2026-07-17

## Универсальные хард-гейты (U1–U6)

| Гейт | Описание | Метод проверки |
|------|----------|----------------|
| U1 | validate_schema/structure/ranges → rc=0 | batch 28 NPZ: 28/28 OK |
| U2 | Ось времени | N/A — нет seq-выхода, только flat features |
| U3 | Различимость между видео | 14/67 полей std>0 на 6 уникальных видео |
| U4 | Expected-empty path | `title=''`→67 ключей NaN by design; `enabled=False`→67 ключей |
| U5 | Golden детерминизм | max\|Δ\|=0.0 на 5 прогонах (кроме timing-полей) |
| U6 | Разные длины | 3 синтетических + 6 storage видео без падений |

## Критерии компонента (C1–C4)

| Критерий | Описание | Вердикт |
|----------|----------|---------|
| C1 | `tp_lex_emoji_diversity` = NaN by design когда нет эмодзи в тексте (все emoji_count=0). Не дефект. | NaN OK |
| C2 | `tp_lex_named_entity_density` = NaN, `tp_lex_named_entity_density_enabled` = 0.0 by design — NER заглушка (spaCy удалён, планируется отдельный компонент). | NaN OK |
| C3 | `tp_lex_compute_ms` и `tp_lex_load_ms` excluded from golden-сравнения (CPU-стенное время, меняется между прогонами). Семантические поля golden=0.0. | excluded |
| C4 | Все transcript-поля (`tp_lex_transcript_*`) = NaN by design когда `tp_lex_present_transcript=0` (нет ASR). Не дефект. | NaN OK |

## NaN-политика

| Поле/группа | Когда NaN | Когда finite |
|-------------|-----------|--------------|
| `tp_lex_title_*` (метрики) | `has_title=False` | `has_title=True` |
| `tp_lex_description_*` (метрики) | `has_description=False` | `has_description=True` |
| `tp_lex_transcript_*` | `has_transcript=False` (нет ASR или текст пуст) | `has_transcript=True` |
| `tp_lex_emoji_diversity` | нет эмодзи в тексте | есть эмодзи |
| `tp_lex_named_entity_density` | всегда (NER-заглушка) | никогда |
| `tp_lex_punctuation_entropy` | `not (has_title or has_description)` | title или desc непустые |
| `tp_lex_title_emoji_count` / `tp_lex_description_emoji_count` | `not enable_emoji` или `not has_title/desc` или нет emoji-lib | `enable_emoji=True` + has + emoji-lib |

## Диапазоны (при finite)

- `_RATIO_0_1` поля (stopword_ratio, type_token_ratio, clickbait_score и т.д.) ∈ [0, 1]
- `tp_lex_upper_lower_ratio_title` ≥ 0 (может быть >1 при all-caps)
- `tp_lex_transcript_readability_score` ≥ 0
- `tp_lex_punctuation_entropy` ≥ 0
- `tp_lex_compute_ms`, `tp_lex_load_ms` ∈ [0, 1e7]
- Все счётчики (`len_words`, `num_urls`, `emoji_count` и т.д.) ≥ 0
- One-hot policy: sum(`tp_lex_transcript_source_policy_*`) = 1.0
- One-hot used: sum(`tp_lex_transcript_source_used_*`) = 1.0
