# `title_to_hashtag_cosine_extractor` — AUDIT (v1)

**Статус**: `done`  
**Компонент**: `src/extractors/title_to_hashtag_cosine_extractor/main.py`

## Изменения для production-grade

- Убраны `glob+mtime` (недетерминировано).
- Чтение только через `doc.tp_artifacts["embeddings"]["title"/"hashtag"]["relpath"]`.
- Выход только `features_flat` (canonical `tp_titlehashcos_*` + legacy aliases `tp_title_hashtag_cosine*`), без путей.
- Safe relpath join + `tp_titlehashcos_unsafe_relpath_flag`.
- Valid empty semantics по умолчанию; fail-fast через `require_title_embedding` / `require_hashtag_embedding`.
- Zero-norm и dim-mismatch не дают “фейковую метрику”: `NaN` + флаги `tp_titlehashcos_zero_norm_flag` / `tp_titlehashcos_dim_mismatch_flag`.
- Feature-gating: `enabled` + `tp_titlehashcos_disabled_by_policy`.

## TODO

- `resource_costs` (CPU very low).


