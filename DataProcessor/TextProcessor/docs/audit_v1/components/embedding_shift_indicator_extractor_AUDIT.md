# `embedding_shift_indicator_extractor` — AUDIT (v1)

**Статус**: `done`  
**Компонент**: `src/extractors/embedding_shift_indicator_extractor/main.py`

## Резюме

Переведён на детерминированный вход через `doc.tp_artifacts` (canonical `transcripts` + legacy `transcript_chunks`, без glob/mtime). Выдаёт только `features_flat` (`tp_embshift_*`), не раскрывает пути.

Prod hardening (A-policy):
- safe relpath join + `tp_embshift_unsafe_relpath_flag`
- stable schema: `tp_embshift_*` ключи всегда присутствуют
- valid empty: если transcript chunks отсутствуют или `n_chunks<require_min_chunks` → `tp_embshift_present=0` и NaN метрики (fail-fast через `require_transcript_chunks=true`)
- параметризован `transcript_source_priority`
- feature-gating: `enabled`, `compute_shift_flag`, `compute_extra_cosines`
- no fake metrics: zero-norm/NaN/Inf → NaN + flags (`tp_embshift_zero_norm_flag`, `tp_embshift_nan_inf_flag`)

## TODO

- `resource_costs` (CPU very low).


