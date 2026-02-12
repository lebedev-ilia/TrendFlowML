# `cosine_metrics_extractor` — AUDIT (v1)

**Статус**: `done`  
**Компонент**: `src/extractors/cosine_metrics_extractor/main.py`

## Резюме

Компонент приведён к A‑policy: детерминированные входы через `doc.tp_artifacts`, безопасная работа с relpath (без traversal), валидная пустота (NaN + `*_present`/`*_empty_*` флаги), feature‑gating и fail-fast для required входов. Возвращает только `features_flat` (`tp_cos_*`) и не раскрывает пути.

Prod hardening (A-policy):
- transcript агрегат читается из canonical `doc.tp_artifacts["transcripts"][source]["agg_mean_relpath"]` с fallback на legacy `doc.tp_artifacts["transcript_aggregates"]`
- по умолчанию считает transcript↔comments по **агрегатам** (`doc.tp_artifacts["comments"]["agg_*_relpath"]`), матричный режим — опционально
- feature-gating для отдельных метрик + `*_enabled` скаляры для стабильной схемы/объяснимости
- добавлены `*_present` флаги, `tp_cos_*_dim_mismatch_flag` и `tp_cos_zero_norm_flag`
- safe relpath join + `tp_cos_unsafe_relpath_flag`
- transcript priority параметризован (`transcript_source_priority`)

## Контракт

- **Вход** (в рамках одного run): `doc.tp_artifacts["embeddings"]`, `doc.tp_artifacts["comments"]`, `doc.tp_artifacts["transcripts"]` (canonical) + legacy alias `doc.tp_artifacts["transcript_aggregates"]`
- **Выход**: `result.features_flat`:
  - `tp_cos_title_desc`
  - `tp_cos_title_transcript`
  - `tp_cos_desc_transcript`
  - `tp_cos_transcript_comments_mean`
  - `tp_cos_transcript_comments_median`

## Соответствие критериям

- ✅ No glob/mtime
- ✅ No abs paths в `result`
- ✅ NPZ-friendly scalars
- ✅ Safe relpath join (no traversal)
- ✅ Valid empty semantics (NaN + flags)
- ✅ No fake metrics for degenerate vectors (zero-norm → NaN)
- ✅ Feature-gating + stable schema (`*_enabled`)

## Resource costs

- CPU-only, без моделей/скачиваний.  
- Пары эмбеддингов: \(O(d)\) на метрику.  
- Matrix режим transcript↔comments: \(O(n \cdot d)\), где \(n\) — число комментариев.


