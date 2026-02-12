# `embedding_pair_topk_extractor` — AUDIT (v1)

**Статус**: `done`  
**Компонент**: `src/extractors/embedding_pair_topk_extractor/main.py`

## Резюме

Убраны `glob+mtime` и чтение “последних” файлов. Экстрактор читает title/description и transcript chunk matrix через `doc.tp_artifacts`, а выход отдаёт как скалярные `features_flat` (`tp_pairtopk_*`).

Prod hardening (A-policy):
- valid empty: если transcript chunks отсутствуют → метрики topK = NaN, `tp_embpair_transcript_chunks_present=0`
- feature-gating для частей расчёта и экспорта (slots/summary)
- стабильная схема фич через `top_k_slots` (top1..topKSlots)
- стандартизация неймспейса: `tp_embpair_*` + back-compat алиасы `tp_pairtopk_*`
- cross-encoder rerank запрещён (privacy + dp_models), включение → fail-fast
- FAISS опционален (`use_faiss`, `require_faiss`)
- safe relpath join + `tp_embpair_unsafe_relpath_flag`
- canonical transcripts key: `tp_artifacts["transcripts"][src]["chunk_embeddings_relpath"]` + legacy fallback flag
- no fake metrics: zero-norm / NaN/Inf → NaN + flags
- (privacy-safe) export top-k chunk indices as slots (`*_top{i}_idx`) when enabled

## TODO

- Решить (отдельно) формат UI‑выхода для top‑k списков, если нужен (сейчас сохраняются только скалярные summary).


