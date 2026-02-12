# `comments_aggregator` — AUDIT (v1)

**Статус**: `done`  
**Компонент**: `src/extractors/comments_aggregator/main.py`

## Резюме

Агрегатор больше не вычисляет hash по raw комментариям для поиска эмбеддингов. Он читает матрицу через `doc.tp_artifacts["embeddings"]["comments"]["relpath"]`, сохраняет mean/median как per-run `*.npy`, регистрирует relpath в `doc.tp_artifacts`, возвращает только `features_flat`.

Prod hardening (A-policy):
- valid empty: если comment embeddings отсутствуют → `tp_comments_agg_present=0`, агрегаты не создаются
- no fake vectors: не создаём нулевые вектора при empty
- фиксированные per-run имена: `comments_agg_mean.npy`, `comments_agg_median.npy`
- weights alignment: веса применяются только при наличии `comments_selected_indices.npy` от `comments_embedder`
- feature-gating: `compute_mean/compute_median/compute_std`, extra metrics опционально
- safe relpath join + `tp_commentsagg_unsafe_relpath_flag`
- stable schema: canonical `tp_commentsagg_*` + legacy aliases

## TODO

- `resource_costs` замеры.


