# `semantic_cluster_extractor` — AUDIT (v1)

**Статус**: `completed`  
**Компонент**: `src/extractors/semantic_cluster_extractor/main.py`

## Резюме

Компонент приведён к A-policy:

- deterministic inputs через `doc.tp_artifacts["embeddings"][...]["relpath"]` (без `glob+mtime`)
- strict model/assets loading через `dp_models` (`semantic_clusters_v1`) + `weights_digest` для воспроизводимости
- valid empty semantics: если embedding отсутствует → `tp_semclust_present=0` и метрики = NaN (без fake vectors)
- прозрачная политика источника: `primary_source`, `allow_fallback_sources`, `require_primary_source`, флаг `tp_semclust_fallback_used`
- dim-mismatch обрабатывается безопасно: `tp_semclust_dim_mismatch_flag=1` + NaNs (или fail-fast при `require_embedding=True`)
- FAISS включается управляемо (`use_faiss` / `require_faiss`), backend отражается в `tp_semclust_backend_faiss` и `semantic_cluster_meta.backend`

## Модели / assets

- `dp_models` spec: `DataProcessor/dp_models/spec_catalog/text/semantic_clusters_v1.yaml`
- assets (bundled): `DataProcessor/dp_models/bundled_models/text/semantic_clusters_v1/{pca.npy, centroids.npy, clusters.jsonl}`

Примечание: `clusters.jsonl` предназначен для UI/интерпретации (id → name/group), при этом компонент в output не пишет текстовые поля.


