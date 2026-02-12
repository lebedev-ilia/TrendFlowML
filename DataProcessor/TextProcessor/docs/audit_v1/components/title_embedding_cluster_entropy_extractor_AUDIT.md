# `title_embedding_cluster_entropy_extractor` — AUDIT (v1)

**Статус**: `done`  
**Компонент**: `src/extractors/title_embedding_cluster_entropy_extractor/main.py`

## Резюме

Компонент приведён к A-policy:

- deterministic input: читает title embedding строго через `doc.tp_artifacts["embeddings"]["title"]["relpath"]` (без `glob+mtime`)
- strict model loading: использует `dp_models` spec `semantic_clusters_v1` (PCA + centroids) и фиксирует `weights_digest`
- shared taxonomy: энтропия считается по тем же centroid’ам, что и `semantic_cluster_extractor` (через PCA‑reduced space)
- valid empty semantics: при отсутствии входа возвращает `tp_titleclent_present=0` и метрики = NaN (без fake vectors)
- fail-fast режим: `require_title_embedding=True` делает отсутствие/несовместимость входа ошибкой
- path safety: relpath не может “вылезти” за пределы `artifacts_dir`
- управляемый FAISS backend: `use_faiss/require_faiss` + флаг `tp_titleclent_backend_faiss`

### Выходные фичи (ключевые)

- `tp_titleclent_entropy_raw`, `tp_titleclent_entropy_norm`, `tp_titleclent_perplexity`
- `tp_titleclent_top_k_slots`, `tp_titleclent_top_k_used`, `tp_titleclent_temperature`
- флаги: `tp_titleclent_title_present`, `tp_titleclent_dim_mismatch_flag`, `tp_titleclent_backend_faiss`

## Модели / assets

- `dp_models` spec: `DataProcessor/dp_models/spec_catalog/text/semantic_clusters_v1.yaml`
- assets (bundled): `DataProcessor/dp_models/bundled_models/text/semantic_clusters_v1/{pca.npy, centroids.npy, clusters.jsonl}`


