## `title_embedding_cluster_entropy_extractor` (Text features)

### Назначение

Вычисляет **энтропию распределения** эмбеддинга заголовка по кластерам (общая таксономия `semantic_clusters_v1` через **`dp_models`**). Проекция **PCA**, cosine к **L2-нормированным** центроидам, **softmax** с температурой на **top‑K**, затем энтропия / нормированная энтропия / perplexity.

**Версия**: 1.3.0  
**Категория**: text, clustering, entropy  
**GPU**: не требуется (опционально **FAISS** IndexFlatIP)

**Диапазоны и валидатор среза** (`text_features.npz`): [`docs/FEATURE_DESCRIPTION.md`](docs/FEATURE_DESCRIPTION.md) · [`utils/validate_title_embedding_cluster_entropy_extractor_text_npz.py`](utils/validate_title_embedding_cluster_entropy_extractor_text_npz.py)

**Контракт Audit v3**: [SCHEMA.md](./SCHEMA.md) · machine: [`schemas/title_embedding_cluster_entropy_extractor_output_v1.json`](../../schemas/title_embedding_cluster_entropy_extractor_output_v1.json) · **Audit v4:** [`../../../../docs/audit_v4/components/text_processor/title_embedding_cluster_entropy_extractor_audit_v4.md`](../../../../docs/audit_v4/components/text_processor/title_embedding_cluster_entropy_extractor_audit_v4.md) · **L2 stats:** [`../../../../storage/audit_v4/title_embedding_cluster_entropy_extractor_l2/title_embedding_cluster_entropy_extractor_audit_v4_stats.json`](../../../../storage/audit_v4/title_embedding_cluster_entropy_extractor_l2/title_embedding_cluster_entropy_extractor_audit_v4_stats.json) (tooling: `scripts/audit_v4_npz_stats.py`)

### Входы

- **Title embedding**: `title_embedder` → `doc.tp_artifacts["embeddings"]["title"]["relpath"]`
- **Кластера / PCA**: `dp_models`, spec `clusters_spec_name` (по умолчанию `semantic_clusters_v1`)

### Выходы

`result.features_flat` — **24** фиксированных ключа (все ветки); `result.title_cluster_entropy_meta`.

#### `features_flat` (сводка)

- **Зеркала:** `tp_titleclent_emit_extra_metrics_enabled`, `tp_titleclent_require_title_embedding_enabled`, `tp_titleclent_use_faiss_enabled`, `tp_titleclent_require_faiss_enabled`, `tp_titleclent_export_topk_distribution_enabled`
- **Top‑K / кламп:** `tp_titleclent_schema_top_k_slots_max` (**8**), `tp_titleclent_top_k_slots_requested`, `tp_titleclent_top_k_slots`, `tp_titleclent_top_k_slots_clamped`
- **Сигнал:** `tp_titleclent_present`, `tp_titleclent_title_present`, `tp_titleclent_entropy_raw`, `tp_titleclent_entropy_norm` (при **K≤1** → **0.0**), `tp_titleclent_perplexity`, `tp_titleclent_top_k_used`, `tp_titleclent_distinct_clusters_topk`, `tp_titleclent_temperature`, `tp_titleclent_dim_mismatch_flag`, `tp_titleclent_backend_faiss`
- **Extra (NaN** если `emit_extra_metrics=False` **или** empty**):** `tp_titleclent_n_clusters`, `tp_titleclent_model_orig_dim`, `tp_titleclent_model_reduced_dim`, `tp_titleclent_margin_top2`, `tp_titleclent_compute_ms`

Полный перечень и правила — в [SCHEMA.md](./SCHEMA.md).

#### `title_cluster_entropy_meta`

`clusters_spec_name`, `clusters_spec_version`, `clusters_weights_digest`, `cluster_db_version`, `backend`. При `export_topk_distribution=True` — `topk` (ids / probs / scores), без raw текста.

#### Верхний уровень ответа

`model_name` / `model_version` / `weights_digest`: **`null`**. `system`: снимки **`pre_init`/`post_init`** из `__init__` и **`post_process`**, **`gpu_peak_mb`**.

### Алгоритм

1. **`__init__`**: загрузка PCA и centroids через ModelManager (fail-fast); опционально FAISS; **`_init_metrics`**
2. **`extract`**: чтение `.npy` по безопасному `relpath`; проверка `orig_dim`
3. `reduced = title @ PCA` → L2 normalize
4. Top‑**K** = **min(`top_k_slots`, n_clusters)**; `top_k_slots` в рантайме — уже после клампа к **8**
5. Softmax(scores / temperature) → **H**, **H_norm**, perplexity

### Конфигурация

```python
TitleEmbeddingClusterEntropyExtractor(
    artifacts_dir=None,
    clusters_spec_name="semantic_clusters_v1",
    top_k_slots=5,
    temperature=0.1,
    export_topk_distribution=False,
    require_title_embedding=False,
    require_faiss=False,
    use_faiss=True,
    emit_extra_metrics=False,
)
```

`clusters_path` запрещён (RuntimeError).

### Valid empty semantics

- Нет `relpath` / нет файла / dim mismatch (и `require_title_embedding=False`): **`tp_titleclent_present=0`**, метрики **NaN**, ключи **все 24** присутствуют
- `require_title_embedding=True`: отсутствие или несовместимость → **RuntimeError**

### Архитектура (фактическая)

1. **Инициализация**: резолв spec → загрузка **PCA** и **centroids**; построение **FAISS** при `use_faiss` и наличии пакета
2. **Извлечение**: только **`tp_artifacts`** + `artifacts_dir` (без glob/mtime)
3. **Вычисление**: numpy / FAISS inner product на нормализованных векторах
4. **Результат**: фиксированный `features_flat` через **`_pack_features_flat`**

### Связанные компоненты

- **TitleEmbedder** — источник вектора заголовка
- **dp_models** — PCA/centroids taxonomy
- **BaseExtractor** — контракт `extract(doc)`

### Примечания

До явной фиксации corpus pack для downstream ML трактуйте фичи как **analytics** (preflight §7).
