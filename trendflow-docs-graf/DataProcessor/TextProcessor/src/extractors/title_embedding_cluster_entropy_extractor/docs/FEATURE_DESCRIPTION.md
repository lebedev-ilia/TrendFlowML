# `title_embedding_cluster_entropy_extractor` — описание фич и артефактов

**Компонент:** `TitleEmbeddingClusterEntropyExtractor` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **24** скаляра `tp_titleclent_*` в `text_processor/text_features.npz`.  
**Контракт:** [`../../../../schemas/title_embedding_cluster_entropy_extractor_output_v1.json`](../../../../schemas/title_embedding_cluster_entropy_extractor_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

Мета: `result.title_cluster_entropy_meta` (без сырого текста).

**Версия:** 1.3.0 (`TitleEmbeddingClusterEntropyExtractor.VERSION`).

---

## 1. Назначение

- Загрузить **title**-эмбеддинг по `doc.tp_artifacts["embeddings"]["title"]["relpath"]` (совместимый `orig_dim` с PCA).  
- `reduced = pca @ v` → L2-норма → top‑**K** сходств к **L2**-нормированным центроидам (**FAISS** IP или **NumPy**).  
- **Softmax(scores / T)** → энтропия **H**, **H / log(K_used)** (при **K_used ≤ 1** нормировка **0.0**), **perp = exp(H)**.  
- **K_used** = `min(top_k_slots, n_clusters)`.

---

## 2. Группы

| Группа | Заметки |
|--------|---------|
| Gating | `tp_titleclent_present` — **1** только на успешном пути с валидным вектором |
| | `tp_titleclent_title_present` — relpath в `tp_artifacts` (не гарантирует `present=1`: файл/размерность) |
| Зеркала | `emit_extra_metrics`, `require_title_embedding`, `use_faiss`, `require_faiss`, `export_topk_distribution` |
| Слоты | `schema_top_k_slots_max` = **8**; requested / actual / `clamped` |
| | `use_faiss` vs факт: `tp_titleclent_backend_faiss` |
| **Extra-блок** | `n_clusters`, `model_*_dim`, `margin_top2`, `compute_ms` — **NaN** при **`emit_extra_metrics=False`** **или** на **empty**-ветке (`_empty_shell`) |

---

## 3. Нормальные диапазоны (`--ranges`)

| Категория | Ожидание |
|-----------|----------|
| Бинарные 0/1 (finite) | зеркала, `dim_mismatch`, `backend_faiss`, `top_k_slots_clamped` |
| `tp_titleclent_schema_top_k_slots_max` | **8** |
| `tp_titleclent_top_k_slots` | **1..8**; `requested` **≥ 1** |
| `tp_titleclent_temperature` | **> 0** (в softmax — с clamp снизу **1e-6**) |
| `entropy_raw` (finite) | **≥ 0**; `entropy_norm` **∈ [0,1]** при **K_used > 1**; при **K_used ≤ 1** нормировка **0.0** по коду |
| `perplexity` (finite) | **≥ 1** |
| `top_k_used`, `distinct_clusters_topk` (finite) | **≥ 1** при успехе |
| `margin_top2` (finite) | типично **[-2, 2]** (разность двух cosines) |
| Extra (finite) | `n_clusters` / `dims` **≥ 1**; `compute_ms` **∈ [0, 1e7]**; при `emit=0` — **все 5** extra **NaN** |

---

## 4. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_title_embedding_cluster_entropy_extractor_text_npz.py`](../utils/validate_title_embedding_cluster_entropy_extractor_text_npz.py)

---

## 5. Чеклист

1. **24** ключа, порядок — JSON / `_FEATURES_FLAT_KEYS` в `main.py`.  
2. **Empty:** метрики сигнала **NaN**, конфиг-зеркала **заполнены**.  
3. `entropy_norm=0` при **K≤1** — **не** ошибка.
---

## Навигация

[README (root)](../README.md) · [SCHEMA (root)](../SCHEMA.md) · [TextProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
