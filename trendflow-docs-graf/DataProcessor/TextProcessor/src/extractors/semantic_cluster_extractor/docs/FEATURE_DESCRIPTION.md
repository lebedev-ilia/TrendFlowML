# `semantic_cluster_extractor` — описание фич и артефактов

**Компонент:** `SemanticClusterExtractor` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **31** скаляров `tp_semclust_*` в `text_processor/text_features.npz`.  
**Контракт:** [`../../../../schemas/semantic_cluster_extractor_output_v1.json`](../../../../schemas/semantic_cluster_extractor_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

Мета таксономии — в **`result.semantic_cluster_meta`**, не в NPZ-скалярах.

**Версия:** 1.3.0 (`SemanticClusterExtractor.VERSION`).

---

## 1. Назначение

- Загрузить эмбеддинг **title / description / hashtag** из `doc.tp_artifacts` (primary + fallback).  
- **PCA** → L2-норма → ближайший центроид (**FAISS** inner product или **NumPy** cosine).  
- Выход: **`tp_semclust_id`**, **`similarity`**, **`distance`** = `1 - similarity` (косинус на нормированных векторах).

---

## 2. Группы

| Группа | Заметки |
|--------|---------|
| Зеркала конфига | `require_*`, `use_faiss`, `require_faiss`, `emit_extra_metrics` → `tp_semclust_*_enabled` |
| One-hot primary | `tp_semclust_config_primary_{title,description,hashtag}` — ровно **один** **1** |
| Наличие файла эмбеддинга | `tp_semclust_*_present` — **1** только после успешной загрузки **.npy** |
| One-hot фактического источника | `tp_semclust_source_*` — **0** или **1** (не более одной **1**); **все 0** при valid empty |
| `tp_semclust_fallback_used` | **1**, если взяли не `primary_source` |
| `tp_semclust_backend_faiss` | **1** при использовании FAISS, иначе NumPy |
| Метрики | `id`, `similarity`, `distance`; **NaN** при отсутствии вектора / mismatch |
| **Extra-блок** | `n_clusters`, `model_*_dim`, `embedding_dim`, `margin_top2`, `compute_ms` — **NaN** при **`emit_extra_metrics=False`** (в т.ч. на успешном пути; см. `_apply_extra_block`) |

---

## 3. Нормальные диапазоны (`--ranges`)

| Категория | Ожидание |
|-----------|----------|
| Бинарные 0/1 (при finite) | флаги, `tp_semclust_present`, зеркала, `config_primary_*` |
| One-hot `config_primary` | **Сумма = 1** |
| `tp_semclust_source_*` (finite) | **Сумма ∈ {0, 1}** |
| `tp_semclust_similarity` (finite) | **[-1, 1]** |
| `tp_semclust_distance` (finite) | **≈ 1 - similarity**; **∈ [0, 2]** |
| `tp_semclust_margin_top2` (finite) | обычно **[-2, 2]** (разность двух косинусов) |
| `tp_semclust_id` (finite) | **≥ 0** (id кластера) |
| `tp_semclust_compute_ms` (finite, extra) | **≥ 0**, **&lt; 1e7** мс |
| при **`emit_extra_metrics_enabled=0`** | `n_clusters`, `model_*_dim`, `embedding_dim`, `margin_top2`, `compute_ms` → **NaN** |

---

## 4. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_semantic_cluster_extractor_text_npz.py`](../utils/validate_semantic_cluster_extractor_text_npz.py)

---

## 5. Чеклист

1. **31** имя в срезе = JSON.  
2. Согласованность **`distance`** с **`1 - similarity`** при finite.  
3. Для публичных таблиц — те же коридоры, что в валидаторе.
---

## Навигация

[README (root)](../README.md) · [SCHEMA (root)](../SCHEMA.md) · [TextProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
