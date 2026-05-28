# `topk_similar_titles_extractor` — описание фич и артефактов

**Компонент:** `TopKSimilarCorpusTitlesExtractor` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **29** скаляров `tp_topktitles_*` в `text_processor/text_features.npz`.  
**Контракт:** [`../../../../schemas/topk_similar_titles_extractor_output_v1.json`](../../../../schemas/topk_similar_titles_extractor_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

Top‑K id/score’ы (переменной длины) живут в **`result.topk_similar_corpus_titles`**, не в `features_flat`.

**Версия:** 1.3.0 (`TopKSimilarCorpusTitlesExtractor.VERSION`).

---

## 1. Назначение

- По **L2-нормированному** title-эмбеддингу (из `doc.tp_artifacts["embeddings"]["title"]["relpath"]`) искать **top‑K** в **офлайн-корпусе** (`dp_models`: `embeddings.npy` + `ids`).
- **FAISS HNSW + inner product** (на единичных векторах ≈ **косинус**; ответ **приближённый**) или **NumPy** полный косинус.
- `tp_topktitles_present=1` только на успешном поиске (валидный title, норма >0, dim = dim корпуса, корпус загружен в `__init__`).

---

## 2. Ключи (смысл)

| Группа | Заметки |
|--------|---------|
| `present` | **1** после успешного `search`; иначе **0** (empty / policy / флаги ошибок) |
| `disabled_by_policy`, `enabled` | зеркала `enabled` у экстрактора; при `enabled=false` — **disabled_by_policy=1**, ранний выход |
| `require_title_embedding_enabled` | при **true** — RuntimeError вместо valid empty на отсутствии/битом title |
| `k`, `corpus_size`, `dim` | запрошенный K, размер корпуса, размерность эмбеддинга |
| `backend_faiss` | **1** если фактически FAISS-индекс; **0** на numpy-ветке |
| `faiss_available` | **1** если пакет `faiss` импортируется |
| `require_faiss_*`, `allow_numpy_large_corpus_*`, `max_corpus_for_numpy` | лимиты и политика бэкенда (см. `README`) |
| `cache_*` | зеркала кэша глобального индекса |
| `export_topk_mode_*` (3 one-hot) | ровно одна **1.0** остальные **0.0** |
| `max_export_k`, `export_k_used`, `export_k_truncated_flag` | лимит экспорта в payload; `k > max_export_k` ⇒ truncated flag |
| `top1_score`, `topk_mean_score` | **NaN** при `present=0`; иначе inner product (косинус) ∈ **[-1, 1]** |
| флаги `unsafe_*`, `title_embed_missing_*`, `dim_mismatch_*`, `zero_norm_*`, `nan_inf_*` | диагностика пути/файла/вектора |

---

## 3. Нормальные диапазоны (`--ranges`)

| Категория | Ожидание |
|-----------|----------|
| Бинарные поля (см. валидатор) | **0/1** (finite) |
| `tp_topktitles_export_topk_mode_ids_only` + `..._ids_and_scores` + `..._none` | сумма **1.0** (one-hot) |
| `tp_topktitles_top1_score`, `tp_topktitles_topk_mean_score` (finite) | **[-1, 1]** |
| при `present=1` | `top1_score`, `topk_mean_score` finite; `export_k_used` finite, **≥ 0** |
| `k`, `corpus_size`, `dim`, `require_faiss_above_corpus_size`, `max_corpus_for_numpy`, `cache_ttl_s`, `cache_max_entries`, `max_export_k` (finite) | **≥ 0** |

---

## 4. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_topk_similar_titles_extractor_text_npz.py`](../utils/validate_topk_similar_titles_extractor_text_npz.py)

---

## 5. Чеклист

1. **29** имён = `topk_similar_titles_extractor_output_v1` (`allow_extra_keys: false`).  
2. Сравнение рангов между **FAISS** и **NumPy** на одном корпусе без оговорок некорректно (HNSW приближённый).
