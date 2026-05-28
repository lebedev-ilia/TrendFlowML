# `embedding_pair_topk_extractor` — описание фич и артефактов

**Компонент:** `EmbeddingPairTopKExtractor` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **69** скаляров (`tp_embpair_*` и legacy `tp_pairtopk_*`) в `text_processor/text_features.npz`.  
**Контракт:** [`../../../../schemas/embedding_pair_topk_extractor_output_v1.json`](../../../../schemas/embedding_pair_topk_extractor_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

**Версия:** 1.3.0 (`EmbeddingPairTopKExtractor.VERSION`).

---

## 1. Назначение

- Косинус **title↔description** (опционально).  
- **Top‑K** косинусов **title** (запрос) с **строками матрицы** chunk-эмбеддингов транскрипта; опционально **FAISS** `IndexFlatIP` на L2-нормализованных векторах.  
- Модель **не** загружается; только `*.npy` из `doc.tp_artifacts`. В ответе `model_name` / `model_version` / `weights_digest` = **null**.  
- Слоты экспорта **фиксированы: 8** (`top1`…`top8`); `top_k_slots` в конфиге **клампится** к 8: `tp_embpair_top_k_slots_requested`, `tp_embpair_top_k_slots`, `tp_embpair_top_k_slots_clamped`, `tp_embpair_schema_slots_max=8`.

---

## 2. Ключевые группы

| Группа | Примеры | Заметки |
|--------|---------|--------|
| Сводный present | `tp_embpair_present` | 1, если есть **хотя бы** title↔desc **или** title↔chunks top-k |
| Legacy present | `tp_pairtopk_present` | Только **title↔transcript top-k**, **не** зеркалит `tp_embpair_present` |
| Косинусы / слоты | `tp_embpair_title_desc_cosine`, `tp_embpair_title_transcript_top1..8`, `tp_pairtopk_title_transcript_top*`, `topk_max` / `mean` | Косинусы **∈ [−1, 1]** при finite; слоты **NaN** если не экспортированы / нет данных |
| Индексы | `tp_embpair_title_transcript_top{i}_idx` | Индексы чанков; **NaN** если нет; при finite **≥ 0** |
| Флаги входов / ошибок | `*_present`, `*_flag` | 0/1 |
| Конфиг (аудит) | `tp_embpair_top_k`, `top_k_slots`, FAISS triplet `use_faiss_mode_auto/never/always`, `min_corpus_for_faiss`, `require_*_enabled` | Ровно **один** из triplet = 1.0 (режим FAISS) |
| Extra (`emit_extra_metrics`) | `tp_embpair_n_chunks`, one-hot `transcript_source_*`, `tp_embpair_use_faiss_mode` (0 / 0.5 / 1), `tp_embpair_require_faiss` | При **`emit_extra_metrics=False`** → **все NaN** в этом блоке |

**Тайминги:** в NPZ **нет** `*_ms`; только `timings_s.total` в ответе `extract()`.

---

## 3. Нормальные диапазоны (`--ranges`)

| Категория | Ожидание |
|-----------|----------|
| Бинарные 0/1 (при finite) | presence/diagnostic/export/require/FAISS one-hot (кроме отдельных полей ниже) |
| Косинус-поля (при finite) | **[-1, 1]** |
| `tp_embpair_schema_slots_max` | **8.0** |
| `tp_embpair_top_k`, `tp_embpair_top_k_slots`, `tp_pairtopk_top_k`, `tp_embpair_top_k_slots_requested` | **≥ 1** (как в коде) |
| `tp_embpair_min_corpus_for_faiss` | **≥ 0** |
| `tp_embpair_title_transcript_top*_idx` (finite) | **≥ 0** |
| `tp_embpair_use_faiss_mode` (finite, extra) | **0.0, 0.5 или 1.0** |
| `tp_embpair_n_chunks` (finite, extra) | **≥ 0** |

---

## 4. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_embedding_pair_topk_extractor_text_npz.py`](../utils/validate_embedding_pair_topk_extractor_text_npz.py)  
- HTML: `text_processor/_render/embedding_pair_topk_extractor_report.html` ([`../render.py`](../render.py))

---

## 5. Чеклист

1. `meta.status=ok` → **69** имён схемы в `feature_names`.  
2. `len(feature_values)==len(feature_names)`.  
3. Legacy: `tp_pairtopk_title_transcript_top{i}` = `tp_embpair_title_transcript_top{i}` (дублирование по смыслу).
