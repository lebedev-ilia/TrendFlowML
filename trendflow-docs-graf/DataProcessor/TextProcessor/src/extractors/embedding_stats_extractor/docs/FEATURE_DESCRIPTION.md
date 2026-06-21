# `embedding_stats_extractor` — описание фич и артефактов

**Компонент:** `EmbeddingStatsExtractor` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **39** скаляров `tp_embstats_*` в `text_processor/text_features.npz`.  
**Контракт:** [`../../../../schemas/embedding_stats_extractor_output_v1.json`](../../../../schemas/embedding_stats_extractor_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

**Версия:** 1.2.0 (`EmbeddingStatsExtractor.VERSION`).

---

## 1. Назначение

- По матрице **chunk embeddings** транскрипта: L2-норма вектора покомпонентной **дисперсии** по чанкам (`l2_variance`), **топ-вариации** по компонентам в **8** фиксированных слотах `topvar_1…8` (кламп `top_k_slots` ≤ 8).  
- Опционально: **энтропия** (и нормировка, perplexity) по `topic_probs` из `doc.tp_artifacts["topics"]["topk_distribution"]` (upstream `semantics_topics_keyphrases`).  
- Источник чанков: только `whisper` / `youtube_auto` в схеме; приоритет задаётся конфигом (`transcript_source_priority` фильтруется к этим ключам).

---

## 2. Ключевые поля

| Группа | Заметки |
|--------|---------|
| `tp_embstats_present` | **1** только если рассчитан блок дисперсии (`l2_variance` не `None` → `n_chunks ≥ min_chunks_required` и валидная матрица) |
| Слоты `topvar_1..8` | Убывающие по величине (max component variance = `topvar_1`); лишние слоты **NaN** |
| `tp_embstats_schema_topvar_slots_max` | **8.0** |
| `tp_embstats_*_slots*`, `min_chunks`, `topk`, `variance_ddof` | Аудит конфигурации (float-скаляры) |
| Topic | При отсутствии/невалидных probs — **NaN** и флаги `topic_probs_*` / `topic_entropy_present` |
| `tp_embstats_load_ms`, `tp_embstats_compute_ms` | **NaN** при **`emit_extra_metrics=False`**; иначе **мс**, **≥ 0** при finite |
| Source | `source_used_whisper` + `source_used_youtube_auto` → сумма **0** или **1** |

---

## 3. Нормальные диапазоны (`--ranges`)

| Категория | Ожидание |
|-----------|----------|
| Бинарные флаги (finite) | 0/1: `enabled`, `disabled`, gating, диагностика, `present` (кроме отдельных float-полей ниже) |
| `tp_embstats_schema_topvar_slots_max` | **8.0** |
| `tp_embstats_l2_variance`, `tp_embstats_topvar_*` (finite) | **≥ 0** |
| `tp_embstats_topic_entropy` (finite) | **≥ 0** |
| `tp_embstats_topic_entropy_norm` (finite) | **∈ [0, 1]** (для \(K>1\); иначе в коде может быть **NaN**) |
| `tp_embstats_topic_perplexity` (finite) | **≥ 1** |
| `tp_embstats_n_chunks`, `tp_embstats_dim` (finite) | **≥ 0** / **≥ 1** соответственно |
| `tp_embstats_min_chunks_required` | **≥ 0** (в типичном конфиге **≥ 1**) |

---

## 4. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_embedding_stats_extractor_text_npz.py`](../utils/validate_embedding_stats_extractor_text_npz.py)

---

## 5. Чеклист

1. Срез в NPZ: **39** имён, совпадающих с machine JSON.  
2. `tp_embstats_source_used_whisper + tp_embstats_source_used_youtube_auto` ∈ **{0, 1}** при обоих finite.  
3. One-hot политик по слотам: ровно **8** ключей `topvar_*` в схеме; экспорт — до `top_k_slots` (после клампа).
---

## Навигация

[README (root)](../README.md) · [SCHEMA (root)](../SCHEMA.md) · [TextProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
