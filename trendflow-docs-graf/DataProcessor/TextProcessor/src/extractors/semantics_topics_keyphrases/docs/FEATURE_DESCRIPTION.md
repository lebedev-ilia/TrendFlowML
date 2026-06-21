# `semantics_topics_keyphrases` — описание фич и артефактов

**Компонент:** `SemanticTopicExtractor` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **116** скаляров `tp_topics_*` в `text_processor/text_features.npz`.  
**Контракт:** [`../../../../schemas/semantics_topics_keyphrases_output_v1.json`](../../../../schemas/semantics_topics_keyphrases_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

**Не в `features_flat`:** список сырьевых keyphrases — только в `result.tp_topics_keyphrases_raw` при `export_keyphrases_mode=raw` и наличии фраз; плотный `.npy` — `tp_topics_keyphrase_embeddings.npy` при `enable_keyphrase_embeddings`.

**Версия:** 2.1.0 (`SemanticTopicExtractor.VERSION`).

---

## 1. Назначение

- **Темы:** retrieval по фиксированной taxonomy (промпты + эмбеддинги, `dp_models`), top-K по `max` score с промпта, softmax по температуре, слоты **1..8** (`SCHEMA_MAX_TOPIC_SLOTS`); фактически заполняется префикс длины `top_k_slots` (остальные **NaN**).  
- **Keyphрызы:** детерминированный n-gram scorer (1–3 слова); **hashed**-слоты **1..16** (`SCHEMA_MAX_KP_SLOTS`), пишутся не больше `keyphrase_slots` и не больше числа фраз.  
- **Extra:** 5 полей `tp_topics_extra_*` (тайминги и digest u24) — **NaN** при `emit_extra_metrics=False` (см. `_fill_extra_metrics` / `_apply_extra_nans`).

---

## 2. Группы полей

| Группа | Заметки |
|--------|---------|
| Gating | `tp_topics_present` (**1** только при непустом объединённом тексте, `enabled=True`) |
| | `tp_topics_disabled_by_policy` (**1** при `enabled=False`) |
| One-hot `transcript_source_policy` | `asr_only` / `asr_then_legacy` / `legacy_only` — **ровно один 1.0** |
| One-hot `export_keyphrases_mode` | `raw` / `hashed` / `none` — **ровно один 1.0** |
| Слоты top-K | `top_k_slots_requested` / `top_k_slots` / `clamped`, `keyphrase_*` — clamp к **8** и **16** |
| Topics | `tp_topics_topic_top{i}_id|score|prob` — **NaN** в неиспользуемых слотах; score ~ cosine **[-1,1]**, prob **softmax** по выбранному top-K **≥0**, сумма **= 1** по конечным prob |
| | `entropy_topk` / `norm` / `perplexity` — **NaN** без распределения (нет тем, `enable_topic_distribution=False`, `present=0`, …) |
| Keyphrases | `count`, `dim`, `keyphrase_score_top1|mean` — **NaN** score при отсутствии фраз |
| KP-слоты | `present` 0/1; при **1** — `hash01` = первый байт SHA256 **0..255**, `len` = длина строки; при **0** — `hash`/`len` **NaN** (см. `_nan_kp_slots`) |
| Style | `style_faq_qmarks` = число символов **`?`** в тексте; остальные — 0/1 |
| Extra | 5 полей; при `emit_extra_metrics` и успешных данных — тайминги **≥ 0** мс, digest u24 **0..16777215** |

---

## 3. Нормальные диапазоны (`--ranges`)

| Категория | Ожидание |
|-----------|----------|
| Бинарные 0/1 (finite) | зеркала `enable_*`, `has_*`, флаги стиля, `emit_extra_metrics`, clamp-флаги, `tp_topics_kp_top*_present` |
| `tp_topics_text_chars` | **≥ 0** |
| `tp_topics_schema_topic_slots_max` | **8**; `tp_topics_schema_kp_slots_max` — **16** |
| `tp_topics_temperature` | **> 0** (в коде clamp снизу **1e-6**) |
| `tp_topics_top_k_slots` / requested | **1..8**; `keyphrase_slots` **0..16** |
| `entropy` (finite) | **≥ 0**; `entropy_norm` **∈ [0,1]** при **K>1**; `perplexity` **≥ 1** |
| `keyphrases_count` (finite) | **≥ 0** |
| `kp_top*_hash01` (present=1) | **0..255** |
| Extra (finite, `emit_extra`) | ms **∈ [0, 1e7]**; digests u24 **∈ [0, 0xFFFFFF]** |
| при `emit_extra_metrics_enabled=0` | все **`tp_topics_extra_*`** **NaN** |

---

## 4. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_semantics_topics_keyphrases_text_npz.py`](../utils/validate_semantics_topics_keyphrases_text_npz.py)

---

## 5. Чеклист

1. **116** имён = `semantics_topics_keyphrases_output_v1`.  
2. One-hot: transcript + export mode, сумма **1** каждая.  
3. Consistency: `keyphrase_slots` ≤ 16, `top_k_slots` ≤ 8; **extra** **NaN** при выключенном `emit`.
---

## Навигация

[README (root)](../README.md) · [SCHEMA (root)](../SCHEMA.md) · [TextProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
