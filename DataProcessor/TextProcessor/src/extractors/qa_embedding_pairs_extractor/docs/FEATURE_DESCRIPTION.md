# `qa_embedding_pairs_extractor` — описание фич и артефактов

**Компонент:** `QAEmbeddingPairsExtractor` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **34** скаляров `tp_qa_*` в `text_processor/text_features.npz`.  
**Контракт:** [`../../../../schemas/qa_embedding_pairs_extractor_output_v1.json`](../../../../schemas/qa_embedding_pairs_extractor_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

Плотная матрица вопросов: **`qa_question_embeddings.npy`** (не попадает в `features_flat`).

**Версия:** 1.3.0 (`QAEmbeddingPairsExtractor.VERSION`).

---

## 1. Назначение

- Найти **вопросоподобные** сегменты (regex: слова из словаря + `?` / `？`) в **title / description / transcript / comments**, в лимитах `max_questions_*`, опционально **dedup**.  
- Закодировать фразы через **sentence-transformers** (`get_model_with_meta` / `dp_models`), L2-нормировка **по строкам**.  
- Имя «pairs» **историческое** — **пар Q–A** в смысле датасета **не** строятся.

---

## 2. Группы полей

| Группа | Заметки |
|--------|---------|
| Сводка | `tp_qa_present` — **1** только если есть ≥1 вопрос и записан артефакт эмбеддингов |
| Счётчики | `tp_qa_num_questions`, `tp_qa_q_title` … `tp_qa_q_comments` — целые как float; **сумма по источникам = `num_questions`** |
| Dim | `tp_qa_embedding_dim` — **NaN** при `present=0` |
| Политика транскрипта | one-hot: `tp_qa_transcript_source_policy_*` (ровно один **1**) |
| Gating | `tp_qa_use_*`, `tp_qa_enabled`, `tp_qa_disabled_by_policy` |
| Конфиг (аудит) | `require_min_questions`, `max_*`, `min_chars_per_question`, … |
| Артефакты | `tp_qa_write_*_enabled`, `tp_qa_hashes_written`, `tp_qa_source_ids_written` |
| Extra (`emit_extra_metrics`) | `tp_qa_questions_per_min`, `tp_qa_questions_per_1k_chars` — **NaN**, если **extra выкл**; при **extra вкл** и **0** вопросов — per_min может быть **0.0** при валидной длительности, per_1k — **NaN** |
| Центроид | `tp_qa_mean_cosine_to_centroid` — **NaN** если &lt;2 эмбеддингов / extra off; `tp_qa_mean_cosine_to_centroid_present` **1** только при **extra**, **N≥2** и валидном центроиде |

В **NPZ** **нет** флага `emit_extra_metrics` — валидатор не восстанавливает его, только **коридоры** и согласованность счётчиков.

---

## 3. Нормальные диапазоны (`--ranges`)

| Категория | Ожидание |
|-----------|----------|
| Бинарные 0/1 (при finite) | `enabled`, `disabled`, `present`, policy/use/dedup/write флаги, `hashes_written`, `source_ids_written`, `mean_cosine_to_centroid_present` |
| one-hot policy (3) | **Сумма = 1** |
| `tp_qa_num_questions` и сумма `tp_qa_q_*` | **Равенство** при finite |
| `tp_qa_mean_cosine_to_centroid` (finite) | **[-1, 1]** |
| Числовые лимиты конфига (finite) | **≥ 0** |
| `tp_qa_present=1` | `tp_qa_embedding_dim` **≥ 1** (finite) |

---

## 4. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_qa_embedding_pairs_extractor_text_npz.py`](../utils/validate_qa_embedding_pairs_extractor_text_npz.py)

---

## 5. Чеклист

1. Срез **34** имён = `qa_embedding_pairs_extractor_output_v1`.  
2. Сумма счётчиков по источникам = `num_questions`.  
3. Матрица — в `qa_question_embeddings.npy`, не в NPZ-таблице.
