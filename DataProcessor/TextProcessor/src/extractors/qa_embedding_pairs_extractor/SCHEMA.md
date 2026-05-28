# `qa_embedding_pairs_extractor` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `qa_embedding_pairs_extractor` |
| Класс | `QAEmbeddingPairsExtractor` |
| Machine schema | `DataProcessor/TextProcessor/schemas/qa_embedding_pairs_extractor_output_v1.json` |
| `schema_version` | `qa_embedding_pairs_extractor_output_v1` |
| Версия реализации | `1.3.0` |

## Назначение

Детерминированное извлечение **вопросоподобных** фрагментов (есть `?` / `？`, список вопросительных слов по языкам) из **title**, **description**, **транскрипта** (`doc.asr.segments`, опционально legacy `doc.transcripts`), **комментариев**; **L2-нормализованные эмбеддинги** строк через **`get_model_with_meta`** (`dp_models`). Артефакт: **`qa_question_embeddings.npy`** `(N, D)`.

**Имя класса историческое:** пары «вопрос–ответ» **не** строятся — только матрица вопросов и счётчики по источникам.

## Входы / артефакты

- Реестр: **`doc.tp_artifacts["qa"]["question_embeddings"]`**: `relpath`, `num_questions`, `embedding_dim`, `per_source_counts`, `model_name`, `model_version`, `weights_digest`, опционально `hashes_relpath` / `source_ids_relpath`.
- Опционально: **`qa_question_hashes.npy`**, **`qa_question_source_ids.npy`** (privacy-safe).

## `features_flat` (34 ключа)

Фиксированный порядок: `_FEATURES_FLAT_KEYS` в `main.py` ↔ JSON-схема. **`allow_extra_keys: false`**.

### `emit_extra_metrics`

При **`emit_extra_metrics=False`** поля **`tp_qa_questions_per_min`**, **`tp_qa_questions_per_1k_chars`**, **`tp_qa_mean_cosine_to_centroid`** остаются **NaN**; **`tp_qa_mean_cosine_to_centroid_present`** = **0**.

При **`emit_extra_metrics=True`** и **valid empty** (нет вопросов): **`tp_qa_questions_per_min`** = **0.0** если **`audio_duration_sec`** конечен и **> 0**, иначе **NaN**; **`tp_qa_questions_per_1k_chars`** = **NaN**.

## Зависимости оркестратора

В **`MainProcessor`** заявлена зависимость от **`TranscriptChunkEmbedder`** (операционный порядок конвейера; из чанковых файлов код не читает).

## Метаданные ответа

Верхний уровень **`extract()`**: **`model_name`**, **`model_version`**, **`weights_digest`**; **`system.pre_init` / `post_init`** из **`_init_metrics`**; **`gpu_peak_mb`** по снимкам GPU (как у других эмбеддеров).

## Версионирование

Смена ключей → **`qa_embedding_pairs_extractor_output_v2`** + запись в `DataProcessor/docs/audit_v3/RUN_LOG.md`.
