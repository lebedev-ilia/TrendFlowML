# `embedding_pair_topk_extractor` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `embedding_pair_topk_extractor` |
| Класс | `EmbeddingPairTopKExtractor` |
| Machine schema | `DataProcessor/TextProcessor/schemas/embedding_pair_topk_extractor_output_v1.json` |
| `schema_version` | `embedding_pair_topk_extractor_output_v1` |
| Версия реализации | `1.3.0` |

## Назначение

- **Cosine similarity** между **вектором title** и **вектором description** (опционально).
- **Top‑K** косинусного сходства **title** (запрос) с **строками матрицы chunk embeddings** транскрипта (corpus), с опциональным **FAISS** (`IndexFlatIP` после L2-нормализации).

Модель **не загружается**: работа только с уже посчитанными **`*.npy`** из **`doc.tp_artifacts`**. В ответе **`model_name` / `model_version` / `weights_digest`** = **`null`**.

## Входы / артефакты

- **Title / description**: `doc.tp_artifacts["embeddings"]["title|description"]["relpath"]`.
- **Чанки**: canonical `doc.tp_artifacts["transcripts"][source]["chunk_embeddings_relpath"]`, legacy `transcript_chunks[src].embeddings_relpath`.
- Источник транскрипта выбирается по **`transcript_source_priority`** (первый доступный). Ключ **`combined`** поддерживается, если в документе есть такой блок с `chunk_embeddings_relpath` (в текущем **`TranscriptChunkEmbedder`** отдельного канала `combined` нет — см. конфиг и приоритеты).

## `features_flat` (69 ключей)

- Фиксированный порядок: **`_FEATURES_FLAT_KEYS`** в `main.py` ↔ JSON.
- **Ровно 8** слотов экспорта: **`tp_embpair_title_transcript_top1..top8`**, зеркала **`tp_pairtopk_*`**, индексы **`tp_embpair_title_transcript_top{i}_idx`**.
- Параметр **`top_k_slots`** в конфиге **клампится** до **8**: **`tp_embpair_top_k_slots`** = эффективное значение, **`tp_embpair_top_k_slots_requested`** = как в конфиге, **`tp_embpair_top_k_slots_clamped`** = 1.0 при превышении.
- **`tp_embpair_schema_slots_max`** всегда **8.0**.

## `emit_extra_metrics`

Поля **`tp_embpair_n_chunks`**, **`tp_embpair_transcript_source_*`**, **`tp_embpair_use_faiss_mode`** (скаляр 0 / 0.5 / 1, не путать с **`tp_embpair_use_faiss_mode_auto`** и т.д.), **`tp_embpair_require_faiss`**:

- при **`emit_extra_metrics=False`** → **NaN**;
- при **`True`** → числовые/булевы значения по факту прогона.

## Legacy

**`tp_pairtopk_present`** по-прежнему отражает только **наличие top‑k title↔transcript**, а не общий **`tp_embpair_present`** (который учитывает ещё и title–description).

## Метаданные ответа

**`system`**: **`pre_init` / `post_init`** из **`_init_metrics`** при создании экстрактора; **`gpu_peak_mb`** по снимкам (часто 0 — numpy/FAISS CPU).

## Версионирование

Смена ключей → **`embedding_pair_topk_extractor_output_v2`** + `RUN_LOG.md`.
---

## Навигация

[README](README.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
