# `transcript_chunk_embedder` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `transcript_chunk_embedder` |
| Класс | `TranscriptChunkEmbedder` |
| Machine schema | `DataProcessor/TextProcessor/schemas/transcript_chunk_embedder_output_v1.json` |
| `schema_version` (`features_flat`) | `transcript_chunk_embedder_output_v1` |
| Версия реализации | `1.3.0` (см. `TranscriptChunkEmbedder.VERSION`) |

## Назначение

Эмбеддинги **по чанкам** транскрипта: token-aware разбиение (`shared_tokenizer_v1`), L2-норма на чанк, артефакты **`transcript_{source}_chunk_embeddings.npy`**, метаданные в **`doc.tp_artifacts["transcripts"][source]`** (и legacy **`transcript_chunks`**). Скаляры — **`result.features_flat`** (`tp_tchunk_*`), **всегда 16 ключей** на любой ветке (см. ниже).

## Имя источника `whisper`

Ключ **`whisper`** — **логическое имя канала** для основного ASR-текста (склейка **`doc.asr.segments`** и, при отсутствии сегментов, fallback на **`doc.transcripts["whisper"]`** после upstream decode). Это **не** утверждение о конкретной модели распознавания.

## Политика источников и Audit v3

- **Канон preflight**: транскрипт из пайплайна **Segmenter → AudioProcessor ASR** в том же **run**; профиль рекомендует **`use_asr=true`**, **`use_youtube_auto=false`** (см. `global_config.yaml`).
- Fallback **`transcripts["whisper"]`** при пустых сегментах в **`doc.asr`** задокументирован здесь и в коде (`_get_sources`); для строгого «только сегменты ASR» используйте контроль на стороне ingest / отдельный decision record.
- Опционально **`youtube_auto`**: **`use_youtube_auto`**, не включается в baseline preflight.

## Зависимости

В **`MainProcessor`** формально **`[]`**; **логически** требуется материализованный **`VideoDocument`** с **`doc.asr`** / **`transcripts`** согласно политике выше. **Downstream:** `transcript_aggregator`, Q&A, темы, stats и др.

## Модель (preflight)

Канон Audit v3: **`intfloat/multilingual-e5-large`** ([preflight §0.5](../../../../docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md)).

## `features_flat` — стабильные 16 ключей

`main.py` → `_build_features_flat`.

| Ключ | Примечание |
|------|------------|
| `tp_tchunk_present` | 1.0 если есть хотя бы один обработанный источник |
| `tp_tchunk_sources_count` | Число ключей в `results_by_source` |
| `tp_tchunk_whisper_present` | Канал `whisper` |
| `tp_tchunk_youtube_auto_present` | Канал `youtube_auto` |
| `tp_tchunk_whisper_chunks` / `tp_tchunk_youtube_chunks` | Число чанков |
| `tp_tchunk_embedding_dim` | Из `whisper`, иначе из `youtube_auto`, иначе NaN |
| `tp_tchunk_conf_*` | При **`emit_confidence_metrics=false`** → 0 / NaN; иначе из stats ASR-чанков (`whisper`) |

### Параметры чанкинга / кеша в `features_flat`

Пять ключей **`tp_tchunk_batch_size`**, **`tp_tchunk_max_chunk_tokens_model`**, **`tp_tchunk_overlap_ratio`**, **`tp_tchunk_max_chunks_total`**, **`tp_tchunk_cache_enabled`**: при **`emit_extra_metrics=false`** значения **NaN** (ключи всё равно присутствуют). При **`emit_extra_metrics=true`** — фактические параметры.

## Верхний уровень `result`

На всех ветках: **`model_name`**, **`model_version`**, **`weights_digest`** (рядом с **`device`**, **`version`**).

## Версионирование

Смена смысла ключей или их набора → bump **`transcript_chunk_embedder_output_v2`** + **`RUN_LOG.md`** + отчёт.
