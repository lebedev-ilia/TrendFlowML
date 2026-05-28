# `transcript_chunk_embedder` — описание фич и артефактов

**Компонент:** `TranscriptChunkEmbedder` ([`../main.py`](../main.py))  
**Вклад в NPZ:** **16** скаляров `tp_tchunk_*` в `text_processor/text_features.npz`.  
**Контракт:** [`../../../../schemas/transcript_chunk_embedder_output_v1.json`](../../../../schemas/transcript_chunk_embedder_output_v1.json) · [`../SCHEMA.md`](../SCHEMA.md) · [`../README.md`](../README.md).

Матрицы чанков: `transcript_{whisper|youtube_auto}_chunk_embeddings.npy` в per-run `artifacts` (relpath в `doc.tp_artifacts`, не в `features_flat`).

**Версия:** 1.3.0 (`TranscriptChunkEmbedder.VERSION`).

---

## 1. Назначение

- Текст транскрипта по источникам (**whisper** из `doc.asr` / `transcripts`, **youtube_auto** из `transcripts`) → token-aware чанки (`shared_tokenizer_v1`) → батч-энкод → L2-норм на чанк.
- **`tp_tchunk_present=1`**, если есть хотя бы один непустой `results_by_source`.

---

## 2. Ключи (смысл)

| Группа | Заметки |
|--------|---------|
| `present`, `sources_count` | агрегация выполнена; число источников с чанками |
| `*_present` (whisper / youtube_auto) | **0/1** по факту наличия источника в `results_by_source` |
| `*_chunks` | число чанков по источнику (0 если источника не было) |
| `embedding_dim` | dim вектора; **NaN** при полном empty |
| Confidence (gated `emit_confidence_metrics`) | `conf_present` **0/1**; mean/min/max по сегментам whisper — **NaN** при выключенной ветке или без метрик |
| Extra (gated `emit_extra_metrics`) | `batch_size`, `max_chunk_tokens_model`, `overlap_ratio`, `max_chunks_total`, `cache_enabled` — при **off** все **пять NaN** (`cache_enabled` тогда не 0/1) |

---

## 3. Нормальные диапазоны (`--ranges`)

| Категория | Ожидание |
|-----------|----------|
| `tp_tchunk_present`, `tp_tchunk_whisper_present`, `tp_tchunk_youtube_auto_present`, `tp_tchunk_conf_present` | **0/1** (finite) |
| `tp_tchunk_sources_count` | **≥ 0** (целое в float) |
| `tp_tchunk_whisper_chunks`, `tp_tchunk_youtube_chunks` | **≥ 0** |
| `tp_tchunk_embedding_dim` | **NaN** (empty) или finite **≥ 1** |
| `tp_tchunk_conf_mean` / `min` / `max` (finite) | типично **[0, 1]** (ASR confidence) |
| при `emit_extra_metrics` (любое из пяти полей finite) | `overlap_ratio` ∈ **[0, 1)**; остальные tuning-поля **> 0**; `tp_tchunk_cache_enabled` **0/1** |

---

## 4. Тайминги

В ответе экстрактора: **`timings_s.total`** (секунды, ≥ 0).  
В NPZ: **`payload.timings_by_extractor["TranscriptChunkEmbedder"]`** — ожидается ключ **`total`**. Проверка: **`--timings`** в валидаторе.

---

## 5. Инструменты

- L2: [`../scripts/audit_v4_npz_stats.py`](../scripts/audit_v4_npz_stats.py)  
- Валидатор: [`../utils/validate_transcript_chunk_embedder_text_npz.py`](../utils/validate_transcript_chunk_embedder_text_npz.py)

---

## 6. Чеклист

1. **16** имён = `transcript_chunk_embedder_output_v1` (`allow_extra_keys: false`).  
2. Дефолт **`emit_extra_metrics=false`** → пять полей NaN в merged NPZ — **норма**, не ошибка таблицы.
