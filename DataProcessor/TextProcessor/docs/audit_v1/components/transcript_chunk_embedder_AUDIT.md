# `transcript_chunk_embedder` — AUDIT (v1)

**Статус**: `done`  
**Компонент**: `DataProcessor/TextProcessor/src/extractors/transcript_chunk_embedder/main.py`  
**Критерии**: `TextProcessor/docs/audit_v1/TP_AUDIT_CRITERIA.md`

## 1) Назначение

Строит chunk-level embeddings транскрипта (в первую очередь `doc.asr.segments`) и сохраняет матрицу эмбеддингов как per-run sub‑artifact (`*.npy`). Для детерминированного связывания downstream‑экстракторов записывает relpath в `doc.tp_artifacts`.

## 2) Контракт входа

- **Preferred**: `VideoDocument.asr.segments[]` (AudioProcessor contract; текст берётся из `segments[].text`).
- **Legacy**: `VideoDocument.transcripts` — только если включён соответствующий флаг (gated).

## 3) Контракт выхода

- **Per-run sub-artifact**: `text_processor/_artifacts/transcript_<source>_chunk_embeddings.npy` (fixed per-run name)
- **NPZ-friendly**: `result.features_flat` (скалярные признаки наличия/объёма)
- **In-memory registry**:
  - `doc.tp_artifacts["transcript_chunks"][source] = { "transcript_id", "embeddings_relpath", "embeddings_path"(alias), "n_chunks", "embedding_dim" }`

## 4) Per-run storage / determinism

- ✅ Пишет `.npy` строго в per-run `text_processor/_artifacts/` через `artifacts_dir`.
- ✅ Не делает `glob+mtime` и не сканирует глобальные директории.
- ✅ Не возвращает абсолютные пути в `result`.

## 5) Privacy

- ✅ Raw transcript/chunks не экспортируются в `result` по умолчанию.
- ✅ В `doc.tp_artifacts` хранятся только relpath’ы (in-memory).

## 6) Model system

- Использует общий загрузчик модели (`src.core.model_registry.get_model`) и локальные веса (no-network).
- Token counting/ chunking строго использует `dp_models` tokenizer (`shared_tokenizer_v1`) без fallback.

## 7) Observability

- ✅ `system_snapshot()` + `process_memory_bytes()` присутствуют.
- ✅ `timings_s.total` возвращается.

## 8) Известные TODO / риски

TODO (не блокирует контракт):
- **resource_costs**: требуется замер и запись в `docs/models_docs/resource_costs/`.


