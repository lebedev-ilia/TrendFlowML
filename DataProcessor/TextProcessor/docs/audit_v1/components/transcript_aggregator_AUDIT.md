# Audit: `transcript_aggregator` (TranscriptAggregatorExtractor)

**Дата**: 2026-01-29  
**Статус**: `done`  
**Критерии**: `TextProcessor/docs/audit_v1/TP_AUDIT_CRITERIA.md`

---

## 1) Summary

Приведено к принятым решениям (Round‑1):
- **A1/A2/A3/A4/A5/A6/A7/A8/A10/A11/A12/A13/A14**
- per-run sub-artifacts для `.npy` в `text_processor/_artifacts/`
- source-of-truth transcript: **AudioProcessor `doc.asr`** (legacy `doc.transcripts` ограничен)
- deterministic linking между chunk embedder ↔ aggregator через `doc.tp_artifacts` (без glob/mtime)
- no absolute paths в `result`, только `features_flat` + manifest artifacts
- streaming aggregation + `compute_std` (по аргументу; иначе `NaN`)

---

## 2) Архитектура и зависимости

### 2.1 Внутренний граф (TextProcessor)

Зависимости:
- `TranscriptAggregatorExtractor` **требует** выполнения `TranscriptChunkEmbedder` раньше в том же run.
- `EmbeddingSourceIdExtractor` должен выбирать primary embedding **детерминированно** (без glob/mtime).

### 2.2 Per-run storage

Все `.npy` сохраняются в:
- `result_store/<platform>/<video>/<run>/text_processor/_artifacts/*.npy`

и перечисляются в `manifest.json.components[].artifacts[]`.

---

## 3) Контракт входа/выхода

### Input

- `doc.asr` (preferred) или legacy источники (ограниченно)
- `doc.tp_artifacts` от `TranscriptChunkEmbedder`:
  - canonical: `doc.tp_artifacts["transcripts"][source]["chunk_embeddings_relpath"]`
  - legacy: `doc.tp_artifacts["transcript_chunks"][source]["embeddings_relpath"]`

### Output

- `result.features_flat` (`tp_tragg_*`) — только числовые скаляры
- `.npy` sub-artifacts (fixed per-run names): `transcript_<source>_agg_mean.npy`, `transcript_<source>_agg_max.npy`
  - combined: `transcript_combined_agg_mean.npy`, `transcript_combined_agg_max.npy`

---

## 4) Empty / Error семантика

- Если нет transcript → valid empty (на уровне chunk embedder), aggregator не должен запускаться (profile-level) либо будет fail-fast из-за missing dependency.
- Если transcript есть, но нет chunk embeddings:
  - `require_chunks=True` → **error** (dependency missing)
  - иначе → source absent (valid empty на уровне источника), без fake vectors

---

## 5) Privacy / Observability

- Raw transcript не пишется в output/manifest/logs.
- Любые empty/error должны логгироваться и фиксироваться в manifest (глобальный критерий TP).

---

## 6) Открытые задачи для закрытия аудита

1. Довести весь embedding пайплайн до полного отсутствия JSON sidecar’ов (если какие-то ещё остались вне cache).
2. Добавить measured resource_costs JSON в `docs/models_docs/resource_costs/`.
3. Воспроизводимый smoke-run окружения (с `numpy/torch`).


