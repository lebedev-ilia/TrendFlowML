# Audit v3 — `embedding_shift_indicator_extractor` — отчёт

**Дата**: 2026-04-01  
**Версия компонента**: `1.3.0`  
**Machine schema (`features_flat`)**: `embedding_shift_indicator_extractor_output_v1`  
**Human schema**: [`src/extractors/embedding_shift_indicator_extractor/SCHEMA.md`](../../../src/extractors/embedding_shift_indicator_extractor/SCHEMA.md)

## TL;DR

**27** фиксированных ключей **`tp_embshift_*`**; **`tp_embshift_chunk_embed_missing_flag`**; зеркало **`tp_embshift_emit_extra_metrics_enabled`**; **`load_ms`/`compute_ms`** — **NaN** при **`emit_extra_metrics=False`**; исправлен **`sys_after`** на ветке «файл не найден»; поиск relpath по **`transcripts`[][]** не требует **`transcript_chunks`**. **`model_*`/`weights_digest`** = **`null`**; **`_init_metrics`**, **`gpu_peak_mb`**.

## Acceptance

- [x] `SCHEMA.md` + `embedding_shift_indicator_extractor_output_v1.json` ↔ `main.py`.
- [x] Tier **analytics**; legacy путь задокументирован.
- [x] Hotfix: missing-file branch; registry canonical без `transcript_chunks`.

## Файлы

- `DataProcessor/TextProcessor/src/extractors/embedding_shift_indicator_extractor/main.py`
- `DataProcessor/TextProcessor/schemas/embedding_shift_indicator_extractor_output_v1.json`

## Run-log

[`DataProcessor/docs/audit_v3/RUN_LOG.md`](../../../../docs/audit_v3/RUN_LOG.md)
