# Audit v3 — `embedding_source_id_extractor` — отчёт

**Дата**: 2026-04-01  
**Версия компонента**: `1.3.0`  
**Machine schema (`features_flat`)**: `embedding_source_id_extractor_output_v1`  
**Human schema**: [`src/extractors/embedding_source_id_extractor/SCHEMA.md`](../../../src/extractors/embedding_source_id_extractor/SCHEMA.md)

## TL;DR

**13** фиксированных **`tp_embid_*`**; **`strict_missing_primary`** управляет и отсутствием primary, и post-path ошибками (unsafe / missing file / load / empty / non-finite): при **`False`** — soft empty + **`embedding_source_id.error`**; разведены **`model_name`** и **`model_version`** во вложенном dict; верхний уровень **`model_*`/`weights_digest`** = **`null`**; **`_init_metrics`**, **`gpu_peak_mb`**.

## Acceptance

- [x] `SCHEMA.md` + `embedding_source_id_extractor_output_v1.json` ↔ код.
- [x] Одинаковый состав `features_flat` на всех ветках.
- [x] Закрыт последний экстрактор из списка **22** preflight.

## Файлы

- `DataProcessor/TextProcessor/src/extractors/embedding_source_id_extractor/main.py`
- `DataProcessor/TextProcessor/schemas/embedding_source_id_extractor_output_v1.json`

## Run-log

[`DataProcessor/docs/audit_v3/RUN_LOG.md`](../../../../docs/audit_v3/RUN_LOG.md)
