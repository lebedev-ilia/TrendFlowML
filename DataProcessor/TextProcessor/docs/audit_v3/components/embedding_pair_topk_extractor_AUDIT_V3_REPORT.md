# Audit v3 — `embedding_pair_topk_extractor` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `EmbeddingPairTopKExtractor.VERSION = 1.3.0`  
**Machine schema (`features_flat`)**: `embedding_pair_topk_extractor_output_v1`  
**Human schema**: [`src/extractors/embedding_pair_topk_extractor/SCHEMA.md`](../../../src/extractors/embedding_pair_topk_extractor/SCHEMA.md)

## TL;DR

**69** фиксированных ключей `tp_embpair_*` + legacy `tp_pairtopk_*`; **8** слотов top‑K/индексов; **кламп** `top_k_slots` ≤ **8** с флагами requested/clamped. **`emit_extra_metrics`**: блок источника/FAISS/chunk count **всегда в schema**, при выкл. → **NaN**. **`_init_metrics`**, **`_gpu_peak_mb`**, верхний уровень **`model_*`/`weights_digest`**: **`null`**. Legacy **`tp_pairtopk_present`** семантика **не** сливалась с **`tp_embpair_present`**.

## Входы / выходы

- Входы: **`tp_artifacts`** title/description vectors, transcript chunk matrix relpath.
- Выходы: **`features_flat`** только скаляры (без сырых векторов в NPZ).

## Acceptance

- [x] `SCHEMA.md` + `embedding_pair_topk_extractor_output_v1.json` (69 keys ↔ `main.py`).
- [x] Фиксированные слоты, extra block, init/GPU snapshots.
- [ ] Полный smoke — `RUN_LOG.md` при прогоне.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/embedding_pair_topk_extractor/main.py`
- `DataProcessor/TextProcessor/schemas/embedding_pair_topk_extractor_output_v1.json`
