# Audit v3 — `hashtag_embedder` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `HashtagEmbedder.VERSION = 1.2.0`  
**Machine schema (`features_flat`)**: `hashtag_embedder_output_v1`  
**Human schema**: [`src/extractors/hashtag_embedder/SCHEMA.md`](../../../src/extractors/hashtag_embedder/SCHEMA.md)

## TL;DR

Агрегированный эмбеддинг по **`doc.hashtags`** (canonicalize → per-tag encode → mean/max/logsumexp, опционально веса частот), артефакт **`hashtag_embedding.npy`**, **`tp_hashemb_*`** в NPZ. Исправлен дефолт **`strict_missing_hashtags=False`**, чтобы **`require_hashtags: false`** из конфига работал; **`extract_batch`** при **`require_hashtags`** снова **fail-fast** при отсутствии/неверном типе списка, как **`extract`**. Пустые ветки **`result`** дополняются **`model_name`** / **`model_version`** / **`weights_digest`**.

## Входы / выходы

- Входы: `VideoDocument.hashtags`, опционально `tp_artifacts.tags` (hint), параметры агрегации и лимитов.
- Выходы: `result` с метаданными модели на всех путях; `result.features_flat` (23 ключа); вектор в `.npy` + `tp_artifacts`.

## Acceptance

- [x] `SCHEMA.md` + `hashtag_embedder_output_v1.json` (ключи ↔ `main.py`).
- [x] Семантика `require_hashtags` согласована `extract` / `extract_batch`.
- [ ] Полный smoke с `DP_MODELS_ROOT` — при необходимости отдельная запись в `RUN_LOG.md`.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/hashtag_embedder/main.py`
- `DataProcessor/TextProcessor/schemas/hashtag_embedder_output_v1.json`
