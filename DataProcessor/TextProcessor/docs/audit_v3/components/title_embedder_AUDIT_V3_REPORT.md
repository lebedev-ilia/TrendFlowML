# Audit v3 — `title_embedder` — отчёт

**Дата**: 2026-04-01  
**Версия кода**: `TitleEmbedder.VERSION = 1.2.0`  
**Machine schema (`features_flat`)**: `title_embedder_output_v1`  
**Human schema**: [`src/extractors/title_embedder/SCHEMA.md`](../../../src/extractors/title_embedder/SCHEMA.md)

## TL;DR

Экстрактор считает L2-нормализованный эмбеддинг **`doc.title`** через `dp_models`, опционально пишет **`title_embedding.npy`**, регистрирует относительный путь в **`doc.tp_artifacts`**, отдаёт **16** скалярных полей в **`result.features_flat`** (`tp_titleemb_*`). Пустой title — валидный empty без encode; **`require_title`** — fail-fast. Полный Audit v3 preflight задаёт модель **`intfloat/multilingual-e5-large`** (профиль/config, не обязательно default в коде).

## Входы / выходы

- Входы: `VideoDocument.title`, параметры кеша/устройства/модели, флаги `require_title`, `compute_embedding`, `write_artifact`, `compute_raw_norm`.
- Выходы: `result.features_flat`; top-level `model_name`, `model_version`, `weights_digest`, `system`, `timings_s`, `error`; плотный вектор только в `.npy` + `tp_artifacts`.

## Принятые решения

1. Контракт скаляров фиксируется JSON-схемой; вектор — отдельный артефакт.
2. Для downstream в том же процессе — `doc.tp_artifacts["embeddings"]["title"]` (без абсолютных путей в result).
3. `emit_extra_metrics` не расширяет `features_flat` в v1.2.0.

## Acceptance

- [x] `SCHEMA.md` + `title_embedder_output_v1.json` (16 keys) в реестре.
- [ ] Smoke с реальным `DP_MODELS_ROOT` и encode — вне этой записи (требуется модель в кеше); сверка ключей выполнена по `main.py`.

## Связанные файлы

- `DataProcessor/TextProcessor/src/extractors/title_embedder/main.py`
- `DataProcessor/TextProcessor/schemas/title_embedder_output_v1.json`
