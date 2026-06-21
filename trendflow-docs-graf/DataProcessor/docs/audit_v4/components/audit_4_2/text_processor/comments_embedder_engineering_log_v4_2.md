# Audit v4.2 — engineering log: `comments_embedder`

**Дата:** 2026-04-14  
**Компонент:** `comments_embedder` (TextProcessor; табличный срез `tp_commentsemb_*` в `text_processor/text_features.npz` + артефакт `comments_embeddings.npy`)

## Цель

Подготовить контур Audit v4 **L2** (A+B) для `comments_embedder`: срез табличных ключей и проверка артефакта эмбеддингов.

## Инструментарий

- Скрипт: `DataProcessor/TextProcessor/src/extractors/comments_embedder/scripts/audit_v4_npz_stats.py`
- Выход: `storage/audit_v4/comments_embedder_l2/comments_embedder_audit_v4_stats.json`
- Опции: `--seed 0`; явные `--npz` при необходимости.

## Наблюдения по `result_store` (youtube)

На текущем B-наборе из `RUN_LOG` (5 путей A+B):

- **2/5** файлов `text_features.npz` имеют `meta.status=ok` и содержат **18** ключей `tp_commentsemb_*`
- **3/5** файлов имеют `meta.status=error`, `feature_names` пустой, артефакт `comments_embeddings.npy` отсутствует

Причина блокировки не в `comments_embedder`, а в **сбое всего `text_processor`** (в этих mock-run часто падает `TitleEmbedder` из-за CUDA OOM — см. `text_processor_error` в JSON).

## Что в JSON

- `dataset_quality`: счётчики OK vs error.
- `per_file`: табличный срез, `meta_flat`, `text_processor_error` (если был), и проверка `comments_embeddings.npy` (exists/shape/finite/диапазон L2-норм по строкам).
- `aggregate_ok_subset`: агрегат только по OK строкам (сейчас `n_rows=2`).

## Следующие шаги (чтобы закрыть L2)

1. Перепрогнать `text_processor` для трёх проблемных `run_id` с устойчивой конфигурацией эмбеддеров (CPU / меньшая модель / достаточный GPU).
2. Повторить скрипт — получить **5** OK строк и осмысленные агрегаты/корреляции.
3. Опционально: golden §4.8 по фиксированному A (`e2bc964f-…`) для `tp_commentsemb_*` и `comments_embeddings.npy`.
---

## Навигация

[Audit v4 hub](../README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
