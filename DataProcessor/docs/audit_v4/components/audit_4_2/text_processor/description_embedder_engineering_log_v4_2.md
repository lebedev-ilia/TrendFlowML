# Audit v4.2 — engineering log: `description_embedder`

**Дата:** 2026-04-14  
**Компонент:** `description_embedder` (TextProcessor; табличный срез `tp_descemb_*` + артефакт `description_embedding.npy`)

## Цель

Подготовить контур Audit v4 **L2** (A+B) для `description_embedder`: табличный слой и проверка артефакта эмбеддинга.

## Инструментарий

- Скрипт: `DataProcessor/TextProcessor/src/extractors/description_embedder/scripts/audit_v4_npz_stats.py`
- Выход: `storage/audit_v4/description_embedder_l2/description_embedder_audit_v4_stats.json`
- Опции: `--seed 0`; явные `--npz` при необходимости.

## Наблюдения по `result_store` (youtube)

На текущем B-наборе из `RUN_LOG` (5 путей A+B):

- **2/5** файлов `text_features.npz` имеют `meta.status=ok` и содержат полный срез `tp_descemb_*` (**19** ключей) и артефакт `description_embedding.npy`
- **3/5** файлов имеют `meta.status=error`, табличный слой пустой (`feature_names` пустой), артефакт отсутствует

Причина блокировки — сбой всего `text_processor` (часто OOM в эмбеддерах до выполнения downstream шагов), а не логика `description_embedder`.

## Что в JSON

- `dataset_quality`: OK vs error.
- `per_file`: табличный срез, `meta_flat`, `text_processor_error` (если был), и проверка `description_embedding.npy` (exists/shape/finite/l2).
- `aggregate_ok_subset`: агрегат только по OK строкам (сейчас `n_rows=2`).

## Следующие шаги (чтобы закрыть L2)

1. Перепрогнать `text_processor` для трёх проблемных `run_id` (CPU / меньшая модель / достаточный GPU).
2. Повторить скрипт — получить 5 OK строк, затем golden §4.8 по фиксированному A (`e2bc964f-…`).

