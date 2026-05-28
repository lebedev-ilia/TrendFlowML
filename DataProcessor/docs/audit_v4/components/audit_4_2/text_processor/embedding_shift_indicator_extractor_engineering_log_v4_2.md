# Audit v4.2 — engineering log: `embedding_shift_indicator_extractor`

**Дата:** 2026-04-14  
**Компонент:** `embedding_shift_indicator_extractor` (TextProcessor; табличный срез `tp_embshift_*` в `text_processor/text_features.npz`)

## Цель

Подготовить контур Audit v4 **L2** (A+B) для `tp_embshift_*`: сводка по табличному слою и качеству `result_store`.

## Инструментарий

- Скрипт: `DataProcessor/TextProcessor/src/extractors/embedding_shift_indicator_extractor/scripts/audit_v4_npz_stats.py`
- Выход: `storage/audit_v4/embedding_shift_indicator_extractor_l2/embedding_shift_indicator_extractor_audit_v4_stats.json`
- Опции: `--seed 0`; явные `--npz` при необходимости.

## Наблюдения по `result_store` (youtube)

На текущем B-наборе из `RUN_LOG` (5 путей A+B):

- **2/5** файлов `text_features.npz` имеют `meta.status=ok` и содержат полный срез `tp_embshift_*` (**27** ключей)
- **3/5** файлов имеют `meta.status=error`, табличный слой пустой (`feature_names` пустой)

Причина блокировки — сбой всего `text_processor` на части mock-run (ошибка до tabular merge), а не логика `embedding_shift_indicator_extractor`.

## Что в JSON

- `dataset_quality`: OK vs error.
- `per_file`: табличный срез, `meta_flat`, `text_processor_error` (если был).
- `aggregate_ok_subset`: агрегат только по OK строкам (сейчас `n_rows=2`).

## Следующие шаги (чтобы закрыть L2)

1. Перепрогнать `text_processor` для трёх проблемных `run_id` (CPU / меньшая модель / достаточный GPU).
2. Повторить скрипт — получить 5 OK строк, затем golden §4.8 по фиксированному A (`e2bc964f-…`).

