# Audit v4.2 — engineering log: `embedding_source_id_extractor`

**Дата:** 2026-04-14  
**Компонент:** `embedding_source_id_extractor` (TextProcessor; табличный срез `tp_embid_*` + nested `payload["embedding_source_id"]`)

## Цель

Подготовить контур Audit v4 **L2** (A+B) для `tp_embid_*` и nested `payload["embedding_source_id"]`, включая сверку `vector_id` по файлу эмбеддинга.

## Инструментарий

- Скрипт: `DataProcessor/TextProcessor/src/extractors/embedding_source_id_extractor/scripts/audit_v4_npz_stats.py`
- Выход: `storage/audit_v4/embedding_source_id_extractor_l2/embedding_source_id_extractor_audit_v4_stats.json`
- Опции: `--seed 0`; явные `--npz` при необходимости.

## Наблюдения по `result_store` (youtube)

На текущем B-наборе из `RUN_LOG` (5 путей A+B):

- **2/5** файлов `text_features.npz` имеют `meta.status=ok`, содержат полный срез `tp_embid_*` (**13** ключей) и имеют nested `payload["embedding_source_id"]`.
  - На этих 2 run `vector_id` в payload **совпадает** с вычисленным sha256 по `float32` байтам файла `text_processor/_artifacts/<embedding_relpath>`.
- **3/5** файлов имеют `meta.status=error`, табличный слой пустой (`feature_names` пустой), а `payload["embedding_source_id"]` отсутствует.

Причина блокировки — сбой всего `text_processor` на части mock-run (ошибка до tabular merge), а не логика `embedding_source_id_extractor`.

## Что в JSON

- `dataset_quality`: OK vs error/missing.
- `per_file`: `tp_embid_*`, `meta_flat`, `payload.embedding_source_id` (subset), `vector_id_check` (exists/shape/dtype/computed/match), `text_processor_error`.
- `aggregate_ok_subset`: агрегат только по OK строкам (сейчас `n_rows=2`).

## Следующие шаги (чтобы закрыть L2)

1. Перепрогнать `text_processor` для трёх проблемных `run_id` (CPU / меньшая модель / достаточный GPU).
2. Повторить скрипт — получить 5 OK строк, затем golden §4.8 по фиксированному A (`e2bc964f-…`) с проверкой `vector_id`.

