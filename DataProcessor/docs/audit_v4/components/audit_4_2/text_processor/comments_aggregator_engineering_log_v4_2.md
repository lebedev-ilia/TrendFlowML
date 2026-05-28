# Audit v4.2 — engineering log: `comments_aggregator`

**Дата:** 2026-04-14  
**Компонент:** `comments_aggregator` (TextProcessor; табличный срез `tp_commentsagg_*` + legacy + артефакты mean/median)

## Цель

Подготовить контур Audit v4 **L2** (A+B) для `comments_aggregator`:

- табличный слой: 39 ключей `tp_commentsagg_*` / `tp_comments_agg_*` / `tp_cagg_*`
- проверка артефактов: `comments_agg_{mean,median}.npy`, `comments_selected_indices.npy` (и наличие `comments_embeddings.npy` как upstream)

## Инструментарий

- Скрипт: `DataProcessor/TextProcessor/src/extractors/comments_aggregator/scripts/audit_v4_npz_stats.py`
- Выход: `storage/audit_v4/comments_aggregator_l2/comments_aggregator_audit_v4_stats.json`
- Опции: `--seed 0`; явные `--npz` при необходимости.

## Наблюдения по `result_store` (youtube)

На текущем B-наборе из `RUN_LOG` (5 путей A+B):

- **2/5** файлов `text_features.npz` имеют `meta.status=ok` и содержат полный табличный срез (39 ключей)
- на этих же run присутствуют артефакты `comments_agg_mean.npy`, `comments_agg_median.npy`, `comments_selected_indices.npy`
- **3/5** файлов имеют `meta.status=error`, табличный слой пустой (`feature_names` пустой), артефакты отсутствуют

Причина блокировки — сбой всего `text_processor` (часто OOM в эмбеддерах до выполнения downstream шагов), а не логика `comments_aggregator`.

## Что в JSON

- `dataset_quality`: OK vs error.
- `per_file`: табличный срез, `meta_flat`, `text_processor_error` (если был), и сводка по артефактам `.npy` (exists/shape/finite).
- `aggregate_ok_subset`: агрегат только по OK строкам (сейчас `n_rows=2`).

## Следующие шаги (чтобы закрыть L2)

1. Перепрогнать `text_processor` для трёх проблемных `run_id` (CPU / меньшая модель / достаточный GPU).
2. Повторить скрипт — получить 5 OK строк, затем golden §4.8 по фиксированному A (`e2bc964f-…`).

