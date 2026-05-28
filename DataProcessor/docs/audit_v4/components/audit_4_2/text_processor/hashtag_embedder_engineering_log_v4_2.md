# Audit v4.2 — engineering log: `hashtag_embedder`

**Дата:** 2026-04-14  
**Компонент:** `hashtag_embedder` (TextProcessor; табличный срез `tp_hashemb_*` + артефакт `hashtag_embedding.npy`)

## Цель

Подготовить контур Audit v4 **L2** (A+B) для `tp_hashemb_*` и проверить бинарный артефакт вектора эмбеддинга.

## Инструментарий

- Скрипт: `DataProcessor/TextProcessor/src/extractors/hashtag_embedder/scripts/audit_v4_npz_stats.py`
- Выход: `storage/audit_v4/hashtag_embedder_l2/hashtag_embedder_audit_v4_stats.json`
- Проверка артефакта: `text_processor/_artifacts/hashtag_embedding.npy` (shape/dtype/L2 + согласование с `tp_hashemb_dim`)
- Опции: `--seed 0`; явные `--npz` при необходимости.

## Наблюдения по `result_store` (youtube)

На текущем B-наборе из `RUN_LOG` (5 путей A+B):

- **2/5** файлов `text_features.npz` имеют `meta.status=ok`, содержат полный срез `tp_hashemb_*` (**23** ключа) и имеют артефакт `hashtag_embedding.npy` (ожидаемо **(1024,)** `float32`, конечный, L2≈1).
- **3/5** файлов имеют `meta.status=error`, табличный слой пустой (`feature_names` пустой), артефакт отсутствует.

Причина блокировки — сбой всего `text_processor` на части mock-run (ошибка до tabular merge), а не логика `hashtag_embedder`.

## Что в JSON

- `dataset_quality`: OK vs error/missing.
- `per_file`: табличный срез, `meta_flat`, `artifact_vector` (exists/shape/dtype/l2/finite), `consistency.dim_match`, `text_processor_error`.
- `aggregate_ok_subset`: агрегат только по OK строкам (сейчас `n_rows=2`).

## Следующие шаги (чтобы закрыть L2)

1. Перепрогнать `text_processor` для трёх проблемных `run_id` (CPU / меньшая модель / достаточный GPU).
2. Повторить скрипт — получить 5 OK строк, затем golden §4.8 по фиксированному A (`e2bc964f-…`).

