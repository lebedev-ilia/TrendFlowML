# Audit v4.2 — engineering log: `semantics_topics_keyphrases`

**Дата:** 2026-04-14  
**Компонент:** `semantics_topics_keyphrases` (TextProcessor; табличный срез `tp_topics_*` + артефакт `tp_topics_keyphrase_embeddings.npy`)

## Цель

Подготовить контур Audit v4 **L2** (A+B) для `tp_topics_*` и проверить бинарный артефакт эмбеддингов keyphrases.

## Инструментарий

- Скрипт: `DataProcessor/TextProcessor/src/extractors/semantics_topics_keyphrases/scripts/audit_v4_npz_stats.py`
- Выход: `storage/audit_v4/semantics_topics_keyphrases_l2/semantics_topics_keyphrases_audit_v4_stats.json`
- Артефакт: `text_processor/_artifacts/tp_topics_keyphrase_embeddings.npy` (shape/dtype/finite + согласование с `tp_topics_keyphrases_count/dim`)
- Опции: `--seed 0`; явные `--npz` при необходимости.

## Наблюдения по `result_store` (youtube)

На текущем B-наборе из `RUN_LOG` (5 путей A+B):

- **2/5** файлов `text_features.npz` имеют `meta.status=ok`, содержат полный срез `tp_topics_*` (**116** ключей) и имеют артефакт `tp_topics_keyphrase_embeddings.npy` (ожидаемо **(10, 1024)** `float32`).
- **3/5** файлов имеют `meta.status=error`, табличный слой пустой (`feature_names` пустой).
  - При этом артефакт `tp_topics_keyphrase_embeddings.npy` на этих run **может присутствовать** (частичный выход до падения пайплайна). Для диагностики успеха ориентироваться на `meta.status` и наличие tabular slice.

Причина блокировки — сбой всего `text_processor` на части mock-run (ошибка до tabular merge), а не логика `semantics_topics_keyphrases`.

## Что в JSON

- `dataset_quality`: OK vs error/empty slice.
- `per_file`: табличный срез, `meta_flat`, `artifact_keyphrase_embeddings` (exists/shape/dtype/finite), `consistency` (ожидаемость артефакта и match), `text_processor_error`.
- `aggregate_ok_subset`: агрегат только по OK строкам (сейчас `n_rows=2`).

## Следующие шаги (чтобы закрыть L2)

1. Перепрогнать `text_processor` для трёх проблемных `run_id` (CPU / меньшая модель / достаточный GPU).
2. Повторить скрипт — получить 5 OK строк, затем golden §4.8 по фиксированному A (`e2bc964f-…`).
---

## Навигация

[Audit v4 hub](../README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
