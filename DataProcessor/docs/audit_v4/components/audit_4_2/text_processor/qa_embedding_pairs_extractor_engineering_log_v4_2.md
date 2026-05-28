# Audit v4.2 — engineering log: `qa_embedding_pairs_extractor`

**Дата:** 2026-04-14  
**Компонент:** `qa_embedding_pairs_extractor` (TextProcessor; табличный срез `tp_qa_*` + опциональный артефакт `qa_question_embeddings.npy`)

## Цель

Подготовить контур Audit v4 **L2** (A+B) для `tp_qa_*` и проверить бинарный артефакт эмбеддингов вопросов, если вопросы извлечены.

## Инструментарий

- Скрипт: `DataProcessor/TextProcessor/src/extractors/qa_embedding_pairs_extractor/scripts/audit_v4_npz_stats.py`
- Выход: `storage/audit_v4/qa_embedding_pairs_extractor_l2/qa_embedding_pairs_extractor_audit_v4_stats.json`
- Артефакт (если `tp_qa_present=1`): `text_processor/_artifacts/qa_question_embeddings.npy`
- Опции: `--seed 0`; явные `--npz` при необходимости.

## Наблюдения по `result_store` (youtube)

На текущем B-наборе из `RUN_LOG` (5 путей A+B):

- **2/5** файлов `text_features.npz` имеют `meta.status=ok` и содержат полный срез `tp_qa_*` (**34** ключа).
  - На одном OK-run `tp_qa_present=1`, `tp_qa_num_questions=2`, и присутствует `qa_question_embeddings.npy` формы **(2, 1024)**.
  - На одном OK-run `tp_qa_present=0`, `tp_qa_num_questions=0` — **валидный пустой исход**, артефакт отсутствует (ожидаемо).
- **3/5** файлов имеют `meta.status=error`, табличный слой пустой (`feature_names` пустой).

Причина блокировки — сбой всего `text_processor` на части mock-run (ошибка до tabular merge), а не логика `qa_embedding_pairs_extractor`.

## Что в JSON

- `dataset_quality`: OK vs error/inconsistent.
- `per_file`: табличный срез, `meta_flat`, `artifact_embeddings` (exists/shape/dtype/finite), `consistency` (валидность «present↔артефакт»), `text_processor_error`.
- `aggregate_ok_subset`: агрегат только по OK строкам (сейчас `n_rows=2`).

## Следующие шаги (чтобы закрыть L2)

1. Перепрогнать `text_processor` для трёх проблемных `run_id` (CPU / меньшая модель / достаточный GPU).
2. Повторить скрипт — получить 5 OK строк, затем golden §4.8 по фиксированному A (`e2bc964f-…`).

