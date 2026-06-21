# Audit v4.2 — engineering log: `tags_extractor`

**Дата:** 2026-04-14  
**Компонент:** `tags_extractor` (TextProcessor; табличный срез `tp_tags_*` в `text_processor/text_features.npz`, слоты `topK` — динамические)

## Цель

Подготовить контур Audit v4 **L2** (A+B) для `tp_tags_*`, учитывая `allow_extra_keys: true` (число слотов top‑K зависит от `top_k_slots`).

## Инструментарий

- Скрипт: `DataProcessor/TextProcessor/src/extractors/tags_extractor/scripts/audit_v4_npz_stats.py`
- Выход: `storage/audit_v4/tags_extractor_l2/tags_extractor_audit_v4_stats.json`
- Опции: `--seed 0`; явные `--npz` при необходимости.

## Наблюдения по `result_store` (youtube)

На текущем B-наборе из `RUN_LOG` (5 путей A+B):

- **2/5** файлов `text_features.npz` имеют `meta.status=ok` и содержат полный набор `tp_tags_*` для текущего `top_k_slots=5`: **43** ключа (= 28 базовых + 15 slot‑ключей `top1..5 × {present,hash01,len}`).
- **3/5** файлов имеют `meta.status=error`, табличный слой пустой (`feature_names` пустой).

Причина блокировки — сбой всего `text_processor` на части mock-run (ошибка до tabular merge), а не логика `tags_extractor`.

## Что в JSON

- `dataset_quality`: OK vs error/inconsistent.
- `per_file`: полный `tp_tags_*` срез, `meta_flat`, проверка слотов top1..top5:
  - наличие `present/hash01/len`,
  - NaN-консистентность: при `present=0` поля `hash01/len` должны быть NaN.
- `aggregate_ok_subset`: агрегат только по OK строкам (сейчас `n_rows=2`).

## Следующие шаги (чтобы закрыть L2)

1. Перепрогнать `text_processor` для трёх проблемных `run_id` (CPU / меньшая модель / достаточный GPU).
2. При изменении `top_k_slots` в YAML — ожидать расширения набора ключей `tp_tags_top{i}_*` (это валидно по контракту).
---

## Навигация

[Audit v4 hub](../README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
