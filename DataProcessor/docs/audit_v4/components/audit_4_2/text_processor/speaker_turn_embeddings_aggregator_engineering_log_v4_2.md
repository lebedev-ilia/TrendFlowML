# Audit v4.2 — engineering log: `speaker_turn_embeddings_aggregator`

**Дата:** 2026-04-14  
**Компонент:** `speaker_turn_embeddings_aggregator` (TextProcessor; табличный срез `tp_spkemb_*` + опциональные per-speaker `speaker_spkXXX_{mean,max}.npy`)

## Цель

Подготовить контур Audit v4 **L2** (A+B) для `tp_spkemb_*` и проверить per-speaker артефакты, если компонент действительно отработал (`present=1`).

## Инструментарий

- Скрипт: `DataProcessor/TextProcessor/src/extractors/speaker_turn_embeddings_aggregator/scripts/audit_v4_npz_stats.py`
- Выход: `storage/audit_v4/speaker_turn_embeddings_aggregator_l2/speaker_turn_embeddings_aggregator_audit_v4_stats.json`
- Артефакты (если `present=1`): `text_processor/_artifacts/speaker_spkXXX_{mean,max}.npy`
- Опции: `--seed 0`; явные `--npz` при необходимости.

## Наблюдения по `result_store` (youtube)

На текущем B-наборе из `RUN_LOG` (5 путей A+B):

- **2/5** файлов `text_features.npz` имеют `meta.status=ok` и содержат полный срез `tp_spkemb_*` (**17** ключей).
  - На этих run `tp_spkemb_present=0` (валидный пустой исход: нет diar+ASR с таймингами и нет legacy `doc.speakers`), поэтому `speaker_spk*.npy` **ожидаемо отсутствуют**.
- **3/5** файлов имеют `meta.status=error`, табличный слой пустой (`feature_names` пустой).

Причина блокировки — сбой всего `text_processor` на части mock-run (ошибка до tabular merge), а «счастливый» путь `speaker_turn_embeddings_aggregator` также не демонстрируется на текущем A/B (нет входа).

## Что в JSON

- `dataset_quality`: OK vs error/inconsistent.
- `per_file`: табличный срез, `meta_flat`, `speaker_artifacts` (n_mean/n_max + список имён), `consistency` (валидность «present↔артефакты»), `text_processor_error`.
- `aggregate_ok_subset`: агрегат только по OK строкам (сейчас `n_rows=2`).

## Следующие шаги (чтобы закрыть L2)

1. Перепрогнать `text_processor` для трёх проблемных `run_id` (CPU / меньшая модель / достаточный GPU).
2. Для содержательного L2 по компоненту подготовить B-run с входом diar+ASR (таймкодированные сегменты) или legacy `doc.speakers` (см. отчёт L1) — иначе `present` останется 0.

