# `transcript_aggregator` — engineering log (audit v4.2)

Цель: закрыть L2-часть плана audit v4 для `transcript_aggregator` (TextProcessor) на фиксированном наборе **A+B** из **5** `result_store` прогонов и зафиксировать блокировки пайплайна.

---

## Контекст

Компонент пишет:

- табличный срез в `text_processor/text_features.npz` как строки `feature_names`/`feature_values` с префиксом `tp_tragg_*` (ожидается **19** ключей);
- при `write_artifacts=1` и соответствующих `present_*` / `compute_*` — фиксированные `.npy` в `text_processor/_artifacts/`:
  - `transcript_{whisper|youtube_auto}_agg_{mean,max}.npy`;
  - при `present_combined=1` и `compute_combined=1` — `transcript_combined_agg_{mean,max}.npy`.

L2 по плану требует **5/5** успешных прогонов `text_processor`. Сейчас на **3/5** путей `meta.status=error`, поэтому полный L2 **blocked**.

---

## L2 stats tooling

Скрипт:

- `DataProcessor/TextProcessor/src/extractors/transcript_aggregator/scripts/audit_v4_npz_stats.py`

Запуск:

```bash
cd DataProcessor/TextProcessor
../.data_venv/bin/python \
  src/extractors/transcript_aggregator/scripts/audit_v4_npz_stats.py \
  --out-dir ../../storage/audit_v4/transcript_aggregator_l2 \
  --seed 0
```

Выход:

- JSON: `storage/audit_v4/transcript_aggregator_l2/transcript_aggregator_audit_v4_stats.json`

Скрипт сверяет ожидаемый набор `.npy` с флагами в `tp_tragg_*`, проверяет форму **(D,)**, конечность и \(L2 \approx 1\) (как после нормировки в экстракторе).

---

## Результат по 5 путям (A+B)

- **2/5 OK**: полный `tp_tragg_*`, ожидаемые агрегаты на диске (**(1024,)** каждый), `consistency.artifacts_ok=true`.
- **3/5 error**: пустой табличный слой — падение `text_processor` (см. `text_processor_error` в JSON).

---

## Ссылки

- Канонический отчёт: `DataProcessor/docs/audit_v4/components/text_processor/transcript_aggregator_audit_v4.md`
- L2 JSON: `storage/audit_v4/transcript_aggregator_l2/transcript_aggregator_audit_v4_stats.json`
---

## Навигация

[Audit v4 hub](../README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
