# `title_embedder` — engineering log (audit v4.2)

Цель: закрыть L2-часть плана audit v4 для `title_embedder` (TextProcessor) на фиксированном наборе **A+B** из **5** `result_store` прогонов и зафиксировать блокировки пайплайна.

---

## Контекст

Компонент пишет:

- табличный срез в `text_processor/text_features.npz` как строки `feature_names`/`feature_values` с префиксом `tp_titleemb_*` (ожидается **16** ключей);
- плотный вектор в `text_processor/_artifacts/title_embedding.npy` (**(D,)** float32).

L2 по плану требует **5/5** успешных прогонов `text_processor` для корректных агрегатов. Сейчас `text_processor` нестабилен на части набора (типично — CUDA OOM/ошибка загрузки embedding-модели), поэтому компонентные L2 остаются **blocked** до фикса инфраструктуры.

---

## L2 stats tooling

Скрипт:

- `DataProcessor/TextProcessor/src/extractors/title_embedder/scripts/audit_v4_npz_stats.py`

Запуск:

```bash
cd DataProcessor/TextProcessor
../.data_venv/bin/python \
  src/extractors/title_embedder/scripts/audit_v4_npz_stats.py \
  --out-dir ../../storage/audit_v4/title_embedder_l2 \
  --seed 0
```

Выход:

- JSON: `storage/audit_v4/title_embedder_l2/title_embedder_audit_v4_stats.json`

---

## Результат по 5 путям (A+B)

Фиксированный набор путей (как в других TextProcessor L2):

- A: `youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759`
- B: `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`

Наблюдения:

- **2/5 OK**: `meta.status=ok`, в `text_features.npz` есть полный `tp_titleemb_*` (**16** ключей), `title_embedding.npy` присутствует, shape **(1024,)** и \(L2 \approx 1\), `tp_titleemb_dim` согласован с размерностью артефакта.
- **3/5 error**: `meta.status=error`, `feature_names` пустой → компонентный срез отсутствует, `title_embedding.npy` не записан (падение `text_processor` до табличного слоя). Текст ошибки сохранён в JSON как `text_processor_error` (усечён).

Итог: **L2 blocked** для `title_embedder` из-за ошибок пайплайна `text_processor` на 3 из 5 прогонов. Скрипт сохраняет `dataset_quality` и `aggregate_ok_subset` для частичного анализа на OK subset.

---

## Ссылки

- Канонический отчёт: `DataProcessor/docs/audit_v4/components/text_processor/title_embedder_audit_v4.md`
- L2 JSON: `storage/audit_v4/title_embedder_l2/title_embedder_audit_v4_stats.json`
---

## Навигация

[Audit v4 hub](../README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
