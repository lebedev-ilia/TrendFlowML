# `transcript_chunk_embedder` — engineering log (audit v4.2)

Цель: закрыть L2-часть плана audit v4 для `transcript_chunk_embedder` (TextProcessor) на фиксированном наборе **A+B** из **5** `result_store` прогонов и зафиксировать блокировки пайплайна.

---

## Контекст

Компонент пишет:

- табличный срез в `text_processor/text_features.npz` как строки `feature_names`/`feature_values` с префиксом `tp_tchunk_*` (ожидается **16** ключей);
- матрицу чанков в `text_processor/_artifacts/transcript_whisper_chunk_embeddings.npy` (на текущем наборе прогонов используется источник `whisper`).

L2 по плану требует **5/5** успешных прогонов `text_processor` для корректных агрегатов. Сейчас `text_processor` нестабилен на части набора, поэтому компонентные L2 остаются **blocked** до фикса инфраструктуры.

---

## L2 stats tooling

Скрипт:

- `DataProcessor/TextProcessor/src/extractors/transcript_chunk_embedder/scripts/audit_v4_npz_stats.py`

Запуск:

```bash
cd DataProcessor/TextProcessor
../.data_venv/bin/python \
  src/extractors/transcript_chunk_embedder/scripts/audit_v4_npz_stats.py \
  --out-dir ../../storage/audit_v4/transcript_chunk_embedder_l2 \
  --seed 0
```

Выход:

- JSON: `storage/audit_v4/transcript_chunk_embedder_l2/transcript_chunk_embedder_audit_v4_stats.json`

---

## Результат по 5 путям (A+B)

Фиксированный набор путей (как в других TextProcessor L2):

- A: `youtube/-Q6fnPIybEI/e2bc964f-1983-4075-a523-1a6cd0cf0759`
- B: `-15jH8mtfJw/30e1183d-…`, `-5EYUqIlyJU/b9761f4a-…`, `-7Ei8e05x30/45c451ad-…`, `-Ga4edhrfog/e2dc8851-…`

Наблюдения:

- **2/5 OK**: `meta.status=ok`, в `text_features.npz` есть полный `tp_tchunk_*` (**16** ключей); `transcript_whisper_chunk_embeddings.npy` присутствует (shape **(1, 1024)**), значения конечны, \(L2\) нормы строк ~1; `tp_tchunk_embedding_dim` согласован со второй размерностью матрицы.
- **3/5 error**: `meta.status=error`, `feature_names` пустой → компонентный срез отсутствует, артефакт не записан (падение `text_processor` до табличного слоя). Текст ошибки сохранён в JSON как `text_processor_error` (усечён).

Итог: **L2 blocked** для `transcript_chunk_embedder` из-за ошибок пайплайна `text_processor` на 3 из 5 прогонов. Скрипт сохраняет `dataset_quality` и `aggregate_ok_subset` для частичного анализа на OK subset.

---

## Ссылки

- Канонический отчёт: `DataProcessor/docs/audit_v4/components/text_processor/transcript_chunk_embedder_audit_v4.md`
- L2 JSON: `storage/audit_v4/transcript_chunk_embedder_l2/transcript_chunk_embedder_audit_v4_stats.json`
---

## Навигация

[Audit v4 hub](../README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
