# `comments_aggregator` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `comments_aggregator` |
| Класс | `CommentsAggregationExtractor` |
| Machine schema | `DataProcessor/TextProcessor/schemas/comments_aggregator_output_v1.json` |
| `schema_version` | `comments_aggregator_output_v1` |
| Версия реализации | `1.3.0` |

## Назначение

Агрегация **уже посчитанных** эмбеддингов комментариев (**`CommentsEmbedder`**) в векторы **взвешенного среднего** (опциональные веса через `selected_indices` и поля документа) и **покомпонентной медианы** (L2 после агрегации). Модель **не исполняется**; **`model_name` / `model_version` / `weights_digest`** задают **то же пространство**, что и у эмбеддера комментариев (**resolve** через **`dp_models`** без forward).

## Входы / артефакты

- Чтение: **`doc.tp_artifacts["embeddings"]["comments"]["relpath"]`** → матрица `(N, D)`.
- Опционально: **`doc.tp_artifacts["comments"]["selected_indices_relpath"]`** для выравнивания лайков/authority/recency с выбранными строками матрицы.
- Запись: **`comments_agg_mean.npy`**, **`comments_agg_median.npy`** (per-run); регистрация путей в **`tp_artifacts["comments"]`** и зеркале в **`embeddings`**.

## `features_flat` (39 ключей)

Три семейства имён (все ключи **всегда** присутствуют, порядок фиксирован в `main.py` / JSON-схеме):

1. **`tp_commentsagg_*`** (22): ядро, флаги конфигурации, веса, безопасность, **2 extra-тайминга** в миллисекундах.
2. **`tp_comments_agg_*`** (12): legacy + зеркала весов и **`compute_*`** для downstream.
3. **`tp_cagg_*`** (5): короткий legacy-слой (present, count, dim, std слоты).

### Extra-тайминги

- **`tp_commentsagg_agg_mean_ms`**, **`tp_commentsagg_agg_median_ms`**: при **`emit_extra_metrics=False`** → **NaN**; при **`True`** и выключенном соответствующем **`compute_*`** → **NaN**; иначе — время шага агрегации в **мс**.

### `tp_commentsagg_dim_mismatch_flag` (семантика без изменений)

- **`1.0`** только на ветке «эмбеддинги отсутствуют / невалидны», если **`embeddings` удалось загрузить как `np.ndarray`, но форма не `(N, D)` с `N>0`, `D>0`** (в т.ч. неверная размерность массива).
- **`0.0`**, если матрицы не было (**`None`**) или на **успешной** ветке.

## Ошибки

- **`require_comment_embeddings=True`** и невалидная/отсутствующая матрица: **RuntimeError**.

## Метаданные ответа

Верхний уровень **`extract()`**: **`model_name`**, **`model_version`**, **`weights_digest`** (согласованы с **`CommentsEmbedder`** для того же прогона). **`gpu_peak_mb`** — по снимкам GPU, как у других компонентов (агрегатор на CPU, пик обычно из init/process).

## Версионирование

Смена ключей/семантики → **`comments_aggregator_output_v2`** + запись в `DataProcessor/docs/audit_v3/RUN_LOG.md`.
---

## Навигация

[README](README.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
