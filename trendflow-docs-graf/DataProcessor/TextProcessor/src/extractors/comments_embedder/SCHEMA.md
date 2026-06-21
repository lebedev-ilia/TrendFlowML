# `comments_embedder` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `comments_embedder` |
| Класс | `CommentsEmbedder` |
| Machine schema | `DataProcessor/TextProcessor/schemas/comments_embedder_output_v1.json` |
| `schema_version` (логический контракт `features_flat`) | `comments_embedder_output_v1` |
| Версия реализации | `1.3.0` (см. `CommentsEmbedder.VERSION`) |

## Назначение

L2-нормализованные эмбеддинги **выбранных** комментариев (`VideoDocument.comments`), артефакт **`comments_embeddings.npy`** (опционально), индексы выбора **`comments_selected_indices.npy`** для выравнивания весов в **`comments_aggregator`**. Скаляры — **`tp_commentsemb_*`** в **`result.features_flat`**, ровно **18** ключей на каждой ветке.

## Audit v3 preflight (модель)

Канон — **`intfloat/multilingual-e5-large`** ([preflight §0.5](../../../../docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md)). Default в коде **1.3.0** совпадает с этим именем; фактическая модель по-прежнему из `model_name` / `global_config.yaml`.

## `emit_extra_metrics`

- **`False`** (default в типичном прогоне): **10** полей из блока «extra» (см. JSON) — **NaN**; сохраняются **8** «core» метрик (отбор/объём текста/`present`/`count`/`dim`/truncation).
- **`True`**: заполняются кеш, fp16/cuda, digest, флаги compute/write, `artifact_written`, **select_ms**, **encode_ms**.

## `extract` vs `extract_batch`

- **`extract_batch`**: общий encode по всем выбранным комментариям; **`tp_commentsemb_encode_ms`** и **`timings_s.encode`** — **доля** общего времени encode, пропорциональная **числу закодированных комментариев** этого документа к сумме по батчу (не деление на `n_docs`).
- **`tp_commentsemb_cache_hit`**: в batch-пути при **`emit_extra_metrics=True`** — **NaN** (per-doc кеш на этом пути не используется; не интерпретировать как «miss»).
- Семантика **core** и gating **`emit_extra_metrics`** совпадает с **`extract`**.

## Артефакты vs `features_flat`

| Что | Где |
|-----|-----|
| Матрица `(N, D)` | `comments_embeddings.npy`, `doc.tp_artifacts["embeddings"]["comments"]` |
| Индексы исходных комментариев | `comments_selected_indices.npy`, `tp_artifacts["comments"]` |
| Скаляры | `result.features_flat` |

Абсолютные пути в **NPZ/result** не отдаём (privacy).

## GPU

**`system.peaks.gpu_peak_mb`** — max по снимкам init + post (как у других эмбеддеров).

## Версионирование

Изменение набора или смысла ключей → bump **`comments_embedder_output_v2`** + `RUN_LOG.md` + отчёт компонента.
---

## Навигация

[README](README.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
