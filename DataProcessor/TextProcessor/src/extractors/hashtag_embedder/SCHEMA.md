# `hashtag_embedder` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `hashtag_embedder` |
| Класс | `HashtagEmbedder` |
| Machine schema | `DataProcessor/TextProcessor/schemas/hashtag_embedder_output_v1.json` |
| `schema_version` (логический контракт `features_flat`) | `hashtag_embedder_output_v1` |
| Версия реализации | `1.2.0` (см. `HashtagEmbedder.VERSION`) |

## Назначение

Построить **один** L2-нормализованный агрегированный эмбеддинг по списку **`doc.hashtags`** (после canonicalize: strip, опционально снятие `#`, casefold, лимиты `max_tag_len` / `max_tags`), записать **`hashtag_embedding.npy`** (опционально) и скаляры в **`result.features_flat`** (`tp_hashemb_*`).

## Audit v3 preflight (модель)

Каноническая модель эмбеддингов — **`intfloat/multilingual-e5-large`** ([preflight §0.5](../../../../docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md)). Фактическая модель задаётся **`model_name`** в прогоне (`global_config.yaml` уже выставляет e5-large для этого компонента).

## Upstream

- **`TagsExtractor`** обычно заполняет **`doc.hashtags`** (inline + merge JSON); опционально кладёт хинт **`tp_artifacts["tags"]["hashtags_disabled_by_policy"]`** → **`tp_hashemb_disabled_by_policy_hint`**.
- **`require_hashtags=true`**: отсутствие или неверный тип **`doc.hashtags`** → **RuntimeError** в **`extract`** и **`extract_batch`** (семантика совпадает).

### Устаревший параметр

- **`strict_missing_hashtags`** (deprecated): при **`True`** эквивалентен **`require_hashtags=True`**. По умолчанию **`False`**, чтобы **`require_hashtags: false`** из конфига не перекрывался молча.

## Артефакты vs `features_flat`

| Что | Где |
|-----|-----|
| Вектор dim D | `hashtag_embedding.npy`, `doc.tp_artifacts["embeddings"]["hashtag"]` |
| Скаляры | `result.features_flat` |

В **`result`** на всех ветках присутствуют **`model_name`**, **`model_version`**, **`weights_digest`** (для логов и трассировки).

## Полный перечень `features_flat`

Source of truth: `main.py` → `_stable_features_template()` и `features_flat.update(...)`.

### Зарезервировано

- **`emit_extra_metrics`**: в v1.2.0 **не добавляет** ключей в `features_flat` (как у других embedder’ов).

### `extract_batch` / тайминги

- **`tp_hashemb_encode_ms`**: в batch-режиме — **амортизированная доля** общего времени encode уникальных тегов: \(t_{\mathrm{enc}} \times 1000 / n_{\mathrm{docs}}\) (не wall-clock одного документа).
- **`tp_hashemb_cache_hit`**: в batch-пути **0.0** (дисковый per-doc кеш не используется).

## Версионирование

Изменение набора или смысла ключей → bump **`hashtag_embedder_output_v2`** + `RUN_LOG.md` + отчёт компонента.
