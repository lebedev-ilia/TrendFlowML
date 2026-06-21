# `semantics_topics_keyphrases` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `semantics_topics_keyphrases` |
| Класс | `SemanticTopicExtractor` |
| Machine schema | `DataProcessor/TextProcessor/schemas/semantics_topics_keyphrases_output_v1.json` |
| `schema_version` | `semantics_topics_keyphrases_output_v1` |
| Версия реализации | `2.1.0` |

## Назначение

- **Темы**: retrieval по зашитой taxonomy (`topics.jsonl`) + эмбеддинги промптов (кеш на диске) и запрос = нормализованный объединённый текст (ASR/legacy + title + description).
- **Ключевые фразы**: детерминированный простой скоринг без внешних библиотек; опционально эмбеддинги фраз и экспорт `raw` / `hashed` / `none`.
- **Стиль**: эвристические флаги (FAQ `?`, instruction/audience/CTA словари из конфига).

Модель sentence-transformers загружается **лениво** в `extract()` только на ветке с непустым текстом и `enabled=True`.

## `features_flat` (116 ключей)

- Фиксированный порядок: **`_FEATURES_FLAT_KEYS`** в `main.py` ↔ JSON (**`allow_extra_keys: false`**).
- **8** topic-слотов: `tp_topics_topic_top{i}_{id|score|prob}`, `i = 1..8`. Заполняются только до эффективного `top_k_slots` (после клампа ≤ 8) и фактической длины retrieval.
- **16** keyphrase-слотов: `tp_topics_kp_top{i}_{present|hash01|len}` (`hashed`‑режим; иначе слоты «пустые»).
- Кламп: `tp_topics_top_k_slots_requested` / `tp_topics_top_k_slots` / `tp_topics_top_k_slots_clamped`; то же для keyphrase-слотов (`*_keyphrase_slots*`).
- One-hot политики транскрипта: **`tp_topics_transcript_source_policy_{asr_only|asr_then_legacy|legacy_only}`** (ровно одна 1.0 при валидной политике).

## `emit_extra_metrics`

Поля `tp_topics_extra_*` (тайминги загрузки модели, topics-пайплайна, encode keyphrases, `digest` первых 6 hex-символов весов БД и модели как float):

- при **`emit_extra_metrics=False`** → **NaN**;
- при **`True`** → числа там, где ветка выполнялась; иначе **NaN** (например, topics выключены → `tp_topics_extra_topics_pipeline_ms` = **NaN**).

Флаг зеркала конфига: **`tp_topics_emit_extra_metrics_enabled`** (0.0 / 1.0), всегда задан.

## `result` (не `features_flat`)

- При **`export_keyphrases_mode=raw`**: опциональный список строк **`tp_topics_keyphrases_raw`** мержится в **`result`** рядом с **`features_flat`** (не входит в machine-schema `features_flat`; в NPZ-агрегате может отсутствовать или храниться иначе — см. пайплайн сохранения).

## Верхний уровень ответа `extract()`

- **`model_name`**, **`model_version`**, **`weights_digest`**: при **`enabled=False`** или пустом тексте → **`null`**; на успехе → из **`get_model_with_meta`**.
- **`system.pre_init` / `post_init`**: снимки из **`_init_metrics`** конструктора; **`gpu_peak_mb`** — по `system_snapshot` (пик по доступным GPU).

## Версионирование

Смена ключей или семантики слотов → **`semantics_topics_keyphrases_output_v2`** + запись в `RUN_LOG.md`.
---

## Навигация

[README](README.md) · [TextProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
