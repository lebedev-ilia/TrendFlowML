# `embedding_stats_extractor` — SCHEMA (Audit v3 / production)

## Идентификация

| Поле | Значение |
|------|----------|
| Компонент | `embedding_stats_extractor` |
| Класс | `EmbeddingStatsExtractor` |
| Machine schema | `DataProcessor/TextProcessor/schemas/embedding_stats_extractor_output_v1.json` |
| `schema_version` | `embedding_stats_extractor_output_v1` |
| Версия реализации | `1.2.0` |

## Назначение

- **Дисперсия по чанкам**: загрузка матрицы эмбеддингов чанков транскрипта из `doc.tp_artifacts` (canonical path от `TranscriptChunkEmbedder`), затем \( \| \mathrm{var}_{\text{chunks}}(\mathbf{X}) \|_2 \) и top component variances в фиксированных слотах.
- **Энтропия тем (опционально)**: по списку **`topic_probs`**, уже вычисленному **`semantics_topics_keyphrases`** (softmax/temperature upstream). Здесь повторная нормализация до распределения и счёт \(H\), \(H/\log K\), \(e^H\).

**Не входит в scope**: title / description / comments embeddings (другие экстракторы).

## Входы

- **Обязательные для `present=1`**: валидная матрица `(N, D)` с `N >= min_chunks_required`.
- **Canonical**: `tp_artifacts["transcripts"][src]["chunk_embeddings_relpath"]`.
- **Legacy**: `tp_artifacts["transcript_chunks"][src]["embeddings_relpath"]` → `tp_embstats_used_legacy_key_flag=1`.
- **Источники в конфиге**: только `whisper` и `youtube_auto` (остальные имена отбрасываются). По умолчанию приоритет **`["whisper"]`**.
- **Топики**: `tp_artifacts["topics"]["topk_distribution"]["topic_probs"]` — опционально, best-effort относительно порядка запуска `semantics_topics_keyphrases`.

## `features_flat` (39 ключей)

- Фиксированный порядок: `_FEATURES_FLAT_KEYS` в `main.py` ↔ JSON (`allow_extra_keys: false`).
- **8** слотов `tp_embstats_topvar_1..8`; параметр **`top_k_slots`** клампится к **8**, флаги **`tp_embstats_top_k_slots_requested`**, **`tp_embstats_top_k_slots`**, **`tp_embstats_top_k_slots_clamped`**.
- **`tp_embstats_schema_topvar_slots_max`**: константа **8**.

## `emit_extra_metrics`

- **`tp_embstats_load_ms`**, **`tp_embstats_compute_ms`**: при **`False`** → **NaN**; при **`True`** — числа (если соответствующая фаза не выполнялась, остаётся **NaN**).
- **`tp_embstats_emit_extra_metrics_enabled`**: зеркало конфига (0/1).

## Ответ `extract()`

- **`model_name` / `model_version` / `weights_digest`**: всегда **`null`**.
- **`system`**: `pre_init`/`post_init` из **`_init_metrics`**, **`gpu_peak_mb`** из снимков.

## Версионирование

Смена ключей → **`embedding_stats_extractor_output_v2`** + запись в `RUN_LOG.md`.
