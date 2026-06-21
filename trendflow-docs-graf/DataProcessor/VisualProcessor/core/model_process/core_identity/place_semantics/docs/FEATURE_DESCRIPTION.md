# place_semantics — описание фич (Audit)

**Компонент:** `place_semantics` (core identity, VisualProcessor)  
**schema_version NPZ:** `place_semantics_npz_v2`  
**Артефакт:** `rs_path/place_semantics/place_semantics.npz`

## Назначение

Поиск **мест** по кадровым эмбеддингам `core_clip/embeddings.npz` и **Embedding Service** (категория `place`): cosine top-K, треки по времени, per-frame / per-track топ-K id/score, маски уверенности.

## Ключи NPZ (сжатый обзор)

Ось **N** = `frame_indices` (как в `core_object_detections`); **T** треков; **K** = `topk` (контракт: 5); **A** — каноническое label-space.

См. полную таблицу: `docs/SCHEMA.md`, схема: `DataProcessor/VisualProcessor/schemas/place_semantics_npz_v2.json` (если есть).

## Meta

`stage_timings_ms`: `initialization`, `load_deps`, `process_frames`, `saving`, `total` (мс) → в CSV `meta_timing_initialization` … `meta_timing_total`.  
Параметры поиска: `topk`, `similarity_threshold`, `threshold_global`, `min_track_length`, `max_gap_sec`, счётчики `num_tracks`, `num_places`, `num_frames`, `place_category`, provenance `db_*`, `embedding_service_url`.

**Пусто:** `empty_reason` в т.ч. `no_places_detected` при `status=empty`.

## Схема

`docs/SCHEMA.md`, `README.md`.
---

## Навигация

[SCHEMA](SCHEMA.md) · [Module README](../README.md) · [VisualProcessor](../../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../../docs/MAIN_INDEX.md)
