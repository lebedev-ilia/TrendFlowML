## clap_extractor — AUDIT V3 REPORT

Дата: 2026-03-13  
Компонент: `DataProcessor/AudioProcessor/src/extractors/clap_extractor`  
Статус: **passed (Audit v3)**  

### TL;DR (что изменилось)

- Перевели `clap_extractor` на **per-extractor контракт**: `schema_version="clap_extractor_npz_v1"` (+ machine schema + `SCHEMA.md`).
- Зафиксировали **ModelManager-only / no-network**: убрали любые неявные попытки загрузки весов CLAP без явного локального `ckpt_path`.
- Ввели **TokenStreams-ready sequence контракт**: `embedding_sequence` + `segment_start/end/center_sec` + `segment_mask` (strict alignment; missing → `mask=false` + `NaN`).
- Добавили **robust aggregation** (trim по нормам сегментов) вместо простого mean “по всем подряд”.
- Добавили наблюдаемость **trim policy**: если окно > 10s, сигнал обрезается и это репортится через `trimmed_segments_count/trimmed_ratio`.
- Отключили legacy `run()` (full-audio + `.npy`) — в audited режиме только `run_segments()` по Segmenter-окнам.
- Переписали HTML render на **полностью offline** (без CDN).

### Ownership / Versions

- **Producer**: `clap_extractor`
- **Producer version**: `1.1.0`
- **Schema version**: `clap_extractor_npz_v1`

### Inputs

- **Upstream**: Segmenter
- **Required files**:
  - `frames_dir/audio/audio.wav` (только если `audio_present=true`)
  - `frames_dir/audio/segments.json` (`audio_segments_v1`)
- **Required sampling family**: `families.clap.segments[]` (no-fallback)

### Outputs (tiers)

#### model_facing

- `feature_names`, `feature_values` (minimal frozen subset):
  - `embedding_dim`
  - `segments_count` (число валидных сегментов)
  - `clap_norm`
  - `clap_magnitude_mean`
  - `clap_magnitude_std`
- `embedding: float32[D]` (robust aggregation)
- `embedding_sequence: float32[N,D]` (strict-aligned; masked rows are `NaN`)

#### analytics

- `segment_start_sec`, `segment_end_sec`, `segment_center_sec` (`float32[N]`)
- `segment_mask` (`bool[N]`)
- `segment_embedding_norm` (`float32[N]`, NaN для masked)
- `embedding_present` (`bool`)

#### debug

- `meta` (run identity, versions, models_used/model_signature, stage timings, scheduler knobs, error/empty_reason)

### Empty vs Error semantics

- **Valid empty**: только upstream `audio_present=false` (AudioProcessor не запускает extractor, но пишет NPZ со `status="empty"`).
- **Error**:
  - `families.clap.segments` отсутствует или пустой при `audio_present=true`
  - локальные веса CLAP не резолвятся через `dp_models` (ModelManager-only)
  - если **все** сегменты стали invalid → `segment_mask` all-false → error (нет валидного эмбеддинга)

### Sampling / alignment decisions

- Segmenter остаётся единственным владельцем sampling.
- `clap_extractor` не делает fallback на другие families.
- Для сегментов, которые не удалось обработать, применяется **strict alignment**:
  - `segment_mask[i]=false`
  - `embedding_sequence[i,:]=NaN`
  - агрегаты считают только валидные сегменты.

### Privacy

- Raw audio/текст не публикуются. NPZ содержит только embeddings и derived scalar stats.

### ModelManager / reproducibility

- Модель CLAP загружается только через `dp_models` (`laion_clap`), веса строго локальные.
- Информация о модели попадает в `meta.models_used[]` + `meta.model_signature` (через общий meta_builder).

### Render (dev-only)

- `render.py` больше не использует CDN (offline-only).
- Timeline рисует `segment_embedding_norm` по `segment_center_sec` с разрывами на masked сегментах.

### Файлы изменены / добавлены

- `DataProcessor/AudioProcessor/run_cli.py`
  - добавлен per-extractor `schema_version` mapping для `clap_extractor`.
- `DataProcessor/AudioProcessor/schemas/clap_extractor_npz_v1.json` (NEW)
- `DataProcessor/AudioProcessor/src/extractors/clap_extractor/SCHEMA.md` (NEW)
- `DataProcessor/AudioProcessor/src/core/npz_savers/clap.py`
  - обновлён saver под `clap_extractor_npz_v1` (явные поля, без legacy `.npy`).
- `DataProcessor/AudioProcessor/src/extractors/clap_extractor/__init__.py`
  - ModelManager-only fail-fast, strict-aligned mask/NaN contract, robust aggregation, trim stats, `run()` disabled.
- `DataProcessor/AudioProcessor/src/extractors/clap_extractor/render.py`
  - offline HTML render, обновлены ключи (`segment_center_sec`, `segment_mask`, …).
- `DataProcessor/AudioProcessor/src/extractors/clap_extractor/README.md`
  - обновлён под Audit v3 контракт и schema.
- `DataProcessor/AudioProcessor/docs/MAIN_INDEX.md`
  - обновлено описание компонента (Audit v3).
- `DataProcessor/AudioProcessor/README.md`
  - добавлен `clap_extractor_npz_v1` в список audited extractors.
- `DataProcessor/docs/COMPONENTS_DESC.md`
  - обновлено описание `clap_extractor` (версия/поля/схема).

### Open items

- В будущем можно вернуть cross-video batching в `extract_batch_segments()` при сохранении strict mask контракта (сейчас correctness-first).

