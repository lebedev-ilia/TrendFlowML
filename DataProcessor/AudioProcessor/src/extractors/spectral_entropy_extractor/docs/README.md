## `spectral_entropy_extractor` — Audit v3

### Назначение

Извлекает **spectral entropy** (и опционально flatness/spread) и публикует:

- стабильные **tabular** скаляры для моделей (`feature_names/feature_values`)
- **per-segment** массивы, выровненные по Segmenter sampling axis (`segment_*_sec`, `segment_mask`)

### Версии

- **producer_version**: `2.0.1`
- **schema_version**: `spectral_entropy_extractor_npz_v2`
- **human schema**: `SCHEMA.md`

### Sampling requirements (Segmenter-owned)

- Required family: **`spectral`** (`frames_dir/audio/segments.json`)
  - В audited режиме это **shared-family** (как для `pitch` / `band_energy` / `spectral_entropy`).

### Outputs (NPZ)

#### Audit v4 — заметки по NPZ

- **Tabular** на reference **A**: **F=2** (`spectral_entropy_mean`, `spectral_entropy_std`), NaN **0**; конфиг и **`device_used`** дублируются в **`meta`**, не в tabular.
- Ось **`spectral`**: **N=12** на A при полной маске; детали рядом — `entropy_*_by_segment`.
- Observability (audit v4.2): `meta.stage_timings_ms`, `meta.spectral_entropy_resource_profile` (env: `AP_SPECTRAL_ENTROPY_RESOURCE_PROFILE=1`)

#### Model-facing (frozen minimal)

В `feature_names/feature_values` (NaN = missing):

- `spectral_entropy_mean`
- `spectral_entropy_std`

#### Analytics

- Canonical axis: `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`
- Per-segment:
  - required: `entropy_mean_by_segment[N]`, `entropy_std_by_segment[N]`
  - optional: `entropy_min_by_segment[N]`, `entropy_max_by_segment[N]` (если включён extended_stats)
  - optional: `flatness_mean_by_segment[N]`, `flatness_std_by_segment[N]` (если включён flatness)
  - optional: `spread_mean_by_segment[N]`, `spread_std_by_segment[N]` (если включён spread)

### Empty / error semantics

- **short audio** (< 1s): `status="empty"`, `empty_reason="audio_too_short"`
- **all segments failed**: `status="empty"`, `empty_reason="spectral_entropy_all_segments_failed"` (axis присутствует, mask=false)
- missing family / invalid input: `status="error"` (no-fallback)

### Render (dev-only)

Offline-only HTML render (без CDN): vanilla `<canvas>` график `entropy_mean_by_segment` по `segment_center_sec`.

