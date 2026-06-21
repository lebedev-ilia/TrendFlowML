# `speaker_diarization_extractor` — SCHEMA (Audit v3)

## Status

- **Producer**: `speaker_diarization_extractor`
- **Schema version**: `speaker_diarization_extractor_npz_v2`
- **Audit v3**: in progress (this schema is designed for audit v3 “passed” requirements)

## Purpose

Speaker diarization: determine “who speaks when” on the full audio, producing **speaker turn events** and a small set of stable aggregates.

Audit v3 constraints:

- **No-network runtime**: Model loading **only via ModelManager** (`dp_models`).
- **No raw text retention**: no transcript/words are stored in this component.
- **Segmenter-owned sampling**: requires `families.diarization.segments` with **exactly one** window covering the whole audio.
- **NPZ is source-of-truth**: no additional JSON artifacts; renderer reads NPZ.

## Inputs

- **Audio**: `audio/audio.wav` (from Segmenter frames_dir)
- **Segments**: `audio/segments.json` must contain:
  - `families.diarization.segments`: list with **one** segment:
    - `start_sec` (float)
    - `end_sec` (float)

## Outputs (NPZ)

### 1) Model-facing tabular scalars

Stored as:

- `feature_names`: `object[F]`
- `feature_values`: `float32[F]`

Frozen tabular set (порядок как в `npz_savers/speaker_diarization.py`, **F=10**):

- `speaker_count`
- `duration_sec`
- `sample_rate`
- `rms`
- `peak`
- `speaker_balance_score`
- `dominant_speaker_id`
- `speaker_turns_count`
- `speaker_turns_density`
- `speaker_transitions_count`

### 2) Canonical sampling axis (Segmenter window)

These arrays describe the **Segmenter sampling window(s)** for this component. For audit v3 the expectation is \(N=1\).

- `segment_start_sec`: `float32[N]`
- `segment_end_sec`: `float32[N]`
- `segment_center_sec`: `float32[N]`
- `segment_mask`: `bool[N]`

### 3) Speaker turns (token-ready events)

Speaker turns are represented as **flat aligned arrays** of length \(K\):

- `turn_start_sec`: `float32[K]`
- `turn_end_sec`: `float32[K]`
- `turn_speaker_id`: `int32[K]` (speaker IDs are `0..S-1`)
- `turn_mask`: `bool[K]` (for audit v3: currently all `true` when `status="ok"`)

### 4) Per-speaker structured arrays

All arrays are length \(S = speaker_count\):

- `speaker_ids`: `int32[S]` (typically `[0, 1, ..., S-1]`)
- `speaker_duration_sec`: `float32[S]`
- `speaker_time_ratio`: `float32[S]` (speaker_duration_sec / duration_sec)
- `speaker_turns_count_by_speaker`: `int32[S]`

### 5) Meta

- `meta`: `object` dict (see global meta contract).

Important meta extras for this component:

- **`device_used`** (str) — только в **`meta`**, не в `feature_values`
- `model_name` (from ModelManager spec)
- `weights_digest`
- `diarization_contract_version`
- `features_enabled`
- observability (audit v4.2, optional): `stage_timings_ms`, `speaker_diarization_resource_profile`
- observability (audit v4.2, optional): `stage_timings_ms`, `speaker_diarization_resource_profile`

## Empty / Error semantics

### `status="empty"`

Canonical empty reasons:

- `audio_missing_or_extract_failed`: upstream reports no audio
- `audio_silent`: silence detection triggers

In empty cases, arrays may be length 0 for turns/speakers; the Segmenter axis should still be present when running via `run_segments()`.

### `status="error"`

Typical reasons:

- `dependency_missing`
- ModelManager cannot resolve/load required spec `pyannote_speaker_diarization`
- runtime errors inside pyannote pipeline

## Privacy / retention

- **No transcripts / words** are produced or stored by this audited contract.
- Do not log raw audio content; logs should contain only counters/timings/status.
---

## Навигация

[README](README.md) · [FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
