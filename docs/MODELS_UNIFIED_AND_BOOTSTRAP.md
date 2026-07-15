# Unified models + bootstrap (2026-06-29)

This document covers tasks 3 and 4: a single source of truth for model weights on
Hugging Face, a verified download script that lays them into the canonical local
paths, and a one-command bootstrap that brings the whole project up from a fresh
`git clone`.

## TL;DR

- Two HF repos were consolidated into **one**: `Ilialebedev/trendflow_models`
  (dataset). The old `trendflow_output` was **renamed** into it (server-side,
  no byte transfer); the 40 MB of visual semantic bases from
  `trendflow_artifact_0_1` were merged in. The clean repo is **3.4 GB / 431
  files** (was 12.6 GB — duplicates, HF caches and source/`.tmp` cruft removed).
- Authoritative list: **`configs/models_manifest.json`** (also copied into the HF
  repo). Each entry `path` is **both** the HF path and the local path relative to
  the repo root, so download is a verified mirror.
- Versioning: manifest carries `schema_version` + `manifest_version`; the HF repo
  has a `v1.0.0` ref.
- Download: `DataProcessor/scripts/download_models.py` (stdlib-only, sha256
  verify, idempotent, parallel, `--dry-run`, `--groups`, `--revision`).
- Bootstrap: `./bootstrap.sh` (prereqs → env → venvs → models → stack → smoke).

## Canonical layout

`DP_MODELS_ROOT = DataProcessor/dp_models`. With this root, both path conventions
present in `dp_models/spec_catalog/*.yaml` resolve correctly (no double
`bundled_models/`):

```
DataProcessor/dp_models/
  audio/
    emotion_diarization/wavlm_large/      # speechbrain SpeechEmotionDiarization
    emotion_recognition/wav2vec2/         # speechbrain wav2vec2 IEMOCAP (neu/ang/hap/sad)
    whisper/whisper_small_hf/             # openai/whisper-small (HF transformers)
  visual/
    object_detection/yolo11l/             # baseline yolo11l (.pt/.onnx)
    pose/yolo11l_pose/                    # yolo11l-pose (.pt/.onnx)
    action_recognition/videomae_kinetics400/   # VideoMAE (Kinetics-400)
  bundled_models/visual/emonet/           # emonet_8.pth (+emonet_5.pth)
DataProcessor/VisualProcessor/core/model_process/core_identity/...   # semantic bases
```

## Manifest schema (`configs/models_manifest.json`)

```jsonc
{
  "schema_version": "1.0",
  "manifest_version": "2026-06-29",
  "repo_id": "Ilialebedev/trendflow_models",
  "repo_type": "dataset",
  "revision": "main",
  "dp_models_root": "DataProcessor/dp_models",
  "entries": [
    { "path": "<repo == local path>", "sha256": "...", "size": 123,
      "lfs": true, "group": "audio|visual|semantic_bases" }
  ],
  "public_base_models": [ /* models fetched from their public source, not stored here */ ],
  "summary": { "file_count": 430, "total_bytes": 3.42e9, "by_group": {...} }
}
```

For LFS entries `sha256` equals the Git-LFS oid; for small inlined files it is
computed from content. The download script verifies it after every download.

## Download script

```bash
# dry-run (groups + how many already present locally)
python DataProcessor/scripts/download_models.py --dry-run

# full download (token only needed because the repo is private)
HF_TOKEN=hf_xxx python DataProcessor/scripts/download_models.py

# only model weights, skip the 409 semantic-base images (~38 MB)
python DataProcessor/scripts/download_models.py --groups audio visual

# pin the versioned snapshot
python DataProcessor/scripts/download_models.py --revision v1.0.0
```

Idempotent: re-running skips files already present with a matching sha256. Atomic
writes + retry + 3-attempt backoff. Token read from `HF_TOKEN` /
`HUGGINGFACE_HUB_TOKEN`.

## Bootstrap

```bash
HF_TOKEN=hf_xxx ./bootstrap.sh            # full: prereqs, venvs+deps, models, stack, smoke
./bootstrap.sh --check                    # prereqs + model dry-run only (no changes)
./bootstrap.sh --skip-deps                # create venvs but skip pip install
./bootstrap.sh --models-groups "audio visual"
./bootstrap.sh --with-triton              # also start Triton
./bootstrap.sh --no-start                 # infra + DB migrations only
```

It orchestrates the existing E2E scripts (`backend/scripts/setup_e2e_infra.sh`,
`start_e2e_stack.sh --with-infra`) and adds venv creation, dependency install and
model download. **Полный E2E runbook (git clone → зелёный прогон):** [backend/docs/E2E_RUNBOOK.md](../backend/docs/E2E_RUNBOOK.md) § 0.

Stop with `./backend/scripts/stop_e2e_stack.sh --with-infra`.

## Spec reconciliation (status)

| Model | Spec | What's in the repo | Status |
|---|---|---|---|
| emonet_8 | `vision/emonet_8_inprocess.yaml` → `bundled_models/visual/emonet/emonet_8.pth` | same path | ✅ matches |
| emotion_diarization | `audio/emotion_diarization_*` → `audio/emotion_diarization/wavlm_large` (dir) | same path | ✅ matches |
| emotion_recognition | **new** `audio/emotion_recognition_inprocess.yaml` | `audio/emotion_recognition/wav2vec2` | ✅ spec added (wiring deferred) |
| action_recognition | `vision/slowfast_r50_action_recognition.yaml` (SlowFast `.pyth`) | VideoMAE (Kinetics-400) | ⚠️ **new** `videomae_kinetics400_inprocess.yaml` added; slowfast weights not shipped — pick one in task 6 |
| whisper | `audio/whisper_*_inprocess.yaml` → `audio/whisper/small.pt` (openai torch) | `audio/whisper/whisper_small_hf/` (HF transformers) | ⚠️ format differs; needs loader decision in task 6 |
| yolo | only `vision/yolo11x_*_triton.yaml` | baseline `yolo11l` (.pt/.onnx) | ℹ️ baseline shipped; your fine-tuned YOLO (task 1) plugs in here |
| CLAP / places365 / slowfast / e5 / pyannote / source_separation / speaker_diarization | specs exist | not in repo | ℹ️ `public_base_models` — fetch from public source (existing `scripts/save_*` / `download_*`) |

The ⚠️ items require small code changes in the components and are intentionally
left for the "feature quality / adapt to our models" workstream (task 6), not for
this download/bootstrap layer.

## Security note

The HF token used to perform the migration was shared in plaintext during the
session. **Rotate it** (HF → Settings → Access Tokens → revoke + create new) and
pass the new one only via the `HF_TOKEN` environment variable. No token is stored
in any committed file.

## Old repos

`trendflow_output` no longer exists (renamed to `trendflow_models`); its full
pre-cleanup snapshot remains in git history. `trendflow_artifact_0_1` is now
redundant (its content was merged) — keep as archive or delete at your
discretion.
