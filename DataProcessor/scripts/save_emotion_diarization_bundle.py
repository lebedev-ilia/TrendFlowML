#!/usr/bin/env python3
"""
Download SpeechBrain emotion-diarization-wavlm-large into DP_MODELS_ROOT.

Usage:
    python scripts/save_emotion_diarization_bundle.py
    python scripts/save_emotion_diarization_bundle.py --models-root dp_models/bundled_models

Output:
    dp_models/bundled_models/audio/emotion_diarization/wavlm_large/
      hyperparams.yaml, model.ckpt, wav2vec2.ckpt, ...

Also caches microsoft/wavlm-large into bundled_models/hf_cache (required offline).
"""

from __future__ import annotations

import argparse
import os
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _default_models_root(repo_root: str) -> str:
    return os.path.join(repo_root, "dp_models", "bundled_models")


def main() -> int:
    ap = argparse.ArgumentParser(description="Save SpeechBrain emotion diarization bundle locally")
    ap.add_argument(
        "--models-root",
        default=None,
        help="DP_MODELS_ROOT (default: <repo>/dp_models/bundled_models)",
    )
    ap.add_argument(
        "--skip-wavlm-cache",
        action="store_true",
        help="Do not download microsoft/wavlm-large into hf_cache",
    )
    args = ap.parse_args()

    repo = _repo_root()
    models_root = os.path.abspath(args.models_root or _default_models_root(repo))
    out_dir = os.path.join(models_root, "audio", "emotion_diarization", "wavlm_large")
    hyperparams = os.path.join(out_dir, "hyperparams.yaml")

    if os.path.isfile(hyperparams):
        print(f"[emotion_diarization] already present: {hyperparams}")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print(f"ERROR: huggingface_hub required: {e}", file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)
    print(f"[emotion_diarization] downloading speechbrain/emotion-diarization-wavlm-large → {out_dir}")
    snapshot_download("speechbrain/emotion-diarization-wavlm-large", local_dir=out_dir)

    if not os.path.isfile(hyperparams):
        print(f"ERROR: hyperparams.yaml missing after download: {hyperparams}", file=sys.stderr)
        return 1

    if not args.skip_wavlm_cache:
        hf_home = os.path.join(models_root, "hf_cache")
        os.makedirs(hf_home, exist_ok=True)
        os.environ["HF_HOME"] = hf_home
        print("[emotion_diarization] caching microsoft/wavlm-large for offline WavLM backbone …")
        snapshot_download("microsoft/wavlm-large")

    print(f"[emotion_diarization] OK: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
