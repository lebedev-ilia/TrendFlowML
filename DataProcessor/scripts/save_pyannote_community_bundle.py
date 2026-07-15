#!/usr/bin/env python3
"""
Download pyannote/speaker-diarization-community-1 into DP_MODELS_ROOT.

Requires HF_TOKEN (gated repo — accept terms at hf.co/pyannote/speaker-diarization-community-1).

Usage:
    HF_TOKEN=hf_xxx python scripts/save_pyannote_community_bundle.py
    python scripts/save_pyannote_community_bundle.py --huggingface-token hf_xxx

Output:
    dp_models/bundled_models/audio/pyannote_speaker_diarization/config.yaml + subdirs
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
    ap = argparse.ArgumentParser(description="Save pyannote community diarization bundle")
    ap.add_argument("--models-root", default=None)
    ap.add_argument("--huggingface-token", default=None)
    args = ap.parse_args()

    token = args.huggingface_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print(
            "ERROR: HF_TOKEN required (gated model). "
            "Accept terms at https://hf.co/pyannote/speaker-diarization-community-1",
            file=sys.stderr,
        )
        return 1

    repo = _repo_root()
    models_root = os.path.abspath(args.models_root or _default_models_root(repo))
    out_dir = os.path.join(models_root, "audio", "pyannote_speaker_diarization")
    cfg = os.path.join(out_dir, "config.yaml")

    if os.path.isfile(cfg):
        print(f"[pyannote] already present: {cfg}")
        return 0

    try:
        from huggingface_hub import snapshot_download, login
    except ImportError as e:
        print(f"ERROR: huggingface_hub required: {e}", file=sys.stderr)
        return 1

    login(token=token, add_to_git_credential=False)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[pyannote] downloading pyannote/speaker-diarization-community-1 → {out_dir}")
    snapshot_download(
        "pyannote/speaker-diarization-community-1",
        local_dir=out_dir,
        token=token,
    )

    if not os.path.isfile(cfg):
        print(f"ERROR: config.yaml missing after download: {cfg}", file=sys.stderr)
        return 1

    print(f"[pyannote] OK: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
