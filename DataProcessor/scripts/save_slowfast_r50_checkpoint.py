#!/usr/bin/env python3
"""
Save SlowFast R50 (Kinetics-400) weights into DP_MODELS_ROOT for action_recognition.

Usage:
    python scripts/save_slowfast_r50_checkpoint.py
    python scripts/save_slowfast_r50_checkpoint.py --models-root dp_models/bundled_models

Output:
    dp_models/bundled_models/visual/action_recognition/slowfast_r50/slowfast_r50.pyth
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
    ap = argparse.ArgumentParser(description="Save SlowFast R50 checkpoint for action_recognition")
    ap.add_argument(
        "--models-root",
        default=None,
        help="DP_MODELS_ROOT (default: <repo>/dp_models/bundled_models)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Override output .pyth path",
    )
    ap.add_argument(
        "--input-checkpoint",
        default=None,
        help="Pre-downloaded SLOWFAST_8x8_R50.pyth (skip torch.hub download)",
    )
    args = ap.parse_args()

    repo = _repo_root()
    models_root = os.path.abspath(args.models_root or _default_models_root(repo))
    out_path = args.output or os.path.join(
        models_root,
        "visual",
        "action_recognition",
        "slowfast_r50",
        "slowfast_r50.pyth",
    )

    if os.path.isfile(out_path):
        print(f"[save_slowfast] already present: {out_path}")
        return 0

    try:
        import torch
        from pytorchvideo.models.hub import slowfast_r50
    except ImportError as e:
        print(f"ERROR: torch and pytorchvideo required: {e}", file=sys.stderr)
        return 1

    if args.input_checkpoint and os.path.isfile(args.input_checkpoint):
        print(f"[save_slowfast] loading from {args.input_checkpoint}")
        hub_ckpt = args.input_checkpoint
        os.environ.setdefault(
            "TORCH_HOME",
            os.path.join(models_root, "torch_cache"),
        )
        hub_dir = os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints")
        os.makedirs(hub_dir, exist_ok=True)
        link_dst = os.path.join(hub_dir, "SLOWFAST_8x8_R50.pyth")
        if not os.path.isfile(link_dst):
            try:
                os.symlink(os.path.abspath(hub_ckpt), link_dst)
            except OSError:
                import shutil
                shutil.copy2(hub_ckpt, link_dst)
        model = slowfast_r50(pretrained=True)
    else:
        print("[save_slowfast] loading slowfast_r50(pretrained=True) — may download ~250MB …")
        model = slowfast_r50(pretrained=True)
    model.eval()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    checkpoint = {
        "model_state": model.state_dict(),
        "model_name": "slowfast_r50",
        "pretrained": True,
    }
    torch.save(checkpoint, tmp)
    os.replace(tmp, out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[save_slowfast] saved {out_path} ({size_mb:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
