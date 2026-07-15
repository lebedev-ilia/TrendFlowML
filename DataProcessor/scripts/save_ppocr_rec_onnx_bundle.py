#!/usr/bin/env python3
"""
Download PP-OCR recognizer ONNX + dict into DP_MODELS_ROOT for ocr_extractor.

Usage:
    python scripts/save_ppocr_rec_onnx_bundle.py
    python scripts/save_ppocr_rec_onnx_bundle.py --lang eslav

Output:
    dp_models/bundled_models/visual/ocr/ppocr_rec_onnx_v1/model.onnx
    dp_models/bundled_models/visual/ocr/ppocr_rec_onnx_v1/dict.txt

Source: monkt/paddleocr-onnx (open HF repo, no auth).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _default_models_root(repo_root: str) -> str:
    return os.path.join(repo_root, "dp_models", "bundled_models")


def main() -> int:
    ap = argparse.ArgumentParser(description="Save PP-OCR rec ONNX pack for ocr_extractor")
    ap.add_argument(
        "--models-root",
        default=None,
        help="bundled_models root (default: <repo>/dp_models/bundled_models)",
    )
    ap.add_argument(
        "--lang",
        default="eslav",
        choices=("english", "eslav", "latin"),
        help="Recognizer language pack (eslav = RU/UA/BG/BY)",
    )
    args = ap.parse_args()

    repo = _repo_root()
    models_root = os.path.abspath(args.models_root or _default_models_root(repo))
    out_dir = os.path.join(models_root, "visual", "ocr", "ppocr_rec_onnx_v1")
    onnx_dst = os.path.join(out_dir, "model.onnx")
    dict_dst = os.path.join(out_dir, "dict.txt")

    if os.path.isfile(onnx_dst) and os.path.isfile(dict_dst):
        print(f"[ppocr] already present: {onnx_dst}")
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print(f"ERROR: huggingface_hub required: {e}", file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)
    rec_rel = f"languages/{args.lang}/rec.onnx"
    dict_rel = f"languages/{args.lang}/dict.txt"
    print(f"[ppocr] downloading monkt/paddleocr-onnx {args.lang} …")
    rec_path = hf_hub_download("monkt/paddleocr-onnx", rec_rel)
    dict_path = hf_hub_download("monkt/paddleocr-onnx", dict_rel)

    for src, dst in ((rec_path, onnx_dst), (dict_path, dict_dst)):
        tmp = dst + ".tmp"
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)

    print(f"[ppocr] OK: {onnx_dst} ({os.path.getsize(onnx_dst)} bytes)")
    print(f"[ppocr] OK: {dict_dst} ({os.path.getsize(dict_dst)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
