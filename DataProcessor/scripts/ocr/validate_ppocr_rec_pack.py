#!/usr/bin/env python3
"""
Validate a PP-OCR recognizer ONNX + dict pack for TrendFlowML DataProcessor.

This script is intentionally lightweight:
- uses ONNXRuntime (already present in VisualProcessor venv)
- uses the same preprocess + greedy CTC decode logic as ocr_extractor

Usage (recommended, run in VisualProcessor venv):
  DataProcessor/VisualProcessor/.vp_venv/bin/python DataProcessor/scripts/ocr/validate_ppocr_rec_pack.py \
    --onnx DataProcessor/dp_models/bundled_models/visual/ocr/ppocr_rec_onnx_v1/model.onnx \
    --dict DataProcessor/dp_models/bundled_models/visual/ocr/ppocr_rec_onnx_v1/dict.txt \
    --image /path/to/crop_or_frame.png
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np


def load_dict(dict_path: str) -> List[str]:
    if not os.path.isfile(dict_path):
        raise FileNotFoundError(dict_path)
    chars: List[str] = []
    with open(dict_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ch = line.rstrip("\n\r")
            if ch:
                chars.append(ch)
    if not chars:
        raise RuntimeError(f"dict is empty: {dict_path}")
    return chars


def preprocess_rgb_uint8(img_rgb: np.ndarray, *, img_h: int, img_w: int) -> np.ndarray:
    from PIL import Image  # type: ignore

    img_h = int(img_h)
    img_w = int(img_w)
    im = Image.fromarray(img_rgb.astype(np.uint8)).convert("RGB")
    w, h = im.size
    new_w = int(round(float(img_h) * float(w) / float(h)))
    new_w = max(1, min(new_w, img_w))
    im = im.resize((new_w, img_h), resample=Image.BICUBIC)
    canvas = Image.new("RGB", (img_w, img_h), color=(0, 0, 0))
    canvas.paste(im, (0, 0))
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.transpose(x, (2, 0, 1)).reshape(1, 3, img_h, img_w).astype(np.float32)
    return x


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.maximum(1e-9, np.sum(ex, axis=1, keepdims=True))


def ctc_greedy_decode(logits_1tc: np.ndarray, chars: List[str]) -> Tuple[str, float]:
    if logits_1tc.ndim != 3 or logits_1tc.shape[0] != 1:
        raise RuntimeError(f"unexpected logits shape: {logits_1tc.shape}")
    probs = softmax(np.asarray(logits_1tc[0], dtype=np.float32))  # (T,C)
    idx = np.argmax(probs, axis=1).astype(np.int32)
    conf = np.max(probs, axis=1).astype(np.float32)
    out: List[str] = []
    out_conf: List[float] = []
    prev = -1
    for t in range(int(idx.shape[0])):
        i = int(idx[t])
        if i == 0:
            prev = i
            continue
        if i == prev:
            continue
        j = i - 1
        if 0 <= j < len(chars):
            out.append(chars[j])
            out_conf.append(float(conf[t]))
        prev = i
    s = "".join(out).strip()
    score = float(sum(out_conf) / max(1, len(out_conf))) if out_conf else 0.0
    return s, score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--dict", required=True)
    ap.add_argument("--image", required=True, help="Any RGB image (crop preferred).")
    ap.add_argument("--img-h", type=int, default=48)
    ap.add_argument("--img-w", type=int, default=320)
    args = ap.parse_args()

    import onnxruntime as ort  # type: ignore
    from PIL import Image  # type: ignore

    if not os.path.isfile(args.onnx):
        raise FileNotFoundError(args.onnx)
    if not os.path.isfile(args.dict):
        raise FileNotFoundError(args.dict)
    if not os.path.isfile(args.image):
        raise FileNotFoundError(args.image)

    chars = load_dict(args.dict)
    providers = ort.get_available_providers()
    use = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx, providers=use)
    inp = sess.get_inputs()[0]
    print("ONNX input:", inp.name, inp.shape, inp.type)
    outs = sess.get_outputs()
    print("ONNX outputs:", [(o.name, o.shape, o.type) for o in outs])

    img = Image.open(args.image).convert("RGB")
    x = np.asarray(img, dtype=np.uint8)
    x_in = preprocess_rgb_uint8(x, img_h=int(args.img_h), img_w=int(args.img_w))
    y = sess.run(None, {inp.name: x_in})[0]
    y = np.asarray(y)
    print("logits shape:", y.shape, "dtype:", y.dtype)
    text, score = ctc_greedy_decode(y, chars)
    print("decoded:", repr(text))
    print("score:", score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


