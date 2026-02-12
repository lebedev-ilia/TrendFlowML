#!/usr/bin/env python3
from __future__ import annotations

"""
V2: minimal training loop for v1 skeleton.

This is a "skeleton" trainer: it is designed to be reproducible and contract-correct,
not necessarily fast.

Inputs:
- v1_dataset_index.parquet (or csv/jsonl fallback)
- per-run artifacts referenced by the index:
  - core_clip_npz_path (frame_embeddings, frame_indices)
  - segmenter_metadata_path (union_timestamps_sec)

Outputs:
- checkpoint.pt
- training_run_manifest.json
- metrics.json (Spearman/MAE on val/test)
"""

import argparse
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _require_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore

        return torch, nn
    except Exception as e:
        raise RuntimeError("train_v1_skeleton requires torch. Install torch in your venv.") from e


def _require_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as e:
        raise RuntimeError("train_v1_skeleton requires numpy to read NPZ/NPY artifacts.") from e


def load_index(path: str) -> "pd.DataFrame":
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("train_v1_skeleton requires pandas.") from e

    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower().endswith(".jsonl"):
        return pd.read_json(p, lines=True)
    raise ValueError(f"Unsupported v1 index format: {path}")


def spearmanr(y_true: List[float], y_pred: List[float]) -> float:
    # simple pure-python spearman (copied style from baseline utils_metrics)
    import math

    def _rankdata(x: List[float]) -> List[float]:
        pairs = [(v, i) for i, v in enumerate(x)]
        pairs.sort(key=lambda t: t[0])
        ranks = [0.0] * len(x)
        i = 0
        while i < len(pairs):
            j = i
            while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
                j += 1
            avg = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                ranks[pairs[k][1]] = avg
            i = j + 1
        return ranks

    xs = []
    ys = []
    for a, b in zip(y_true, y_pred):
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        xs.append(a)
        ys.append(b)
    if len(xs) < 2:
        return float("nan")
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = 0.0
    dx = 0.0
    dy = 0.0
    for a, b in zip(rx, ry):
        xa = a - mx
        yb = b - my
        num += xa * yb
        dx += xa * xa
        dy += yb * yb
    if dx <= 0.0 or dy <= 0.0:
        return float("nan")
    return num / math.sqrt(dx * dy)


def mae(y_true: List[float], y_pred: List[float]) -> float:
    import math

    s = 0.0
    n = 0
    for a, b in zip(y_true, y_pred):
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        s += abs(a - b)
        n += 1
    return float("nan") if n == 0 else s / n


def load_visual_seq(row: Dict[str, Any]) -> Tuple["np.ndarray", "np.ndarray", float]:
    np = _require_numpy()

    npz_path = str(row.get("core_clip_npz_path") or "")
    seg_path = str(row.get("segmenter_metadata_path") or "")
    if not npz_path or not os.path.exists(npz_path):
        raise FileNotFoundError(f"core_clip_npz_path not found: {npz_path}")
    if not seg_path or not os.path.exists(seg_path):
        raise FileNotFoundError(f"segmenter_metadata_path not found: {seg_path}")

    with np.load(npz_path, allow_pickle=True) as z:
        frame_idx = z["frame_indices"].astype("int64")
        frame_emb = z["frame_embeddings"].astype("float32")

    seg = json.loads(Path(seg_path).read_text(encoding="utf-8"))
    union_t = seg.get("union_timestamps_sec") or []
    union_t = np.asarray(union_t, dtype="float32")
    times = union_t[frame_idx]
    duration_sec = float(row.get("duration_sec") or (union_t[-1] if union_t.size else 0.0))
    return frame_emb, times, duration_sec


def main() -> int:
    p = argparse.ArgumentParser(description="Train v1 skeleton (V2)")
    p.add_argument("--v1-index", type=str, required=True)
    p.add_argument("--text-index", type=str, default="", help="Optional text embeddings index (video_id -> text_npz_path)")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder", type=str, default="v0", choices=["v0", "v1"], help="Which encoder to use for visual modality")
    p.add_argument("--quantiles", type=str, default="0.5", help="Comma-separated quantiles, e.g. '0.1,0.5,0.9'")
    args = p.parse_args()

    torch, nn = _require_torch()
    np = _require_numpy()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_index(args.v1_index)
    if df.shape[0] == 0:
        raise ValueError("Empty v1 index")

    # Hybrid split (same policy as baseline). If channel_id absent, fallback to channelTitle then video_id.
    from Models.v1.common.split import split_hybrid_time_channel  # noqa: E402

    channel_col = "channel_id" if "channel_id" in df.columns else ("channelTitle" if "channelTitle" in df.columns else "video_id")
    df = df.assign(_split=split_hybrid_time_channel(df, channel_col=channel_col, published_col="publishedAt"))

    # Optional join text index
    if args.text_index:
        import pandas as pd  # type: ignore

        tdf = load_index(args.text_index)
        if "video_id" in tdf.columns and "text_npz_path" in tdf.columns:
            df = df.merge(tdf[["video_id", "text_npz_path"]], on="video_id", how="left")

    from Models.v1.encoder.encoder_v0 import EncoderV0, EncoderV0Config, validate_encoder_output  # noqa: E402
    from Models.v1.encoder.encoder_v1 import EncoderV1, EncoderV1Config  # noqa: E402
    from Models.v1.model.v1_skeleton import V1Skeleton, V1SkeletonConfig  # noqa: E402

    if args.encoder == "v0":
        cfg_enc = EncoderV0Config(d_model=768, seed=args.seed)
        enc_v = EncoderV0(cfg=cfg_enc, d_in=512)  # core_clip frame_embeddings: (N,512)
        enc_is_torch = False
    else:
        cfg_enc = EncoderV1Config(d_model=768)
        enc_v = EncoderV1(cfg=cfg_enc, d_in=512)  # trainable
        enc_is_torch = True

    # meta vector = snapshot_0 numeric + channel stats + duration + age proxy
    meta_cols = [
        "views_0",
        "likes_0",
        "comments_0",
        "channel_subscribers_0",
        "channel_total_views_0",
        "channel_total_videos_0",
        "duration_sec",
    ]
    d_meta = len(meta_cols)
    qs = tuple([float(x.strip()) for x in args.quantiles.split(",") if x.strip()])
    if not qs:
        qs = (0.5,)
    model = V1Skeleton(cfg=V1SkeletonConfig(d_model=768, quantiles=qs), d_meta=d_meta)
    model_torch = model

    params = list(model_torch.parameters())
    if enc_is_torch:
        params += list(enc_v.parameters())  # type: ignore[attr-defined]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    def _iter_rows(split: str):
        sub = df[df["_split"] == split]
        for _, r in sub.iterrows():
            yield r.to_dict()

    def _load_text_tokens(row: Dict[str, Any]) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        np = _require_numpy()
        pth = str(row.get("text_npz_path") or "")
        if not pth or not os.path.exists(pth):
            return None, None
        with np.load(pth, allow_pickle=True) as z:
            tok = z["text_tokens"].astype("float32")  # (Kc+1,D)
            msk = z["text_mask"].astype("float32")  # (Kc+1,)
        return torch.from_numpy(tok).unsqueeze(0), torch.from_numpy(msk).unsqueeze(0)

    # training (very small, single-sample batches)
    model_torch.train()
    for epoch in range(args.epochs):
        for row in _iter_rows("train"):
            emb, times, dur = load_visual_seq(row)
            out = enc_v(times_s=torch.from_numpy(times), x=torch.from_numpy(emb), duration_sec=dur)
            validate_encoder_output(out, d_model=cfg_enc.d_model)

            # build tensors
            visual_tokens = out["summary_tokens"].unsqueeze(0)
            visual_times = out["summary_times_s"].unsqueeze(0)
            visual_mask = out["summary_mask"].unsqueeze(0)
            meta_vec = torch.tensor([[float(row.get(c, 0.0)) for c in meta_cols]], dtype=torch.float32)
            duration = torch.tensor([float(dur)], dtype=torch.float32)
            text_tokens, text_mask = _load_text_tokens(row)

            pred = model_torch.forward(
                visual_tokens=visual_tokens,
                visual_times_s=visual_times,
                visual_mask=visual_mask,
                text_tokens=text_tokens,
                text_mask=text_mask,
                meta_vec=meta_vec,
                duration_sec=duration,
            )
            tv = torch.tensor(
                [[float(row.get("target_views_7d", float("nan"))), float(row.get("target_views_14d", float("nan"))), float(row.get("target_views_21d", float("nan")))]],
                dtype=torch.float32,
            )
            tl = torch.tensor(
                [[float(row.get("target_likes_7d", float("nan"))), float(row.get("target_likes_14d", float("nan"))), float(row.get("target_likes_21d", float("nan")))]],
                dtype=torch.float32,
            )
            m7 = torch.tensor([float(row.get("mask_7d", 0.0))], dtype=torch.float32)

            losses = model_torch.loss(pred=pred, target_views=tv, target_likes=tl, mask_7d=m7)
            loss = losses["loss"]

            opt.zero_grad()
            loss.backward()
            opt.step()

    # evaluation (test)
    model_torch.eval()
    metrics: Dict[str, Any] = {"created_at": _now_utc(), "per_head": {}}
    for split in ["val", "test"]:
        ys_true = {"views_14d": [], "likes_14d": []}
        ys_pred = {"views_14d": [], "likes_14d": []}
        for row in _iter_rows(split):
            emb, times, dur = load_visual_seq(row)
            out = enc_v(times_s=torch.from_numpy(times), x=torch.from_numpy(emb), duration_sec=dur)
            visual_tokens = out["summary_tokens"].unsqueeze(0)
            visual_times = out["summary_times_s"].unsqueeze(0)
            visual_mask = out["summary_mask"].unsqueeze(0)
            meta_vec = torch.tensor([[float(row.get(c, 0.0)) for c in meta_cols]], dtype=torch.float32)
            duration = torch.tensor([float(dur)], dtype=torch.float32)

            pred = model_torch.forward(
                visual_tokens=visual_tokens,
                visual_times_s=visual_times,
                visual_mask=visual_mask,
                meta_vec=meta_vec,
                duration_sec=duration,
            )
            # take 14d index 1
            pv14 = float(pred["views"][0, 1].detach().cpu().item())
            pl14 = float(pred["likes"][0, 1].detach().cpu().item())
            tv14 = float(row.get("target_views_14d", float("nan")))
            tl14 = float(row.get("target_likes_14d", float("nan")))
            ys_true["views_14d"].append(tv14)
            ys_true["likes_14d"].append(tl14)
            ys_pred["views_14d"].append(pv14)
            ys_pred["likes_14d"].append(pl14)

        metrics[split] = {
            "views_14d": {"spearman": spearmanr(ys_true["views_14d"], ys_pred["views_14d"]), "mae": mae(ys_true["views_14d"], ys_pred["views_14d"]), "n": len(ys_pred["views_14d"])},
            "likes_14d": {"spearman": spearmanr(ys_true["likes_14d"], ys_pred["likes_14d"]), "mae": mae(ys_true["likes_14d"], ys_pred["likes_14d"]), "n": len(ys_pred["likes_14d"])},
        }

    # save checkpoint
    ckpt = {
        "created_at": _now_utc(),
        "seed": args.seed,
        "encoder": args.encoder,
        "quantiles": list(qs),
        "model_state_dict": model_torch.state_dict(),
    }
    if enc_is_torch:
        ckpt["encoder_state_dict"] = enc_v.state_dict()  # type: ignore[attr-defined]
    torch.save(ckpt, out_dir / "checkpoint.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "training_run_manifest.json").write_text(
        json.dumps(
            {
                "created_at": _now_utc(),
                "seed": args.seed,
                "v1_index": args.v1_index,
                "text_index": args.text_index or None,
                "encoder": args.encoder,
                "quantiles": list(qs),
                "channel_col_used": channel_col,
                "note": "skeleton trainer. Evaluation should be done via Models/v1/training/evaluate_v1.py",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[ok] wrote v1 skeleton artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


