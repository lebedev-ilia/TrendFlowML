#!/usr/bin/env python3
from __future__ import annotations

"""
V3: Build text/comment embeddings and aggregate into Kc tokens (no raw storage).

Inputs:
- data_00.json (video_id -> snapshot_0..3 + metadata)
- v1_dataset_index.* (to get video_ids to process)

Outputs:
- per-video NPZ artifacts with:
  - comment_embeddings (Nc, D)
  - comment_scores (Nc,)
  - text_tokens (Kc+1, D): [global_token, topk_tokens...]
  - text_mask (Kc+1,)
  - meta (dict) WITHOUT raw
- an index file mapping video_id -> text_npz_path
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Models.v1.common.utils_bigjson import load_video_records_subset
from Models.v1.text.text_encoder import SentenceTransformerEncoder, TextEncoderSpec


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def load_index(path: str) -> "pd.DataFrame":
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("build_text_embeddings requires pandas.") from e

    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower().endswith(".jsonl"):
        return pd.read_json(p, lines=True)
    raise ValueError(f"Unsupported v1 index format: {path}")


def _comment_score(c: Dict[str, Any]) -> float:
    # deterministic heuristic (no model needed): likes + replies + small length bonus
    try:
        like = float(c.get("likeCount") or 0.0)
    except Exception:
        like = 0.0
    try:
        rep = float(c.get("repliesCount") or 0.0)
    except Exception:
        rep = 0.0
    txt = str(c.get("text") or "")
    length_bonus = min(len(txt), 200) / 200.0
    return float(like + 0.5 * rep + 0.1 * length_bonus)


def build_tokens(
    *,
    comment_embeddings: "np.ndarray",  # (Nc,D)
    comment_scores: "np.ndarray",  # (Nc,)
    Kc: int,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    import numpy as np  # type: ignore

    Nc, D = comment_embeddings.shape
    if Nc == 0:
        tokens = np.zeros((Kc + 1, D), dtype="float32")
        mask = np.zeros((Kc + 1,), dtype="float32")
        mask[0] = 0.0
        return tokens, mask, np.asarray([], dtype="int64")

    # global token = mean over all comments
    global_tok = comment_embeddings.mean(axis=0, keepdims=True).astype("float32")

    # top-K by score (stable: tie-breaker by index)
    idx = np.arange(Nc, dtype="int64")
    order = np.lexsort((idx, -comment_scores.astype("float32")))
    topk = order[: min(Kc, Nc)]
    topk_tok = comment_embeddings[topk].astype("float32")

    # pad to Kc
    if topk_tok.shape[0] < Kc:
        pad = np.zeros((Kc - topk_tok.shape[0], D), dtype="float32")
        topk_tok = np.concatenate([topk_tok, pad], axis=0)

    tokens = np.concatenate([global_tok, topk_tok], axis=0)  # (Kc+1,D)
    mask = np.zeros((Kc + 1,), dtype="float32")
    mask[0] = 1.0
    mask[1 : 1 + min(Kc, Nc)] = 1.0
    return tokens, mask, topk


def main() -> int:
    p = argparse.ArgumentParser(description="Build v1 text/comment embeddings (V3)")
    p.add_argument("--data-json", type=str, required=True)
    p.add_argument("--v1-index", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True, help="Directory to write per-video NPZ artifacts")
    p.add_argument("--out-index", type=str, required=True, help="Output index mapping (parquet/csv/jsonl)")
    p.add_argument("--kc", type=int, default=8, help="Number of top comment tokens (Kc). Total tokens = Kc+1 with global token.")
    p.add_argument("--max-comments", type=int, default=100)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    df = load_index(args.v1_index)
    video_ids = sorted(set([str(v) for v in df["video_id"].astype(str).tolist()]))
    subset, stats = load_video_records_subset(args.data_json, include_video_ids=set(video_ids))

    enc = SentenceTransformerEncoder(TextEncoderSpec(device=args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows: List[Dict[str, Any]] = []
    missing = 0
    for vid in video_ids:
        rec = subset.get(vid)
        if rec is None:
            missing += 1
            continue
        s0 = rec.get("snapshot_0") or {}
        comments = s0.get("comments") or []
        if not isinstance(comments, list):
            comments = []
        comments = comments[: args.max_comments]

        texts: List[str] = []
        scores: List[float] = []
        for c in comments:
            if not isinstance(c, dict):
                continue
            t = str(c.get("text") or "").strip()
            if not t:
                continue
            texts.append(t)
            scores.append(_comment_score(c))

        if texts:
            emb = enc.encode(texts)  # (Nc,D)
            score_arr = np.asarray(scores, dtype="float32")
        else:
            emb = np.zeros((0, enc.dim), dtype="float32")
            score_arr = np.zeros((0,), dtype="float32")

        tokens, mask, topk_idx = build_tokens(comment_embeddings=emb, comment_scores=score_arr, Kc=args.kc)

        # Write NPZ (no raw text)
        npz_path = out_dir / f"{vid}.npz"
        meta = {
            "created_at": _now_utc(),
            "video_id": vid,
            "model_name": enc.spec.model_name,
            "device": enc.spec.device,
            "max_comments": args.max_comments,
            "comments_used": int(emb.shape[0]),
            "kc": args.kc,
        }
        np.savez_compressed(
            npz_path,
            comment_embeddings=emb,
            comment_scores=score_arr,
            text_tokens=tokens,
            text_mask=mask,
            topk_indices=topk_idx.astype("int64"),
            meta=np.asarray(meta, dtype=object),
        )

        index_rows.append(
            {
                "video_id": vid,
                "text_npz_path": str(npz_path),
                "comments_used": int(emb.shape[0]),
                "text_dim": int(enc.dim),
                "kc": int(args.kc),
            }
        )

    out_index = Path(args.out_index)
    out_index.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(index_rows)
    if out_index.suffix.lower() == ".parquet":
        df_out.to_parquet(out_index, index=False)
        fmt = "parquet"
    else:
        df_out.to_csv(out_index, index=False)
        fmt = "csv"

    summary = {
        "created_at": _now_utc(),
        "data_json_seen": stats.total_records_seen,
        "data_json_loaded": stats.total_records_yielded,
        "video_ids": len(video_ids),
        "missing_data_json": missing,
        "written": len(index_rows),
        "out_index": str(out_index),
        "out_dir": str(out_dir),
        "format": fmt,
    }
    (out_index.parent / (out_index.stem + "_meta.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote text embeddings index -> {out_index} ({fmt}, rows={len(index_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


