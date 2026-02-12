#!/usr/bin/env python3
"""
Human-friendly quality demo for shot_quality.

Creates an HTML report with:
- per-frame curves for selected features
- per-frame top-1 class from quality_probs
- shot boundaries from shot_start_frame/shot_end_frame

Usage:
  python scripts/baseline/demo_shot_quality_quality.py \
    --video-path /path/to/video.mp4 \
    --out-dir /tmp/tf_demo \
    [--visual-cfg-path DataProcessor/configs/visual_config.yaml] \
    [--preset fast|default|quality]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import plotly.graph_objects as go
except Exception as e:
    raise RuntimeError("plotly is required for this demo script") from e


_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "VisualProcessor"))

from utils.logger import get_logger
from modules.cut_detection.cut_detection import CutDetectionPipeline
from modules.shot_quality.shot_quality import ShotQualityModule

logger = get_logger("demo_shot_quality_quality")


def run_segmenter(video_path: str, out_dir: str, visual_cfg_path: Optional[str]) -> str:
    segmenter_script = _root / "Segmenter" / "segmenter.py"
    video_id = Path(video_path).stem
    frames_dir = os.path.join(out_dir, video_id, "video")
    metadata_path = os.path.join(frames_dir, "metadata.json")

    if os.path.exists(metadata_path):
        return frames_dir

    cmd = [
        sys.executable,
        str(segmenter_script),
        "--video-path",
        video_path,
        "--output",
        out_dir,
        "--platform-id",
        "demo",
        "--video-id",
        video_id,
        "--run-id",
        "demo_run",
        "--sampling-policy-version",
        "v1",
        "--config-hash",
        "demo_demo_run",
    ]
    if visual_cfg_path:
        cmd.extend(["--visual-cfg-path", visual_cfg_path])

    logger.info("Running Segmenter: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Segmenter failed:\n{r.stdout}\n{r.stderr}")
    if not os.path.exists(metadata_path):
        raise RuntimeError(f"Segmenter did not create frames_dir: {frames_dir}")
    return frames_dir


def load_latest_npz(rs_path: str, component: str, filename: str) -> str:
    p = Path(rs_path) / component / filename
    if not p.exists():
        raise FileNotFoundError(str(p))
    return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--video-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--visual-cfg-path", default=None)
    ap.add_argument("--preset", default="default", choices=["fast", "default", "quality"])
    args = ap.parse_args()

    out_dir = str(Path(args.out_dir).resolve())
    os.makedirs(out_dir, exist_ok=True)

    frames_dir = run_segmenter(args.video_path, out_dir, args.visual_cfg_path)

    # result_store path is produced by Segmenter under out_dir/<video_id>/result_store
    video_id = Path(args.video_path).stem
    rs_path = os.path.join(out_dir, video_id, "result_store", "demo", video_id, "demo_run")
    os.makedirs(rs_path, exist_ok=True)

    # Run cut_detection (dependency for shots)
    cd = CutDetectionPipeline(rs_path=rs_path)
    cd.run(frames_dir=frames_dir, config={"preset": "default"})

    # Run shot_quality
    sq = ShotQualityModule(rs_path=rs_path, device="cuda")
    sq.run(
        frames_dir=frames_dir,
        config={
            "preset": args.preset,
            "progress_every_n_frames": 25,
        },
    )

    npz_path = load_latest_npz(rs_path, "shot_quality", "shot_quality.npz")
    data = np.load(npz_path, allow_pickle=True)

    meta = {}
    try:
        meta = (data["meta"].item() if "meta" in data else {})
    except Exception:
        meta = {}

    times_s = data["times_s"].astype(np.float32)
    feature_names = [str(x) for x in data["feature_names"].tolist()]
    X = data["frame_features"].astype(np.float32)
    probs = data["quality_probs"].astype(np.float32)

    ui = meta.get("ui_payload") or {}
    q = ui.get("quality") or {}
    frame_conf = np.asarray(q.get("frame_confidence") or [], dtype=np.float32)
    frame_ent = np.asarray(q.get("frame_entropy") or [], dtype=np.float32)

    # pick a few stable signals
    wanted = ["sharpness_tenengrad", "contrast_global", "underexposure_ratio", "overexposure_ratio"]
    series: Dict[str, np.ndarray] = {}
    for name in wanted:
        if name in feature_names:
            j = feature_names.index(name)
            series[name] = X[:, j]

    top1 = np.argmax(probs, axis=1).astype(np.int32)

    fig = go.Figure()
    for k, v in series.items():
        fig.add_trace(go.Scatter(x=times_s, y=v, mode="lines", name=k))

    if frame_conf.size == times_s.size:
        fig.add_trace(go.Scatter(x=times_s, y=frame_conf, mode="lines", name="quality_confidence_max"))
    if frame_ent.size == times_s.size:
        fig.add_trace(go.Scatter(x=times_s, y=frame_ent, mode="lines", name="quality_entropy"))

    fig.add_trace(go.Scatter(x=times_s, y=top1, mode="lines", name="quality_top1_id", yaxis="y2"))

    fig.update_layout(
        title=f"shot_quality demo | preset={args.preset} | status={meta.get('status')}",
        xaxis=dict(title="time (s)"),
        yaxis=dict(title="feature value"),
        yaxis2=dict(title="top1 class id", overlaying="y", side="right"),
        height=700,
        legend=dict(orientation="h"),
    )

    report = {
        "npz_path": npz_path,
        "meta_status": meta.get("status"),
        "empty_reason": meta.get("empty_reason"),
        "ui_payload_present": bool(meta.get("ui_payload")),
    }

    html_path = os.path.join(out_dir, f"shot_quality_demo_{video_id}.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    with open(os.path.join(out_dir, f"shot_quality_demo_{video_id}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("Saved: %s", html_path)
    logger.info("NPZ: %s", npz_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


