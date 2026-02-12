#!/usr/bin/env python3
"""
Human-friendly quality demo for video_pacing: HTML report with shot boundaries + key metrics.

Flow:
1) Run Segmenter to create frames_dir (or reuse an existing frames_dir)
2) Run video_pacing to produce video_pacing_features.npz (or reuse an existing artifact)
3) Generate an HTML report with:
   - top-level features table
   - shot boundary thumbnails (subset)
   - simple sanity checks (times_s monotonic, boundaries within frame_indices)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2  # type: ignore
import numpy as np

# Make VisualProcessor imports work for FrameManager.
vp_root = Path(__file__).parent.parent.parent / "VisualProcessor"
sys.path.insert(0, str(vp_root))

from utils.frame_manager import FrameManager  # type: ignore  # noqa: E402
from utils.logger import get_logger  # type: ignore  # noqa: E402
from utils.artifact_validator import validate_npz  # type: ignore  # noqa: E402

logger = get_logger("demo_video_pacing_quality")


def _npz_to_dict(npz_path: str) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for k in data.files:
        v = data[k]
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            try:
                out[k] = v.item()
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def _thumb(frame_manager: FrameManager, frame_idx: int, max_side: int = 240) -> Optional[str]:
    try:
        img = frame_manager.get(int(frame_idx))
        h, w = img.shape[:2]
        scale = min(float(max_side) / float(max(h, w)), 1.0)
        nh, nw = int(h * scale), int(w * scale)
        if nh <= 0 or nw <= 0:
            return None
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
    except Exception:
        return None


def run_segmenter(video_path: str, out_dir: str, visual_cfg_path: Optional[str]) -> str:
    seg = Path(__file__).parent.parent.parent / "Segmenter" / "segmenter.py"
    if not seg.exists():
        raise FileNotFoundError(f"Segmenter not found: {seg}")
    video_id = Path(video_path).stem
    frames_dir = os.path.join(out_dir, video_id, "video")
    meta_path = os.path.join(frames_dir, "metadata.json")
    if os.path.isfile(meta_path):
        return frames_dir
    cmd = [
        sys.executable,
        str(seg),
        "--video-path",
        str(video_path),
        "--output",
        str(out_dir),
        "--platform-id",
        "demo",
        f"--video-id={video_id}",
        "--run-id",
        "demo_run",
        "--sampling-policy-version",
        "v1",
        "--config-hash",
        "demo_demo_run",
    ]
    if visual_cfg_path:
        cmd += ["--visual-cfg-path", str(visual_cfg_path)]
    logger.info("Running Segmenter: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    if not os.path.isfile(meta_path):
        raise RuntimeError(f"Segmenter did not create frames_dir: {frames_dir}")
    return frames_dir


def run_video_pacing(frames_dir: str, rs_path: str) -> str:
    module_main = Path(__file__).parent.parent.parent / "VisualProcessor" / "modules" / "video_pacing" / "main.py"
    out_npz = os.path.join(rs_path, "video_pacing", "video_pacing_features.npz")
    if os.path.isfile(out_npz):
        return out_npz
    cmd = [
        sys.executable,
        str(module_main),
        "--frames-dir",
        str(frames_dir),
        "--rs-path",
        str(rs_path),
    ]
    # Ensure repo-root packages are importable (dp_models, etc.) if running in a venv.
    repo_root = str(Path(__file__).parent.parent.parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    logger.info("Running video_pacing: %s", " ".join(cmd))
    subprocess.check_call(cmd, env=env)
    if not os.path.isfile(out_npz):
        raise RuntimeError(f"video_pacing finished but artifact not found: {out_npz}")
    return out_npz


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser("demo_video_pacing_quality")
    ap.add_argument("--video-path", default=None, help="Optional: run Segmenter to create frames_dir")
    ap.add_argument("--frames-dir", default=None, help="Optional: existing frames_dir (with metadata.json)")
    ap.add_argument("--rs-path", required=True, help="Per-run result_store path containing core providers")
    ap.add_argument("--out-dir", required=True, help="Output dir for HTML report (and Segmenter outputs if used)")
    ap.add_argument("--visual-cfg-path", default=None, help="Optional Segmenter visual cfg")
    ap.add_argument("--thumb-max-side", type=int, default=240)
    args = ap.parse_args(argv)

    rs_path = os.path.abspath(str(args.rs_path))
    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    if args.frames_dir:
        frames_dir = os.path.abspath(str(args.frames_dir))
    else:
        if not args.video_path:
            raise SystemExit("Provide either --frames-dir or --video-path")
        frames_dir = run_segmenter(str(args.video_path), out_dir=out_dir, visual_cfg_path=args.visual_cfg_path)

    npz_path = run_video_pacing(frames_dir=frames_dir, rs_path=rs_path)
    ok, issues, _ = validate_npz(npz_path)
    data = _npz_to_dict(npz_path)

    fi_raw = data.get("frame_indices")
    ts_raw = data.get("times_s")
    sb_raw = data.get("shot_boundary_frame_indices")
    motion_raw = data.get("motion_norm_per_sec_mean")
    sem_raw = data.get("semantic_change_rate_per_sec")
    color_raw = data.get("color_change_rate_per_sec")
    fi = np.asarray(fi_raw if fi_raw is not None else [], dtype=np.int32)
    ts = np.asarray(ts_raw if ts_raw is not None else [], dtype=np.float32)
    sb = np.asarray(sb_raw if sb_raw is not None else [], dtype=np.int32)
    motion = np.asarray(motion_raw if motion_raw is not None else [], dtype=np.float32).reshape(-1)
    sem = np.asarray(sem_raw if sem_raw is not None else [], dtype=np.float32).reshape(-1)
    col = np.asarray(color_raw if color_raw is not None else [], dtype=np.float32).reshape(-1)
    feats = data.get("features") if isinstance(data.get("features"), dict) else {}
    meta = data.get("meta")
    if isinstance(meta, np.ndarray) and meta.dtype == object and meta.shape == ():
        try:
            meta = meta.item()
        except Exception:
            meta = None
    ui = meta.get("ui_payload") if isinstance(meta, dict) else None
    if not isinstance(ui, dict):
        ui = {}
    summ = data.get("summary")
    if isinstance(summ, np.ndarray) and summ.dtype == object and summ.shape == ():
        try:
            summ = summ.item()
        except Exception:
            summ = None
    if not isinstance(summ, dict):
        summ = {}

    checks: List[str] = []
    if fi.size >= 2 and ts.size == fi.size:
        if np.any(np.diff(ts) < -1e-3):
            checks.append("FAIL: times_s is not monotonic")
        else:
            checks.append("OK: times_s monotonic")
    else:
        checks.append("WARN: times_s missing or shape mismatch")
    if sb.size and fi.size:
        if np.any((sb < fi.min()) | (sb > fi.max())):
            checks.append("FAIL: shot_boundary_frame_indices out of frame_indices range")
        else:
            checks.append("OK: shot_boundary_frame_indices within range")

    # Map shot boundary frame idx -> time if present in this module's sampling (best-effort).
    boundary_times: List[float] = []
    if fi.size and ts.size == fi.size and sb.size:
        m = {int(f): float(t) for f, t in zip(fi.tolist(), ts.tolist())}
        boundary_times = [m.get(int(x)) for x in sb.tolist() if m.get(int(x)) is not None]

    fm = FrameManager(frames_dir=frames_dir, chunk_size=32, cache_size=2)
    try:
        # Thumbnail a subset of boundaries.
        thumbs_rows: List[str] = []
        pick = sb.tolist()
        if len(pick) > 30:
            pick = [pick[0]] + pick[1:: max(1, len(pick) // 30)]
        for idx in pick:
            uri = _thumb(fm, int(idx), max_side=int(args.thumb_max_side))
            if not uri:
                continue
            thumbs_rows.append(f"<tr><td>{int(idx)}</td><td><img src='{uri}'/></td></tr>")
    finally:
        fm.close()

    # Render feature table (top-level keys only).
    feat_rows: List[str] = []
    if isinstance(feats, dict):
        for k in sorted(feats.keys()):
            v = feats.get(k)
            if isinstance(v, (int, float, str)):
                feat_rows.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
            elif isinstance(v, (list, tuple)) and len(v) <= 16:
                feat_rows.append(f"<tr><td>{k}</td><td>{list(v)}</td></tr>")

    issues_html = ""
    if not ok:
        issues_html = "<br/>".join([f"{i.level}: {i.message}" for i in issues])
    else:
        issues_html = "OK"

    # Render curves with plotly if available (preferred); else keep simple HTML without charts.
    plot_html = ""
    try:
        import plotly.graph_objects as go  # type: ignore
        from plotly.subplots import make_subplots  # type: ignore

        if ts.size and motion.size == ts.size and sem.size == ts.size and col.size == ts.size:
            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=("Motion (per-sec mean)", "Semantic change rate (/s)", "Color change rate (/s)"),
            )
            fig.add_trace(go.Scatter(x=ts, y=motion, mode="lines", name="motion_norm_per_sec_mean"), row=1, col=1)
            fig.add_trace(go.Scatter(x=ts, y=sem, mode="lines", name="semantic_change_rate_per_sec"), row=2, col=1)
            fig.add_trace(go.Scatter(x=ts, y=col, mode="lines", name="color_change_rate_per_sec"), row=3, col=1)

            for t in boundary_times:
                for r in [1, 2, 3]:
                    fig.add_vline(x=float(t), line_width=1, line_dash="dot", line_color="#93c5fd", row=r, col=1)

            fig.update_layout(
                height=760,
                legend=dict(orientation="h"),
                margin=dict(l=40, r=20, t=60, b=40),
                template="plotly_dark",
            )
            fig.update_xaxes(title_text="time (s)", row=3, col=1)
            plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    except Exception:
        plot_html = ""

    ui_debug = ""
    try:
        ui_debug = json.dumps({"ui_payload": ui, "summary": summ}, ensure_ascii=False, indent=2)
    except Exception:
        ui_debug = "{}"

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>video_pacing quality demo</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; background: #0b1220; color: #e5e7eb; }}
    a {{ color: #93c5fd; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #334155; padding: 6px; vertical-align: top; }}
    th {{ background: #0f172a; }}
    .card {{ background: #0f172a; padding: 12px; border: 1px solid #334155; border-radius: 10px; margin: 12px 0; }}
    img {{ max-width: 240px; height: auto; border-radius: 8px; }}
    code {{ background: #111827; padding: 2px 6px; border-radius: 6px; }}
    pre {{ background: #111827; padding: 10px; border-radius: 10px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h2>video_pacing — quality demo</h2>
  <div class="card">
    <div><b>NPZ</b>: <code>{npz_path}</code></div>
    <div><b>validate_npz</b>: {issues_html}</div>
    <div><b>sanity</b>:<br/>{"<br/>".join(checks)}</div>
  </div>
  <div class="card">
    <h3>Curves</h3>
    {plot_html if plot_html else "<div>plotly not available (no curves rendered)</div>"}
  </div>
  <div class="card">
    <h3>Top-level features (subset)</h3>
    <table><tr><th>key</th><th>value</th></tr>
      {"".join(feat_rows) if feat_rows else "<tr><td colspan='2'>no scalar features</td></tr>"}
    </table>
  </div>
  <div class="card">
    <h3>Shot boundary thumbnails (subset)</h3>
    <table><tr><th>frame_idx</th><th>thumb</th></tr>
      {"".join(thumbs_rows) if thumbs_rows else "<tr><td colspan='2'>no thumbnails</td></tr>"}
    </table>
  </div>
  <div class="card">
    <h3>meta.ui_payload / summary (debug)</h3>
    <pre>{ui_debug}</pre>
  </div>
</body>
</html>
"""

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    out_path = os.path.join(out_dir, f"demo_video_pacing_quality_{ts}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Wrote HTML: %s", out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


