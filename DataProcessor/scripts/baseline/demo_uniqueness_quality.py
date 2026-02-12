#!/usr/bin/env python3
"""
Human-friendly quality demo for uniqueness: HTML report with top repeated frames + key metrics.

Flow:
1) Optionally run Segmenter to create frames_dir (or reuse an existing frames_dir)
2) Run uniqueness to produce uniqueness_features.npz (or reuse an existing artifact)
3) Generate an HTML report with:
   - top-level features table
   - thumbnails for the most "repeated" frames (highest max_sim_to_other)
   - simple sanity checks (times_s monotonic, shapes, validate_npz)
"""

from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np

# Make VisualProcessor imports work for FrameManager.
vp_root = Path(__file__).parent.parent.parent / "VisualProcessor"
sys.path.insert(0, str(vp_root))

from utils.frame_manager import FrameManager  # type: ignore  # noqa: E402
from utils.logger import get_logger  # type: ignore  # noqa: E402
from utils.artifact_validator import validate_npz  # type: ignore  # noqa: E402

logger = get_logger("demo_uniqueness_quality")


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


def run_uniqueness(frames_dir: str, rs_path: str, repeat_threshold: float, max_frames: int) -> str:
    module_main = Path(__file__).parent.parent.parent / "VisualProcessor" / "modules" / "uniqueness" / "main.py"
    out_npz_new = os.path.join(rs_path, "uniqueness", "uniqueness.npz")
    out_npz_old = os.path.join(rs_path, "uniqueness", "uniqueness_features.npz")
    if os.path.isfile(out_npz_new):
        return out_npz_new
    if os.path.isfile(out_npz_old):
        return out_npz_old
    cmd = [
        sys.executable,
        str(module_main),
        "--frames-dir",
        str(frames_dir),
        "--rs-path",
        str(rs_path),
        "--repeat-threshold",
        str(float(repeat_threshold)),
        "--repeat-threshold-mode",
        "auto",
        "--max-frames",
        str(int(max_frames)),
    ]
    # Ensure repo-root packages are importable (dp_models, etc.) if running in a venv.
    repo_root = str(Path(__file__).parent.parent.parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    logger.info("Running uniqueness: %s", " ".join(cmd))
    subprocess.check_call(cmd, env=env)
    if not os.path.isfile(out_npz):
        raise RuntimeError(f"uniqueness finished but artifact not found: {out_npz}")
    return out_npz


def _summarize_array(x: np.ndarray) -> str:
    if x.size == 0:
        return "empty"
    return f"min={float(np.min(x)):.4g} mean={float(np.mean(x)):.4g} p50={float(np.median(x)):.4g} p95={float(np.percentile(x, 95)):.4g} max={float(np.max(x)):.4g}"


def _arr_or_empty(data: Dict[str, Any], key: str, dtype) -> np.ndarray:
    v = data.get(key, None)
    if v is None:
        return np.asarray([], dtype=dtype)
    return np.asarray(v, dtype=dtype)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser("demo_uniqueness_quality")
    ap.add_argument("--video-path", default=None, help="Optional: run Segmenter to create frames_dir")
    ap.add_argument("--frames-dir", default=None, help="Optional: existing frames_dir (with metadata.json)")
    ap.add_argument("--rs-path", required=True, help="Per-run result_store path containing core providers")
    ap.add_argument("--out-dir", required=True, help="Output dir for HTML report (and Segmenter outputs if used)")
    ap.add_argument("--visual-cfg-path", default=None, help="Optional Segmenter visual cfg")
    ap.add_argument("--thumb-max-side", type=int, default=240)
    ap.add_argument("--topk", type=int, default=24, help="How many top repeated frames to show")
    ap.add_argument("--repeat-threshold", type=float, default=0.97)
    ap.add_argument("--max-frames", type=int, default=200)
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

    npz_path = run_uniqueness(
        frames_dir=frames_dir,
        rs_path=rs_path,
        repeat_threshold=float(args.repeat_threshold),
        max_frames=int(args.max_frames),
    )
    ok, issues, _meta = validate_npz(npz_path)
    data = _npz_to_dict(npz_path)

    fi = _arr_or_empty(data, "frame_indices", np.int32)
    ts = _arr_or_empty(data, "times_s", np.float32)
    max_sim = _arr_or_empty(data, "max_sim_to_other", np.float32)
    cos_dist_next = _arr_or_empty(data, "cos_dist_next", np.float32)
    feats = data.get("features") if isinstance(data.get("features"), dict) else {}

    checks: List[str] = []
    if fi.size == 0:
        checks.append("FAIL: frame_indices is empty")
    if ts.size == fi.size and ts.size >= 2:
        checks.append("OK: times_s present") if not np.any(np.diff(ts) < -1e-3) else checks.append("FAIL: times_s not monotonic")
    else:
        checks.append("WARN: times_s missing or shape mismatch")
    if max_sim.size == fi.size:
        checks.append("OK: max_sim_to_other shape")
    else:
        checks.append("FAIL: max_sim_to_other shape mismatch")
    if fi.size >= 2:
        if cos_dist_next.size == fi.size - 1:
            checks.append("OK: cos_dist_next shape")
        else:
            checks.append("FAIL: cos_dist_next shape mismatch")

    # Prepare top-k repeated frames (highest max_sim_to_other).
    top_rows: List[str] = []
    fm = FrameManager(frames_dir=frames_dir, chunk_size=32, cache_size=2)
    try:
        k = int(max(0, args.topk))
        if fi.size and max_sim.size == fi.size and k > 0:
            order = np.argsort(-max_sim)[: min(k, int(fi.size))]
            for pos in order.tolist():
                fidx = int(fi[pos])
                simv = float(max_sim[pos])
                uri = _thumb(fm, fidx, max_side=int(args.thumb_max_side))
                if not uri:
                    continue
                top_rows.append(
                    "<tr>"
                    f"<td>{fidx}</td>"
                    f"<td>{simv:.6f}</td>"
                    f"<td><img src='{uri}'/></td>"
                    "</tr>"
                )
    finally:
        fm.close()

    feat_rows: List[str] = []
    if isinstance(feats, dict):
        for k in sorted(feats.keys()):
            v = feats.get(k)
            if isinstance(v, (int, float, str)):
                feat_rows.append(f"<tr><td>{k}</td><td>{v}</td></tr>")

    issues_html = "<br/>".join([f"{i.level}: {i.message}" for i in issues]) if not ok else "OK"
    stats_rows: List[str] = [
        f"<tr><td>max_sim_to_other</td><td>{_summarize_array(max_sim)}</td></tr>",
        f"<tr><td>cos_dist_next</td><td>{_summarize_array(cos_dist_next)}</td></tr>",
    ]

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>uniqueness quality demo</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; background: #0b1220; color: #e5e7eb; }}
    a {{ color: #93c5fd; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #334155; padding: 6px; vertical-align: top; }}
    th {{ background: #0f172a; }}
    .card {{ background: #0f172a; padding: 12px; border: 1px solid #334155; border-radius: 10px; margin: 12px 0; }}
    img {{ max-width: 240px; height: auto; border-radius: 8px; }}
    code {{ background: #111827; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h2>uniqueness — quality demo</h2>
  <div class="card">
    <div><b>NPZ</b>: <code>{npz_path}</code></div>
    <div><b>validate_npz</b>: {issues_html}</div>
    <div><b>sanity</b>:<br/>{"<br/>".join(checks)}</div>
  </div>
  <div class="card">
    <h3>Top-level features (subset)</h3>
    <table><tr><th>key</th><th>value</th></tr>
      {"".join(feat_rows) if feat_rows else "<tr><td colspan='2'>no scalar features</td></tr>"}
    </table>
  </div>
  <div class="card">
    <h3>Distributions (summary)</h3>
    <table><tr><th>array</th><th>summary</th></tr>
      {"".join(stats_rows)}
    </table>
  </div>
  <div class="card">
    <h3>Top repeated frames (by max_sim_to_other)</h3>
    <table><tr><th>frame_idx</th><th>max_sim_to_other</th><th>thumb</th></tr>
      {"".join(top_rows) if top_rows else "<tr><td colspan='3'>no thumbnails</td></tr>"}
    </table>
  </div>
</body>
</html>
"""

    ts_tag = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    out_path = os.path.join(out_dir, f"demo_uniqueness_quality_{ts_tag}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Wrote HTML: %s", out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


