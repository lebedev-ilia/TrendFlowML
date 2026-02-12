#!/usr/bin/env python3
"""
Human-friendly quality demo for tempo_extractor: HTML report with windowed BPM timeline + key metrics.

Inputs:
- rs-path: per-run result_store directory (<...>/<platform>/<video>/<run_id>)

Output:
- HTML file with validate_npz result, sanity checks, and a simple SVG plot of windowed_bpm over time.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Reuse VisualProcessor validator (meta contract)
vp_root = Path(__file__).parent.parent.parent / "VisualProcessor"
sys.path.insert(0, str(vp_root))
from utils.artifact_validator import validate_npz  # type: ignore  # noqa: E402
from utils.logger import get_logger  # type: ignore  # noqa: E402

logger = get_logger("demo_tempo_extractor_quality")


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


def _svg_plot(xs: np.ndarray, ys: np.ndarray, *, w: int = 900, h: int = 220) -> str:
    xs = np.asarray(xs, dtype=float).reshape(-1)
    ys = np.asarray(ys, dtype=float).reshape(-1)
    if xs.size == 0 or ys.size == 0 or xs.size != ys.size:
        return "<div>no data</div>"

    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    if abs(xmax - xmin) < 1e-9:
        xmax = xmin + 1.0
    if abs(ymax - ymin) < 1e-9:
        ymax = ymin + 1.0

    pad = 20
    def sx(x: float) -> float:
        return pad + (x - xmin) / (xmax - xmin) * (w - 2 * pad)
    def sy(y: float) -> float:
        return (h - pad) - (y - ymin) / (ymax - ymin) * (h - 2 * pad)

    pts = " ".join([f"{sx(float(x)):.2f},{sy(float(y)):.2f}" for x, y in zip(xs, ys)])
    return f"""
<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="background:#0b1220;border:1px solid #334155;border-radius:10px">
  <polyline points="{pts}" fill="none" stroke="#93c5fd" stroke-width="2"/>
  <text x="{pad}" y="{pad}" fill="#e5e7eb" font-size="12">windowed_bpm over time</text>
  <text x="{pad}" y="{h-4}" fill="#94a3b8" font-size="11">t: {xmin:.1f}..{xmax:.1f}s</text>
  <text x="{w-140}" y="{h-4}" fill="#94a3b8" font-size="11">bpm: {ymin:.1f}..{ymax:.1f}</text>
</svg>
"""


def _feature_lookup(names: np.ndarray, values: np.ndarray) -> Dict[str, float]:
    try:
        n = [str(x) for x in np.asarray(names, dtype=object).reshape(-1).tolist()]
        v = np.asarray(values, dtype=np.float32).reshape(-1)
        return {k: float(v[i]) for i, k in enumerate(n) if i < v.size}
    except Exception:
        return {}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser("demo_tempo_extractor_quality")
    ap.add_argument("--rs-path", required=True, help="Per-run result_store directory (<...>/<platform>/<video>/<run_id>)")
    ap.add_argument("--out-dir", required=True, help="Output dir for HTML report")
    args = ap.parse_args(argv)

    rs_path = os.path.abspath(str(args.rs_path))
    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(rs_path, "tempo_extractor", "tempo_extractor_features.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"tempo_extractor artifact not found: {npz_path}")

    ok, issues, _ = validate_npz(npz_path)
    data = _npz_to_dict(npz_path)

    feat = _feature_lookup(data.get("feature_names"), data.get("feature_values"))
    w_times = np.asarray(data.get("windowed_times_sec") if data.get("windowed_times_sec") is not None else [], dtype=np.float32).reshape(-1)
    w_bpm = np.asarray(data.get("windowed_bpm") if data.get("windowed_bpm") is not None else [], dtype=np.float32).reshape(-1)
    warnings = data.get("warnings")
    if isinstance(warnings, np.ndarray):
        warnings_list = [str(x) for x in warnings.tolist()]
    else:
        warnings_list = []

    checks: List[str] = []
    if w_times.size != w_bpm.size:
        checks.append("FAIL: windowed_times_sec and windowed_bpm shape mismatch")
    elif w_times.size > 1 and np.any(np.diff(w_times) < -1e-3):
        checks.append("FAIL: windowed_times_sec is not monotonic")
    else:
        checks.append("OK: windowed series shapes/monotonicity")

    issues_html = "<br/>".join([f"{i.level}: {i.message}" for i in issues]) if not ok else "OK"

    # Render small feature subset
    keys_show = [
        "tempo_bpm",
        "tempo_bpm_mean",
        "tempo_bpm_median",
        "tempo_bpm_std",
        "tempo_confidence",
        "tempo_windowed_bpm_mean",
        "tempo_windowed_bpm_median",
        "tempo_windowed_bpm_std",
        "duration_sec",
        "segments_count",
    ]
    rows = "".join([f"<tr><td>{k}</td><td>{feat.get(k)}</td></tr>" for k in keys_show if k in feat])

    plot = _svg_plot(w_times, w_bpm)
    warn_html = ", ".join(warnings_list) if warnings_list else "(none)"

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>tempo_extractor quality demo</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; background: #0b1220; color: #e5e7eb; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border: 1px solid #334155; padding: 6px; vertical-align: top; }}
    th {{ background: #0f172a; }}
    .card {{ background: #0f172a; padding: 12px; border: 1px solid #334155; border-radius: 10px; margin: 12px 0; }}
    code {{ background: #111827; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h2>tempo_extractor — quality demo</h2>
  <div class="card">
    <div><b>NPZ</b>: <code>{npz_path}</code></div>
    <div><b>validate_npz</b>: {issues_html}</div>
    <div><b>sanity</b>:<br/>{"<br/>".join(checks)}</div>
    <div><b>warnings</b>: {warn_html}</div>
  </div>
  <div class="card">
    <h3>Key metrics</h3>
    <table><tr><th>key</th><th>value</th></tr>
      {rows if rows else "<tr><td colspan='2'>no metrics</td></tr>"}
    </table>
  </div>
  <div class="card">
    <h3>Windowed BPM</h3>
    {plot}
    <div style="margin-top:8px;color:#94a3b8;font-size:12px">points: {int(w_times.size)}</div>
  </div>
</body>
</html>
"""

    ts_tag = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    out_path = os.path.join(out_dir, f"demo_tempo_extractor_quality_{ts_tag}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Wrote HTML: %s", out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


