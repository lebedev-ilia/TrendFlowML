"""
Renderer для quality_extractor: генерация render-context JSON и HTML debug страницы.
Audit v3: feature_names/feature_values, arrays from NPZ, vanilla canvas (no CDN).
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)

from ....core.renderer import load_npz, extract_meta


def _to_list(v: Any) -> list:
    """Convert value to list for JSON/timeline."""
    if v is None:
        return []
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size > 0 else []
    if isinstance(v, list):
        return v
    return []


def render_quality_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для quality_extractor (Audit v3: feature_names/feature_values)."""
    render = {
        "component": "quality_extractor",
        "summary": {},
        "basic_metrics": {},
        "dynamic_metrics": {},
        "frame_analysis": {},
        "timeline": [],
        "time_series_paths": {},
    }

    # Build features from feature_names/feature_values
    feature_names = npz_data.get("feature_names", [])
    feature_values = npz_data.get("feature_values", [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    if isinstance(feature_values, np.ndarray):
        feature_values = feature_values.tolist()

    features: Dict[str, Any] = {}
    for i, name in enumerate(feature_names):
        if i < len(feature_values):
            try:
                v = feature_values[i]
                features[str(name)] = float(v) if np.isfinite(float(v)) else None
            except (ValueError, TypeError):
                features[str(name)] = None

    # Summary
    render["summary"] = {
        "sample_rate": int(features.get("sample_rate", 0) or 0),
        "duration": float(features.get("duration", 0.0) or 0.0),
        "segments_count": int(features.get("segments_count", 0) or 0),
        "device_used": meta.get("device_used", "cpu"),
    }

    # Basic metrics (feature-gated)
    if "dc_offset" in features:
        render["basic_metrics"] = {
            "dc_offset": features.get("dc_offset"),
            "clipping_ratio": features.get("clipping_ratio"),
            "crest_factor_db": features.get("crest_factor_db"),
        }

    # Dynamic metrics (feature-gated)
    if "dynamic_range_db" in features:
        render["dynamic_metrics"] = {
            "dynamic_range_db": features.get("dynamic_range_db"),
            "dynamic_range_stability": features.get("dynamic_range_stability"),
        }

    # Frame analysis (feature-gated)
    if "frame_levels_mean" in features:
        render["frame_analysis"] = {
            "frame_levels_mean": features.get("frame_levels_mean"),
            "frame_levels_std": features.get("frame_levels_std"),
            "frame_levels_min": features.get("frame_levels_min"),
            "frame_levels_max": features.get("frame_levels_max"),
            "frame_levels_median": features.get("frame_levels_median"),
        }

    # Timeline (canonical segment axis)
    centers = _to_list(npz_data.get("segment_center_sec"))
    mask = _to_list(npz_data.get("segment_mask"))
    if centers:
        if not mask:
            mask = [True] * len(centers)
        timeline = []
        for i in range(min(len(centers), len(mask))):
            timeline.append({
                "center_sec": float(centers[i]),
                "segment_index": i,
                "segment_mask": bool(mask[i]),
            })
        render["timeline"] = timeline

    # Time series paths (meta: .npy paths from extra)
    for key in ["dc_offset_series_npy", "clipping_ratio_series_npy", "crest_factor_db_series_npy",
                "dynamic_range_db_series_npy", "frame_levels_db_series_npy", "frame_rms_series_npy",
                "clipping_segments_series_npy"]:
        path = meta.get(key) if isinstance(meta, dict) else None
        if path:
            render["time_series_paths"][key] = path

    return render


def safe_float(value: Any, default: float = 0.0) -> float:
    """Безопасное преобразование в float для HTML."""
    if value is None:
        return default
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (ValueError, TypeError):
        return default


def render_quality_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага quality_extractor (Audit v3: vanilla canvas, no CDN).
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_quality_extractor(npz_data, meta)

    summary = render.get("summary", {})
    basic_metrics = render.get("basic_metrics", {})
    dynamic_metrics = render.get("dynamic_metrics", {})
    frame_analysis = render.get("frame_analysis", {})
    timeline = render.get("timeline", [])
    time_series_paths = render.get("time_series_paths", {})

    # Load time series from .npy if paths exist (relative to NPZ dir)
    npz_dir = str(Path(npz_path).parent)
    series_data: Dict[str, List[float]] = {}
    for key, path in time_series_paths.items():
        try:
            full_path = Path(path)
            if not full_path.is_absolute():
                full_path = Path(npz_dir) / path
            if full_path.exists():
                arr = np.load(str(full_path))
                series_data[key.replace("_npy", "")] = arr.tolist() if arr.size > 0 else []
        except Exception:
            pass

    # Timeline data for charts
    centers = [float(t.get("center_sec", 0.0)) for t in timeline if t.get("segment_mask")]
    dc_series = series_data.get("dc_offset_series", [])
    clip_series = series_data.get("clipping_ratio_series", [])
    crest_series = series_data.get("crest_factor_db_series", [])
    dr_series = series_data.get("dynamic_range_db_series", [])

    # Use segment count for per-segment series; frame series may be longer
    def _align(xs: list, n: int) -> list:
        if len(xs) <= n:
            return xs
        return xs[:n]

    n_centers = len(centers)
    dc_series = _align(dc_series, n_centers) if n_centers else []
    clip_series = _align(clip_series, n_centers) if n_centers else []
    crest_series = _align(crest_series, n_centers) if n_centers else []
    dr_series = _align(dr_series, n_centers) if n_centers else []

    # Basic metrics HTML
    basic_html = ""
    if basic_metrics:
        basic_html = f"""
    <div class="metrics">
        <h2>Basic Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td><strong>DC Offset</strong></td><td>{safe_float(basic_metrics.get('dc_offset'), 0.0):.6f}</td></tr>
            <tr><td><strong>Clipping Ratio</strong></td><td>{safe_float(basic_metrics.get('clipping_ratio'), 0.0):.4f}</td></tr>
            <tr><td><strong>Crest Factor (dB)</strong></td><td>{safe_float(basic_metrics.get('crest_factor_db'), 0.0):.2f} dB</td></tr>
        </table>
    </div>
"""

    # Dynamic metrics HTML
    dynamic_html = ""
    if dynamic_metrics:
        dynamic_html = f"""
    <div class="metrics">
        <h2>Dynamic Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td><strong>Dynamic Range (dB)</strong></td><td>{safe_float(dynamic_metrics.get('dynamic_range_db'), 0.0):.2f} dB</td></tr>
            <tr><td><strong>Dynamic Range Stability</strong></td><td>{safe_float(dynamic_metrics.get('dynamic_range_stability'), 0.0):.4f}</td></tr>
        </table>
    </div>
"""

    # Frame analysis HTML
    frame_html = ""
    if frame_analysis:
        frame_html = f"""
    <div class="metrics">
        <h2>Frame Analysis</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td><strong>Frame Levels Mean</strong></td><td>{safe_float(frame_analysis.get('frame_levels_mean'), 0.0):.2f}</td></tr>
            <tr><td><strong>Frame Levels Std</strong></td><td>{safe_float(frame_analysis.get('frame_levels_std'), 0.0):.2f}</td></tr>
            <tr><td><strong>Frame Levels Min</strong></td><td>{safe_float(frame_analysis.get('frame_levels_min'), 0.0):.2f}</td></tr>
            <tr><td><strong>Frame Levels Max</strong></td><td>{safe_float(frame_analysis.get('frame_levels_max'), 0.0):.2f}</td></tr>
            <tr><td><strong>Frame Levels Median</strong></td><td>{safe_float(frame_analysis.get('frame_levels_median'), 0.0):.2f}</td></tr>
        </table>
    </div>
"""

    raw_json = json.dumps(render, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Quality Extractor Debug</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, Arial, sans-serif; margin: 20px; background: #f4f4f4; color: #111; }}
    .container {{ background: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); max-width: 1200px; margin: 0 auto; }}
    h1, h2 {{ color: #0b3d91; }}
    .summary {{ background: #eaf4ff; padding: 15px; border-radius: 8px; margin: 16px 0; border: 1px solid #cce0ff; }}
    .metrics {{ margin: 10px 0; }}
    .metrics table {{ border-collapse: collapse; width: 100%; }}
    .metrics th, .metrics td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    .metrics th {{ background-color: #4CAF50; color: white; }}
    .chart {{ margin: 18px 0; }}
    canvas {{ width: 100%; max-width: 1100px; height: 280px; border: 1px solid #eee; border-radius: 8px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Quality Extractor Debug</h1>
    <div class="summary">
      <h2>Summary</h2>
      <p><strong>Sample Rate:</strong> {int(summary.get('sample_rate', 0) or 0)} Hz</p>
      <p><strong>Duration:</strong> {safe_float(summary.get('duration', 0.0), 0.0):.2f} sec</p>
      <p><strong>Segments:</strong> {int(summary.get('segments_count', 0) or 0)}</p>
    </div>
    {basic_html}
    {dynamic_html}
    {frame_html}

    <div class="chart"><h2>DC Offset (segments)</h2><canvas id="dcCanvas" width="1100" height="280"></canvas></div>
    <div class="chart"><h2>Clipping Ratio (segments)</h2><canvas id="clipCanvas" width="1100" height="280"></canvas></div>
    <div class="chart"><h2>Crest Factor dB (segments)</h2><canvas id="crestCanvas" width="1100" height="280"></canvas></div>
    <div class="chart"><h2>Dynamic Range dB (segments)</h2><canvas id="drCanvas" width="1100" height="280"></canvas></div>

    <div class="summary"><h2>Raw Data (JSON)</h2><pre>{raw_json}</pre></div>
  </div>
  <script>
    const centers = {json.dumps(centers)};
    const dcSeries = {json.dumps(dc_series)};
    const clipSeries = {json.dumps(clip_series)};
    const crestSeries = {json.dumps(crest_series)};
    const drSeries = {json.dumps(dr_series)};

    function isFiniteNum(x) {{
      const v = Number(x);
      return Number.isFinite(v);
    }}

    function drawLine(canvasId, xs, ys, color, yLabel) {{
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;
      const pad = 44;
      ctx.clearRect(0,0,w,h);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0,0,w,h);
      ctx.strokeStyle = '#ddd';
      ctx.strokeRect(0.5,0.5,w-1,h-1);

      const pts = [];
      for (let i=0;i<Math.min(xs.length, ys.length);i++) {{
        if (isFiniteNum(xs[i]) && isFiniteNum(ys[i])) pts.push([Number(xs[i]), Number(ys[i])]);
      }}
      if (pts.length < 2) {{
        ctx.fillStyle = '#666';
        ctx.font = '14px sans-serif';
        ctx.fillText('no data', 12, 24);
        return;
      }}

      let xMin = pts[0][0], xMax = pts[0][0], yMin = pts[0][1], yMax = pts[0][1];
      for (const [x,y] of pts) {{
        if (x < xMin) xMin = x;
        if (x > xMax) xMax = x;
        if (y < yMin) yMin = y;
        if (y > yMax) yMax = y;
      }}
      if (xMax <= xMin) xMax = xMin + 1;
      if (yMax <= yMin) yMax = yMin + 1e-6;

      const xScale = (w - 2*pad) / (xMax - xMin);
      const yScale = (h - 2*pad) / (yMax - yMin);

      ctx.strokeStyle = '#bbb';
      ctx.beginPath();
      ctx.moveTo(pad, pad);
      ctx.lineTo(pad, h-pad);
      ctx.lineTo(w-pad, h-pad);
      ctx.stroke();

      ctx.fillStyle = '#666';
      ctx.font = '12px sans-serif';
      ctx.fillText(yLabel, 8, 18);

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i=0;i<pts.length;i++) {{
        const x = pad + (pts[i][0] - xMin) * xScale;
        const y = h - pad - (pts[i][1] - yMin) * yScale;
        if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }}
      ctx.stroke();

      ctx.fillStyle = color;
      for (let i=0;i<pts.length;i++) {{
        const x = pad + (pts[i][0] - xMin) * xScale;
        const y = h - pad - (pts[i][1] - yMin) * yScale;
        ctx.beginPath();
        ctx.arc(x,y,2.5,0,2*Math.PI);
        ctx.fill();
      }}
    }}

    drawLine('dcCanvas', centers, dcSeries, '#22a6b3', 'DC Offset');
    drawLine('clipCanvas', centers, clipSeries, '#eb4d4b', 'Clipping Ratio');
    drawLine('crestCanvas', centers, crestSeries, '#6c5ce7', 'Crest Factor dB');
    drawLine('drCanvas', centers, drSeries, '#00b894', 'Dynamic Range dB');
  </script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    os.replace(tmp_path, output_path)
    logger.info(f"Saved Quality HTML render to {output_path}")
    return output_path


__all__ = ["render_quality_extractor", "render_quality_extractor_html"]
