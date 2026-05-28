"""
Renderer для loudness_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        xf = float(x)
        if np.isnan(xf):
            return int(default)
        return int(xf)
    except Exception:
        return int(default)


def render_loudness_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для loudness_extractor."""
    render = {
        "component": "loudness_extractor",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract scalar features
    feature_names = npz_data.get("feature_names", [])
    feature_values = npz_data.get("feature_values", [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    if isinstance(feature_values, np.ndarray):
        feature_values = feature_values.tolist()
    
    features = {}
    for i, name in enumerate(feature_names):
        if i < len(feature_values):
            features[name] = feature_values[i]
    
    # Summary
    lufs_present_flag = bool(npz_data.get("lufs_present", False))
    lufs_value = features.get("loudness_lufs") or features.get("lufs")
    # Проверяем, что lufs не NaN
    if lufs_value is not None:
        try:
            lufs_float = float(lufs_value)
            if np.isnan(lufs_float):
                lufs_value = None
                lufs_present_flag = False
        except (ValueError, TypeError):
            lufs_value = None
            lufs_present_flag = False
    
    render["summary"] = {
        "rms": features.get("loudness_rms") or features.get("rms", 0.0),
        "peak": features.get("loudness_peak") or features.get("peak", 0.0),
        "dbfs": features.get("loudness_dbfs") or features.get("dbfs", 0.0),
        "rms_mean": features.get("segment_rms_mean", 0.0),
        "rms_std": features.get("segment_rms_std", 0.0),
        "peak_mean": features.get("peak_mean", 0.0),
        "dbfs_mean": features.get("dbfs_mean", 0.0),
        "lufs_present": lufs_present_flag,
        "lufs": float(lufs_value) if lufs_value is not None and not np.isnan(float(lufs_value)) else None,
        "segments_count": _safe_int(features.get("segments_count", 0), default=0),
    }
    
    # Timeline data (Audit v3: prefer segment_center_sec + segment_mask; fallback legacy)
    def _to_list(v: Any) -> list:
        if v is None:
            return []
        if isinstance(v, np.ndarray):
            return v.tolist() if v.size > 0 else []
        if isinstance(v, list):
            return v
        return []

    centers = _to_list(npz_data.get("segment_center_sec") or npz_data.get("segment_centers_sec"))
    mask = _to_list(npz_data.get("segment_mask"))
    seg_rms = _to_list(npz_data.get("segment_rms"))
    seg_dbfs = _to_list(npz_data.get("segment_dbfs"))
    seg_lufs = _to_list(npz_data.get("segment_lufs"))

    if centers:
        if not mask:
            mask = [True] * len(centers)

        timeline = []
        n = min(len(centers), len(mask))
        for i in range(n):
            entry: Dict[str, Any] = {
                "center_sec": float(centers[i]),
                "segment_index": i,
                "segment_mask": bool(mask[i]),
            }

            if i < len(seg_rms):
                try:
                    rv = float(seg_rms[i])
                    entry["rms"] = None if np.isnan(rv) or np.isinf(rv) else rv
                except Exception:
                    entry["rms"] = None

            if i < len(seg_dbfs):
                try:
                    dv = float(seg_dbfs[i])
                    entry["dbfs"] = None if np.isnan(dv) or np.isinf(dv) else dv
                except Exception:
                    entry["dbfs"] = None

            if i < len(seg_lufs) and render["summary"]["lufs_present"]:
                try:
                    lv = float(seg_lufs[i])
                    entry["lufs"] = None if np.isnan(lv) or np.isinf(lv) else lv
                except Exception:
                    entry["lufs"] = None

            timeline.append(entry)

        render["timeline"] = timeline

        # Distributions (valid segments only, finite values only)
        if timeline:
            rms_values = [t.get("rms") for t in timeline if t.get("segment_mask") and isinstance(t.get("rms"), (int, float)) and np.isfinite(float(t.get("rms")))]
            dbfs_values = [t.get("dbfs") for t in timeline if t.get("segment_mask") and isinstance(t.get("dbfs"), (int, float)) and np.isfinite(float(t.get("dbfs")))]
            lufs_values = [t.get("lufs") for t in timeline if t.get("segment_mask") and isinstance(t.get("lufs"), (int, float)) and np.isfinite(float(t.get("lufs")))]

            if rms_values:
                render["distributions"]["rms"] = {
                    "min": float(np.min(rms_values)),
                    "max": float(np.max(rms_values)),
                    "mean": float(np.mean(rms_values)),
                    "std": float(np.std(rms_values)),
                }

            if dbfs_values:
                render["distributions"]["dbfs"] = {
                    "min": float(np.min(dbfs_values)),
                    "max": float(np.max(dbfs_values)),
                    "mean": float(np.mean(dbfs_values)),
                    "std": float(np.std(dbfs_values)),
                }

            if lufs_values:
                render["distributions"]["lufs"] = {
                    "min": float(np.min(lufs_values)),
                    "max": float(np.max(lufs_values)),
                    "mean": float(np.mean(lufs_values)),
                    "std": float(np.std(lufs_values)),
                }
    
    return render


def render_loudness_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага loudness_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    ap_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(ap_root / "src") not in sys.path:
        sys.path.insert(0, str(ap_root / "src"))
    from ....core.renderer import load_npz, extract_meta
    
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_loudness_extractor(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    
    # Offline-only HTML (Audit v3): no CDN/Chart.js.
    lufs_present = bool(summary.get("lufs_present", False))
    lufs_value = summary.get("lufs")

    centers = [float(t.get("center_sec", 0.0)) for t in timeline if t.get("segment_mask")]
    rms = [t.get("rms") for t in timeline if t.get("segment_mask")]
    dbfs = [t.get("dbfs") for t in timeline if t.get("segment_mask")]
    lufs = [t.get("lufs") for t in timeline if t.get("segment_mask")]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Loudness Extractor Debug Render</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; background: #f4f4f4; color: #111; }}
    .container {{ background: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); max-width: 1200px; margin: 0 auto; }}
    h1, h2 {{ color: #0b3d91; }}
    .meta-info {{ background: #f7f7f7; padding: 10px; border-radius: 8px; margin: 12px 0; font-size: 0.95em; color: #444; }}
    .summary {{ background: #eaf4ff; padding: 15px; border-radius: 8px; margin: 16px 0; border: 1px solid #cce0ff; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 12px 0; }}
    .metric-card {{ background: #fff; padding: 12px; border-radius: 8px; border: 1px solid #e6e6e6; }}
    .metric-card strong {{ color: #0b3d91; display: block; margin-bottom: 6px; }}
    .metric-value {{ font-variant-numeric: tabular-nums; font-size: 1.15em; font-weight: 700; }}
    .lufs-info {{ background: {('#d4edda' if lufs_present else '#fff3cd')}; padding: 10px; border-radius: 8px; margin: 12px 0; border: 1px solid {('#c3e6cb' if lufs_present else '#ffc107')}; }}
    .chart {{ margin: 18px 0; }}
    canvas {{ width: 100%; max-width: 1100px; height: 320px; border: 1px solid #eee; border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #eee; }}
    th {{ background: #0b3d91; color: #fff; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Loudness Extractor Debug Render</h1>
    <div class="meta-info">
      <div><strong>Status:</strong> {meta.get('status', 'unknown')}</div>
      <div><strong>Producer version:</strong> {meta.get('producer_version', 'unknown')}</div>
      <div><strong>Schema version:</strong> {meta.get('schema_version', 'unknown')}</div>
    </div>

    <div class="summary">
      <h2>Summary</h2>
      <div class="metric-grid">
        <div class="metric-card"><strong>RMS</strong><div class="metric-value">{float(summary.get('rms', 0.0) or 0.0):.6f}</div></div>
        <div class="metric-card"><strong>Peak</strong><div class="metric-value">{float(summary.get('peak', 0.0) or 0.0):.6f}</div></div>
        <div class="metric-card"><strong>dBFS</strong><div class="metric-value">{float(summary.get('dbfs', 0.0) or 0.0):.2f} dB</div></div>
        <div class="metric-card"><strong>Segments</strong><div class="metric-value">{int(summary.get('segments_count', 0) or 0)}</div></div>
        <div class="metric-card"><strong>LUFS present</strong><div class="metric-value">{'Yes' if lufs_present else 'No'}</div></div>
        <div class="metric-card"><strong>LUFS</strong><div class="metric-value">{(f'{float(lufs_value):.2f} LUFS' if (lufs_present and lufs_value is not None) else 'N/A')}</div></div>
      </div>
    </div>

    <div class="lufs-info">
      <strong>LUFS Status:</strong> {'LUFS computation available' if lufs_present else 'LUFS computation not available (pyloudnorm missing/failed)'}
    </div>

    <div class="chart">
      <h2>Timeline: RMS</h2>
      <canvas id="rmsCanvas" width="1100" height="320"></canvas>
    </div>
    <div class="chart">
      <h2>Timeline: dBFS</h2>
      <canvas id="dbfsCanvas" width="1100" height="320"></canvas>
    </div>
    <div class="chart">
      <h2>Timeline: LUFS</h2>
      <canvas id="lufsCanvas" width="1100" height="320"></canvas>
    </div>

    <div class="chart">
      <h2>Distributions (valid segments)</h2>
      <table>
        <thead><tr><th>Metric</th><th>Min</th><th>Max</th><th>Mean</th><th>Std</th></tr></thead>
        <tbody>
          <tr>
            <td><strong>RMS</strong></td>
            <td>{float(distributions.get('rms', {{}}).get('min', float('nan'))):.6f}</td>
            <td>{float(distributions.get('rms', {{}}).get('max', float('nan'))):.6f}</td>
            <td>{float(distributions.get('rms', {{}}).get('mean', float('nan'))):.6f}</td>
            <td>{float(distributions.get('rms', {{}}).get('std', float('nan'))):.6f}</td>
          </tr>
          <tr>
            <td><strong>dBFS</strong></td>
            <td>{float(distributions.get('dbfs', {{}}).get('min', float('nan'))):.2f}</td>
            <td>{float(distributions.get('dbfs', {{}}).get('max', float('nan'))):.2f}</td>
            <td>{float(distributions.get('dbfs', {{}}).get('mean', float('nan'))):.2f}</td>
            <td>{float(distributions.get('dbfs', {{}}).get('std', float('nan'))):.2f}</td>
          </tr>
          <tr>
            <td><strong>LUFS</strong></td>
            <td>{float(distributions.get('lufs', {{}}).get('min', float('nan'))):.2f}</td>
            <td>{float(distributions.get('lufs', {{}}).get('max', float('nan'))):.2f}</td>
            <td>{float(distributions.get('lufs', {{}}).get('mean', float('nan'))):.2f}</td>
            <td>{float(distributions.get('lufs', {{}}).get('std', float('nan'))):.2f}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <script>
      const centers = {json.dumps(centers)};
      const rms = {json.dumps(rms)};
      const dbfs = {json.dumps(dbfs)};
      const lufs = {json.dumps(lufs)};

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
        ctx.fillText('t, sec', w-pad-40, h-10);

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

      drawLine('rmsCanvas', centers, rms, '#22a6b3', 'RMS');
      drawLine('dbfsCanvas', centers, dbfs, '#eb4d4b', 'dBFS');
      drawLine('lufsCanvas', centers, lufs, '#6c5ce7', 'LUFS');
    </script>
  </div>
</body>
</html>"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    os.replace(tmp_path, output_path)
    
    logger.info(f"Saved Loudness HTML render to {output_path}")
    return output_path


__all__ = ["render_loudness_extractor", "render_loudness_extractor_html"]

