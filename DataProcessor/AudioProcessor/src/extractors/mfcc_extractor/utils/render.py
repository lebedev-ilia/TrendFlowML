"""
Renderer для mfcc_extractor: генерация render-context JSON и HTML debug страницы.
Audit v3: offline-only (vanilla canvas/SVG, без CDN).
"""

import json
import logging
import os
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ....core.renderer import load_npz, extract_meta


def render_mfcc_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для mfcc_extractor."""
    render = {
        "component": "mfcc_extractor",
        "summary": {},
        "basic_features": {},
        "deltas": {},
        "time_series": {},
        "additional_metrics": {},
    }

    feature_names = npz_data.get("feature_names", [])
    feature_values = npz_data.get("feature_values", [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    if isinstance(feature_values, np.ndarray):
        feature_values = feature_values.tolist()
    features = {}
    if isinstance(feature_names, list) and isinstance(feature_values, list):
        for i, name in enumerate(feature_names):
            if i < len(feature_values):
                features[str(name)] = feature_values[i]

    def get_feature(key: str, default: Any = None) -> Any:
        return features.get(key, default)

    def safe_get(key, default=None):
        value = npz_data.get(key, default)
        if value is None:
            return default
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return value.item()
            elif value.dtype == object and value.size == 1:
                return value.item()
        if isinstance(value, list):
            if len(value) == 0:
                return default
            if len(value) == 1 and not isinstance(value[0], (list, np.ndarray)):
                return value[0]
        return value

    def is_non_empty(value):
        if value is None:
            return False
        if isinstance(value, list):
            return len(value) > 0
        if isinstance(value, np.ndarray):
            return value.size > 0
        return value != 0 and value != 0.0

    _device = get_feature("device_used", None)
    if _device is not None and isinstance(_device, (float, np.floating)) and np.isnan(_device):
        _device = None
    if _device is None:
        _device = meta.get("device_used", "cpu")

    render["summary"] = {
        "sample_rate": int(get_feature("sample_rate", 22050) or 22050),
        "n_mfcc": int(get_feature("n_mfcc", 13) or 13),
        "n_fft": int(get_feature("n_fft", 2048) or 2048),
        "hop_length": int(get_feature("hop_length", 512) or 512),
        "n_mels": int(get_feature("n_mels", 128) or 128),
        "fmin": float(get_feature("fmin", 0.0) or 0.0),
        "fmax": get_feature("fmax"),
        "device_used": str(_device),
        "duration": float(get_feature("duration_sec", 0.0) or 0.0),
        "segments_count": int(float(get_feature("segments_count", 0.0) or 0.0)),
    }

    mfcc_mean = safe_get("mfcc_mean")
    if is_non_empty(mfcc_mean):
        render["basic_features"] = {
            "mfcc_mean": mfcc_mean if isinstance(mfcc_mean, list) else (mfcc_mean.tolist() if isinstance(mfcc_mean, np.ndarray) else mfcc_mean),
            "mfcc_std": safe_get("mfcc_std"),
            "mfcc_min": safe_get("mfcc_min"),
            "mfcc_max": safe_get("mfcc_max"),
        }

    delta_mean = safe_get("delta_mean")
    if is_non_empty(delta_mean):
        render["deltas"] = {
            "delta_mean": delta_mean if isinstance(delta_mean, list) else (delta_mean.tolist() if isinstance(delta_mean, np.ndarray) else delta_mean),
            "delta_std": safe_get("delta_std"),
            "delta_delta_mean": safe_get("delta_delta_mean"),
            "delta_delta_std": safe_get("delta_delta_std"),
        }

    render["additional_metrics"] = {
        "mfcc_energy": float(get_feature("mfcc_energy", 0.0) or 0.0),
        "mfcc_centroid": float(get_feature("mfcc_centroid", 0.0) or 0.0),
        "mfcc_bandwidth": float(get_feature("mfcc_bandwidth", 0.0) or 0.0),
        "mfcc_stability": float(get_feature("mfcc_stability", 0.0) or 0.0),
    }

    segment_keys = ["segment_start_sec", "segment_end_sec", "segment_center_sec", "segment_mask"]
    seq_keys = ["mfcc_mean_by_segment", "mfcc_energy_by_segment", "delta_mean_by_segment"]
    render["time_series"] = {}
    for key in segment_keys + seq_keys:
        series = safe_get(key)
        if is_non_empty(series):
            if isinstance(series, np.ndarray):
                render["time_series"][key] = series.tolist()
            else:
                render["time_series"][key] = series

    return render


def render_mfcc_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага mfcc_extractor (Audit v3: offline-only, vanilla canvas).
    """
    npz_raw = np.load(npz_path, allow_pickle=True)
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_mfcc_extractor(npz_data, meta)

    summary = render.get("summary", {})
    basic_features = render.get("basic_features", {})
    additional_metrics = render.get("additional_metrics", {})
    time_series = render.get("time_series", {})

    def safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    segment_center_sec = time_series.get("segment_center_sec", [])
    segment_mask = time_series.get("segment_mask", [])
    mfcc_mean_stats = basic_features.get("mfcc_mean", [])
    mfcc_mean_by_segment = time_series.get("mfcc_mean_by_segment", [])
    mfcc_energy_by_segment = time_series.get("mfcc_energy_by_segment", [])
    delta_mean_by_segment = time_series.get("delta_mean_by_segment", [])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MFCC Extractor Debug</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; background: #f5f6fa; color: #111; }}
    .container {{ max-width: 1200px; margin: 0 auto; background: #fff; border-radius: 12px; padding: 20px; box-shadow: 0 2px 14px rgba(0,0,0,0.06); }}
    h1 {{ margin: 0 0 10px 0; }}
    h2 {{ margin: 18px 0 10px 0; }}
    .meta {{ color: #555; font-size: 0.95em; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 14px 0; }}
    .card {{ border: 1px solid #eee; border-radius: 10px; padding: 12px; background: #fff; }}
    .label {{ color: #666; font-size: 0.85em; margin-bottom: 4px; }}
    .value {{ font-variant-numeric: tabular-nums; font-size: 1.15em; font-weight: 700; }}
    canvas {{ width: 100%; border: 1px solid #eee; border-radius: 10px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>MFCC Extractor Debug</h1>
    <div class="meta">
      <div><b>Status:</b> {meta.get("status", "unknown")} {(" | " + str(meta.get("empty_reason"))) if meta.get("empty_reason") else ""}</div>
      <div><b>Schema:</b> {meta.get("schema_version", "unknown")} | <b>Producer:</b> {meta.get("producer_version", "unknown")}</div>
      <div><b>Features enabled:</b> {", ".join(meta.get("features_enabled", []) or [])}</div>
    </div>

    <div class="grid">
      <div class="card"><div class="label">sample_rate</div><div class="value">{summary.get("sample_rate")}</div></div>
      <div class="card"><div class="label">n_mfcc</div><div class="value">{summary.get("n_mfcc")}</div></div>
      <div class="card"><div class="label">duration_sec</div><div class="value">{safe_float(summary.get("duration", 0.0)):.2f}</div></div>
      <div class="card"><div class="label">segments_count</div><div class="value">{summary.get("segments_count")}</div></div>
    </div>

    <div class="grid">
      <div class="card"><div class="label">mfcc_energy</div><div class="value">{safe_float(additional_metrics.get("mfcc_energy", 0.0)):.4f}</div></div>
      <div class="card"><div class="label">mfcc_centroid</div><div class="value">{safe_float(additional_metrics.get("mfcc_centroid", 0.0)):.4f}</div></div>
      <div class="card"><div class="label">mfcc_bandwidth</div><div class="value">{safe_float(additional_metrics.get("mfcc_bandwidth", 0.0)):.4f}</div></div>
      <div class="card"><div class="label">mfcc_stability</div><div class="value">{safe_float(additional_metrics.get("mfcc_stability", 0.0)):.4f}</div></div>
    </div>

    <h2>mfcc_mean (basic_features)</h2>
    <canvas id="mfccMeanCanvas" width="1100" height="280"></canvas>

    <h2>Segment-aligned heatmap (mfcc_mean_by_segment)</h2>
    <canvas id="mfccHeatmapCanvas" width="1100" height="360"></canvas>

    <h2>Segment timelines</h2>
    <canvas id="energyCanvas" width="1100" height="260"></canvas>

    <script>
      const centers = {json.dumps(segment_center_sec)};
      const mask = {json.dumps(segment_mask)};
      const mfccMean = {json.dumps(mfcc_mean_stats)};
      const mfccMeanBySeg = {json.dumps(mfcc_mean_by_segment)};
      const energy = {json.dumps(mfcc_energy_by_segment)};

      function isFiniteNum(x) {{
        const v = Number(x);
        return Number.isFinite(v);
      }}

      function drawVector(canvasId, vec, color, yLabel) {{
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
        if (!Array.isArray(vec) || vec.length < 2) {{
          ctx.fillStyle = '#666';
          ctx.font = '14px sans-serif';
          ctx.fillText('no data', 12, 24);
          return;
        }}
        const pts = [];
        for (let i=0;i<vec.length;i++) {{
          if (isFiniteNum(vec[i])) pts.push([i, Number(vec[i])]);
        }}
        if (pts.length < 2) return;
        let yMin = pts[0][1], yMax = pts[0][1];
        for (const [,y] of pts) {{
          if (y < yMin) yMin = y;
          if (y > yMax) yMax = y;
        }}
        if (yMax <= yMin) yMax = yMin + 1e-6;
        const xScale = (w - 2*pad) / Math.max(1, (vec.length - 1));
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
          const x = pad + pts[i][0] * xScale;
          const y = h - pad - (pts[i][1] - yMin) * yScale;
          if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        }}
        ctx.stroke();
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
        for (let i=0;i<Math.min(xs.length, ys.length, mask.length);i++) {{
          if (!mask[i]) continue;
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
      }}

      function drawHeatmap(canvasId, mat, mask, maxRows, maxCols) {{
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0,0,w,h);
        if (!Array.isArray(mat) || mat.length === 0) {{
          ctx.fillStyle = '#666';
          ctx.font = '14px sans-serif';
          ctx.fillText('no data', 12, 24);
          return;
        }}
        const N = mat.length;
        const M = Array.isArray(mat[0]) ? mat[0].length : 0;
        if (M === 0) return;
        const rowStep = Math.max(1, Math.floor(N / Math.max(1, maxRows)));
        const colStep = Math.max(1, Math.floor(M / Math.max(1, maxCols)));
        let maxV = 0.0;
        for (let i=0;i<N;i+=rowStep) {{
          if (mask && mask.length > i && !mask[i]) continue;
          for (let j=0;j<M;j+=colStep) {{
            const v = Number(mat[i][j]);
            if (Number.isFinite(v)) maxV = Math.max(maxV, Math.abs(v));
          }}
        }}
        if (maxV <= 0) maxV = 1.0;
        const rows = Math.ceil(N / rowStep);
        const cols = Math.ceil(M / colStep);
        const cellW = w / cols;
        const cellH = h / rows;
        for (let ri=0; ri<rows; ri++) {{
          const i = ri * rowStep;
          if (mask && mask.length > i && !mask[i]) continue;
          for (let cj=0; cj<cols; cj++) {{
            const j = cj * colStep;
            const v = Number(mat[i][j]);
            const t = Number.isFinite(v) ? Math.max(0, Math.min(1, (v / maxV + 1) * 0.5)) : 0.0;
            const r = Math.floor(20 + 230 * t);
            const g = Math.floor(20 + 160 * (1 - Math.abs(t - 0.5) * 2));
            const b = Math.floor(30 + 220 * (1 - t));
            ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
            ctx.fillRect(cj * cellW, (rows - 1 - ri) * cellH, Math.ceil(cellW), Math.ceil(cellH));
          }}
        }}
      }}

      drawVector('mfccMeanCanvas', mfccMean, '#0b3d91', 'mfcc_mean');
      drawHeatmap('mfccHeatmapCanvas', mfccMeanBySeg, mask, 220, 128);
      drawLine('energyCanvas', centers, energy, '#22a6b3', 'energy');
    </script>
  </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    os.replace(tmp_path, output_path)
    return output_path


__all__ = ["render_mfcc_extractor", "render_mfcc_extractor_html"]
