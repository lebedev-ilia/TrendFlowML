"""
Renderer для mel_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ....core.renderer import load_npz, extract_meta

def render_mel_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для mel_extractor."""
    render = {
        "component": "mel_extractor",
        "summary": {},
        "basic_features": {},
        "statistics": {},
        "spectral_features": {},
        "time_series": {},
        "additional_metrics": {},
    }

    # Scalars are stored in feature_names/feature_values (tabular path).
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
    
    # Helper function to safely extract values from NPZ
    # Note: load_npz converts numpy arrays to lists, so we need to handle both
    def safe_get(key, default=None):
        value = npz_data.get(key, default)
        if value is None:
            return default
        # Handle numpy arrays (if not converted yet)
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return value.item()
            elif value.dtype == object and value.size == 1:
                return value.item()
        # Handle lists (converted from numpy arrays by load_npz)
        if isinstance(value, list):
            if len(value) == 0:
                return default
            # For scalar values stored as single-element lists
            if len(value) == 1 and not isinstance(value[0], (list, np.ndarray)):
                return value[0]
        return value
    
    # Helper to check if value is non-empty (handles lists and arrays)
    def is_non_empty(value):
        if value is None:
            return False
        if isinstance(value, list):
            return len(value) > 0
        if isinstance(value, np.ndarray):
            return value.size > 0
        return value != 0 and value != 0.0
    
    # Summary (tabular scalars)
    render["summary"] = {
        "sample_rate": int(get_feature("sample_rate", 22050) or 22050),
        "n_fft": int(get_feature("n_fft", 2048) or 2048),
        "hop_length": int(get_feature("hop_length", 512) or 512),
        "n_mels": int(get_feature("n_mels", 128) or 128),
        "fmin": float(get_feature("fmin", 0.0) or 0.0),
        "fmax": get_feature("fmax"),
        "power": float(get_feature("power", 2.0) or 2.0),
        "device_used": str(get_feature("device_used", meta.get("device_used", "cpu"))),
        "duration": float(get_feature("duration_sec", 0.0) or 0.0),
        "segments_count": int(float(get_feature("segments_count", 0.0) or 0.0)),
    }
    
    # Basic features (feature-gated) - read from NPZ
    mel_shape_0 = get_feature("mel_shape_0")
    mel_shape_1 = get_feature("mel_shape_1")
    if mel_shape_0 is not None and mel_shape_1 is not None:
        render["basic_features"] = {
            "mel_shape": (int(mel_shape_0), int(mel_shape_1)),
            "mel_elements": int(get_feature("mel_elements", 0) or 0),
        }
    
    # Statistics (feature-gated) - read from NPZ
    mel_mean = safe_get("mel_mean")
    if is_non_empty(mel_mean):
        render["statistics"] = {}
        if is_non_empty(mel_mean):
            render["statistics"]["mel_mean"] = mel_mean if isinstance(mel_mean, list) else (mel_mean.tolist() if isinstance(mel_mean, np.ndarray) else mel_mean)
        mel_std = safe_get("mel_std")
        if is_non_empty(mel_std):
            render["statistics"]["mel_std"] = mel_std if isinstance(mel_std, list) else (mel_std.tolist() if isinstance(mel_std, np.ndarray) else mel_std)
        mel_min = safe_get("mel_min")
        if is_non_empty(mel_min):
            render["statistics"]["mel_min"] = mel_min if isinstance(mel_min, list) else (mel_min.tolist() if isinstance(mel_min, np.ndarray) else mel_min)
        mel_max = safe_get("mel_max")
        if is_non_empty(mel_max):
            render["statistics"]["mel_max"] = mel_max if isinstance(mel_max, list) else (mel_max.tolist() if isinstance(mel_max, np.ndarray) else mel_max)
        freq_mean = safe_get("freq_mean")
        if is_non_empty(freq_mean):
            render["statistics"]["freq_mean"] = freq_mean if isinstance(freq_mean, list) else (freq_mean.tolist() if isinstance(freq_mean, np.ndarray) else freq_mean)
        freq_std = safe_get("freq_std")
        if is_non_empty(freq_std):
            render["statistics"]["freq_std"] = freq_std if isinstance(freq_std, list) else (freq_std.tolist() if isinstance(freq_std, np.ndarray) else freq_std)
    
    # Spectral features (feature-gated) - read from NPZ
    spectral_centroid = safe_get("spectral_centroid")
    if is_non_empty(spectral_centroid):
        render["spectral_features"] = {}
        render["spectral_features"]["spectral_centroid"] = spectral_centroid if isinstance(spectral_centroid, list) else (spectral_centroid.tolist() if isinstance(spectral_centroid, np.ndarray) else spectral_centroid)
        spectral_bandwidth = safe_get("spectral_bandwidth")
        if is_non_empty(spectral_bandwidth):
            render["spectral_features"]["spectral_bandwidth"] = spectral_bandwidth if isinstance(spectral_bandwidth, list) else (spectral_bandwidth.tolist() if isinstance(spectral_bandwidth, np.ndarray) else spectral_bandwidth)
    
    # Additional ML/analytics metrics - read from NPZ
    mel_energy = get_feature("mel_energy")
    if mel_energy is not None:
        render["additional_metrics"] = {
            "mel_energy": float(get_feature("mel_energy", 0.0) or 0.0),
            "mel_centroid_mean": float(get_feature("mel_centroid_mean", 0.0) or 0.0),
            "mel_centroid_std": float(get_feature("mel_centroid_std", 0.0) or 0.0),
            "mel_bandwidth_mean": float(get_feature("mel_bandwidth_mean", 0.0) or 0.0),
            "mel_bandwidth_std": float(get_feature("mel_bandwidth_std", 0.0) or 0.0),
            "mel_spectrogram_entropy": float(get_feature("mel_spectrogram_entropy", 0.0) or 0.0),
            "mel_spectrogram_contrast": float(get_feature("mel_spectrogram_contrast", 0.0) or 0.0),
            "mel_rolloff": float(get_feature("mel_rolloff", 0.0) or 0.0),
            "mel_flatness": float(get_feature("mel_flatness", 0.0) or 0.0),
            "mel_stability": float(get_feature("mel_stability", 0.0) or 0.0),
        }
    
    # Segment-aligned sequences (Audit v3): keep as lists for HTML debug.
    segment_keys = ["segment_start_sec", "segment_end_sec", "segment_center_sec", "segment_mask"]
    seq_keys = [
        "mel_mean_by_segment",
        "mel_energy_by_segment",
        "mel_centroid_mean_by_segment",
        "mel_bandwidth_mean_by_segment",
    ]
    render["time_series"] = {}
    for key in segment_keys + seq_keys:
        series = safe_get(key)
        if is_non_empty(series):
            if isinstance(series, np.ndarray):
                render["time_series"][key] = series.tolist()
            else:
                render["time_series"][key] = series
    
    return render


def render_mel_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага mel_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    # Load NPZ directly to get raw arrays (not converted to lists)
    import numpy as np
    npz_raw = np.load(npz_path, allow_pickle=True)
    
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_mel_extractor(npz_data, meta)
    
    # Безопасное извлечение данных для форматирования
    summary = render.get("summary", {})
    basic_features = render.get("basic_features", {})
    statistics = render.get("statistics", {})
    spectral_features = render.get("spectral_features", {})
    additional_metrics = render.get("additional_metrics", {})
    time_series = render.get("time_series", {})
    
    # Форматируем значения заранее, чтобы избежать вложенных f-строк
    def safe_float(value, default=0.0):
        """Безопасное преобразование в float."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # Audit v3: offline-only renderer (no Plotly/CDN). We generate a self-contained HTML
    # and return early; legacy Plotly renderer below is kept but unreachable.
    segment_center_sec = time_series.get("segment_center_sec", [])
    segment_mask = time_series.get("segment_mask", [])
    mel_mean_by_segment = time_series.get("mel_mean_by_segment", [])
    mel_energy_by_segment = time_series.get("mel_energy_by_segment", [])
    mel_centroid_mean_by_segment = time_series.get("mel_centroid_mean_by_segment", [])
    mel_bandwidth_mean_by_segment = time_series.get("mel_bandwidth_mean_by_segment", [])

    mel_mean_stats = statistics.get("mel_mean", []) if isinstance(statistics, dict) else []

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Mel Extractor Debug</title>
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
    <h1>Mel Extractor Debug</h1>
    <div class="meta">
      <div><b>Status:</b> {meta.get("status", "unknown")} {(" | " + str(meta.get("empty_reason"))) if meta.get("empty_reason") else ""}</div>
      <div><b>Schema:</b> {meta.get("schema_version", "unknown")} | <b>Producer:</b> {meta.get("producer_version", "unknown")}</div>
      <div><b>Features enabled:</b> {", ".join(meta.get("features_enabled", []) or [])}</div>
    </div>

    <div class="grid">
      <div class="card"><div class="label">sample_rate</div><div class="value">{summary.get("sample_rate")}</div></div>
      <div class="card"><div class="label">n_mels</div><div class="value">{summary.get("n_mels")}</div></div>
      <div class="card"><div class="label">duration_sec</div><div class="value">{safe_float(summary.get("duration", 0.0)):.2f}</div></div>
      <div class="card"><div class="label">segments_count</div><div class="value">{summary.get("segments_count")}</div></div>
    </div>

    <div class="grid">
      <div class="card"><div class="label">mel_energy</div><div class="value">{safe_float(additional_metrics.get("mel_energy", 0.0)):.4f}</div></div>
      <div class="card"><div class="label">mel_centroid_mean/std (Hz)</div><div class="value">{safe_float(additional_metrics.get("mel_centroid_mean", 0.0)):.1f} / {safe_float(additional_metrics.get("mel_centroid_std", 0.0)):.1f}</div></div>
      <div class="card"><div class="label">mel_bandwidth_mean/std (Hz)</div><div class="value">{safe_float(additional_metrics.get("mel_bandwidth_mean", 0.0)):.1f} / {safe_float(additional_metrics.get("mel_bandwidth_std", 0.0)):.1f}</div></div>
      <div class="card"><div class="label">entropy / contrast</div><div class="value">{safe_float(additional_metrics.get("mel_spectrogram_entropy", 0.0)):.4f} / {safe_float(additional_metrics.get("mel_spectrogram_contrast", 0.0)):.4f}</div></div>
      <div class="card"><div class="label">rolloff (Hz)</div><div class="value">{safe_float(additional_metrics.get("mel_rolloff", 0.0)):.1f}</div></div>
      <div class="card"><div class="label">flatness / stability</div><div class="value">{safe_float(additional_metrics.get("mel_flatness", 0.0)):.4f} / {safe_float(additional_metrics.get("mel_stability", 0.0)):.4f}</div></div>
    </div>

    <h2>mel_mean (if statistics enabled)</h2>
    <canvas id="melMeanCanvas" width="1100" height="280"></canvas>

    <h2>Segment-aligned heatmap (mel_mean_by_segment)</h2>
    <canvas id="melHeatmapCanvas" width="1100" height="360"></canvas>

    <h2>Segment timelines</h2>
    <canvas id="energyCanvas" width="1100" height="260"></canvas>
    <div style="height:10px"></div>
    <canvas id="centroidCanvas" width="1100" height="260"></canvas>
    <div style="height:10px"></div>
    <canvas id="bandwidthCanvas" width="1100" height="260"></canvas>

    <script>
      const centers = {json.dumps(segment_center_sec)};
      const mask = {json.dumps(segment_mask)};
      const melMean = {json.dumps(mel_mean_stats)};
      const melMeanBySeg = {json.dumps(mel_mean_by_segment)};
      const energy = {json.dumps(mel_energy_by_segment)};
      const centroid = {json.dumps(mel_centroid_mean_by_segment)};
      const bandwidth = {json.dumps(mel_bandwidth_mean_by_segment)};

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

      drawVector('melMeanCanvas', melMean, '#0b3d91', 'mel_mean (dB)');
      drawHeatmap('melHeatmapCanvas', melMeanBySeg, mask, 220, 128);
      drawLine('energyCanvas', centers, energy, '#22a6b3', 'energy');
      drawLine('centroidCanvas', centers, centroid, '#6c5ce7', 'centroid_mean (Hz)');
      drawLine('bandwidthCanvas', centers, bandwidth, '#eb4d4b', 'bandwidth_mean (Hz)');
    </script>
  </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    os.replace(tmp_path, output_path)
    return output_path
    
    # Extract time series data for visualization
    mel_series_data = None
    segment_centers = time_series.get("segment_centers_sec", [])
    
    # Try to get data from render context or load directly from NPZ
    if "mel_series" in time_series and "data" in time_series["mel_series"]:
        mel_series_data = time_series["mel_series"]["data"]
    elif "mel_series" in npz_raw.files:
        arr = npz_raw["mel_series"]
        if arr.size > 0:
            # Flatten if 2D and sample if too large
            if arr.ndim == 2:
                arr = arr.flatten()
            if arr.size > 10000:
                step = arr.size // 10000
                arr = arr[::step]
            mel_series_data = arr.tolist()[:10000]
    
    # Statistics for distribution charts
    mel_mean_stats = statistics.get("mel_mean", [])
    mel_std_stats = statistics.get("mel_std", [])
    spectral_centroid_data = spectral_features.get("spectral_centroid", [])
    spectral_bandwidth_data = spectral_features.get("spectral_bandwidth", [])
    
    # Ensure lists are not empty and have valid data
    if isinstance(mel_mean_stats, np.ndarray):
        mel_mean_stats = mel_mean_stats.tolist() if mel_mean_stats.size > 0 else []
    if isinstance(mel_std_stats, np.ndarray):
        mel_std_stats = mel_std_stats.tolist() if mel_std_stats.size > 0 else []
    if isinstance(spectral_centroid_data, np.ndarray):
        spectral_centroid_data = spectral_centroid_data.tolist() if spectral_centroid_data.size > 0 else []
    if isinstance(spectral_bandwidth_data, np.ndarray):
        spectral_bandwidth_data = spectral_bandwidth_data.tolist() if spectral_bandwidth_data.size > 0 else []
    
    # Ensure we have lists, not other types
    if not isinstance(mel_mean_stats, list):
        mel_mean_stats = []
    if not isinstance(mel_std_stats, list):
        mel_std_stats = []
    if not isinstance(spectral_centroid_data, list):
        spectral_centroid_data = []
    if not isinstance(spectral_bandwidth_data, list):
        spectral_bandwidth_data = []
    
    # Build HTML with Plotly charts
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>🎵 Mel Extractor Debug</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 12px; 
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 30px;
        }}
        h1 {{ 
            font-size: 2.5em; 
            margin-bottom: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        h2 {{ 
            font-size: 1.8em; 
            margin: 30px 0 15px 0; 
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        h3 {{
            font-size: 1.3em;
            margin: 20px 0 10px 0;
            color: #764ba2;
        }}
        .summary {{ 
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        .section {{
            margin: 30px 0;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .chart-container {{
            margin: 20px 0;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .meta-info {{
            background: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .meta-info p {{
            margin: 5px 0;
            color: #555;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Mel Extractor</h1>
        <div class="meta-info">
            <p><strong>Status:</strong> <span style="color: {'green' if meta.get('status') == 'ok' else 'orange' if meta.get('status') == 'empty' else 'red'}">{meta.get('status', 'unknown')}</span></p>
            <p><strong>Producer:</strong> {meta.get('producer', 'unknown')} v{meta.get('producer_version', 'unknown')}</p>
            <p><strong>Contract Version:</strong> {meta.get('mel_contract_version', 'unknown')}</p>
        </div>
        
        <h2>📊 Summary</h2>
        <div class="summary">
            <div class="stat-card">
                <div class="stat-label">Sample Rate</div>
                <div class="stat-value">{summary.get('sample_rate', 'N/A')} Hz</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">N FFT</div>
                <div class="stat-value">{summary.get('n_fft', 'N/A')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Hop Length</div>
                <div class="stat-value">{summary.get('hop_length', 'N/A')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">N Mels</div>
                <div class="stat-value">{summary.get('n_mels', 'N/A')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Power</div>
                <div class="stat-value">{summary.get('power', 'N/A')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Device</div>
                <div class="stat-value">{summary.get('device_used', 'N/A')}</div>
            </div>
            {f'<div class="stat-card"><div class="stat-label">Segments</div><div class="stat-value">{summary.get("segments_count", "N/A")}</div></div>' if summary.get('segments_count') is not None else ''}
        </div>
"""
    
    # Always show at least basic info, even if no features enabled
    if not basic_features and not statistics and not spectral_features and not additional_metrics:
        html_content += """
        <div class="section">
            <h2>⚠️ No Features Enabled</h2>
            <p>No features were enabled for this extraction. Please enable at least one feature flag in the configuration.</p>
        </div>
"""
    
    # Basic features section
    if basic_features:
        mel_shape = basic_features.get("mel_shape", "N/A")
        mel_elements = basic_features.get("mel_elements", "N/A")
        html_content += f"""
        <div class="section">
            <h2>📈 Basic Features</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-label">Mel Shape</div>
                    <div class="metric-value">{mel_shape}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mel Elements</div>
                    <div class="metric-value">{mel_elements}</div>
                </div>
            </div>
        </div>
"""
    
    # Additional metrics section
    if additional_metrics:
        html_content += """
        <div class="section">
            <h2>📈 Additional Metrics</h2>
            <div class="metrics">
"""
        for key, value in additional_metrics.items():
            formatted_value = f"{safe_float(value, 0.0):.4f}" if isinstance(value, (int, float)) else str(value)
            html_content += f"""
                <div class="metric-card">
                    <div class="metric-label">{key.replace('_', ' ').title()}</div>
                    <div class="metric-value">{formatted_value}</div>
                </div>
"""
        html_content += """
            </div>
        </div>
"""
    
    # Mel Statistics Distribution
    if mel_mean_stats and isinstance(mel_mean_stats, list) and len(mel_mean_stats) > 0:
        html_content += f"""
        <div class="section">
            <h2>📊 Mel Statistics</h2>
            <h3>Mean Values by Mel Bin</h3>
            <div class="chart-container">
                <div id="mel-mean-chart" style="height: 400px;"></div>
            </div>
            <script>
                var meanData = [{{
                    x: {json.dumps(list(range(len(mel_mean_stats))))},
                    y: {json.dumps(mel_mean_stats)},
                    type: 'bar',
                    marker: {{
                        color: 'rgba(102, 126, 234, 0.7)',
                        line: {{ color: 'rgba(102, 126, 234, 1.0)', width: 1 }}
                    }},
                    name: 'Mel Mean'
                }}];
                var meanLayout = {{
                    title: {{
                        text: 'Mel Mean Values by Bin',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'Mel Bin Index' }},
                    yaxis: {{ title: 'Mean Value (dB)' }},
                    height: 400
                }};
                Plotly.newPlot('mel-mean-chart', meanData, meanLayout);
            </script>
"""
        if mel_std_stats and len(mel_std_stats) > 0:
            html_content += f"""
            <h3>Standard Deviation by Mel Bin</h3>
            <div class="chart-container">
                <div id="mel-std-chart" style="height: 400px;"></div>
            </div>
            <script>
                var stdData = [{{
                    x: {json.dumps(list(range(len(mel_std_stats))))},
                    y: {json.dumps(mel_std_stats)},
                    type: 'bar',
                    marker: {{
                        color: 'rgba(118, 75, 162, 0.7)',
                        line: {{ color: 'rgba(118, 75, 162, 1.0)', width: 1 }}
                    }},
                    name: 'Mel Std'
                }}];
                var stdLayout = {{
                    title: {{
                        text: 'Mel Standard Deviation by Bin',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'Mel Bin Index' }},
                    yaxis: {{ title: 'Standard Deviation (dB)' }},
                    height: 400
                }};
                Plotly.newPlot('mel-std-chart', stdData, stdLayout);
            </script>
"""
        html_content += """
        </div>
"""
    
    # Spectral features charts
    if spectral_centroid_data and isinstance(spectral_centroid_data, list) and len(spectral_centroid_data) > 0:
        centroid_time_indices = list(range(len(spectral_centroid_data)))
        html_content += f"""
        <div class="section">
            <h2>📈 Spectral Features</h2>
            <h3>Spectral Centroid</h3>
            <div class="chart-container">
                <div id="spectral-centroid-chart" style="height: 400px;"></div>
            </div>
            <script>
                var centroidData = [{{
                    x: {json.dumps(centroid_time_indices)},
                    y: {json.dumps(spectral_centroid_data)},
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#667eea', width: 2 }},
                    name: 'Spectral Centroid'
                }}];
                var centroidLayout = {{
                    title: {{
                        text: 'Spectral Centroid Over Time',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'Time Index' }},
                    yaxis: {{ title: 'Frequency (Hz)' }},
                    height: 400
                }};
                Plotly.newPlot('spectral-centroid-chart', centroidData, centroidLayout);
            </script>
"""
        
        if spectral_bandwidth_data and len(spectral_bandwidth_data) > 0:
            bandwidth_time_indices = list(range(len(spectral_bandwidth_data)))
            html_content += f"""
            <h3>Spectral Bandwidth</h3>
            <div class="chart-container">
                <div id="spectral-bandwidth-chart" style="height: 400px;"></div>
            </div>
            <script>
                var bandwidthData = [{{
                    x: {json.dumps(bandwidth_time_indices)},
                    y: {json.dumps(spectral_bandwidth_data)},
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#764ba2', width: 2 }},
                    name: 'Spectral Bandwidth'
                }}];
                var bandwidthLayout = {{
                    title: {{
                        text: 'Spectral Bandwidth Over Time',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'Time Index' }},
                    yaxis: {{ title: 'Frequency (Hz)' }},
                    height: 400
                }};
                Plotly.newPlot('spectral-bandwidth-chart', bandwidthData, bandwidthLayout);
            </script>
"""
        
        html_content += """
        </div>
"""
    
    # Time series chart
    if mel_series_data and isinstance(mel_series_data, list) and len(mel_series_data) > 0:
        time_indices = list(range(len(mel_series_data)))
        html_content += f"""
        <div class="section">
            <h2>📈 Time Series</h2>
            <h3>Mel Series</h3>
            <div class="chart-container">
                <div id="mel-series-chart" style="height: 400px;"></div>
            </div>
            <script>
                var seriesData = [{{
                    x: {json.dumps(time_indices)},
                    y: {json.dumps(mel_series_data)},
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#667eea', width: 1 }},
                    name: 'Mel Series'
                }}];
                var seriesLayout = {{
                    title: {{
                        text: 'Mel Time Series',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'Time Index' }},
                    yaxis: {{ title: 'Mel Value (dB)' }},
                    height: 400
                }};
                Plotly.newPlot('mel-series-chart', seriesData, seriesLayout);
            </script>
        </div>
"""
    
    # Distribution histogram
    if mel_series_data and len(mel_series_data) > 0:
        html_content += f"""
        <div class="section">
            <h2>📊 Distribution</h2>
            <h3>Mel Values Distribution</h3>
            <div class="chart-container">
                <div id="mel-distribution-chart" style="height: 400px;"></div>
            </div>
            <script>
                var distData = [{{
                    x: {json.dumps(mel_series_data)},
                    type: 'histogram',
                    marker: {{
                        color: 'rgba(102, 126, 234, 0.7)',
                        line: {{ color: 'rgba(102, 126, 234, 1.0)', width: 1 }}
                    }},
                    name: 'Mel Distribution',
                    nbinsx: 50
                }}];
                var distLayout = {{
                    title: {{
                        text: 'Distribution of Mel Values',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'Mel Value (dB)' }},
                    yaxis: {{ title: 'Frequency' }},
                    height: 400
                }};
                Plotly.newPlot('mel-distribution-chart', distData, distLayout);
            </script>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Mel extractor HTML render saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save mel extractor HTML render to {output_path}: {e}")
        raise

__all__ = ["render_mel_extractor", "render_mel_extractor_html"]
