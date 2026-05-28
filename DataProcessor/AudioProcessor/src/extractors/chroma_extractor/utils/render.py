"""
Renderer для chroma_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ....core.renderer import load_npz, extract_meta

def render_chroma_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для chroma_extractor."""
    render = {
        "component": "chroma_extractor",
        "summary": {},
        "basic_stats": {},
        "extended_stats": {},
        "additional_metrics": {},
        "time_series": {},
    }
    
    features_enabled = meta.get("features_enabled", [])
    
    # Basic stats
    if "basic_stats" in features_enabled:
        chroma_mean = npz_data.get("chroma_mean", np.zeros(12, dtype=np.float32))
        chroma_std = npz_data.get("chroma_std", np.zeros(12, dtype=np.float32))
        chroma_min = npz_data.get("chroma_min", np.zeros(12, dtype=np.float32))
        chroma_max = npz_data.get("chroma_max", np.zeros(12, dtype=np.float32))
        
        if isinstance(chroma_mean, np.ndarray):
            chroma_mean = chroma_mean.tolist()
        if isinstance(chroma_std, np.ndarray):
            chroma_std = chroma_std.tolist()
        if isinstance(chroma_min, np.ndarray):
            chroma_min = chroma_min.tolist()
        if isinstance(chroma_max, np.ndarray):
            chroma_max = chroma_max.tolist()
        
        render["basic_stats"] = {
            "chroma_mean": chroma_mean,
            "chroma_std": chroma_std,
            "chroma_min": chroma_min,
            "chroma_max": chroma_max,
        }
    
    # Extended stats
    if "extended_stats" in features_enabled:
        chroma_median = npz_data.get("chroma_median", np.zeros(12, dtype=np.float32))
        chroma_p25 = npz_data.get("chroma_p25", np.zeros(12, dtype=np.float32))
        chroma_p75 = npz_data.get("chroma_p75", np.zeros(12, dtype=np.float32))
        
        if isinstance(chroma_median, np.ndarray):
            chroma_median = chroma_median.tolist()
        if isinstance(chroma_p25, np.ndarray):
            chroma_p25 = chroma_p25.tolist()
        if isinstance(chroma_p75, np.ndarray):
            chroma_p75 = chroma_p75.tolist()
        
        render["extended_stats"] = {
            "chroma_median": chroma_median,
            "chroma_p25": chroma_p25,
            "chroma_p75": chroma_p75,
        }
    
    # Helper function to safely get value from NPZ data
    def safe_get(key: str, default: Any = None) -> Any:
        value = npz_data.get(key, default)
        if value is None:
            return default
        # Handle numpy arrays (if not converted yet by load_npz)
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            if value.size == 1:
                val = value.item()
                # Convert NaN/Inf to None for JSON compatibility
                if isinstance(val, (float, np.floating)):
                    if np.isnan(val) or np.isinf(val):
                        return None
                return val
            # For multi-element arrays, convert to list
            return value.tolist()
        # Handle lists (converted from numpy arrays by load_npz)
        if isinstance(value, list):
            if len(value) == 0:
                return default
            # For scalar values stored as single-element lists
            if len(value) == 1 and not isinstance(value[0], (list, np.ndarray)):
                val = value[0]
                # Convert NaN/Inf to None for JSON compatibility
                if isinstance(val, (float, np.floating)):
                    if np.isnan(val) or np.isinf(val):
                        return None
                return val
            return value
        # Convert NaN/Inf to None for JSON compatibility
        if isinstance(value, (float, np.floating)):
            if np.isnan(value) or np.isinf(value):
                return None
        return value
    
    # Extract feature_names and feature_values to build features dict
    feature_names = safe_get("feature_names", [])
    feature_values = safe_get("feature_values", [])
    
    # Build features dict from feature_names/feature_values
    features = {}
    if isinstance(feature_names, list) and isinstance(feature_values, list):
        for i, name in enumerate(feature_names):
            if i < len(feature_values):
                value = feature_values[i]
                # Convert NaN/Inf to None for JSON compatibility
                if isinstance(value, (float, np.floating)):
                    if np.isnan(value) or np.isinf(value):
                        features[name] = None
                    else:
                        features[name] = value
                else:
                    features[name] = value
    
    # Helper to get value from features dict or directly from NPZ
    def get_feature(key: str, default: Any = None) -> Any:
        # First try features dict (from feature_names/feature_values)
        if key in features:
            return features[key]
        # Then try direct NPZ key
        return safe_get(key, default)
    
    # Summary (always available) - read from NPZ directly
    render["summary"] = {
        "chroma_type": get_feature("chroma_type", meta.get("chroma_type", "unknown")),
        "normalize": get_feature("normalize", meta.get("normalize", "unknown")),
        "n_chroma": get_feature("n_chroma", 12),
        "sample_rate": get_feature("sample_rate"),
        "hop_length": get_feature("hop_length"),
        "n_fft": get_feature("n_fft"),
        "chroma_frames": get_feature("chroma_frames"),
        "segments_count": get_feature("segments_count"),
    }
    
    # Additional metrics (always computed)
    render["additional_metrics"] = {
        "tuning_estimate": get_feature("tuning_estimate"),
        "chroma_dominant_class": get_feature("chroma_dominant_class"),
        "chroma_dominant_energy": get_feature("chroma_dominant_energy"),
        "chroma_harmonic_stability": get_feature("chroma_harmonic_stability"),
        "chroma_entropy": get_feature("chroma_entropy"),
        "chroma_contrast": get_feature("chroma_contrast"),
        "chroma_centroid": get_feature("chroma_centroid"),
        "chroma_rolloff": get_feature("chroma_rolloff"),
    }
    
    # Time series (feature-gated) - chroma is stored as separate key in NPZ
    chroma = safe_get("chroma", [])
    if not isinstance(chroma, list):
        chroma = []
    
    render["time_series"] = {
        "chroma": chroma,
        "duration": get_feature("duration"),
    }
    
    return render


def render_chroma_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага chroma_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML файла
        
    Returns:
        Путь к сохраненному HTML файлу
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_chroma_extractor(npz_data, meta)
    
    chroma = render.get("time_series", {}).get("chroma")
    chroma_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    # Helper to safely get chroma_dominant_class as int
    def get_dominant_class_name():
        dominant_class = render.get("additional_metrics", {}).get("chroma_dominant_class")
        if dominant_class is None:
            return "N/A"
        try:
            # Convert to int (handles float values like 0.0)
            idx = int(dominant_class)
            if 0 <= idx < len(chroma_names):
                return chroma_names[idx]
            return "N/A"
        except (ValueError, TypeError):
            return "N/A"
    
    # Get dominant class name before using in f-string
    dominant_class_name = get_dominant_class_name()
    
    # Offline-only HTML (Audit v3): no CDN / no network.
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Chroma Extractor Debug</title>
    <style>
        body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .metric-label {{ font-weight: bold; }}
        .metric-value {{ color: #333; }}
        .row {{ display: grid; grid-template-columns: 52px 1fr 64px; gap: 10px; align-items: center; margin: 6px 0; }}
        .bar {{ height: 12px; background: #eee; border-radius: 999px; overflow: hidden; }}
        .fill {{ height: 100%; background: linear-gradient(90deg, #16a34a, #22c55e); }}
        .val {{ text-align: right; font-variant-numeric: tabular-nums; }}
        canvas {{ width: 100%; max-width: 980px; border: 1px solid #eee; border-radius: 6px; }}
    </style>
</head>
<body>
    <h1>Chroma Extractor Debug</h1>
    
    <div class="section">
        <h2>Summary</h2>
        <div class="metric">
            <span class="metric-label">Chroma Type:</span>
            <span class="metric-value">{render.get("summary", {}).get("chroma_type", "unknown")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Normalize:</span>
            <span class="metric-value">{render.get("summary", {}).get("normalize", "unknown")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">N Chroma:</span>
            <span class="metric-value">{render.get("summary", {}).get("n_chroma", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Sample Rate:</span>
            <span class="metric-value">{render.get("summary", {}).get("sample_rate", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Hop Length:</span>
            <span class="metric-value">{render.get("summary", {}).get("hop_length", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Chroma Frames:</span>
            <span class="metric-value">{render.get("summary", {}).get("chroma_frames", "N/A")}</span>
        </div>
    </div>
    
    <div class="section">
        <h2>Chroma mean (12 classes)</h2>
        <div id="chroma-mean-bars"></div>
    </div>
    
    <div class="section">
        <h2>Additional Metrics</h2>
        <div class="metric">
            <span class="metric-label">Tuning Estimate:</span>
            <span class="metric-value">{render.get("additional_metrics", {}).get("tuning_estimate", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Dominant Class:</span>
            <span class="metric-value">{dominant_class_name}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Dominant Energy:</span>
            <span class="metric-value">{render.get("additional_metrics", {}).get("chroma_dominant_energy", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Harmonic Stability:</span>
            <span class="metric-value">{render.get("additional_metrics", {}).get("chroma_harmonic_stability", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Entropy:</span>
            <span class="metric-value">{render.get("additional_metrics", {}).get("chroma_entropy", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Contrast:</span>
            <span class="metric-value">{render.get("additional_metrics", {}).get("chroma_contrast", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Centroid:</span>
            <span class="metric-value">{render.get("additional_metrics", {}).get("chroma_centroid", "N/A")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Rolloff:</span>
            <span class="metric-value">{render.get("additional_metrics", {}).get("chroma_rolloff", "N/A")}</span>
        </div>
    </div>
    
    <div class="section">
        <h2>Chroma Spectrogram</h2>
        <div class="metric"><span class="metric-value">If time series wasn't saved, this will be empty.</span></div>
        <canvas id="chroma-canvas" width="980" height="320"></canvas>
    </div>
    
    <script>
        var chromaNames = {json.dumps(chroma_names)};
        var chroma = {json.dumps(chroma) if chroma is not None else "null"};
        var chromaMean = {json.dumps(npz_data.get("chroma_mean").tolist() if isinstance(npz_data.get("chroma_mean"), np.ndarray) else [])};
        
        // Simple bars for chroma mean
        var container = document.getElementById('chroma-mean-bars');
        function clamp01(x) {{ return Math.max(0, Math.min(1, x)); }}
        for (var i = 0; i < chromaNames.length; i++) {{
          var v = (i < chromaMean.length) ? chromaMean[i] : 0.0;
          if (v === null || Number.isNaN(v) || !Number.isFinite(v)) v = 0.0;
          v = clamp01(v);
          var pct = Math.round(v * 100);
          var row = document.createElement('div');
          row.className = 'row';
          row.innerHTML = `
            <div><b>${{chromaNames[i]}}</b></div>
            <div class="bar"><div class="fill" style="width:${{pct}}%"></div></div>
            <div class="val">${{v.toFixed(3)}}</div>
          `;
          container.appendChild(row);
        }}
        
        // Offline heatmap on canvas
        var canvas = document.getElementById('chroma-canvas');
        var ctx = canvas.getContext('2d');
        ctx.clearRect(0,0,canvas.width,canvas.height);
        if (chroma !== null && Array.isArray(chroma) && chroma.length === 12 && chroma[0].length > 0) {{
          var rows = 12;
          var cols = chroma[0].length;
          var cellW = canvas.width / cols;
          var cellH = canvas.height / rows;
          // find max for normalization
          var maxV = 0.0;
          for (var r = 0; r < rows; r++) {{
            for (var c = 0; c < cols; c++) {{
              var vv = chroma[r][c];
              if (Number.isFinite(vv) && vv > maxV) maxV = vv;
            }}
          }}
          if (maxV <= 0) maxV = 1.0;
          for (var r = 0; r < rows; r++) {{
            for (var c = 0; c < cols; c++) {{
              var vv = chroma[r][c];
              if (!Number.isFinite(vv)) vv = 0.0;
              var t = clamp01(vv / maxV);
              // simple green palette
              var g = Math.floor(30 + 200 * t);
              var b = Math.floor(30 + 40 * (1-t));
              ctx.fillStyle = `rgb(20,${{g}},${{b}})`;
              ctx.fillRect(c*cellW, (rows-1-r)*cellH, Math.ceil(cellW), Math.ceil(cellH));
            }}
          }}
        }} else {{
          ctx.fillStyle = '#666';
          ctx.font = '16px sans-serif';
          ctx.fillText('chroma time series not available', 12, 28);
        }}
    </script>
</body>
</html>"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return output_path

__all__ = ["render_chroma_extractor", "render_chroma_extractor_html"]
