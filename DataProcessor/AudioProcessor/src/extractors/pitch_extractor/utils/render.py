"""
Renderer для pitch_extractor: генерация render-context JSON и HTML debug страницы.
Audit v3: vanilla canvas (no CDN). f0_series из meta.f0_series_npy / meta.f0_series_torchcrepe_npy.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ....core.renderer import load_npz, extract_meta


def _load_f0_from_npy(npz_path: str, meta: Dict[str, Any]) -> list:
    """Загрузить f0_series из .npy (meta.f0_series_npy или meta.f0_series_torchcrepe_npy)."""
    for key in ["f0_series_npy", "f0_series_torchcrepe_npy"]:
        path = meta.get(key)
        if isinstance(path, str) and path:
            full = path if os.path.isabs(path) else os.path.join(os.path.dirname(npz_path), path)
            if os.path.exists(full):
                try:
                    arr = np.load(full)
                    return np.asarray(arr, dtype=np.float64).tolist()
                except Exception:
                    pass
    return []


def render_pitch_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для pitch_extractor."""
    render = {
        "component": "pitch_extractor",
        "summary": {},
        "basic_stats": {},
        "stability_metrics": {},
        "delta_features": {},
        "method_stats": {},
        "time_series": {},
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
    render["summary"] = {
        "sample_rate": int(features.get("sample_rate", 22050)),
        "device_used": "cuda" if features.get("device_used", 0.0) > 0.5 else "cpu",
        "f0_method": meta.get("f0_method", features.get("f0_method", "unknown")),
        "duration": float(features.get("duration", 0.0) or 0.0),
        "hop_length": int(features.get("hop_length", 512) or 512),
    }

    # Basic stats (feature-gated)
    if "f0_mean" in features:
        render["basic_stats"] = {
            "f0_mean": float(features.get("f0_mean", 0.0)),
            "f0_std": float(features.get("f0_std", 0.0)),
            "f0_min": float(features.get("f0_min", 0.0)),
            "f0_max": float(features.get("f0_max", 0.0)),
            "f0_median": float(features.get("f0_median", 0.0)),
        }

        # Additional ML/analytics metrics (Q6: pitch_centroid removed)
        if "pitch_contour_smoothness" in features:
            render["basic_stats"].update({
                "pitch_contour_smoothness": float(features.get("pitch_contour_smoothness", 0.0)),
                "pitch_jump_count": int(features.get("pitch_jump_count", 0)),
                "pitch_skewness": float(features.get("pitch_skewness", 0.0)),
                "pitch_kurtosis": float(features.get("pitch_kurtosis", 0.0)),
            })

        # Pitch octave distribution
        pitch_octave_distribution = npz_data.get("pitch_octave_distribution")
        if pitch_octave_distribution is not None:
            if isinstance(pitch_octave_distribution, np.ndarray) and pitch_octave_distribution.dtype == object:
                pitch_octave_distribution = pitch_octave_distribution.item() if pitch_octave_distribution.size == 1 else {}
            if isinstance(pitch_octave_distribution, dict):
                render["basic_stats"]["pitch_octave_distribution"] = {str(k): float(v) for k, v in pitch_octave_distribution.items()}

    # Stability metrics (feature-gated)
    if "pitch_variation" in features:
        render["stability_metrics"] = {
            "pitch_variation": float(features.get("pitch_variation", 0.0)),
            "pitch_stability": float(features.get("pitch_stability", 0.0)),
            "pitch_range": float(features.get("pitch_range", 0.0)),
        }

    # Delta features (feature-gated)
    if "f0_delta_mean" in features:
        render["delta_features"] = {
            "f0_delta_mean": float(features.get("f0_delta_mean", 0.0)),
            "f0_delta_std": float(features.get("f0_delta_std", 0.0)),
            "f0_delta_abs_mean": float(features.get("f0_delta_abs_mean", 0.0)),
        }

    # Method stats (feature-gated)
    if "f0_mean_pyin" in features:
        render["method_stats"] = {
            "pyin": {
                "f0_mean": float(features.get("f0_mean_pyin", 0.0)),
                "f0_std": float(features.get("f0_std_pyin", 0.0)),
                "f0_min": float(features.get("f0_min_pyin", 0.0)),
                "f0_max": float(features.get("f0_max_pyin", 0.0)),
                "f0_median": float(features.get("f0_median_pyin", 0.0)),
                "f0_count": int(features.get("f0_count_pyin", 0)),
                "voiced_fraction": float(features.get("voiced_fraction_pyin", 0.0)),
                "voiced_probability_mean": float(features.get("voiced_probability_mean_pyin", 0.0)),
            },
            "yin": {
                "f0_mean": float(features.get("f0_mean_yin", 0.0)),
                "f0_std": float(features.get("f0_std_yin", 0.0)),
                "f0_min": float(features.get("f0_min_yin", 0.0)),
                "f0_max": float(features.get("f0_max_yin", 0.0)),
                "f0_median": float(features.get("f0_median_yin", 0.0)),
                "f0_count": int(features.get("f0_count_yin", 0)),
            },
        }

        if "f0_mean_torchcrepe" in features:
            render["method_stats"]["torchcrepe"] = {
                "f0_mean": float(features.get("f0_mean_torchcrepe", 0.0)),
                "f0_std": float(features.get("f0_std_torchcrepe", 0.0)),
                "f0_min": float(features.get("f0_min_torchcrepe", 0.0)),
                "f0_max": float(features.get("f0_max_torchcrepe", 0.0)),
                "f0_median": float(features.get("f0_median_torchcrepe", 0.0)),
                "f0_count": int(features.get("f0_count_torchcrepe", 0)),
            }

    # Time series: loaded from .npy (Q7), not from NPZ
    render["time_series"] = {}
    return render


def render_pitch_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML render для pitch_extractor (Audit v3: vanilla canvas, no CDN).
    f0_series загружается из meta.f0_series_npy или meta.f0_series_torchcrepe_npy.
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_pitch_extractor(npz_data, meta)

    # Load f0_series from .npy
    f0_series = _load_f0_from_npy(npz_path, meta)

    summary = render.get("summary", {})
    basic_stats = render.get("basic_stats", {})
    stability_metrics = render.get("stability_metrics", {})
    delta_features = render.get("delta_features", {})
    method_stats = render.get("method_stats", {})

    # Build features for duration/hop_length (from feature_names/feature_values)
    feature_names = npz_data.get("feature_names", [])
    feature_values = npz_data.get("feature_values", [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    if isinstance(feature_values, np.ndarray):
        feature_values = feature_values.tolist()
    fmap = {feature_names[i]: feature_values[i] for i in range(min(len(feature_names), len(feature_values)))}
    duration = float(fmap.get("duration", summary.get("duration", 0.0)) or 0.0)
    sample_rate = int(summary.get("sample_rate", 22050) or 22050)
    hop_length = int(fmap.get("hop_length", summary.get("hop_length", 512)) or 512)

    def safe_float(v, default=0.0):
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    # Build time axis for f0_series (frame indices -> seconds)
    n_frames = len(f0_series) if f0_series else 0
    time_sec = [i * hop_length / sample_rate for i in range(n_frames)] if n_frames else []

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Pitch Extractor Debug</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; background: #f5f6fa; color: #111; }}
    .container {{ max-width: 1200px; margin: 0 auto; background: #fff; border-radius: 12px; padding: 20px; box-shadow: 0 2px 14px rgba(0,0,0,0.06); }}
    h1 {{ margin: 0 0 10px 0; }}
    h2 {{ margin: 18px 0 10px 0; }}
    .meta {{ color: #555; font-size: 0.95em; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 14px 0; }}
    .card {{ border: 1px solid #eee; border-radius: 10px; padding: 12px; background: #fff; }}
    .label {{ color: #666; font-size: 0.85em; margin-bottom: 4px; }}
    .value {{ font-variant-numeric: tabular-nums; font-size: 1.15em; font-weight: 700; }}
    canvas {{ width: 100%; border: 1px solid #eee; border-radius: 10px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Pitch Extractor Debug</h1>
    <div class="meta">
      <div><b>Status:</b> {meta.get("status", "unknown")} {(" | " + str(meta.get("empty_reason"))) if meta.get("empty_reason") else ""}</div>
      <div><b>Producer:</b> {meta.get("producer", "unknown")} v{meta.get("producer_version", "unknown")} | <b>Method:</b> {summary.get("f0_method", "unknown")}</div>
    </div>

    <div class="grid">
      <div class="card"><div class="label">sample_rate</div><div class="value">{summary.get("sample_rate", "N/A")}</div></div>
      <div class="card"><div class="label">device</div><div class="value">{summary.get("device_used", "cpu")}</div></div>
      <div class="card"><div class="label">duration</div><div class="value">{duration:.2f}s</div></div>
    </div>
"""

    if basic_stats:
        html_content += """
    <h2>Basic Statistics</h2>
    <div class="grid">
"""
        for key, value in basic_stats.items():
            if key != "pitch_octave_distribution":
                fmt = f"{value:.2f}" if isinstance(value, float) else str(value)
                html_content += f'      <div class="card"><div class="label">{key.replace("_", " ").title()}</div><div class="value">{fmt}</div></div>\n'
        html_content += "    </div>\n"
        if "pitch_octave_distribution" in basic_stats:
            html_content += '    <h3>Pitch Octave Distribution</h3>\n    <div class="grid">\n'
            for octave, ratio in basic_stats["pitch_octave_distribution"].items():
                html_content += f'      <div class="card"><div class="label">{octave}</div><div class="value">{ratio:.1%}</div></div>\n'
            html_content += "    </div>\n"

    if stability_metrics:
        html_content += """
    <h2>Stability Metrics</h2>
    <div class="grid">
"""
        for key, value in stability_metrics.items():
            html_content += f'      <div class="card"><div class="label">{key.replace("_", " ").title()}</div><div class="value">{value:.2f}</div></div>\n'
        html_content += "    </div>\n"

    if delta_features:
        html_content += """
    <h2>Delta Features</h2>
    <div class="grid">
"""
        for key, value in delta_features.items():
            html_content += f'      <div class="card"><div class="label">{key.replace("_", " ").title()}</div><div class="value">{value:.2f}</div></div>\n'
        html_content += "    </div>\n"

    if method_stats:
        html_content += """
    <h2>Method Statistics</h2>
"""
        for method, stats in method_stats.items():
            html_content += f'    <h3>{method.upper()}</h3>\n    <div class="grid">\n'
            for key, value in stats.items():
                fmt = f"{value:.2f}" if isinstance(value, float) else str(value)
                html_content += f'      <div class="card"><div class="label">{key.replace("_", " ").title()}</div><div class="value">{fmt}</div></div>\n'
            html_content += "    </div>\n"

    html_content += """
    <h2>Pitch (f0) Time Series</h2>
    <canvas id="pitchCanvas" width="1100" height="280"></canvas>

    <script>
      const f0Series = """ + json.dumps(f0_series) + """;
      const timeSec = """ + json.dumps(time_sec) + """;
      const duration = """ + json.dumps(duration) + """;

      function drawPitchChart() {
        const canvas = document.getElementById('pitchCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        const pad = 44;
        ctx.clearRect(0,0,w,h);
        ctx.fillStyle = '#fff';
        ctx.fillRect(0,0,w,h);
        ctx.strokeStyle = '#ddd';
        ctx.strokeRect(0.5,0.5,w-1,h-1);
        if (!Array.isArray(f0Series) || f0Series.length === 0) {
          ctx.fillStyle = '#666';
          ctx.font = '14px sans-serif';
          ctx.fillText('no f0 data (enable time_series for .npy)', 12, 24);
          return;
        }
        const pts = [];
        for (let i = 0; i < f0Series.length; i++) {
          const t = Array.isArray(timeSec) && timeSec[i] !== undefined ? Number(timeSec[i]) : i;
          const y = Number(f0Series[i]);
          if (Number.isFinite(y) && y > 0) pts.push([t, y]);
        }
        if (pts.length === 0) return;
        const xMin = pts[0][0], xMax = pts[pts.length-1][0] || 1;
        const yVals = pts.map(p => p[1]);
        const yMin = Math.max(1, Math.min(...yVals) * 0.9);
        const yMax = Math.max(...yVals) * 1.1;
        const xScale = (w - 2*pad) / Math.max(xMax - xMin, 1e-6);
        const yScale = (h - 2*pad) / (yMax - yMin);
        ctx.strokeStyle = '#bbb';
        ctx.beginPath();
        ctx.moveTo(pad, pad);
        ctx.lineTo(pad, h-pad);
        ctx.lineTo(w-pad, h-pad);
        ctx.stroke();
        ctx.strokeStyle = '#1976d2';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        let first = true;
        for (let i = 0; i < pts.length; i++) {
          const x = pad + (pts[i][0] - xMin) * xScale;
          const y = h - pad - (pts[i][1] - yMin) * yScale;
          if (first) { ctx.moveTo(x, y); first = false; }
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      drawPitchChart();
    </script>
  </div>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    os.replace(tmp_path, output_path)
    logger.info(f"Saved pitch HTML render to {output_path}")
    return output_path


__all__ = ["render_pitch_extractor", "render_pitch_extractor_html"]
