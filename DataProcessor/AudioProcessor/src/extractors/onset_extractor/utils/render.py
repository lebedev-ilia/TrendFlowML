"""
Renderer для onset_extractor: генерация render-context JSON и HTML debug страницы.
Audit v3: offline-only (vanilla canvas, без CDN). onset_times из meta.extra.onset_times_npy.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ....core.renderer import load_npz, extract_meta


def render_onset_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для onset_extractor."""
    render = {
        "component": "onset_extractor",
        "summary": {},
        "basic_features": {},
        "interval_stats": {},
        "rhythmic_metrics": {},
        "timeline": {},
    }

    def safe_get(key: str, default: Any = None) -> Any:
        value = npz_data.get(key, default)
        if value is None:
            return default
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            if value.size == 1:
                val = value.item()
                if isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val)):
                    return None
                return val
            return value.tolist()
        if isinstance(value, list):
            if len(value) == 0:
                return default
            if len(value) == 1 and not isinstance(value[0], (list, np.ndarray)):
                val = value[0]
                if isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val)):
                    return None
                return val
            return value
        if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
            return None
        return value

    feature_names = safe_get("feature_names", [])
    feature_values = safe_get("feature_values", [])
    features = {}
    if isinstance(feature_names, list) and isinstance(feature_values, list):
        for i, name in enumerate(feature_names):
            if i < len(feature_values):
                value = feature_values[i]
                if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
                    features[name] = None
                else:
                    features[name] = value

    def get_feature(key: str, default: Any = None) -> Any:
        if key in features:
            return features[key]
        return safe_get(key, default)

    features_enabled = meta.get("features_enabled", [])

    _backend = features.get("backend")
    if _backend is None:
        _backend = meta.get("backend", "unknown")

    render["summary"] = {
        "backend": _backend,
        "sample_rate": get_feature("sample_rate"),
        "hop_length": get_feature("hop_length"),
        "segments_count": get_feature("segments_count"),
    }

    if "basic_features" in features_enabled:
        render["basic_features"] = {
            "onset_count": get_feature("onset_count"),
            "onset_density_per_sec": get_feature("onset_density_per_sec"),
            "insufficient_onsets": get_feature("insufficient_onsets"),
        }

    if "interval_stats" in features_enabled:
        render["interval_stats"] = {
            "avg_interval_sec": get_feature("avg_interval_sec"),
            "interval_std": get_feature("interval_std"),
            "interval_min": get_feature("interval_min"),
            "interval_max": get_feature("interval_max"),
            "interval_median": get_feature("interval_median"),
        }

    if "rhythmic_metrics" in features_enabled:
        render["rhythmic_metrics"] = {
            "onset_regularity_score": get_feature("onset_regularity_score"),
            "onset_tempo_estimate": get_feature("onset_tempo_estimate"),
            "onset_syncopation_score": get_feature("onset_syncopation_score"),
            "onset_strength_mean": get_feature("onset_strength_mean"),
            "onset_strength_std": get_feature("onset_strength_std"),
            "onset_density_variance": get_feature("onset_density_variance"),
            "onset_tempo_consistency": get_feature("onset_tempo_consistency"),
        }

    render["timeline"] = {
        "onset_times": [],
        "duration": get_feature("duration"),
    }
    return render


def render_onset_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага onset_extractor (Audit v3: offline-only, vanilla canvas).
    onset_times загружаются из meta.extra.onset_times_npy (.npy).
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_onset_extractor(npz_data, meta)

    onset_times = []
    onset_times_npy = meta.get("onset_times_npy")
    if isinstance(onset_times_npy, str) and onset_times_npy:
        npy_full = os.path.join(os.path.dirname(npz_path), onset_times_npy)
        if os.path.exists(npy_full):
            try:
                onset_times = np.load(npy_full).tolist()
            except Exception:
                pass

    duration = render.get("timeline", {}).get("duration", 0.0)
    try:
        duration = float(duration) if duration is not None else 0.0
    except (TypeError, ValueError):
        duration = 0.0

    def safe_float(v, default=0.0):
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Onset Extractor Debug</title>
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
    <h1>Onset Extractor Debug</h1>
    <div class="meta">
      <div><b>Status:</b> {meta.get("status", "unknown")} {(" | " + str(meta.get("empty_reason"))) if meta.get("empty_reason") else ""}</div>
      <div><b>Schema:</b> {meta.get("schema_version", "unknown")} | <b>Producer:</b> {meta.get("producer_version", "unknown")}</div>
      <div><b>Features enabled:</b> {", ".join(meta.get("features_enabled", []) or [])}</div>
    </div>

    <div class="grid">
      <div class="card"><div class="label">backend</div><div class="value">{render.get("summary", {}).get("backend", "N/A")}</div></div>
      <div class="card"><div class="label">sample_rate</div><div class="value">{render.get("summary", {}).get("sample_rate", "N/A")}</div></div>
      <div class="card"><div class="label">hop_length</div><div class="value">{render.get("summary", {}).get("hop_length", "N/A")}</div></div>
      <div class="card"><div class="label">segments_count</div><div class="value">{render.get("summary", {}).get("segments_count", "N/A")}</div></div>
      <div class="card"><div class="label">duration</div><div class="value">{duration:.2f}s</div></div>
    </div>

    <div class="grid">
      <div class="card"><div class="label">onset_count</div><div class="value">{render.get("basic_features", {}).get("onset_count", "N/A")}</div></div>
      <div class="card"><div class="label">onset_density_per_sec</div><div class="value">{safe_float(render.get("basic_features", {}).get("onset_density_per_sec")):.4f}</div></div>
      <div class="card"><div class="label">insufficient_onsets</div><div class="value">{render.get("basic_features", {}).get("insufficient_onsets", "N/A")}</div></div>
    </div>

    <div class="grid">
      <div class="card"><div class="label">avg_interval_sec</div><div class="value">{safe_float(render.get("interval_stats", {}).get("avg_interval_sec")):.4f}</div></div>
      <div class="card"><div class="label">interval_std</div><div class="value">{safe_float(render.get("interval_stats", {}).get("interval_std")):.4f}</div></div>
      <div class="card"><div class="label">onset_regularity_score</div><div class="value">{safe_float(render.get("rhythmic_metrics", {}).get("onset_regularity_score")):.4f}</div></div>
      <div class="card"><div class="label">onset_tempo_estimate (BPM)</div><div class="value">{safe_float(render.get("rhythmic_metrics", {}).get("onset_tempo_estimate")):.2f}</div></div>
    </div>

    <h2>Onset Timeline</h2>
    <canvas id="onsetCanvas" width="1100" height="280"></canvas>

    <script>
      const onsetTimes = {json.dumps(onset_times)};
      const duration = {duration};

      function drawOnsetTimeline() {{
        const canvas = document.getElementById('onsetCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        const pad = 44;
        ctx.clearRect(0,0,w,h);
        ctx.fillStyle = '#fff';
        ctx.fillRect(0,0,w,h);
        ctx.strokeStyle = '#ddd';
        ctx.strokeRect(0.5,0.5,w-1,h-1);
        if (!Array.isArray(onsetTimes) || onsetTimes.length === 0) {{
          ctx.fillStyle = '#666';
          ctx.font = '14px sans-serif';
          ctx.fillText('no onset data', 12, 24);
          return;
        }}
        const pts = [];
        for (let i = 0; i < onsetTimes.length; i++) {{
          const t = Number(onsetTimes[i]);
          if (Number.isFinite(t)) pts.push([t, 1]);
        }}
        if (pts.length === 0) return;
        const xMin = 0, xMax = Math.max(duration, 1e-6);
        const xScale = (w - 2*pad) / xMax;
        const yScale = (h - 2*pad) / 2;
        ctx.strokeStyle = '#bbb';
        ctx.beginPath();
        ctx.moveTo(pad, pad);
        ctx.lineTo(pad, h-pad);
        ctx.lineTo(w-pad, h-pad);
        ctx.stroke();
        ctx.fillStyle = '#c41e3a';
        for (let i = 0; i < pts.length; i++) {{
          const x = pad + pts[i][0] * xScale;
          const y = h - pad - (pts[i][1] - 0) * yScale;
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fill();
        }}
      }}
      drawOnsetTimeline();
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


__all__ = ["render_onset_extractor", "render_onset_extractor_html"]
