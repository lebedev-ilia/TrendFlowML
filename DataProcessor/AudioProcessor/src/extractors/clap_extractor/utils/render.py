"""
Renderer для clap_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_clap_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для clap_extractor."""
    render = {
        "component": "clap_extractor",
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
    
    # Build feature dict
    features = {}
    for i, name in enumerate(feature_names):
        if i < len(feature_values):
            features[name] = feature_values[i]
    
    # Summary statistics
    # Helper function to safely convert to int (handles NaN)
    def safe_int(value, default=0):
        if value is None:
            return default
        try:
            val = float(value)
            if np.isnan(val):
                return default
            return int(val)
        except (ValueError, TypeError):
            return default

    def safe_float(value, default=0.0):
        if value is None:
            return default
        try:
            v = float(value)
            if np.isnan(v) or np.isinf(v):
                return default
            return v
        except (ValueError, TypeError):
            return default

    emb_present_raw = npz_data.get("embedding_present", False)
    try:
        emb_present = bool(np.asarray(emb_present_raw).item())
    except Exception:
        emb_present = bool(emb_present_raw)
    
    render["summary"] = {
        "embedding_norm": safe_float(features.get("clap_norm", 0.0)),
        "embedding_magnitude_mean": safe_float(features.get("clap_magnitude_mean", 0.0)),
        "embedding_magnitude_std": safe_float(features.get("clap_magnitude_std", 0.0)),
        "segments_count": safe_int(features.get("segments_count", 0)),
        "embedding_dim": safe_int(features.get("embedding_dim", 0)),
        "embedding_present": emb_present,
        "trimmed_ratio": safe_float(meta.get("trimmed_ratio", 0.0)),
        "trimmed_segments_count": safe_int(meta.get("trimmed_segments_count", 0)),
    }
    
    # Timeline data (if available)
    segment_centers_sec = npz_data.get("segment_center_sec")
    segment_mask = npz_data.get("segment_mask")
    segment_norm = npz_data.get("segment_embedding_norm")
    
    if segment_centers_sec is not None:
        if isinstance(segment_centers_sec, np.ndarray):
            segment_centers_sec = segment_centers_sec.tolist()

        if isinstance(segment_mask, np.ndarray):
            segment_mask = segment_mask.astype(bool).tolist()
        if segment_mask is None:
            segment_mask = [True] * len(segment_centers_sec)

        if isinstance(segment_norm, np.ndarray):
            segment_norm = segment_norm.astype(np.float32).tolist()
        if segment_norm is None:
            segment_norm = [None] * len(segment_centers_sec)

        # Build timeline
        timeline = []
        norms = []
        for i, center_sec in enumerate(segment_centers_sec):
            m = bool(segment_mask[i]) if i < len(segment_mask) else True
            n = None
            if i < len(segment_norm):
                try:
                    n = float(segment_norm[i]) if segment_norm[i] is not None else None
                except Exception:
                    n = None
            if n is None or (isinstance(n, float) and (np.isnan(n) or np.isinf(n))):
                n = None
            timeline.append({
                "center_sec": float(center_sec),
                "embedding_norm": n,
                "masked": (not m),
                "segment_index": i,
            })
            if (n is not None) and m:
                norms.append(float(n))
        render["timeline"] = timeline
        
        # Distribution of embedding norms
        if norms:
            render["distributions"]["embedding_norm"] = {
                "min": float(np.min(norms)),
                "max": float(np.max(norms)),
                "mean": float(np.mean(norms)),
                "std": float(np.std(norms)),
                "median": float(np.median(norms)),
            }
    
    return render


def render_clap_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага clap_extractor результатов.
    
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
    render = render_clap_extractor(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    
    times = [float(t.get("center_sec", 0.0) or 0.0) for t in timeline] if timeline else []
    norms = [t.get("embedding_norm", None) for t in timeline] if timeline else []
    masked = [bool(t.get("masked", False)) for t in timeline] if timeline else []
    
    html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CLAP Extractor Debug Render</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 1200px; margin: 0 auto; }}
        h1, h2 {{ color: #0056b3; }}
        .summary {{ background-color: #eaf4ff; padding: 15px; border-radius: 5px; margin: 20px 0; border: 1px solid #cce0ff; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .metric-card {{ background-color: #f8f9fa; padding: 12px; border-radius: 5px; border: 1px solid #dee2e6; }}
        .metric-card strong {{ color: #0056b3; display: block; margin-bottom: 5px; }}
        .metric-value {{ font-size: 1.2em; color: #333; }}
        .chart-container {{ position: relative; height: 420px; width: 100%; margin: 20px 0; }}
        canvas {{ width: 100%; height: 380px; background: #ffffff; border: 1px solid #dee2e6; border-radius: 6px; }}
        .distributions {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .distributions table {{ width: 100%; border-collapse: collapse; }}
        .distributions th, .distributions td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .distributions th {{ background-color: #0056b3; color: white; }}
        .note {{ font-size: 0.95em; color: #555; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CLAP Extractor Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Embedding Norm</strong>
                    <span class="metric-value">{summary.get('embedding_norm', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Magnitude Mean</strong>
                    <span class="metric-value">{summary.get('embedding_magnitude_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Magnitude Std</strong>
                    <span class="metric-value">{summary.get('embedding_magnitude_std', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Segments Count</strong>
                    <span class="metric-value">{summary.get('segments_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Embedding Dim</strong>
                    <span class="metric-value">{summary.get('embedding_dim', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Embedding Present</strong>
                    <span class="metric-value">{'Yes' if summary.get('embedding_present', False) else 'No'}</span>
                </div>
                <div class="metric-card">
                    <strong>Trimmed Ratio</strong>
                    <span class="metric-value">{float(summary.get('trimmed_ratio', 0.0) or 0.0):.3f}</span>
                </div>
                <div class="metric-card">
                    <strong>Trimmed Segments</strong>
                    <span class="metric-value">{int(summary.get('trimmed_segments_count', 0) or 0)}</span>
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Timeline: Embedding Norm Over Time</h2>
            <p class="note">Masked сегменты (segment_mask=false) отображаются как разрывы линии.</p>
            <canvas id="timelineCanvas" width="1100" height="380"></canvas>
        </div>
        
        {f'''
        <div class="distributions">
            <h2>Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{distributions.get('embedding_norm', {}).get('min', 0.0):.4f}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{distributions.get('embedding_norm', {}).get('max', 0.0):.4f}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{distributions.get('embedding_norm', {}).get('mean', 0.0):.4f}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{distributions.get('embedding_norm', {}).get('std', 0.0):.4f}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{distributions.get('embedding_norm', {}).get('median', 0.0):.4f}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions.get('embedding_norm') else ''}
    </div>
    
    <script>
      const times = {json.dumps(times)};
      const norms = {json.dumps(norms)};
      const masked = {json.dumps(masked)};

      function drawTimeline() {{
        const canvas = document.getElementById('timelineCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        // Axes padding
        const padL = 55, padR = 18, padT = 18, padB = 34;
        const x0 = padL, x1 = W - padR;
        const y0 = H - padB, y1 = padT;

        // Background
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, W, H);

        // If no data, draw message
        if (!times.length) {{
          ctx.fillStyle = '#666';
          ctx.font = '16px Arial';
          ctx.fillText('No timeline data available', 20, 30);
          return;
        }}

        // Compute min/max over valid points
        let tMin = Math.min(...times), tMax = Math.max(...times);
        let validNorms = norms
          .map((v, i) => (!masked[i] && typeof v === 'number' && isFinite(v)) ? v : null)
          .filter(v => v !== null);
        if (!validNorms.length) {{
          ctx.fillStyle = '#666';
          ctx.font = '16px Arial';
          ctx.fillText('All segments masked / no finite norms', 20, 30);
          return;
        }}
        let nMin = Math.min(...validNorms), nMax = Math.max(...validNorms);
        if (tMax === tMin) tMax = tMin + 1.0;
        if (nMax === nMin) nMax = nMin + 1e-6;

        const xScale = (t) => x0 + (t - tMin) * (x1 - x0) / (tMax - tMin);
        const yScale = (n) => y0 - (n - nMin) * (y0 - y1) / (nMax - nMin);

        // Grid
        ctx.strokeStyle = '#eef2f7';
        ctx.lineWidth = 1;
        for (let k = 0; k <= 5; k++) {{
          const y = y1 + k * (y0 - y1) / 5;
          ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
        }}

        // Axes
        ctx.strokeStyle = '#9aa4b2';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x0, y1); ctx.lineTo(x0, y0); ctx.lineTo(x1, y0);
        ctx.stroke();

        // Labels
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.fillText('norm', 10, y1 + 5);
        ctx.fillText('time (s)', x1 - 55, H - 10);

        // Line with gaps for masked points
        ctx.strokeStyle = 'rgb(75, 192, 192)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        let penDown = false;
        for (let i = 0; i < times.length; i++) {{
          const m = !!masked[i];
          const v = norms[i];
          const ok = (!m && typeof v === 'number' && isFinite(v));
          if (!ok) {{
            penDown = false;
            continue;
          }}
          const x = xScale(times[i]);
          const y = yScale(v);
          if (!penDown) {{
            ctx.moveTo(x, y);
            penDown = true;
          }} else {{
            ctx.lineTo(x, y);
          }}
        }}
        ctx.stroke();

        // Dots
        for (let i = 0; i < times.length; i++) {{
          const m = !!masked[i];
          const v = norms[i];
          const ok = (!m && typeof v === 'number' && isFinite(v));
          if (!ok) continue;
          const x = xScale(times[i]);
          const y = yScale(v);
          ctx.fillStyle = 'rgba(75, 192, 192, 0.9)';
          ctx.beginPath(); ctx.arc(x, y, 2.6, 0, Math.PI*2); ctx.fill();
        }}
      }}

      drawTimeline();
    </script>
</body>
</html>"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Saved CLAP HTML render to {output_path}")
    return output_path


__all__ = ["render_clap_extractor", "render_clap_extractor_html"]

