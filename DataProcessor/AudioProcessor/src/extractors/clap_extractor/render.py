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
    
    render["summary"] = {
        "embedding_norm": features.get("clap_norm", 0.0),
        "embedding_magnitude_mean": features.get("clap_magnitude_mean", 0.0),
        "embedding_magnitude_std": features.get("clap_magnitude_std", 0.0),
        "segments_count": safe_int(features.get("segments_count", 0)),
        "embedding_dim": safe_int(features.get("embedding_dim", 0)),
        "embedding_present": bool(features.get("embedding_present", 0.0) > 0.5),
    }
    
    # Timeline data (if available)
    embedding_sequence = npz_data.get("embedding_sequence")
    segment_centers_sec = npz_data.get("segment_centers_sec")
    
    if embedding_sequence is not None and segment_centers_sec is not None:
        if isinstance(embedding_sequence, np.ndarray):
            embedding_sequence = embedding_sequence.tolist()
        if isinstance(segment_centers_sec, np.ndarray):
            segment_centers_sec = segment_centers_sec.tolist()
        
        # Calculate norms for each segment
        timeline = []
        for i, (center_sec, emb) in enumerate(zip(segment_centers_sec, embedding_sequence)):
            if isinstance(emb, list):
                emb_arr = np.array(emb, dtype=np.float32)
                norm = float(np.linalg.norm(emb_arr))
            else:
                norm = 0.0
            timeline.append({
                "center_sec": float(center_sec),
                "embedding_norm": norm,
                "segment_index": i,
            })
        render["timeline"] = timeline
        
        # Distribution of embedding norms
        if timeline:
            norms = [t["embedding_norm"] for t in timeline]
            render["distributions"]["embedding_norm"] = {
                "min": float(np.min(norms)) if norms else 0.0,
                "max": float(np.max(norms)) if norms else 0.0,
                "mean": float(np.mean(norms)) if norms else 0.0,
                "std": float(np.std(norms)) if norms else 0.0,
                "median": float(np.median(norms)) if norms else 0.0,
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
    from core.renderer import load_npz, extract_meta  # type: ignore
    
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_clap_extractor(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    
    # Prepare timeline data for chart
    timeline_js = ""
    if timeline:
        times = [t.get("center_sec", 0.0) for t in timeline]
        norms = [t.get("embedding_norm", 0.0) for t in timeline]
        timeline_js = f"""
        const timelineData = {{
            labels: {json.dumps([f"{t:.2f}s" for t in times])},
            datasets: [{{
                label: 'Embedding Norm',
                data: {json.dumps(norms)},
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1
            }}]
        }};
        """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CLAP Extractor Debug Render</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 1200px; margin: 0 auto; }}
        h1, h2 {{ color: #0056b3; }}
        .summary {{ background-color: #eaf4ff; padding: 15px; border-radius: 5px; margin: 20px 0; border: 1px solid #cce0ff; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .metric-card {{ background-color: #f8f9fa; padding: 12px; border-radius: 5px; border: 1px solid #dee2e6; }}
        .metric-card strong {{ color: #0056b3; display: block; margin-bottom: 5px; }}
        .metric-value {{ font-size: 1.2em; color: #333; }}
        .chart-container {{ position: relative; height: 400px; width: 100%; margin: 20px 0; }}
        .distributions {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .distributions table {{ width: 100%; border-collapse: collapse; }}
        .distributions th, .distributions td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .distributions th {{ background-color: #0056b3; color: white; }}
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
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Embedding Norm Over Time</h2>
            <canvas id="timelineChart"></canvas>
        </div>
        ''' if timeline else '<p>No timeline data available</p>'}
        
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
    
    {f'''
    <script>
        {timeline_js}
        const ctx = document.getElementById('timelineChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: timelineData,
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Embedding Norm'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Time (seconds)'
                        }}
                    }}
                }}
            }}
        }});
    </script>
    ''' if timeline else ''}
</body>
</html>"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Saved CLAP HTML render to {output_path}")
    return output_path


__all__ = ["render_clap_extractor", "render_clap_extractor_html"]

