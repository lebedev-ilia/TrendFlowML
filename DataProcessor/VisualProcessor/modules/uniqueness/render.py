"""
Renderer для uniqueness: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_uniqueness(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для uniqueness."""
    render = {
        "component": "uniqueness",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract uniqueness data
    max_sim_to_other = npz_data.get("max_sim_to_other")
    cos_dist_next = npz_data.get("cos_dist_next")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    features = npz_data.get("features")
    
    # Convert to numpy arrays if needed
    if max_sim_to_other is not None:
        if isinstance(max_sim_to_other, list):
            max_sim_to_other = np.array(max_sim_to_other, dtype=np.float32)
        elif isinstance(max_sim_to_other, np.ndarray):
            max_sim_to_other = np.asarray(max_sim_to_other, dtype=np.float32)
        else:
            max_sim_to_other = None
    
    if cos_dist_next is not None:
        if isinstance(cos_dist_next, list):
            cos_dist_next = np.array(cos_dist_next, dtype=np.float32)
        elif isinstance(cos_dist_next, np.ndarray):
            cos_dist_next = np.asarray(cos_dist_next, dtype=np.float32)
        else:
            cos_dist_next = None
    
    if times_s is not None:
        if isinstance(times_s, list):
            times_s = np.array(times_s, dtype=np.float32)
        elif isinstance(times_s, np.ndarray):
            times_s = np.asarray(times_s, dtype=np.float32)
        else:
            times_s = None
    
    if frame_indices is not None:
        if isinstance(frame_indices, list):
            frame_indices = np.array(frame_indices, dtype=np.int32)
        elif isinstance(frame_indices, np.ndarray):
            frame_indices = np.asarray(frame_indices, dtype=np.int32)
        else:
            frame_indices = None
    
    # Extract features dict
    if features is not None:
        if isinstance(features, np.ndarray) and features.dtype == object:
            if features.size == 1:
                features = features.item() if isinstance(features.item(), dict) else {}
            else:
                features = features.item() if hasattr(features, 'item') else {}
        if not isinstance(features, dict):
            features = {}
    else:
        features = {}
    
    # Summary statistics
    if max_sim_to_other is not None and max_sim_to_other.size > 0:
        valid_sim = max_sim_to_other[np.isfinite(max_sim_to_other)]
        
        render["summary"] = {
            "frames_count": int(max_sim_to_other.shape[0]),
            "repetition_ratio": float(features.get("repetition_ratio", 0.0)) if isinstance(features.get("repetition_ratio"), (int, float)) else 0.0,
            "diversity_score": float(features.get("diversity_score", 0.0)) if isinstance(features.get("diversity_score"), (int, float)) else 0.0,
            "pairwise_sim_mean": float(features.get("pairwise_sim_mean", 0.0)) if isinstance(features.get("pairwise_sim_mean"), (int, float)) else 0.0,
            "pairwise_sim_p95": float(features.get("pairwise_sim_p95", 0.0)) if isinstance(features.get("pairwise_sim_p95"), (int, float)) else 0.0,
            "temporal_change_mean": float(features.get("temporal_change_mean", 0.0)) if isinstance(features.get("temporal_change_mean"), (int, float)) else 0.0,
            "repeat_threshold_used": float(features.get("repeat_threshold_used", 0.97)) if isinstance(features.get("repeat_threshold_used"), (int, float)) else 0.97,
            "repeat_threshold_mode": str(features.get("repeat_threshold_mode", "fixed")),
        }
        
        if valid_sim.size > 0:
            render["summary"]["max_sim_to_other_mean"] = float(np.mean(valid_sim))
            render["summary"]["max_sim_to_other_std"] = float(np.std(valid_sim))
            render["summary"]["max_sim_to_other_min"] = float(np.min(valid_sim))
            render["summary"]["max_sim_to_other_max"] = float(np.max(valid_sim))
            render["summary"]["max_sim_to_other_median"] = float(np.median(valid_sim))
    
    # Timeline data
    if max_sim_to_other is not None and times_s is not None and frame_indices is not None:
        n = len(max_sim_to_other)
        timeline = []
        
        for i in range(n):
            frame_idx = int(frame_indices[i]) if i < len(frame_indices) else i
            time_sec = float(times_s[i]) if i < len(times_s) else 0.0
            sim_val = float(max_sim_to_other[i]) if np.isfinite(max_sim_to_other[i]) else None
            
            # cos_dist_next has N-1 elements
            cos_dist_val = None
            if cos_dist_next is not None and i < len(cos_dist_next):
                cos_dist_val = float(cos_dist_next[i]) if np.isfinite(cos_dist_next[i]) else None
            
            timeline.append({
                "frame_index": frame_idx,
                "time_sec": time_sec,
                "max_sim_to_other": sim_val,
                "cos_dist_next": cos_dist_val,
            })
        
        render["timeline"] = timeline
    
    # Distributions
    distributions = {}
    
    if max_sim_to_other is not None:
        valid_sim = max_sim_to_other[np.isfinite(max_sim_to_other)]
        if valid_sim.size > 0:
            distributions["max_sim_to_other"] = {
                "min": float(np.min(valid_sim)),
                "max": float(np.max(valid_sim)),
                "mean": float(np.mean(valid_sim)),
                "std": float(np.std(valid_sim)),
                "median": float(np.median(valid_sim)),
                "p25": float(np.percentile(valid_sim, 25)),
                "p75": float(np.percentile(valid_sim, 75)),
                "p05": float(np.percentile(valid_sim, 5)),
                "p95": float(np.percentile(valid_sim, 95)),
            }
    
    if cos_dist_next is not None:
        valid_dist = cos_dist_next[np.isfinite(cos_dist_next)]
        if valid_dist.size > 0:
            distributions["cos_dist_next"] = {
                "min": float(np.min(valid_dist)),
                "max": float(np.max(valid_dist)),
                "mean": float(np.mean(valid_dist)),
                "std": float(np.std(valid_dist)),
                "median": float(np.median(valid_dist)),
                "p25": float(np.percentile(valid_dist, 25)),
                "p75": float(np.percentile(valid_dist, 75)),
                "p05": float(np.percentile(valid_dist, 5)),
                "p95": float(np.percentile(valid_dist, 95)),
            }
    
    render["distributions"] = distributions
    
    return render


def render_uniqueness_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага uniqueness результатов.
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    vp_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(vp_root / "modules") not in sys.path:
        sys.path.insert(0, str(vp_root / "modules"))
    
    # Try to import from utils if renderer exists
    try:
        from utils.renderer import load_npz, extract_meta  # type: ignore
    except ImportError:
        # Fallback: direct load
        def load_npz(path: str):
            data = np.load(path, allow_pickle=True)
            result = {}
            for key in data.files:
                arr = data[key]
                if isinstance(arr, np.ndarray):
                    if arr.dtype == object:
                        result[key] = arr.item() if arr.size == 1 else arr.tolist()
                    else:
                        result[key] = arr.tolist() if arr.size > 0 else []
                else:
                    result[key] = arr
            return result
        
        def extract_meta(npz_data: Dict[str, Any]) -> Dict[str, Any]:
            meta = npz_data.get("meta")
            if isinstance(meta, np.ndarray) and meta.dtype == object:
                return meta.item() if meta.size == 1 else meta.tolist()
            return meta if isinstance(meta, dict) else {}
    
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_uniqueness(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    
    # Helper function to format distribution values
    def format_dist_value(dist_key, stat_key):
        dist = distributions.get(dist_key)
        if dist and stat_key in dist:
            return f"{dist[stat_key]:.4f}"
        return "N/A"
    
    # Prepare timeline data for chart
    timeline_js = ""
    y1_scale_js = ""
    if timeline:
        times = [t.get("time_sec", 0.0) for t in timeline]
        max_sims = [t.get("max_sim_to_other") for t in timeline if t.get("max_sim_to_other") is not None]
        cos_dists = [t.get("cos_dist_next") for t in timeline if t.get("cos_dist_next") is not None]
        
        # Build datasets array
        datasets = []
        
        if max_sims:
            datasets.append({
                "label": "Max Similarity to Other",
                "data": max_sims,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "tension": 0.1,
                "yAxisID": "y"
            })
        
        if cos_dists:
            datasets.append({
                "label": "Cosine Distance to Next",
                "data": cos_dists,
                "borderColor": "rgb(255, 99, 132)",
                "backgroundColor": "rgba(255, 99, 132, 0.2)",
                "tension": 0.1,
                "yAxisID": "y1"
            })
            y1_scale_js = """,
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Cosine Distance'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }"""
        
        if datasets:
            # Format time labels
            time_labels = [f"{t:.2f}s" for t in times]
            timeline_js = f"""
        const timelineData = {{
            labels: {json.dumps(time_labels)},
            datasets: {json.dumps(datasets)}
        }};
        """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Uniqueness Debug Render</title>
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
        <h1>Uniqueness Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Repetition Ratio</strong>
                    <span class="metric-value">{summary.get('repetition_ratio', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Diversity Score</strong>
                    <span class="metric-value">{summary.get('diversity_score', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Pairwise Sim Mean</strong>
                    <span class="metric-value">{summary.get('pairwise_sim_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Pairwise Sim P95</strong>
                    <span class="metric-value">{summary.get('pairwise_sim_p95', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Temporal Change Mean</strong>
                    <span class="metric-value">{summary.get('temporal_change_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Repeat Threshold</strong>
                    <span class="metric-value">{summary.get('repeat_threshold_used', 0.97):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Threshold Mode</strong>
                    <span class="metric-value">{summary.get('repeat_threshold_mode', 'fixed')}</span>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Uniqueness Metrics Over Time</h2>
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
                        <th>Max Sim to Other</th>
                        <th>Cosine Dist Next</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('max_sim_to_other', 'min')}</td>
                        <td>{format_dist_value('cos_dist_next', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('max_sim_to_other', 'max')}</td>
                        <td>{format_dist_value('cos_dist_next', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('max_sim_to_other', 'mean')}</td>
                        <td>{format_dist_value('cos_dist_next', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('max_sim_to_other', 'std')}</td>
                        <td>{format_dist_value('cos_dist_next', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('max_sim_to_other', 'median')}</td>
                        <td>{format_dist_value('cos_dist_next', 'median')}</td>
                    </tr>
                    <tr>
                        <td><strong>P25</strong></td>
                        <td>{format_dist_value('max_sim_to_other', 'p25')}</td>
                        <td>{format_dist_value('cos_dist_next', 'p25')}</td>
                    </tr>
                    <tr>
                        <td><strong>P75</strong></td>
                        <td>{format_dist_value('max_sim_to_other', 'p75')}</td>
                        <td>{format_dist_value('cos_dist_next', 'p75')}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions else ''}
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
                            text: 'Max Similarity to Other'
                        }}
                    }}{y1_scale_js}
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
    
    # Show relative path for cleaner output
    rel_output_path = os.path.relpath(output_path, os.getcwd()) if os.path.exists(output_path) else output_path
    logger.info(f"Saved Uniqueness HTML render to {rel_output_path}")
    return output_path


__all__ = ["render_uniqueness", "render_uniqueness_html"]

