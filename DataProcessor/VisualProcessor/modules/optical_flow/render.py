"""
Renderer для optical_flow: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_optical_flow(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для optical_flow."""
    render = {
        "component": "optical_flow",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract optical flow data
    motion_norm_per_sec_mean = npz_data.get("motion_norm_per_sec_mean")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    features = npz_data.get("features")
    
    # Convert to numpy arrays if needed
    if motion_norm_per_sec_mean is not None:
        if isinstance(motion_norm_per_sec_mean, list):
            motion_norm_per_sec_mean = np.array(motion_norm_per_sec_mean, dtype=np.float32)
        elif isinstance(motion_norm_per_sec_mean, np.ndarray):
            motion_norm_per_sec_mean = np.asarray(motion_norm_per_sec_mean, dtype=np.float32)
        else:
            motion_norm_per_sec_mean = None
    
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
            features = features.item() if features.size == 1 else features.tolist()
        if not isinstance(features, dict):
            features = {}
    else:
        features = {}
    
    # Summary statistics
    if motion_norm_per_sec_mean is not None and motion_norm_per_sec_mean.size > 0:
        # Ignore first element (usually 0.0, no previous frame)
        curve_for_stats = motion_norm_per_sec_mean[1:] if motion_norm_per_sec_mean.size >= 2 else motion_norm_per_sec_mean
        valid_curve = curve_for_stats[np.isfinite(curve_for_stats)]
        
        render["summary"] = {
            "frames_count": int(motion_norm_per_sec_mean.size),
            "motion_curve_mean": float(features.get("motion_curve_mean", np.nanmean(valid_curve) if valid_curve.size > 0 else np.nan)),
            "motion_curve_median": float(features.get("motion_curve_median", np.nanmedian(valid_curve) if valid_curve.size > 0 else np.nan)),
            "motion_curve_p90": float(features.get("motion_curve_p90", np.nanpercentile(valid_curve, 90) if valid_curve.size > 0 else np.nan)),
            "motion_curve_variance": float(features.get("motion_curve_variance", np.nanvar(valid_curve) if valid_curve.size > 0 else np.nan)),
        }
        
        if valid_curve.size > 0:
            render["summary"]["motion_curve_min"] = float(np.nanmin(valid_curve))
            render["summary"]["motion_curve_max"] = float(np.nanmax(valid_curve))
            render["summary"]["motion_curve_std"] = float(np.nanstd(valid_curve))
            render["summary"]["motion_curve_p25"] = float(np.nanpercentile(valid_curve, 25))
            render["summary"]["motion_curve_p75"] = float(np.nanpercentile(valid_curve, 75))
            render["summary"]["motion_curve_p05"] = float(np.nanpercentile(valid_curve, 5))
            render["summary"]["motion_curve_p95"] = float(np.nanpercentile(valid_curve, 95))
    
    # Timeline data (per-frame motion)
    if motion_norm_per_sec_mean is not None and times_s is not None and frame_indices is not None:
        n = len(motion_norm_per_sec_mean)
        timeline = []
        
        for i in range(n):
            frame_idx = int(frame_indices[i]) if i < len(frame_indices) else i
            time_sec = float(times_s[i]) if i < len(times_s) else 0.0
            motion_val = float(motion_norm_per_sec_mean[i]) if np.isfinite(motion_norm_per_sec_mean[i]) else None
            
            timeline.append({
                "frame_index": frame_idx,
                "time_sec": time_sec,
                "motion_norm_per_sec_mean": motion_val,
            })
        
        render["timeline"] = timeline
    
    # Distributions
    distributions = {}
    
    if motion_norm_per_sec_mean is not None:
        # Ignore first element for statistics
        curve_for_stats = motion_norm_per_sec_mean[1:] if motion_norm_per_sec_mean.size >= 2 else motion_norm_per_sec_mean
        valid_curve = curve_for_stats[np.isfinite(curve_for_stats)]
        
        if valid_curve.size > 0:
            distributions["motion_norm_per_sec_mean"] = {
                "min": float(np.min(valid_curve)),
                "max": float(np.max(valid_curve)),
                "mean": float(np.mean(valid_curve)),
                "std": float(np.std(valid_curve)),
                "median": float(np.median(valid_curve)),
                "p25": float(np.percentile(valid_curve, 25)),
                "p75": float(np.percentile(valid_curve, 75)),
                "p05": float(np.percentile(valid_curve, 5)),
                "p95": float(np.percentile(valid_curve, 95)),
            }
    
    render["distributions"] = distributions
    
    return render


def render_optical_flow_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага optical_flow результатов.
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    vp_root = Path(__file__).resolve().parent.parent.parent
    if str(vp_root) not in sys.path:
        sys.path.insert(0, str(vp_root))
    
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
    render = render_optical_flow(npz_data, meta)
    
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
    if timeline:
        times = [t.get("time_sec", 0.0) for t in timeline]
        motion_values = [t.get("motion_norm_per_sec_mean") for t in timeline if t.get("motion_norm_per_sec_mean") is not None]
        
        # Build datasets array
        datasets = []
        
        if motion_values:
            datasets.append({
                "label": "Motion Norm (px/sec)",
                "data": motion_values,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "tension": 0.1,
            })
        
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
    <title>Optical Flow Debug Render</title>
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
        <h1>Optical Flow Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Motion Mean</strong>
                    <span class="metric-value">{summary.get('motion_curve_mean', 0.0):.4f} px/sec</span>
                </div>
                <div class="metric-card">
                    <strong>Motion Median</strong>
                    <span class="metric-value">{summary.get('motion_curve_median', 0.0):.4f} px/sec</span>
                </div>
                <div class="metric-card">
                    <strong>Motion P90</strong>
                    <span class="metric-value">{summary.get('motion_curve_p90', 0.0):.4f} px/sec</span>
                </div>
                <div class="metric-card">
                    <strong>Motion Variance</strong>
                    <span class="metric-value">{summary.get('motion_curve_variance', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Motion Std</strong>
                    <span class="metric-value">{summary.get('motion_curve_std', 0.0):.4f} px/sec</span>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Motion Norm Over Time</h2>
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
                        <th>Motion Norm (px/sec)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('motion_norm_per_sec_mean', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('motion_norm_per_sec_mean', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('motion_norm_per_sec_mean', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('motion_norm_per_sec_mean', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('motion_norm_per_sec_mean', 'median')}</td>
                    </tr>
                    <tr>
                        <td><strong>P25</strong></td>
                        <td>{format_dist_value('motion_norm_per_sec_mean', 'p25')}</td>
                    </tr>
                    <tr>
                        <td><strong>P75</strong></td>
                        <td>{format_dist_value('motion_norm_per_sec_mean', 'p75')}</td>
                    </tr>
                    <tr>
                        <td><strong>P05</strong></td>
                        <td>{format_dist_value('motion_norm_per_sec_mean', 'p05')}</td>
                    </tr>
                    <tr>
                        <td><strong>P95</strong></td>
                        <td>{format_dist_value('motion_norm_per_sec_mean', 'p95')}</td>
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
                            text: 'Motion Norm (px/sec)'
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
    
    # Show relative path for cleaner output
    rel_output_path = os.path.relpath(output_path, os.getcwd()) if os.path.exists(output_path) else output_path
    logger.info(f"Saved Optical Flow HTML render to {rel_output_path}")
    return output_path


__all__ = ["render_optical_flow", "render_optical_flow_html"]

