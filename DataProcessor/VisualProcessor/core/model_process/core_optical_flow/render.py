"""
Renderer для core_optical_flow: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_core_optical_flow(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для core_optical_flow."""
    render = {
        "component": "core_optical_flow",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract flow data
    motion_norm_per_sec_mean = npz_data.get("motion_norm_per_sec_mean")
    dt_seconds = npz_data.get("dt_seconds")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    
    # Convert to numpy arrays if needed
    if motion_norm_per_sec_mean is not None:
        if isinstance(motion_norm_per_sec_mean, list):
            motion_norm_per_sec_mean = np.array(motion_norm_per_sec_mean, dtype=np.float32)
        elif isinstance(motion_norm_per_sec_mean, np.ndarray):
            motion_norm_per_sec_mean = np.asarray(motion_norm_per_sec_mean, dtype=np.float32)
        else:
            motion_norm_per_sec_mean = None
    
    if dt_seconds is not None:
        if isinstance(dt_seconds, list):
            dt_seconds = np.array(dt_seconds, dtype=np.float32)
        elif isinstance(dt_seconds, np.ndarray):
            dt_seconds = np.asarray(dt_seconds, dtype=np.float32)
        else:
            dt_seconds = None
    
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
    
    # Summary statistics
    if motion_norm_per_sec_mean is not None and motion_norm_per_sec_mean.size > 0:
        n_frames = len(motion_norm_per_sec_mean)
        
        # Filter valid values (exclude NaN)
        valid_motion = motion_norm_per_sec_mean[np.isfinite(motion_norm_per_sec_mean)]
        valid_dt = dt_seconds[np.isfinite(dt_seconds)] if dt_seconds is not None else None
        
        render["summary"] = {
            "frames_count": int(n_frames),
            "valid_motion_count": int(len(valid_motion)),
        }
        
        if valid_motion.size > 0:
            render["summary"]["motion_mean"] = float(np.mean(valid_motion))
            render["summary"]["motion_std"] = float(np.std(valid_motion))
            render["summary"]["motion_min"] = float(np.min(valid_motion))
            render["summary"]["motion_max"] = float(np.max(valid_motion))
            render["summary"]["motion_median"] = float(np.median(valid_motion))
        
        if valid_dt is not None and valid_dt.size > 0:
            render["summary"]["dt_mean"] = float(np.mean(valid_dt))
            render["summary"]["dt_std"] = float(np.std(valid_dt))
            render["summary"]["dt_min"] = float(np.min(valid_dt))
            render["summary"]["dt_max"] = float(np.max(valid_dt))
    
    # Timeline data (per-frame statistics)
    if motion_norm_per_sec_mean is not None and times_s is not None and frame_indices is not None:
        n = len(motion_norm_per_sec_mean)
        timeline = []
        
        for i in range(n):
            frame_idx = int(frame_indices[i]) if i < len(frame_indices) else i
            time_sec = float(times_s[i]) if i < len(times_s) else 0.0
            motion_val = float(motion_norm_per_sec_mean[i]) if np.isfinite(motion_norm_per_sec_mean[i]) else None
            dt_val = float(dt_seconds[i]) if dt_seconds is not None and i < len(dt_seconds) and np.isfinite(dt_seconds[i]) else None
            
            timeline.append({
                "frame_index": frame_idx,
                "time_sec": time_sec,
                "motion_norm_per_sec": motion_val,
                "dt_seconds": dt_val,
            })
        
        render["timeline"] = timeline
    
    # Distributions
    distributions = {}
    
    if motion_norm_per_sec_mean is not None:
        valid_motion = motion_norm_per_sec_mean[np.isfinite(motion_norm_per_sec_mean)]
        if valid_motion.size > 0:
            distributions["motion_norm_per_sec"] = {
                "min": float(np.min(valid_motion)),
                "max": float(np.max(valid_motion)),
                "mean": float(np.mean(valid_motion)),
                "std": float(np.std(valid_motion)),
                "median": float(np.median(valid_motion)),
                "p25": float(np.percentile(valid_motion, 25)),
                "p75": float(np.percentile(valid_motion, 75)),
                "p05": float(np.percentile(valid_motion, 5)),
                "p95": float(np.percentile(valid_motion, 95)),
            }
    
    if dt_seconds is not None:
        valid_dt = dt_seconds[np.isfinite(dt_seconds)]
        if valid_dt.size > 0:
            distributions["dt_seconds"] = {
                "min": float(np.min(valid_dt)),
                "max": float(np.max(valid_dt)),
                "mean": float(np.mean(valid_dt)),
                "std": float(np.std(valid_dt)),
                "median": float(np.median(valid_dt)),
                "p25": float(np.percentile(valid_dt, 25)),
                "p75": float(np.percentile(valid_dt, 75)),
                "p05": float(np.percentile(valid_dt, 5)),
                "p95": float(np.percentile(valid_dt, 95)),
            }
    
    render["distributions"] = distributions
    
    return render


def render_core_optical_flow_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага core_optical_flow результатов.
    
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
    if str(vp_root / "core" / "model_process") not in sys.path:
        sys.path.insert(0, str(vp_root / "core" / "model_process"))
    
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
    render = render_core_optical_flow(npz_data, meta)
    
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
        motion_vals = [t.get("motion_norm_per_sec") for t in timeline if t.get("motion_norm_per_sec") is not None]
        dt_vals = [t.get("dt_seconds") for t in timeline if t.get("dt_seconds") is not None]
        
        # Build datasets array
        datasets = []
        
        if motion_vals:
            datasets.append({
                "label": "Motion Norm Per Sec",
                "data": motion_vals,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "tension": 0.1,
                "yAxisID": "y"
            })
        
        if dt_vals:
            datasets.append({
                "label": "DT Seconds",
                "data": dt_vals,
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
                            text: 'DT Seconds'
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
    <title>Core Optical Flow Debug Render</title>
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
        <h1>Core Optical Flow Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Valid Motion Count</strong>
                    <span class="metric-value">{summary.get('valid_motion_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Motion Mean</strong>
                    <span class="metric-value">{summary.get('motion_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Motion Std</strong>
                    <span class="metric-value">{summary.get('motion_std', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Motion Min</strong>
                    <span class="metric-value">{summary.get('motion_min', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Motion Max</strong>
                    <span class="metric-value">{summary.get('motion_max', 0.0):.4f}</span>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Motion Norm Per Second Over Time</h2>
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
                        <th>Motion Norm Per Sec</th>
                        <th>DT Seconds</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('motion_norm_per_sec', 'min')}</td>
                        <td>{format_dist_value('dt_seconds', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('motion_norm_per_sec', 'max')}</td>
                        <td>{format_dist_value('dt_seconds', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('motion_norm_per_sec', 'mean')}</td>
                        <td>{format_dist_value('dt_seconds', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('motion_norm_per_sec', 'std')}</td>
                        <td>{format_dist_value('dt_seconds', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('motion_norm_per_sec', 'median')}</td>
                        <td>{format_dist_value('dt_seconds', 'median')}</td>
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
                            text: 'Motion Norm Per Sec'
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
    logger.info(f"Saved Core Optical Flow HTML render to {rel_output_path}")
    return output_path


__all__ = ["render_core_optical_flow", "render_core_optical_flow_html"]

