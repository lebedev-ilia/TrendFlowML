"""
Renderer для core_depth_midas: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_core_depth_midas(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для core_depth_midas."""
    render = {
        "component": "core_depth_midas",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract depth data
    depth_maps = npz_data.get("depth_maps")
    depth_mean = npz_data.get("depth_mean")
    depth_std = npz_data.get("depth_std")
    depth_p05 = npz_data.get("depth_p05")
    depth_p95 = npz_data.get("depth_p95")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    
    # Convert to numpy arrays if needed
    if depth_maps is not None:
        if isinstance(depth_maps, list):
            depth_maps = np.array(depth_maps, dtype=np.float32)
        elif isinstance(depth_maps, np.ndarray):
            depth_maps = np.asarray(depth_maps, dtype=np.float32)
        else:
            depth_maps = None
    
    if depth_mean is not None:
        if isinstance(depth_mean, list):
            depth_mean = np.array(depth_mean, dtype=np.float32)
        elif isinstance(depth_mean, np.ndarray):
            depth_mean = np.asarray(depth_mean, dtype=np.float32)
        else:
            depth_mean = None
    
    if depth_std is not None:
        if isinstance(depth_std, list):
            depth_std = np.array(depth_std, dtype=np.float32)
        elif isinstance(depth_std, np.ndarray):
            depth_std = np.asarray(depth_std, dtype=np.float32)
        else:
            depth_std = None
    
    if depth_p05 is not None:
        if isinstance(depth_p05, list):
            depth_p05 = np.array(depth_p05, dtype=np.float32)
        elif isinstance(depth_p05, np.ndarray):
            depth_p05 = np.asarray(depth_p05, dtype=np.float32)
        else:
            depth_p05 = None
    
    if depth_p95 is not None:
        if isinstance(depth_p95, list):
            depth_p95 = np.array(depth_p95, dtype=np.float32)
        elif isinstance(depth_p95, np.ndarray):
            depth_p95 = np.asarray(depth_p95, dtype=np.float32)
        else:
            depth_p95 = None
    
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
    if depth_maps is not None and depth_maps.size > 0:
        n_frames = depth_maps.shape[0] if depth_maps.ndim >= 2 else 1
        if depth_maps.ndim >= 2:
            depth_h, depth_w = depth_maps.shape[-2], depth_maps.shape[-1]
        else:
            depth_h, depth_w = 1, depth_maps.size
        
        # Flatten all depth maps for global statistics
        all_depths = depth_maps[np.isfinite(depth_maps)]
        
        render["summary"] = {
            "frames_count": int(n_frames),
            "depth_map_height": int(depth_h),
            "depth_map_width": int(depth_w),
            "depth_map_shape": [int(n_frames), int(depth_h), int(depth_w)],
        }
        
        if all_depths.size > 0:
            render["summary"]["global_depth_mean"] = float(np.mean(all_depths))
            render["summary"]["global_depth_std"] = float(np.std(all_depths))
            render["summary"]["global_depth_min"] = float(np.min(all_depths))
            render["summary"]["global_depth_max"] = float(np.max(all_depths))
            render["summary"]["global_depth_median"] = float(np.median(all_depths))
    
    # Timeline data (per-frame statistics)
    if depth_mean is not None and times_s is not None and frame_indices is not None:
        n = len(depth_mean)
        timeline = []
        
        for i in range(n):
            frame_idx = int(frame_indices[i]) if i < len(frame_indices) else i
            time_sec = float(times_s[i]) if i < len(times_s) else 0.0
            mean_val = float(depth_mean[i]) if np.isfinite(depth_mean[i]) else None
            std_val = float(depth_std[i]) if depth_std is not None and i < len(depth_std) and np.isfinite(depth_std[i]) else None
            p05_val = float(depth_p05[i]) if depth_p05 is not None and i < len(depth_p05) and np.isfinite(depth_p05[i]) else None
            p95_val = float(depth_p95[i]) if depth_p95 is not None and i < len(depth_p95) and np.isfinite(depth_p95[i]) else None
            
            timeline.append({
                "frame_index": frame_idx,
                "time_sec": time_sec,
                "depth_mean": mean_val,
                "depth_std": std_val,
                "depth_p05": p05_val,
                "depth_p95": p95_val,
            })
        
        render["timeline"] = timeline
    
    # Distributions
    distributions = {}
    
    if depth_mean is not None:
        valid_means = depth_mean[np.isfinite(depth_mean)]
        if valid_means.size > 0:
            distributions["depth_mean"] = {
                "min": float(np.min(valid_means)),
                "max": float(np.max(valid_means)),
                "mean": float(np.mean(valid_means)),
                "std": float(np.std(valid_means)),
                "median": float(np.median(valid_means)),
                "p25": float(np.percentile(valid_means, 25)),
                "p75": float(np.percentile(valid_means, 75)),
                "p05": float(np.percentile(valid_means, 5)),
                "p95": float(np.percentile(valid_means, 95)),
            }
    
    if depth_std is not None:
        valid_stds = depth_std[np.isfinite(depth_std)]
        if valid_stds.size > 0:
            distributions["depth_std"] = {
                "min": float(np.min(valid_stds)),
                "max": float(np.max(valid_stds)),
                "mean": float(np.mean(valid_stds)),
                "std": float(np.std(valid_stds)),
                "median": float(np.median(valid_stds)),
                "p25": float(np.percentile(valid_stds, 25)),
                "p75": float(np.percentile(valid_stds, 75)),
                "p05": float(np.percentile(valid_stds, 5)),
                "p95": float(np.percentile(valid_stds, 95)),
            }
    
    if depth_maps is not None and depth_maps.size > 0:
        all_depths = depth_maps[np.isfinite(depth_maps)]
        if all_depths.size > 0:
            distributions["depth_maps"] = {
                "min": float(np.min(all_depths)),
                "max": float(np.max(all_depths)),
                "mean": float(np.mean(all_depths)),
                "std": float(np.std(all_depths)),
                "median": float(np.median(all_depths)),
                "p25": float(np.percentile(all_depths, 25)),
                "p75": float(np.percentile(all_depths, 75)),
                "p05": float(np.percentile(all_depths, 5)),
                "p95": float(np.percentile(all_depths, 95)),
            }
    
    render["distributions"] = distributions
    
    return render


def render_core_depth_midas_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага core_depth_midas результатов.
    
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
    render = render_core_depth_midas(npz_data, meta)
    
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
        depth_means = [t.get("depth_mean") for t in timeline if t.get("depth_mean") is not None]
        depth_stds = [t.get("depth_std") for t in timeline if t.get("depth_std") is not None]
        
        # Build datasets array
        datasets = []
        
        if depth_means:
            datasets.append({
                "label": "Depth Mean",
                "data": depth_means,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "tension": 0.1,
                "yAxisID": "y"
            })
        
        if depth_stds:
            datasets.append({
                "label": "Depth Std",
                "data": depth_stds,
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
                            text: 'Depth Std'
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
    <title>Core Depth MiDaS Debug Render</title>
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
        <h1>Core Depth MiDaS Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Depth Map Size</strong>
                    <span class="metric-value">{summary.get('depth_map_height', 0)}×{summary.get('depth_map_width', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Global Depth Mean</strong>
                    <span class="metric-value">{summary.get('global_depth_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Global Depth Std</strong>
                    <span class="metric-value">{summary.get('global_depth_std', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Global Depth Min</strong>
                    <span class="metric-value">{summary.get('global_depth_min', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Global Depth Max</strong>
                    <span class="metric-value">{summary.get('global_depth_max', 0.0):.4f}</span>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Depth Statistics Over Time</h2>
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
                        <th>Depth Mean</th>
                        <th>Depth Std</th>
                        <th>Depth Maps (All)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('depth_mean', 'min')}</td>
                        <td>{format_dist_value('depth_std', 'min')}</td>
                        <td>{format_dist_value('depth_maps', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('depth_mean', 'max')}</td>
                        <td>{format_dist_value('depth_std', 'max')}</td>
                        <td>{format_dist_value('depth_maps', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('depth_mean', 'mean')}</td>
                        <td>{format_dist_value('depth_std', 'mean')}</td>
                        <td>{format_dist_value('depth_maps', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('depth_mean', 'std')}</td>
                        <td>{format_dist_value('depth_std', 'std')}</td>
                        <td>{format_dist_value('depth_maps', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('depth_mean', 'median')}</td>
                        <td>{format_dist_value('depth_std', 'median')}</td>
                        <td>{format_dist_value('depth_maps', 'median')}</td>
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
                            text: 'Depth Mean'
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
    logger.info(f"Saved Core Depth MiDaS HTML render to {rel_output_path}")
    return output_path


__all__ = ["render_core_depth_midas", "render_core_depth_midas_html"]

