"""
Renderer для frames_composition: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_frames_composition(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для frames_composition."""
    render = {
        "component": "frames_composition",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract data
    frame_indices = npz_data.get("frame_indices")
    times_s = npz_data.get("times_s")
    feature_names = npz_data.get("feature_names")
    feature_values = npz_data.get("feature_values")
    frame_feature_names = npz_data.get("frame_feature_names")
    frame_feature_values = npz_data.get("frame_feature_values")
    
    # Convert to numpy arrays if needed
    if frame_indices is not None:
        if isinstance(frame_indices, list):
            frame_indices = np.array(frame_indices, dtype=np.int32)
        elif isinstance(frame_indices, np.ndarray):
            frame_indices = np.asarray(frame_indices, dtype=np.int32)
        else:
            frame_indices = None
    
    if times_s is not None:
        if isinstance(times_s, list):
            times_s = np.array(times_s, dtype=np.float32)
        elif isinstance(times_s, np.ndarray):
            times_s = np.asarray(times_s, dtype=np.float32)
        else:
            times_s = None
    
    if frame_feature_values is not None:
        if isinstance(frame_feature_values, list):
            frame_feature_values = np.array(frame_feature_values, dtype=np.float32)
        elif isinstance(frame_feature_values, np.ndarray):
            frame_feature_values = np.asarray(frame_feature_values, dtype=np.float32)
        else:
            frame_feature_values = None
    
    # Extract feature names
    if feature_names is not None:
        if isinstance(feature_names, np.ndarray) and feature_names.dtype == object:
            feature_names = [str(f) for f in feature_names.flatten()]
        elif isinstance(feature_names, list):
            feature_names = [str(f) for f in feature_names]
        else:
            feature_names = []
    else:
        feature_names = []
    
    if frame_feature_names is not None:
        if isinstance(frame_feature_names, np.ndarray) and frame_feature_names.dtype == object:
            frame_feature_names = [str(f) for f in frame_feature_names.flatten()]
        elif isinstance(frame_feature_names, list):
            frame_feature_names = [str(f) for f in frame_feature_names]
        else:
            frame_feature_names = []
    else:
        frame_feature_names = []
    
    # Summary statistics from video-level features
    if feature_names and feature_values is not None:
        if isinstance(feature_values, list):
            feature_values = np.array(feature_values, dtype=np.float32)
        elif isinstance(feature_values, np.ndarray):
            feature_values = np.asarray(feature_values, dtype=np.float32)
        
        # Build summary from video-level features
        summary = {
            "frames_count": int(len(frame_indices)) if frame_indices is not None else 0,
        }
        
        # Extract key metrics from feature_names/feature_values
        for i, name in enumerate(feature_names):
            if i < len(feature_values):
                val = float(feature_values[i]) if np.isfinite(feature_values[i]) else None
                if val is not None:
                    # Store important metrics
                    if "has_faces" in name:
                        summary["has_faces"] = bool(val > 0.5)
                    elif "frames_n" in name:
                        summary["frames_n"] = int(val)
                    elif "style_dominant_id" in name:
                        summary["style_dominant_id"] = int(val)
                    elif "style_prob__" in name:
                        if "style_probabilities" not in summary:
                            summary["style_probabilities"] = {}
                        style_name = name.replace("style_prob__", "").replace("__mean", "")
                        summary["style_probabilities"][style_name] = val
        
        render["summary"] = summary
    
    # Define key features for timeline and distributions
    key_features = {
        "saliency_center_offset": None,
        "symmetry_score": None,
        "negative_space_ratio": None,
        "edge_density": None,
        "line_strength": None,
        "face_present": None,
        "object_count": None,
        "depth_mean": None,
        "style_minimalist": None,
        "style_cinematic": None,
        "style_vlog": None,
        "style_product_centered": None,
    }
    
    # Timeline data (per-frame features)
    timeline = []
    if frame_feature_values is not None and frame_feature_names and times_s is not None and frame_indices is not None:
        n_frames = len(frame_indices)
        n_features = len(frame_feature_names)
        
        # Find indices of key features for timeline
        
        for i, name in enumerate(frame_feature_names):
            for key in key_features:
                if key in name:
                    key_features[key] = i
                    break
        
        for i in range(n_frames):
            if i >= len(times_s) or i >= len(frame_indices):
                break
            
            frame_idx = int(frame_indices[i])
            time_sec = float(times_s[i])
            
            timeline_entry = {
                "frame_index": frame_idx,
                "time_sec": time_sec,
            }
            
            # Add key features if available
            for key, idx in key_features.items():
                if idx is not None and idx < n_features and i < frame_feature_values.shape[0]:
                    val = float(frame_feature_values[i, idx]) if np.isfinite(frame_feature_values[i, idx]) else None
                    if val is not None:
                        timeline_entry[key] = val
            
            timeline.append(timeline_entry)
    
    render["timeline"] = timeline
    
    # Distribution statistics for per-frame features
    distributions = {}
    
    if timeline and frame_feature_names:
        # Collect values for each key feature
        feature_vals_map = {}
        for key in key_features:
            if key_features[key] is not None:
                idx = key_features[key]
                vals = []
                for entry in timeline:
                    if key in entry and entry[key] is not None:
                        vals.append(entry[key])
                if vals:
                    feature_vals_map[key] = np.array(vals, dtype=np.float32)
        
        # Compute distributions
        for key, vals in feature_vals_map.items():
            if vals.size > 0:
                distributions[key] = {
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "median": float(np.median(vals)),
                    "p25": float(np.percentile(vals, 25)),
                    "p75": float(np.percentile(vals, 75)),
                    "p10": float(np.percentile(vals, 10)),
                    "p90": float(np.percentile(vals, 90)),
                }
    
    render["distributions"] = distributions
    
    return render


def render_frames_composition_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага frames_composition результатов.
    
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
    render = render_frames_composition(npz_data, meta)
    
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
    y2_scale_js = ""
    if timeline:
        times = [t.get("time_sec", 0.0) for t in timeline]
        
        # Build datasets array
        datasets = []
        
        # Composition features
        saliency_vals = [t.get("saliency_center_offset") for t in timeline if t.get("saliency_center_offset") is not None]
        symmetry_vals = [t.get("symmetry_score") for t in timeline if t.get("symmetry_score") is not None]
        neg_space_vals = [t.get("negative_space_ratio") for t in timeline if t.get("negative_space_ratio") is not None]
        edge_density_vals = [t.get("edge_density") for t in timeline if t.get("edge_density") is not None]
        line_strength_vals = [t.get("line_strength") for t in timeline if t.get("line_strength") is not None]
        
        if saliency_vals:
            datasets.append({
                "label": "Saliency Center Offset",
                "data": saliency_vals,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "tension": 0.1,
                "yAxisID": "y"
            })
        
        if symmetry_vals:
            datasets.append({
                "label": "Symmetry Score",
                "data": symmetry_vals,
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
                            text: 'Symmetry Score'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }"""
        
        if neg_space_vals:
            datasets.append({
                "label": "Negative Space Ratio",
                "data": neg_space_vals,
                "borderColor": "rgb(153, 102, 255)",
                "backgroundColor": "rgba(153, 102, 255, 0.2)",
                "tension": 0.1,
                "yAxisID": "y2"
            })
            y2_scale_js = """,
                    y2: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Negative Space'
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
    
    # Style probabilities display
    style_probs_html = ""
    style_probs = summary.get("style_probabilities", {})
    if style_probs:
        style_probs_html = "<div class='style-probs'><h3>Style Probabilities</h3><ul>"
        for style_name, prob in sorted(style_probs.items(), key=lambda x: x[1], reverse=True):
            style_probs_html += f"<li><strong>{style_name.replace('_', ' ').title()}</strong>: {prob:.4f}</li>"
        style_probs_html += "</ul></div>"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Frames Composition Debug Render</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 1200px; margin: 0 auto; }}
        h1, h2, h3 {{ color: #0056b3; }}
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
        .style-probs {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .style-probs ul {{ list-style: none; padding: 0; }}
        .style-probs li {{ padding: 8px; margin: 5px 0; background-color: #fff; border-radius: 4px; border: 1px solid #dee2e6; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Frames Composition Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Has Faces</strong>
                    <span class="metric-value">{'Yes' if summary.get('has_faces', False) else 'No'}</span>
                </div>
                <div class="metric-card">
                    <strong>Style Dominant ID</strong>
                    <span class="metric-value">{summary.get('style_dominant_id', 'N/A')}</span>
                </div>
            </div>
        </div>
        
        {style_probs_html}
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Composition Features Over Time</h2>
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
                        <th>Saliency Offset</th>
                        <th>Symmetry Score</th>
                        <th>Negative Space</th>
                        <th>Edge Density</th>
                        <th>Line Strength</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('saliency_center_offset', 'min')}</td>
                        <td>{format_dist_value('symmetry_score', 'min')}</td>
                        <td>{format_dist_value('negative_space_ratio', 'min')}</td>
                        <td>{format_dist_value('edge_density', 'min')}</td>
                        <td>{format_dist_value('line_strength', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('saliency_center_offset', 'max')}</td>
                        <td>{format_dist_value('symmetry_score', 'max')}</td>
                        <td>{format_dist_value('negative_space_ratio', 'max')}</td>
                        <td>{format_dist_value('edge_density', 'max')}</td>
                        <td>{format_dist_value('line_strength', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('saliency_center_offset', 'mean')}</td>
                        <td>{format_dist_value('symmetry_score', 'mean')}</td>
                        <td>{format_dist_value('negative_space_ratio', 'mean')}</td>
                        <td>{format_dist_value('edge_density', 'mean')}</td>
                        <td>{format_dist_value('line_strength', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('saliency_center_offset', 'std')}</td>
                        <td>{format_dist_value('symmetry_score', 'std')}</td>
                        <td>{format_dist_value('negative_space_ratio', 'std')}</td>
                        <td>{format_dist_value('edge_density', 'std')}</td>
                        <td>{format_dist_value('line_strength', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('saliency_center_offset', 'median')}</td>
                        <td>{format_dist_value('symmetry_score', 'median')}</td>
                        <td>{format_dist_value('negative_space_ratio', 'median')}</td>
                        <td>{format_dist_value('edge_density', 'median')}</td>
                        <td>{format_dist_value('line_strength', 'median')}</td>
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
                            text: 'Saliency Center Offset'
                        }}
                    }}{y1_scale_js}{y2_scale_js}
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
    logger.info(f"Saved Frames Composition HTML render to {rel_output_path}")
    return output_path


__all__ = ["render_frames_composition", "render_frames_composition_html"]

