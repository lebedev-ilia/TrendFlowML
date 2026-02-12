"""
Renderer для color_light: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_color_light(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для color_light."""
    render = {
        "component": "color_light",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract data
    video_features = npz_data.get("video_features", {})
    frames = npz_data.get("frames", {})
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    sequence_frame_indices = npz_data.get("sequence_frame_indices")
    sequence_times_s = npz_data.get("sequence_times_s")
    
    # Convert to numpy arrays if needed
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
    
    if sequence_times_s is not None:
        if isinstance(sequence_times_s, list):
            sequence_times_s = np.array(sequence_times_s, dtype=np.float32)
        elif isinstance(sequence_times_s, np.ndarray):
            sequence_times_s = np.asarray(sequence_times_s, dtype=np.float32)
        else:
            sequence_times_s = None
    
    # Summary statistics from video_features
    if video_features:
        render["summary"] = {
            "frames_count": int(len(frame_indices)) if frame_indices is not None else 0,
            "scenes_count": int(len(frames)) if isinstance(frames, dict) else 0,
            "color_distribution_entropy": float(video_features.get("color_distribution_entropy", 0.0)) if isinstance(video_features.get("color_distribution_entropy"), (int, float)) else None,
            "color_distribution_gini": float(video_features.get("color_distribution_gini", 0.0)) if isinstance(video_features.get("color_distribution_gini"), (int, float)) else None,
            "global_brightness_change_speed": float(video_features.get("global_brightness_change_speed", 0.0)) if isinstance(video_features.get("global_brightness_change_speed"), (int, float)) else None,
            "global_color_change_speed": float(video_features.get("global_color_change_speed", 0.0)) if isinstance(video_features.get("global_color_change_speed"), (int, float)) else None,
            "strobe_transition_frequency": float(video_features.get("strobe_transition_frequency", 0.0)) if isinstance(video_features.get("strobe_transition_frequency"), (int, float)) else None,
        }
        
        # Style probabilities
        style_probs = {k: float(v) for k, v in video_features.items() if k.startswith("style_") and isinstance(v, (int, float))}
        if style_probs:
            render["summary"]["style_probabilities"] = style_probs
    
    # Timeline data (per-frame features)
    timeline = []
    if frames and isinstance(frames, dict) and sequence_times_s is not None:
        # Collect frame features from all scenes
        frame_feat_map: Dict[int, Dict[str, Any]] = {}
        for scene_dict in frames.values():
            if isinstance(scene_dict, dict):
                for frame_idx, frame_obj in scene_dict.items():
                    if isinstance(frame_obj, dict):
                        feat = frame_obj.get("features", {})
                        try:
                            frame_feat_map[int(frame_idx)] = feat
                        except Exception:
                            continue
        
        # Build timeline from sequence
        if sequence_frame_indices is not None:
            if isinstance(sequence_frame_indices, list):
                sequence_frame_indices = np.array(sequence_frame_indices, dtype=np.int32)
            elif isinstance(sequence_frame_indices, np.ndarray):
                sequence_frame_indices = np.asarray(sequence_frame_indices, dtype=np.int32)
            
            n = len(sequence_times_s)
            for i in range(n):
                if i >= len(sequence_frame_indices):
                    break
                frame_idx = int(sequence_frame_indices[i])
                time_sec = float(sequence_times_s[i])
                feat = frame_feat_map.get(frame_idx, {})
                
                timeline.append({
                    "frame_index": frame_idx,
                    "time_sec": time_sec,
                    "hue_mean_norm": float(feat.get("hue_mean_norm", 0.0)) if isinstance(feat.get("hue_mean_norm"), (int, float)) else None,
                    "colorfulness_norm": float(feat.get("colorfulness_norm", 0.0)) if isinstance(feat.get("colorfulness_norm"), (int, float)) else None,
                    "brightness_mean_norm": float(feat.get("brightness_mean", 0.0) / 255.0) if isinstance(feat.get("brightness_mean"), (int, float)) else None,
                    "global_contrast_norm": float(feat.get("global_contrast_norm", 0.0)) if isinstance(feat.get("global_contrast_norm"), (int, float)) else None,
                    "saturation_mean_norm": float(feat.get("sat_mean_norm", 0.0)) if isinstance(feat.get("sat_mean_norm"), (int, float)) else None,
                    "value_mean_norm": float(feat.get("val_mean_norm", 0.0)) if isinstance(feat.get("val_mean_norm"), (int, float)) else None,
                })
    
    render["timeline"] = timeline
    
    # Distribution statistics
    distributions = {}
    
    if timeline:
        hue_vals = [t.get("hue_mean_norm") for t in timeline if t.get("hue_mean_norm") is not None]
        colorfulness_vals = [t.get("colorfulness_norm") for t in timeline if t.get("colorfulness_norm") is not None]
        brightness_vals = [t.get("brightness_mean_norm") for t in timeline if t.get("brightness_mean_norm") is not None]
        contrast_vals = [t.get("global_contrast_norm") for t in timeline if t.get("global_contrast_norm") is not None]
        
        if hue_vals:
            hue_arr = np.array(hue_vals, dtype=np.float32)
            distributions["hue_mean_norm"] = {
                "min": float(np.min(hue_arr)),
                "max": float(np.max(hue_arr)),
                "mean": float(np.mean(hue_arr)),
                "std": float(np.std(hue_arr)),
                "median": float(np.median(hue_arr)),
                "p25": float(np.percentile(hue_arr, 25)),
                "p75": float(np.percentile(hue_arr, 75)),
            }
        
        if colorfulness_vals:
            colorfulness_arr = np.array(colorfulness_vals, dtype=np.float32)
            distributions["colorfulness_norm"] = {
                "min": float(np.min(colorfulness_arr)),
                "max": float(np.max(colorfulness_arr)),
                "mean": float(np.mean(colorfulness_arr)),
                "std": float(np.std(colorfulness_arr)),
                "median": float(np.median(colorfulness_arr)),
                "p25": float(np.percentile(colorfulness_arr, 25)),
                "p75": float(np.percentile(colorfulness_arr, 75)),
            }
        
        if brightness_vals:
            brightness_arr = np.array(brightness_vals, dtype=np.float32)
            distributions["brightness_mean_norm"] = {
                "min": float(np.min(brightness_arr)),
                "max": float(np.max(brightness_arr)),
                "mean": float(np.mean(brightness_arr)),
                "std": float(np.std(brightness_arr)),
                "median": float(np.median(brightness_arr)),
                "p25": float(np.percentile(brightness_arr, 25)),
                "p75": float(np.percentile(brightness_arr, 75)),
            }
        
        if contrast_vals:
            contrast_arr = np.array(contrast_vals, dtype=np.float32)
            distributions["global_contrast_norm"] = {
                "min": float(np.min(contrast_arr)),
                "max": float(np.max(contrast_arr)),
                "mean": float(np.mean(contrast_arr)),
                "std": float(np.std(contrast_arr)),
                "median": float(np.median(contrast_arr)),
                "p25": float(np.percentile(contrast_arr, 25)),
                "p75": float(np.percentile(contrast_arr, 75)),
            }
    
    render["distributions"] = distributions
    
    return render


def render_color_light_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага color_light результатов.
    
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
    render = render_color_light(npz_data, meta)
    
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
        hue_vals = [t.get("hue_mean_norm") for t in timeline if t.get("hue_mean_norm") is not None]
        colorfulness_vals = [t.get("colorfulness_norm") for t in timeline if t.get("colorfulness_norm") is not None]
        brightness_vals = [t.get("brightness_mean_norm") for t in timeline if t.get("brightness_mean_norm") is not None]
        contrast_vals = [t.get("global_contrast_norm") for t in timeline if t.get("global_contrast_norm") is not None]
        
        # Build datasets array
        datasets = []
        
        if hue_vals:
            datasets.append({
                "label": "Hue Mean (norm)",
                "data": hue_vals,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "tension": 0.1,
                "yAxisID": "y"
            })
        
        if colorfulness_vals:
            datasets.append({
                "label": "Colorfulness (norm)",
                "data": colorfulness_vals,
                "borderColor": "rgb(255, 99, 132)",
                "backgroundColor": "rgba(255, 99, 132, 0.2)",
                "tension": 0.1,
                "yAxisID": "y1"
            })
        
        if brightness_vals:
            datasets.append({
                "label": "Brightness Mean (norm)",
                "data": brightness_vals,
                "borderColor": "rgb(153, 102, 255)",
                "backgroundColor": "rgba(153, 102, 255, 0.2)",
                "tension": 0.1,
                "yAxisID": "y2"
            })
        
        if contrast_vals:
            datasets.append({
                "label": "Global Contrast (norm)",
                "data": contrast_vals,
                "borderColor": "rgb(255, 206, 86)",
                "backgroundColor": "rgba(255, 206, 86, 0.2)",
                "tension": 0.1,
                "yAxisID": "y3"
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
    
    # Style probabilities display
    style_probs_html = ""
    style_probs = summary.get("style_probabilities", {})
    if style_probs:
        style_probs_html = "<div class='style-probs'><h3>Style Probabilities</h3><ul>"
        for style_name, prob in sorted(style_probs.items(), key=lambda x: x[1], reverse=True):
            style_probs_html += f"<li><strong>{style_name.replace('style_', '').replace('_', ' ').title()}</strong>: {prob:.4f}</li>"
        style_probs_html += "</ul></div>"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Color & Light Debug Render</title>
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
        <h1>Color & Light Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Scenes Count</strong>
                    <span class="metric-value">{summary.get('scenes_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Color Distribution Entropy</strong>
                    <span class="metric-value">{summary.get('color_distribution_entropy', 'N/A') if summary.get('color_distribution_entropy') is not None else 'N/A'}</span>
                </div>
                <div class="metric-card">
                    <strong>Color Distribution Gini</strong>
                    <span class="metric-value">{summary.get('color_distribution_gini', 'N/A') if summary.get('color_distribution_gini') is not None else 'N/A'}</span>
                </div>
                <div class="metric-card">
                    <strong>Global Brightness Change Speed</strong>
                    <span class="metric-value">{summary.get('global_brightness_change_speed', 'N/A') if summary.get('global_brightness_change_speed') is not None else 'N/A'}</span>
                </div>
                <div class="metric-card">
                    <strong>Global Color Change Speed</strong>
                    <span class="metric-value">{summary.get('global_color_change_speed', 'N/A') if summary.get('global_color_change_speed') is not None else 'N/A'}</span>
                </div>
            </div>
        </div>
        
        {style_probs_html}
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Color & Light Features Over Time</h2>
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
                        <th>Hue Mean</th>
                        <th>Colorfulness</th>
                        <th>Brightness</th>
                        <th>Contrast</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('hue_mean_norm', 'min')}</td>
                        <td>{format_dist_value('colorfulness_norm', 'min')}</td>
                        <td>{format_dist_value('brightness_mean_norm', 'min')}</td>
                        <td>{format_dist_value('global_contrast_norm', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('hue_mean_norm', 'max')}</td>
                        <td>{format_dist_value('colorfulness_norm', 'max')}</td>
                        <td>{format_dist_value('brightness_mean_norm', 'max')}</td>
                        <td>{format_dist_value('global_contrast_norm', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('hue_mean_norm', 'mean')}</td>
                        <td>{format_dist_value('colorfulness_norm', 'mean')}</td>
                        <td>{format_dist_value('brightness_mean_norm', 'mean')}</td>
                        <td>{format_dist_value('global_contrast_norm', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('hue_mean_norm', 'std')}</td>
                        <td>{format_dist_value('colorfulness_norm', 'std')}</td>
                        <td>{format_dist_value('brightness_mean_norm', 'std')}</td>
                        <td>{format_dist_value('global_contrast_norm', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('hue_mean_norm', 'median')}</td>
                        <td>{format_dist_value('colorfulness_norm', 'median')}</td>
                        <td>{format_dist_value('brightness_mean_norm', 'median')}</td>
                        <td>{format_dist_value('global_contrast_norm', 'median')}</td>
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
                            text: 'Hue Mean (norm)'
                        }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'Colorfulness (norm)'
                        }},
                        grid: {{
                            drawOnChartArea: false
                        }}
                    }},
                    y2: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'Brightness (norm)'
                        }},
                        grid: {{
                            drawOnChartArea: false
                        }}
                    }},
                    y3: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'Contrast (norm)'
                        }},
                        grid: {{
                            drawOnChartArea: false
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
    logger.info(f"Saved Color & Light HTML render to {rel_output_path}")
    return output_path


__all__ = ["render_color_light", "render_color_light_html"]

