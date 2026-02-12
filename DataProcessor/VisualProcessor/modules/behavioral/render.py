"""
Renderer для behavioral: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_behavioral(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для behavioral."""
    render = {
        "component": "behavioral",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract behavioral data
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    landmarks_present = npz_data.get("landmarks_present")
    
    # Sequence features
    seq_keys = [
        "num_hands", "hands_visibility", "hand_motion_energy",
        "arm_openness", "pose_expansion", "body_lean_angle", "balance_offset",
        "shoulder_angle", "shoulder_angle_velocity",
        "head_position_x_norm", "head_position_y_norm", "head_motion_energy", "head_stability",
        "mouth_width_norm", "mouth_height_norm", "mouth_area_norm",
        "mouth_velocity", "mouth_open_ratio", "speech_activity_proxy",
        "blink_flag", "blink_rate_short", "self_touch_flag", "fidgeting_energy",
        "timestamp_norm",
    ]
    
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
    
    if landmarks_present is not None:
        if isinstance(landmarks_present, list):
            landmarks_present = np.array(landmarks_present, dtype=bool)
        elif isinstance(landmarks_present, np.ndarray):
            landmarks_present = np.asarray(landmarks_present, dtype=bool)
        else:
            landmarks_present = None
    
    # Extract sequence arrays
    seq_arrays = {}
    for key in seq_keys:
        arr_key = f"seq_{key}"
        arr = npz_data.get(arr_key)
        if arr is not None:
            if isinstance(arr, list):
                arr = np.array(arr, dtype=np.float32)
            elif isinstance(arr, np.ndarray):
                arr = np.asarray(arr, dtype=np.float32)
            seq_arrays[key] = arr
    
    # Extract aggregated results
    aggregated = npz_data.get("aggregated")
    if aggregated is not None:
        if isinstance(aggregated, np.ndarray) and aggregated.dtype == object:
            try:
                aggregated = aggregated.item()
            except Exception:
                aggregated = {}
        if not isinstance(aggregated, dict):
            aggregated = {}
    else:
        aggregated = {}
    
    # Summary statistics
    n_frames = len(times_s) if times_s is not None else 0
    landmarks_ratio = float(np.mean(landmarks_present)) if landmarks_present is not None and landmarks_present.size > 0 else 0.0
    
    render["summary"] = {
        "frames_count": int(n_frames),
        "landmarks_present_ratio": landmarks_ratio,
    }
    
    # Add aggregated metrics to summary
    if aggregated:
        render["summary"].update({
            "avg_engagement": aggregated.get("avg_engagement"),
            "avg_confidence": aggregated.get("avg_confidence"),
            "avg_stress": aggregated.get("avg_stress"),
            "gesture_rate_per_sec": aggregated.get("gesture_rate_per_sec"),
            "hands_visibility_ratio": aggregated.get("hands_visibility_ratio"),
            "face_visibility_ratio": aggregated.get("face_visibility_ratio"),
        })
    
    # Timeline data (per-frame features)
    if times_s is not None and frame_indices is not None and len(times_s) > 0:
        timeline = []
        n = len(times_s)
        
        for i in range(n):
            frame_idx = int(frame_indices[i]) if i < len(frame_indices) else i
            time_sec = float(times_s[i]) if np.isfinite(times_s[i]) else 0.0
            
            timeline_entry = {
                "frame_index": frame_idx,
                "time_s": time_sec,
                "landmarks_present": bool(landmarks_present[i]) if landmarks_present is not None and i < len(landmarks_present) else False,
            }
            
            # Add sequence features to timeline
            for key in seq_keys:
                arr = seq_arrays.get(key)
                if arr is not None and i < len(arr):
                    val = arr[i]
                    if np.isfinite(val):
                        timeline_entry[key] = float(val)
            
            timeline.append(timeline_entry)
        
        render["timeline"] = timeline
    
    # Distribution statistics
    distributions = {}
    
    # Key behavioral features for distributions
    key_features = [
        "speech_activity_proxy",
        "arm_openness",
        "body_lean_angle",
        "hand_motion_energy",
        "head_motion_energy",
        "blink_rate_short",
        "fidgeting_energy",
    ]
    
    for key in key_features:
        arr = seq_arrays.get(key)
        if arr is not None:
            valid_vals = arr[np.isfinite(arr)]
            if valid_vals.size > 0:
                distributions[key] = {
                    "min": float(np.min(valid_vals)),
                    "max": float(np.max(valid_vals)),
                    "mean": float(np.mean(valid_vals)),
                    "std": float(np.std(valid_vals)),
                    "median": float(np.median(valid_vals)),
                    "p25": float(np.percentile(valid_vals, 25)),
                    "p75": float(np.percentile(valid_vals, 75)),
                    "p05": float(np.percentile(valid_vals, 5)),
                    "p95": float(np.percentile(valid_vals, 95)),
                }
    
    # Gesture distribution
    gesture_counts = aggregated.get("gesture_counts", {})
    if gesture_counts:
        distributions["gesture_distribution"] = gesture_counts
    
    render["distributions"] = distributions
    
    return render


def render_behavioral_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага behavioral результатов.
    
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
    render = render_behavioral(npz_data, meta)
    
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
        times = [t.get("time_s", 0.0) for t in timeline]
        speech_activity = [t.get("speech_activity_proxy") for t in timeline if t.get("speech_activity_proxy") is not None]
        arm_openness = [t.get("arm_openness") for t in timeline if t.get("arm_openness") is not None]
        body_lean = [t.get("body_lean_angle") for t in timeline if t.get("body_lean_angle") is not None]
        
        # Build datasets array
        datasets = []
        
        if speech_activity:
            datasets.append({
                "label": "Speech Activity",
                "data": speech_activity,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "tension": 0.1,
                "yAxisID": "y"
            })
        
        if arm_openness:
            datasets.append({
                "label": "Arm Openness",
                "data": arm_openness,
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
                            text: 'Arm Openness'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }"""
        
        if body_lean:
            datasets.append({
                "label": "Body Lean Angle",
                "data": body_lean,
                "borderColor": "rgb(54, 162, 235)",
                "backgroundColor": "rgba(54, 162, 235, 0.2)",
                "tension": 0.1,
                "yAxisID": "y1"
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
    <title>Behavioral Analysis Debug Render</title>
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
        <h1>Behavioral Analysis Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Landmarks Present Ratio</strong>
                    <span class="metric-value">{summary.get('landmarks_present_ratio', 0.0):.2%}</span>
                </div>
                <div class="metric-card">
                    <strong>Avg Engagement</strong>
                    <span class="metric-value">{summary.get('avg_engagement', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Avg Confidence</strong>
                    <span class="metric-value">{summary.get('avg_confidence', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Avg Stress</strong>
                    <span class="metric-value">{summary.get('avg_stress', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Gesture Rate (per sec)</strong>
                    <span class="metric-value">{summary.get('gesture_rate_per_sec', 0.0):.2f}</span>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Behavioral Features Over Time</h2>
            <canvas id="timelineChart"></canvas>
        </div>
        ''' if timeline else '<p>No timeline data available</p>'}
        
        {f'''
        <div class="distributions">
            <h2>Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Mean</th>
                        <th>Std</th>
                        <th>Median</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Speech Activity</strong></td>
                        <td>{format_dist_value('speech_activity_proxy', 'min')}</td>
                        <td>{format_dist_value('speech_activity_proxy', 'max')}</td>
                        <td>{format_dist_value('speech_activity_proxy', 'mean')}</td>
                        <td>{format_dist_value('speech_activity_proxy', 'std')}</td>
                        <td>{format_dist_value('speech_activity_proxy', 'median')}</td>
                    </tr>
                    <tr>
                        <td><strong>Arm Openness</strong></td>
                        <td>{format_dist_value('arm_openness', 'min')}</td>
                        <td>{format_dist_value('arm_openness', 'max')}</td>
                        <td>{format_dist_value('arm_openness', 'mean')}</td>
                        <td>{format_dist_value('arm_openness', 'std')}</td>
                        <td>{format_dist_value('arm_openness', 'median')}</td>
                    </tr>
                    <tr>
                        <td><strong>Body Lean Angle</strong></td>
                        <td>{format_dist_value('body_lean_angle', 'min')}</td>
                        <td>{format_dist_value('body_lean_angle', 'max')}</td>
                        <td>{format_dist_value('body_lean_angle', 'mean')}</td>
                        <td>{format_dist_value('body_lean_angle', 'std')}</td>
                        <td>{format_dist_value('body_lean_angle', 'median')}</td>
                    </tr>
                    <tr>
                        <td><strong>Hand Motion Energy</strong></td>
                        <td>{format_dist_value('hand_motion_energy', 'min')}</td>
                        <td>{format_dist_value('hand_motion_energy', 'max')}</td>
                        <td>{format_dist_value('hand_motion_energy', 'mean')}</td>
                        <td>{format_dist_value('hand_motion_energy', 'std')}</td>
                        <td>{format_dist_value('hand_motion_energy', 'median')}</td>
                    </tr>
                    <tr>
                        <td><strong>Blink Rate</strong></td>
                        <td>{format_dist_value('blink_rate_short', 'min')}</td>
                        <td>{format_dist_value('blink_rate_short', 'max')}</td>
                        <td>{format_dist_value('blink_rate_short', 'mean')}</td>
                        <td>{format_dist_value('blink_rate_short', 'std')}</td>
                        <td>{format_dist_value('blink_rate_short', 'median')}</td>
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
                            text: 'Speech Activity'
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
    
    logger.info(f"Saved Behavioral HTML render to {output_path}")
    return output_path


__all__ = ["render_behavioral", "render_behavioral_html"]

