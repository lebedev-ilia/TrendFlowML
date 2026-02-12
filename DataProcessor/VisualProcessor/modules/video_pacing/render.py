"""
Renderer для video_pacing: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)


def render_video_pacing(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для video_pacing."""
    render = {
        "component": "video_pacing",
        "summary": {},
        "timeline": [],
        "distributions": {},
        "features": {},
    }
    
    # Extract data
    frame_indices = npz_data.get("frame_indices")
    times_s = npz_data.get("times_s")
    shot_boundary_frame_indices = npz_data.get("shot_boundary_frame_indices")
    motion_norm_per_sec_mean = npz_data.get("motion_norm_per_sec_mean")
    semantic_change_rate_per_sec = npz_data.get("semantic_change_rate_per_sec")
    color_change_rate_per_sec = npz_data.get("color_change_rate_per_sec")
    features = npz_data.get("features")
    
    # Convert to numpy arrays if needed
    def to_array(data, dtype=None):
        if data is None:
            return None
        if isinstance(data, list):
            return np.array(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            return np.asarray(data, dtype=dtype) if dtype else data
        elif isinstance(data, np.ndarray) and data.dtype == object:
            return data
        return None
    
    frame_indices = to_array(frame_indices, np.int32)
    times_s = to_array(times_s, np.float32)
    shot_boundary_frame_indices = to_array(shot_boundary_frame_indices, np.int32) if shot_boundary_frame_indices is not None else None
    motion_norm_per_sec_mean = to_array(motion_norm_per_sec_mean, np.float32) if motion_norm_per_sec_mean is not None else None
    semantic_change_rate_per_sec = to_array(semantic_change_rate_per_sec, np.float32) if semantic_change_rate_per_sec is not None else None
    color_change_rate_per_sec = to_array(color_change_rate_per_sec, np.float32) if color_change_rate_per_sec is not None else None
    
    # Extract features dict
    if features is not None:
        if isinstance(features, np.ndarray) and features.dtype == object:
            if features.size > 0:
                features = features.item() if features.size == 1 else features.tolist()[0]
        if not isinstance(features, dict):
            features = {}
    
    # Summary statistics
    if frame_indices is not None and frame_indices.size > 0:
        n_frames = len(frame_indices)
        n_shots = len(shot_boundary_frame_indices) if shot_boundary_frame_indices is not None and shot_boundary_frame_indices.size > 0 else 0
        
        render["summary"] = {
            "frames_count": int(n_frames),
            "shots_count": int(n_shots),
        }
        
        if features and isinstance(features, dict):
            # Extract key features for summary
            if "shots_count" in features:
                render["summary"]["shots_count"] = int(features["shots_count"]) if isinstance(features["shots_count"], (int, float, np.number)) else n_shots
            if "shot_duration_mean" in features:
                render["summary"]["avg_shot_duration_seconds"] = float(features["shot_duration_mean"]) if isinstance(features["shot_duration_mean"], (int, float, np.number)) else None
            if "cuts_per_10s" in features:
                render["summary"]["cuts_per_10s"] = float(features["cuts_per_10s"]) if isinstance(features["cuts_per_10s"], (int, float, np.number)) else None
        
        if motion_norm_per_sec_mean is not None and motion_norm_per_sec_mean.size > 0:
            valid_motion = motion_norm_per_sec_mean[np.isfinite(motion_norm_per_sec_mean)]
            if valid_motion.size > 0:
                render["summary"]["motion_mean"] = float(np.mean(valid_motion))
                render["summary"]["motion_std"] = float(np.std(valid_motion))
                render["summary"]["motion_max"] = float(np.max(valid_motion))
        
        if semantic_change_rate_per_sec is not None and semantic_change_rate_per_sec.size > 0:
            valid_semantic = semantic_change_rate_per_sec[np.isfinite(semantic_change_rate_per_sec)]
            if valid_semantic.size > 0:
                render["summary"]["semantic_change_mean"] = float(np.mean(valid_semantic))
                render["summary"]["semantic_change_std"] = float(np.std(valid_semantic))
        
        if color_change_rate_per_sec is not None and color_change_rate_per_sec.size > 0:
            valid_color = color_change_rate_per_sec[np.isfinite(color_change_rate_per_sec)]
            if valid_color.size > 0:
                render["summary"]["color_change_mean"] = float(np.mean(valid_color))
                render["summary"]["color_change_std"] = float(np.std(valid_color))
    
    # Timeline data (per-frame)
    if frame_indices is not None and times_s is not None and len(frame_indices) == len(times_s):
        timeline = []
        for i in range(len(frame_indices)):
            frame_idx = int(frame_indices[i])
            time_sec = float(times_s[i])
            
            timeline_entry = {
                "frame_index": frame_idx,
                "time_sec": time_sec,
                "is_shot_boundary": bool(shot_boundary_frame_indices is not None and frame_idx in shot_boundary_frame_indices),
            }
            
            if motion_norm_per_sec_mean is not None and i < len(motion_norm_per_sec_mean):
                timeline_entry["motion"] = float(motion_norm_per_sec_mean[i]) if np.isfinite(motion_norm_per_sec_mean[i]) else None
            
            if semantic_change_rate_per_sec is not None and i < len(semantic_change_rate_per_sec):
                timeline_entry["semantic_change"] = float(semantic_change_rate_per_sec[i]) if np.isfinite(semantic_change_rate_per_sec[i]) else None
            
            if color_change_rate_per_sec is not None and i < len(color_change_rate_per_sec):
                timeline_entry["color_change"] = float(color_change_rate_per_sec[i]) if np.isfinite(color_change_rate_per_sec[i]) else None
            
            timeline.append(timeline_entry)
        
        render["timeline"] = timeline
    
    # Features data
    if features and isinstance(features, dict):
        # Convert all numeric features to JSON-serializable format
        features_clean = {}
        for key, value in features.items():
            if isinstance(value, (int, float, np.number)):
                features_clean[key] = float(value) if np.isfinite(value) else None
            elif isinstance(value, (list, np.ndarray)):
                try:
                    arr = np.asarray(value)
                    if arr.size > 0:
                        features_clean[key] = [float(x) if np.isfinite(x) else None for x in arr.flatten()[:100]]  # Limit to first 100 elements
                    else:
                        features_clean[key] = []
                except Exception:
                    features_clean[key] = str(value)
            elif isinstance(value, str):
                features_clean[key] = value
            else:
                features_clean[key] = str(value)
        
        render["features"] = features_clean
    
    # Distributions
    distributions = {}
    
    if motion_norm_per_sec_mean is not None and motion_norm_per_sec_mean.size > 0:
        valid_motion = motion_norm_per_sec_mean[np.isfinite(motion_norm_per_sec_mean)]
        if valid_motion.size > 0:
            distributions["motion"] = {
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
    
    if semantic_change_rate_per_sec is not None and semantic_change_rate_per_sec.size > 0:
        valid_semantic = semantic_change_rate_per_sec[np.isfinite(semantic_change_rate_per_sec)]
        if valid_semantic.size > 0:
            distributions["semantic_change"] = {
                "min": float(np.min(valid_semantic)),
                "max": float(np.max(valid_semantic)),
                "mean": float(np.mean(valid_semantic)),
                "std": float(np.std(valid_semantic)),
                "median": float(np.median(valid_semantic)),
                "p25": float(np.percentile(valid_semantic, 25)),
                "p75": float(np.percentile(valid_semantic, 75)),
                "p05": float(np.percentile(valid_semantic, 5)),
                "p95": float(np.percentile(valid_semantic, 95)),
            }
    
    if color_change_rate_per_sec is not None and color_change_rate_per_sec.size > 0:
        valid_color = color_change_rate_per_sec[np.isfinite(color_change_rate_per_sec)]
        if valid_color.size > 0:
            distributions["color_change"] = {
                "min": float(np.min(valid_color)),
                "max": float(np.max(valid_color)),
                "mean": float(np.mean(valid_color)),
                "std": float(np.std(valid_color)),
                "median": float(np.median(valid_color)),
                "p25": float(np.percentile(valid_color, 25)),
                "p75": float(np.percentile(valid_color, 75)),
                "p05": float(np.percentile(valid_color, 5)),
                "p95": float(np.percentile(valid_color, 95)),
            }
    
    render["distributions"] = distributions
    
    return render


def render_video_pacing_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага video_pacing результатов.
    
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
    render = render_video_pacing(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    features = render.get("features", {})
    
    # Helper function to format distribution values
    def format_dist_value(dist_key, stat_key):
        dist = distributions.get(dist_key)
        if dist and stat_key in dist:
            return f"{dist[stat_key]:.4f}"
        return "N/A"
    
    # Helper function to format feature values
    def format_feature(feature_key, default="N/A"):
        if feature_key in features:
            val = features[feature_key]
            if isinstance(val, (int, float)):
                return f"{val:.4f}"
            return str(val)
        return default
    
    # Prepare timeline data for chart
    timeline_js = ""
    if timeline:
        times = [t.get("time_sec", 0.0) for t in timeline]
        motion = [t.get("motion") for t in timeline if t.get("motion") is not None]
        semantic_change = [t.get("semantic_change") for t in timeline if t.get("semantic_change") is not None]
        color_change = [t.get("color_change") for t in timeline if t.get("color_change") is not None]
        
        # Build datasets array
        datasets = []
        
        if motion:
            datasets.append({
                "label": "Motion (per-sec mean)",
                "data": motion,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "tension": 0.1,
                "yAxisID": "y"
            })
        
        if semantic_change:
            datasets.append({
                "label": "Semantic Change Rate",
                "data": semantic_change,
                "borderColor": "rgb(255, 99, 132)",
                "backgroundColor": "rgba(255, 99, 132, 0.2)",
                "tension": 0.1,
                "yAxisID": "y1"
            })
        
        if color_change:
            datasets.append({
                "label": "Color Change Rate",
                "data": color_change,
                "borderColor": "rgb(153, 102, 255)",
                "backgroundColor": "rgba(153, 102, 255, 0.2)",
                "tension": 0.1,
                "yAxisID": "y2"
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
    
    # Prepare features table
    features_table_html = ""
    if features:
        features_rows = []
        # Select key features to display
        key_features = [
            ("shots_count", "Shots Count"),
            ("shot_duration_mean", "Avg Shot Duration (s)"),
            ("shot_duration_std", "Shot Duration Std (s)"),
            ("cuts_per_10s", "Cuts per 10s"),
            ("mean_motion_speed_per_shot", "Mean Motion Speed"),
            ("frame_embedding_diff_mean", "Semantic Change Mean"),
            ("color_change_rate_mean", "Color Change Rate Mean"),
        ]
        
        for key, label in key_features:
            value = format_feature(key)
            features_rows.append(f"""
                <tr>
                    <td>{label}</td>
                    <td>{value}</td>
                </tr>
            """)
        
        if features_rows:
            features_table_html = f"""
        <h2>Key Features</h2>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {''.join(features_rows)}
            </tbody>
        </table>
        """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Pacing Debug</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .summary-card {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 14px;
            font-weight: normal;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .chart-container {{
            margin: 30px 0;
            position: relative;
            height: 400px;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .data-table th,
        .data-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .data-table th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        .data-table tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Video Pacing Debug</h1>
        
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Frames Count</h3>
                <div class="value">{summary.get('frames_count', 'N/A')}</div>
            </div>
            <div class="summary-card">
                <h3>Shots Count</h3>
                <div class="value">{summary.get('shots_count', 'N/A')}</div>
            </div>
            <div class="summary-card">
                <h3>Avg Shot Duration</h3>
                <div class="value">{summary.get('avg_shot_duration_seconds', 0) if isinstance(summary.get('avg_shot_duration_seconds'), (int, float)) else 0:.2f}s</div>
            </div>
            <div class="summary-card">
                <h3>Cuts per 10s</h3>
                <div class="value">{summary.get('cuts_per_10s', 0) if isinstance(summary.get('cuts_per_10s'), (int, float)) else 0:.2f}</div>
            </div>
            <div class="summary-card">
                <h3>Motion Mean</h3>
                <div class="value">{summary.get('motion_mean', 0) if isinstance(summary.get('motion_mean'), (int, float)) else 0:.4f}</div>
            </div>
            <div class="summary-card">
                <h3>Semantic Change Mean</h3>
                <div class="value">{summary.get('semantic_change_mean', 0) if isinstance(summary.get('semantic_change_mean'), (int, float)) else 0:.4f}</div>
            </div>
        </div>
        
        <h2>Timeline</h2>
        <div class="chart-container">
            <canvas id="timelineChart"></canvas>
        </div>
        
        {features_table_html}
        
        <h2>Distributions</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Motion</h3>
                <div>Min: {format_dist_value('motion', 'min')}</div>
                <div>Max: {format_dist_value('motion', 'max')}</div>
                <div>Mean: {format_dist_value('motion', 'mean')}</div>
                <div>Std: {format_dist_value('motion', 'std')}</div>
            </div>
            <div class="summary-card">
                <h3>Semantic Change</h3>
                <div>Min: {format_dist_value('semantic_change', 'min')}</div>
                <div>Max: {format_dist_value('semantic_change', 'max')}</div>
                <div>Mean: {format_dist_value('semantic_change', 'mean')}</div>
                <div>Std: {format_dist_value('semantic_change', 'std')}</div>
            </div>
            <div class="summary-card">
                <h3>Color Change</h3>
                <div>Min: {format_dist_value('color_change', 'min')}</div>
                <div>Max: {format_dist_value('color_change', 'max')}</div>
                <div>Mean: {format_dist_value('color_change', 'mean')}</div>
                <div>Std: {format_dist_value('color_change', 'std')}</div>
            </div>
        </div>
    </div>
    
    <script>
        {timeline_js}
        
        if (typeof timelineData !== 'undefined') {{
            const ctx = document.getElementById('timelineChart').getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: timelineData,
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: 'Time (seconds)'
                            }}
                        }},
                        y: {{
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {{
                                display: true,
                                text: 'Motion (per-sec mean)'
                            }}
                        }},
                        y1: {{
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {{
                                display: true,
                                text: 'Semantic Change Rate'
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
                                text: 'Color Change Rate'
                            }},
                            grid: {{
                                drawOnChartArea: false
                            }}
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            display: true,
                            position: 'top'
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false
                        }}
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return output_path


__all__ = ["render_video_pacing", "render_video_pacing_html"]

