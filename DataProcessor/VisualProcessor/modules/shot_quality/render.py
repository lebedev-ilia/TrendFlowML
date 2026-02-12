"""
Renderer для shot_quality: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_shot_quality(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для shot_quality."""
    render = {
        "component": "shot_quality",
        "summary": {},
        "timeline": [],
        "distributions": {},
        "shots": [],
    }
    
    # Extract data
    frame_features = npz_data.get("frame_features")
    quality_probs = npz_data.get("quality_probs")
    shot_ids = npz_data.get("shot_ids")
    shot_start_frame = npz_data.get("shot_start_frame")
    shot_end_frame = npz_data.get("shot_end_frame")
    shot_frame_count = npz_data.get("shot_frame_count")
    shot_features_mean = npz_data.get("shot_features_mean")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    feature_names = npz_data.get("feature_names")
    
    # Convert to numpy arrays if needed
    if frame_features is not None:
        if isinstance(frame_features, list):
            frame_features = np.array(frame_features, dtype=np.float32)
        elif isinstance(frame_features, np.ndarray):
            frame_features = np.asarray(frame_features, dtype=np.float32)
        else:
            frame_features = None
    
    if quality_probs is not None:
        if isinstance(quality_probs, list):
            quality_probs = np.array(quality_probs, dtype=np.float32)
        elif isinstance(quality_probs, np.ndarray):
            quality_probs = np.asarray(quality_probs, dtype=np.float32)
        else:
            quality_probs = None
    
    if shot_ids is not None:
        if isinstance(shot_ids, list):
            shot_ids = np.array(shot_ids, dtype=np.int32)
        elif isinstance(shot_ids, np.ndarray):
            shot_ids = np.asarray(shot_ids, dtype=np.int32)
        else:
            shot_ids = None
    
    if shot_features_mean is not None:
        if isinstance(shot_features_mean, list):
            shot_features_mean = np.array(shot_features_mean, dtype=np.float32)
        elif isinstance(shot_features_mean, np.ndarray):
            shot_features_mean = np.asarray(shot_features_mean, dtype=np.float32)
        else:
            shot_features_mean = None
    
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
    if frame_features is not None and isinstance(frame_features, np.ndarray) and frame_features.size > 0:
        n_frames = int(frame_features.shape[0]) if frame_features.ndim == 2 else 1
        n_features = int(frame_features.shape[1]) if frame_features.ndim == 2 else 0
        
        # Calculate per-feature statistics
        feature_stats = {}
        if feature_names is not None:
            if isinstance(feature_names, (list, np.ndarray)):
                feature_names_list = list(feature_names) if isinstance(feature_names, list) else feature_names.tolist()
            else:
                feature_names_list = []
            
            for i, feat_name in enumerate(feature_names_list):
                if frame_features.ndim >= 2 and i < frame_features.shape[1]:
                    feat_data = frame_features[:, i]
                    valid_data = feat_data[np.isfinite(feat_data)]
                    if valid_data.size > 0:
                        feature_stats[str(feat_name)] = {
                            "mean": float(np.mean(valid_data)),
                            "std": float(np.std(valid_data)),
                            "min": float(np.min(valid_data)),
                            "max": float(np.max(valid_data)),
                            "median": float(np.median(valid_data)),
                        }
        
        # Quality probabilities summary
        quality_summary = {}
        if quality_probs is not None and quality_probs.size > 0:
            # Calculate frame confidence (max prob per frame)
            frame_confidence = np.max(quality_probs, axis=1) if quality_probs.ndim == 2 else np.array([np.max(quality_probs)])
            valid_conf = frame_confidence[np.isfinite(frame_confidence)]
            if valid_conf.size > 0:
                quality_summary = {
                    "avg_frame_confidence": float(np.mean(valid_conf)),
                    "min_frame_confidence": float(np.min(valid_conf)),
                    "max_frame_confidence": float(np.max(valid_conf)),
                    "std_frame_confidence": float(np.std(valid_conf)),
                }
        
        render["summary"] = {
            "frames_count": int(n_frames),
            "features_count": int(n_features),
            "shots_count": int(shot_ids.max() + 1) if shot_ids is not None and shot_ids.size > 0 else 0,
            "feature_stats": feature_stats,
            **quality_summary,
        }
    
    # Timeline data (per-frame)
    if times_s is not None and frame_indices is not None:
        n = len(times_s)
        timeline = []
        
        for i in range(n):
            frame_idx = int(frame_indices[i]) if i < len(frame_indices) else i
            time_sec = float(times_s[i]) if i < len(times_s) else 0.0
            shot_id = int(shot_ids[i]) if shot_ids is not None and i < len(shot_ids) else None
            
            # Extract key features for timeline
            timeline_entry = {
                "frame_index": frame_idx,
                "time_sec": time_sec,
                "shot_id": shot_id,
            }
            
            # Add sharpness (if available)
            if frame_features is not None and isinstance(frame_features, np.ndarray) and frame_features.ndim >= 1 and i < frame_features.shape[0]:
                if feature_names is not None:
                    if isinstance(feature_names, (list, np.ndarray)):
                        feature_names_list = list(feature_names) if isinstance(feature_names, list) else feature_names.tolist()
                    else:
                        feature_names_list = []
                    
                    # Try to find sharpness features
                    for feat_idx, feat_name in enumerate(feature_names_list):
                        if "sharpness" in str(feat_name).lower() and frame_features.ndim >= 2 and feat_idx < frame_features.shape[1]:
                            val = float(frame_features[i, feat_idx])
                            if np.isfinite(val):
                                timeline_entry["sharpness"] = val
                                break
                
                # Add quality confidence if available
                if quality_probs is not None and isinstance(quality_probs, np.ndarray) and quality_probs.ndim >= 1 and i < quality_probs.shape[0]:
                    conf = float(np.max(quality_probs[i]))
                    if np.isfinite(conf):
                        timeline_entry["quality_confidence"] = conf
            
            timeline.append(timeline_entry)
        
        render["timeline"] = timeline
    
    # Shots data
    if shot_start_frame is not None and shot_end_frame is not None:
        if isinstance(shot_start_frame, list):
            shot_start_frame = np.array(shot_start_frame, dtype=np.int32)
        elif isinstance(shot_start_frame, np.ndarray):
            shot_start_frame = np.asarray(shot_start_frame, dtype=np.int32)
        
        if isinstance(shot_end_frame, list):
            shot_end_frame = np.array(shot_end_frame, dtype=np.int32)
        elif isinstance(shot_end_frame, np.ndarray):
            shot_end_frame = np.asarray(shot_end_frame, dtype=np.int32)
        
        n_shots = len(shot_start_frame) if shot_start_frame is not None else 0
        shots = []
        
        for sid in range(n_shots):
            shot_entry = {
                "shot_id": int(sid),
                "start_frame": int(shot_start_frame[sid]) if sid < len(shot_start_frame) else None,
                "end_frame": int(shot_end_frame[sid]) if sid < len(shot_end_frame) else None,
            }
            
            if shot_frame_count is not None:
                if isinstance(shot_frame_count, (list, np.ndarray)):
                    shot_frame_count_arr = np.asarray(shot_frame_count, dtype=np.int32)
                else:
                    shot_frame_count_arr = None
                if shot_frame_count_arr is not None and sid < len(shot_frame_count_arr):
                    shot_entry["frame_count"] = int(shot_frame_count_arr[sid])
            
            if shot_features_mean is not None and isinstance(shot_features_mean, np.ndarray) and shot_features_mean.ndim >= 1 and sid < shot_features_mean.shape[0]:
                # Extract key shot-level features
                if feature_names is not None:
                    if isinstance(feature_names, (list, np.ndarray)):
                        feature_names_list = list(feature_names) if isinstance(feature_names, list) else feature_names.tolist()
                    else:
                        feature_names_list = []
                    
                    shot_feats = {}
                    for feat_idx, feat_name in enumerate(feature_names_list):
                        if shot_features_mean.ndim >= 2 and feat_idx < shot_features_mean.shape[1]:
                            val = float(shot_features_mean[sid, feat_idx])
                            if np.isfinite(val):
                                # Include only key features to keep JSON size manageable
                                if any(keyword in str(feat_name).lower() for keyword in ["sharpness", "noise", "contrast", "exposure"]):
                                    shot_feats[str(feat_name)] = val
                    
                    if shot_feats:
                        shot_entry["features"] = shot_feats
            
            shots.append(shot_entry)
        
        render["shots"] = shots
    
    # Distributions
    distributions = {}
    
    if frame_features is not None and feature_names is not None:
        if isinstance(feature_names, (list, np.ndarray)):
            feature_names_list = list(feature_names) if isinstance(feature_names, list) else feature_names.tolist()
        else:
            feature_names_list = []
        
        # Calculate distributions for key features
        key_features = ["sharpness_tenengrad", "sharpness_secondary", "noise_level_luma", "contrast_global"]
        
        for feat_name in key_features:
            if feat_name in feature_names_list:
                feat_idx = feature_names_list.index(feat_name)
                if frame_features is not None and isinstance(frame_features, np.ndarray) and frame_features.ndim >= 2 and feat_idx < frame_features.shape[1]:
                    feat_data = frame_features[:, feat_idx]
                    valid_data = feat_data[np.isfinite(feat_data)]
                    if valid_data.size > 0:
                        distributions[feat_name] = {
                            "min": float(np.min(valid_data)),
                            "max": float(np.max(valid_data)),
                            "mean": float(np.mean(valid_data)),
                            "std": float(np.std(valid_data)),
                            "median": float(np.median(valid_data)),
                            "p25": float(np.percentile(valid_data, 25)),
                            "p75": float(np.percentile(valid_data, 75)),
                            "p05": float(np.percentile(valid_data, 5)),
                            "p95": float(np.percentile(valid_data, 95)),
                        }
    
    if quality_probs is not None and quality_probs.size > 0:
        frame_confidence = np.max(quality_probs, axis=1) if quality_probs.ndim == 2 else np.array([np.max(quality_probs)])
        valid_conf = frame_confidence[np.isfinite(frame_confidence)]
        if valid_conf.size > 0:
            distributions["quality_confidence"] = {
                "min": float(np.min(valid_conf)),
                "max": float(np.max(valid_conf)),
                "mean": float(np.mean(valid_conf)),
                "std": float(np.std(valid_conf)),
                "median": float(np.median(valid_conf)),
                "p25": float(np.percentile(valid_conf, 25)),
                "p75": float(np.percentile(valid_conf, 75)),
                "p05": float(np.percentile(valid_conf, 5)),
                "p95": float(np.percentile(valid_conf, 95)),
            }
    
    render["distributions"] = distributions
    
    return render


def render_shot_quality_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага shot_quality результатов.
    
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
    render = render_shot_quality(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    shots = render.get("shots", [])
    
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
        sharpness_vals = [t.get("sharpness") for t in timeline if t.get("sharpness") is not None]
        quality_conf_vals = [t.get("quality_confidence") for t in timeline if t.get("quality_confidence") is not None]
        
        # Build datasets array
        datasets = []
        
        if sharpness_vals:
            datasets.append({
                "label": "Sharpness",
                "data": sharpness_vals,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "tension": 0.1,
                "yAxisID": "y"
            })
        
        if quality_conf_vals:
            datasets.append({
                "label": "Quality Confidence",
                "data": quality_conf_vals,
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
                            text: 'Quality Confidence'
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
    
    # Prepare shots summary
    shots_summary_html = ""
    if shots:
        shots_table_rows = ""
        for shot in shots[:20]:  # Limit to first 20 shots
            shot_id = shot.get("shot_id", "N/A")
            start_frame = shot.get("start_frame", "N/A")
            end_frame = shot.get("end_frame", "N/A")
            frame_count = shot.get("frame_count", "N/A")
            shots_table_rows += f"""
                    <tr>
                        <td>{shot_id}</td>
                        <td>{start_frame}</td>
                        <td>{end_frame}</td>
                        <td>{frame_count}</td>
                    </tr>
            """
        
        shots_summary_html = f"""
        <div class="shots">
            <h2>Shots Summary (first 20)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Shot ID</th>
                        <th>Start Frame</th>
                        <th>End Frame</th>
                        <th>Frame Count</th>
                    </tr>
                </thead>
                <tbody>
                    {shots_table_rows}
                </tbody>
            </table>
        </div>
        """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shot Quality Debug Render</title>
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
        .shots {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .shots table {{ width: 100%; border-collapse: collapse; }}
        .shots th, .shots td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .shots th {{ background-color: #0056b3; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Shot Quality Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Features Count</strong>
                    <span class="metric-value">{summary.get('features_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Shots Count</strong>
                    <span class="metric-value">{summary.get('shots_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Avg Frame Confidence</strong>
                    <span class="metric-value">{summary.get('avg_frame_confidence', 0.0):.4f}</span>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Sharpness and Quality Confidence Over Time</h2>
            <canvas id="timelineChart"></canvas>
        </div>
        ''' if timeline else '<p>No timeline data available</p>'}
        
        {shots_summary_html}
        
        {f'''
        <div class="distributions">
            <h2>Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Sharpness Tenengrad</th>
                        <th>Noise Level Luma</th>
                        <th>Quality Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('sharpness_tenengrad', 'min')}</td>
                        <td>{format_dist_value('noise_level_luma', 'min')}</td>
                        <td>{format_dist_value('quality_confidence', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('sharpness_tenengrad', 'max')}</td>
                        <td>{format_dist_value('noise_level_luma', 'max')}</td>
                        <td>{format_dist_value('quality_confidence', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('sharpness_tenengrad', 'mean')}</td>
                        <td>{format_dist_value('noise_level_luma', 'mean')}</td>
                        <td>{format_dist_value('quality_confidence', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('sharpness_tenengrad', 'std')}</td>
                        <td>{format_dist_value('noise_level_luma', 'std')}</td>
                        <td>{format_dist_value('quality_confidence', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('sharpness_tenengrad', 'median')}</td>
                        <td>{format_dist_value('noise_level_luma', 'median')}</td>
                        <td>{format_dist_value('quality_confidence', 'median')}</td>
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
                            text: 'Sharpness'
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
    
    logger.info(f"Saved Shot Quality HTML render to {output_path}")
    return output_path


__all__ = ["render_shot_quality", "render_shot_quality_html"]

