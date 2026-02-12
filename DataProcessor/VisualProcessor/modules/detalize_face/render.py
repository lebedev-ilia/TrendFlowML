"""
Renderer для detalize_face: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_detalize_face(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для detalize_face."""
    render = {
        "component": "detalize_face",
        "summary": {},
        "timeline": [],
        "distributions": {},
        "faces": [],
    }
    
    # Extract face data
    face_count = npz_data.get("face_count")
    primary_gaze = npz_data.get("primary_gaze_at_camera_prob")
    primary_blink_rate = npz_data.get("primary_blink_rate")
    primary_attention = npz_data.get("primary_attention_score")
    primary_quality = npz_data.get("primary_quality_proxy_score")
    primary_sharpness = npz_data.get("primary_face_sharpness")
    primary_occlusion = npz_data.get("primary_occlusion_proxy")
    primary_speech = npz_data.get("primary_speech_activity_prob")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    faces_agg = npz_data.get("faces_agg")
    summary = npz_data.get("summary")
    
    # Convert to numpy arrays if needed
    if face_count is not None:
        if isinstance(face_count, list):
            face_count = np.array(face_count, dtype=np.float32)
        elif isinstance(face_count, np.ndarray):
            face_count = np.asarray(face_count, dtype=np.float32)
        else:
            face_count = None
    
    if primary_gaze is not None:
        if isinstance(primary_gaze, list):
            primary_gaze = np.array(primary_gaze, dtype=np.float32)
        elif isinstance(primary_gaze, np.ndarray):
            primary_gaze = np.asarray(primary_gaze, dtype=np.float32)
        else:
            primary_gaze = None
    
    if primary_blink_rate is not None:
        if isinstance(primary_blink_rate, list):
            primary_blink_rate = np.array(primary_blink_rate, dtype=np.float32)
        elif isinstance(primary_blink_rate, np.ndarray):
            primary_blink_rate = np.asarray(primary_blink_rate, dtype=np.float32)
        else:
            primary_blink_rate = None
    
    if primary_attention is not None:
        if isinstance(primary_attention, list):
            primary_attention = np.array(primary_attention, dtype=np.float32)
        elif isinstance(primary_attention, np.ndarray):
            primary_attention = np.asarray(primary_attention, dtype=np.float32)
        else:
            primary_attention = None
    
    if primary_quality is not None:
        if isinstance(primary_quality, list):
            primary_quality = np.array(primary_quality, dtype=np.float32)
        elif isinstance(primary_quality, np.ndarray):
            primary_quality = np.asarray(primary_quality, dtype=np.float32)
        else:
            primary_quality = None
    
    if primary_sharpness is not None:
        if isinstance(primary_sharpness, list):
            primary_sharpness = np.array(primary_sharpness, dtype=np.float32)
        elif isinstance(primary_sharpness, np.ndarray):
            primary_sharpness = np.asarray(primary_sharpness, dtype=np.float32)
        else:
            primary_sharpness = None
    
    if primary_occlusion is not None:
        if isinstance(primary_occlusion, list):
            primary_occlusion = np.array(primary_occlusion, dtype=np.float32)
        elif isinstance(primary_occlusion, np.ndarray):
            primary_occlusion = np.asarray(primary_occlusion, dtype=np.float32)
        else:
            primary_occlusion = None
    
    if primary_speech is not None:
        if isinstance(primary_speech, list):
            primary_speech = np.array(primary_speech, dtype=np.float32)
        elif isinstance(primary_speech, np.ndarray):
            primary_speech = np.asarray(primary_speech, dtype=np.float32)
        else:
            primary_speech = None
    
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
    if summary is not None:
        if isinstance(summary, np.ndarray) and summary.dtype == object:
            summary = summary.item() if summary.size == 1 else {}
        if isinstance(summary, dict):
            render["summary"] = {
                "total_frames": int(summary.get("total_frames", 0)),
                "processed_frames": int(summary.get("processed_frames", 0)),
                "frames_with_faces": int(summary.get("frames_with_faces", 0)),
                "total_faces": int(summary.get("total_faces", 0)),
                "primary_faces": int(summary.get("primary_faces", 0)),
                "avg_faces_per_frame": float(summary.get("avg_faces_per_frame", 0.0)),
            }
    
    # Timeline data
    if times_s is not None and face_count is not None:
        n = len(times_s)
        timeline = []
        
        for i in range(n):
            frame_idx = int(frame_indices[i]) if frame_indices is not None and i < len(frame_indices) else i
            time_sec = float(times_s[i]) if np.isfinite(times_s[i]) else 0.0
            fc = float(face_count[i]) if i < len(face_count) and np.isfinite(face_count[i]) else None
            gaze = float(primary_gaze[i]) if primary_gaze is not None and i < len(primary_gaze) and np.isfinite(primary_gaze[i]) else None
            blink = float(primary_blink_rate[i]) if primary_blink_rate is not None and i < len(primary_blink_rate) and np.isfinite(primary_blink_rate[i]) else None
            attn = float(primary_attention[i]) if primary_attention is not None and i < len(primary_attention) and np.isfinite(primary_attention[i]) else None
            qual = float(primary_quality[i]) if primary_quality is not None and i < len(primary_quality) and np.isfinite(primary_quality[i]) else None
            sharp = float(primary_sharpness[i]) if primary_sharpness is not None and i < len(primary_sharpness) and np.isfinite(primary_sharpness[i]) else None
            occ = float(primary_occlusion[i]) if primary_occlusion is not None and i < len(primary_occlusion) and np.isfinite(primary_occlusion[i]) else None
            speech = float(primary_speech[i]) if primary_speech is not None and i < len(primary_speech) and np.isfinite(primary_speech[i]) else None
            
            timeline.append({
                "frame_index": frame_idx,
                "time_sec": time_sec,
                "face_count": fc,
                "primary_gaze_at_camera_prob": gaze,
                "primary_blink_rate": blink,
                "primary_attention_score": attn,
                "primary_quality_proxy_score": qual,
                "primary_face_sharpness": sharp,
                "primary_occlusion_proxy": occ,
                "primary_speech_activity_prob": speech,
            })
        
        render["timeline"] = timeline
    
    # Distributions
    distributions = {}
    
    if face_count is not None:
        valid_counts = face_count[np.isfinite(face_count)]
        if valid_counts.size > 0:
            distributions["face_count"] = {
                "min": float(np.min(valid_counts)),
                "max": float(np.max(valid_counts)),
                "mean": float(np.mean(valid_counts)),
                "std": float(np.std(valid_counts)),
                "median": float(np.median(valid_counts)),
                "p25": float(np.percentile(valid_counts, 25)),
                "p75": float(np.percentile(valid_counts, 75)),
            }
    
    if primary_gaze is not None:
        valid_gaze = primary_gaze[np.isfinite(primary_gaze)]
        if valid_gaze.size > 0:
            distributions["primary_gaze_at_camera_prob"] = {
                "min": float(np.min(valid_gaze)),
                "max": float(np.max(valid_gaze)),
                "mean": float(np.mean(valid_gaze)),
                "std": float(np.std(valid_gaze)),
                "median": float(np.median(valid_gaze)),
                "p25": float(np.percentile(valid_gaze, 25)),
                "p75": float(np.percentile(valid_gaze, 75)),
            }
    
    if primary_blink_rate is not None:
        valid_blink = primary_blink_rate[np.isfinite(primary_blink_rate)]
        if valid_blink.size > 0:
            distributions["primary_blink_rate"] = {
                "min": float(np.min(valid_blink)),
                "max": float(np.max(valid_blink)),
                "mean": float(np.mean(valid_blink)),
                "std": float(np.std(valid_blink)),
                "median": float(np.median(valid_blink)),
                "p25": float(np.percentile(valid_blink, 25)),
                "p75": float(np.percentile(valid_blink, 75)),
            }
    
    if primary_attention is not None:
        valid_attn = primary_attention[np.isfinite(primary_attention)]
        if valid_attn.size > 0:
            distributions["primary_attention_score"] = {
                "min": float(np.min(valid_attn)),
                "max": float(np.max(valid_attn)),
                "mean": float(np.mean(valid_attn)),
                "std": float(np.std(valid_attn)),
                "median": float(np.median(valid_attn)),
                "p25": float(np.percentile(valid_attn, 25)),
                "p75": float(np.percentile(valid_attn, 75)),
            }
    
    if primary_quality is not None:
        valid_qual = primary_quality[np.isfinite(primary_quality)]
        if valid_qual.size > 0:
            distributions["primary_quality_proxy_score"] = {
                "min": float(np.min(valid_qual)),
                "max": float(np.max(valid_qual)),
                "mean": float(np.mean(valid_qual)),
                "std": float(np.std(valid_qual)),
                "median": float(np.median(valid_qual)),
                "p25": float(np.percentile(valid_qual, 25)),
                "p75": float(np.percentile(valid_qual, 75)),
            }
    
    if primary_sharpness is not None:
        valid_sharp = primary_sharpness[np.isfinite(primary_sharpness)]
        if valid_sharp.size > 0:
            distributions["primary_face_sharpness"] = {
                "min": float(np.min(valid_sharp)),
                "max": float(np.max(valid_sharp)),
                "mean": float(np.mean(valid_sharp)),
                "std": float(np.std(valid_sharp)),
                "median": float(np.median(valid_sharp)),
                "p25": float(np.percentile(valid_sharp, 25)),
                "p75": float(np.percentile(valid_sharp, 75)),
            }
    
    if primary_occlusion is not None:
        valid_occ = primary_occlusion[np.isfinite(primary_occlusion)]
        if valid_occ.size > 0:
            distributions["primary_occlusion_proxy"] = {
                "min": float(np.min(valid_occ)),
                "max": float(np.max(valid_occ)),
                "mean": float(np.mean(valid_occ)),
                "std": float(np.std(valid_occ)),
                "median": float(np.median(valid_occ)),
                "p25": float(np.percentile(valid_occ, 25)),
                "p75": float(np.percentile(valid_occ, 75)),
            }
    
    if primary_speech is not None:
        valid_speech = primary_speech[np.isfinite(primary_speech)]
        if valid_speech.size > 0:
            distributions["primary_speech_activity_prob"] = {
                "min": float(np.min(valid_speech)),
                "max": float(np.max(valid_speech)),
                "mean": float(np.mean(valid_speech)),
                "std": float(np.std(valid_speech)),
                "median": float(np.median(valid_speech)),
                "p25": float(np.percentile(valid_speech, 25)),
                "p75": float(np.percentile(valid_speech, 75)),
            }
    
    render["distributions"] = distributions
    
    # Faces aggregate info
    if faces_agg is not None:
        if isinstance(faces_agg, np.ndarray) and faces_agg.dtype == object:
            faces_agg = faces_agg.item() if faces_agg.size == 1 else {}
        if isinstance(faces_agg, dict):
            render["faces"] = [
                {
                    "tracking_id": int(track_id),
                    "frames_count": int(agg.get("frames_count", 0)) if isinstance(agg, dict) else 0,
                }
                for track_id, agg in faces_agg.items()
            ]
    
    return render


def render_detalize_face_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага detalize_face результатов.
    
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
    render = render_detalize_face(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    faces = render.get("faces", [])
    
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
        face_counts = [t.get("face_count") for t in timeline if t.get("face_count") is not None]
        gaze_values = [t.get("primary_gaze_at_camera_prob") for t in timeline if t.get("primary_gaze_at_camera_prob") is not None]
        blink_values = [t.get("primary_blink_rate") for t in timeline if t.get("primary_blink_rate") is not None]
        attn_values = [t.get("primary_attention_score") for t in timeline if t.get("primary_attention_score") is not None]
        
        # Build datasets array
        datasets = []
        
        if face_counts:
            datasets.append({
                "label": "Face Count",
                "data": face_counts,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "tension": 0.1,
                "yAxisID": "y"
            })
        
        if gaze_values:
            datasets.append({
                "label": "Gaze at Camera",
                "data": gaze_values,
                "borderColor": "rgb(255, 99, 132)",
                "backgroundColor": "rgba(255, 99, 132, 0.2)",
                "tension": 0.1,
                "yAxisID": "y1"
            })
        
        if blink_values:
            datasets.append({
                "label": "Blink Rate",
                "data": blink_values,
                "borderColor": "rgb(54, 162, 235)",
                "backgroundColor": "rgba(54, 162, 235, 0.2)",
                "tension": 0.1,
                "yAxisID": "y1"
            })
        
        if attn_values:
            datasets.append({
                "label": "Attention Score",
                "data": attn_values,
                "borderColor": "rgb(255, 206, 86)",
                "backgroundColor": "rgba(255, 206, 86, 0.2)",
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
    <title>Detalize Face Debug Render</title>
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
        .faces-list {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Detalize Face Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Total Frames</strong>
                    <span class="metric-value">{summary.get('total_frames', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Processed Frames</strong>
                    <span class="metric-value">{summary.get('processed_frames', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Frames with Faces</strong>
                    <span class="metric-value">{summary.get('frames_with_faces', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Total Faces</strong>
                    <span class="metric-value">{summary.get('total_faces', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Primary Faces</strong>
                    <span class="metric-value">{summary.get('primary_faces', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Avg Faces per Frame</strong>
                    <span class="metric-value">{summary.get('avg_faces_per_frame', 0.0):.2f}</span>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Face Metrics Over Time</h2>
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
                        <th>Face Count</th>
                        <th>Gaze at Camera</th>
                        <th>Blink Rate</th>
                        <th>Attention Score</th>
                        <th>Quality Score</th>
                        <th>Sharpness</th>
                        <th>Occlusion</th>
                        <th>Speech Activity</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('face_count', 'mean')}</td>
                        <td>{format_dist_value('primary_gaze_at_camera_prob', 'mean')}</td>
                        <td>{format_dist_value('primary_blink_rate', 'mean')}</td>
                        <td>{format_dist_value('primary_attention_score', 'mean')}</td>
                        <td>{format_dist_value('primary_quality_proxy_score', 'mean')}</td>
                        <td>{format_dist_value('primary_face_sharpness', 'mean')}</td>
                        <td>{format_dist_value('primary_occlusion_proxy', 'mean')}</td>
                        <td>{format_dist_value('primary_speech_activity_prob', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('face_count', 'std')}</td>
                        <td>{format_dist_value('primary_gaze_at_camera_prob', 'std')}</td>
                        <td>{format_dist_value('primary_blink_rate', 'std')}</td>
                        <td>{format_dist_value('primary_attention_score', 'std')}</td>
                        <td>{format_dist_value('primary_quality_proxy_score', 'std')}</td>
                        <td>{format_dist_value('primary_face_sharpness', 'std')}</td>
                        <td>{format_dist_value('primary_occlusion_proxy', 'std')}</td>
                        <td>{format_dist_value('primary_speech_activity_prob', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('face_count', 'min')}</td>
                        <td>{format_dist_value('primary_gaze_at_camera_prob', 'min')}</td>
                        <td>{format_dist_value('primary_blink_rate', 'min')}</td>
                        <td>{format_dist_value('primary_attention_score', 'min')}</td>
                        <td>{format_dist_value('primary_quality_proxy_score', 'min')}</td>
                        <td>{format_dist_value('primary_face_sharpness', 'min')}</td>
                        <td>{format_dist_value('primary_occlusion_proxy', 'min')}</td>
                        <td>{format_dist_value('primary_speech_activity_prob', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('face_count', 'max')}</td>
                        <td>{format_dist_value('primary_gaze_at_camera_prob', 'max')}</td>
                        <td>{format_dist_value('primary_blink_rate', 'max')}</td>
                        <td>{format_dist_value('primary_attention_score', 'max')}</td>
                        <td>{format_dist_value('primary_quality_proxy_score', 'max')}</td>
                        <td>{format_dist_value('primary_face_sharpness', 'max')}</td>
                        <td>{format_dist_value('primary_occlusion_proxy', 'max')}</td>
                        <td>{format_dist_value('primary_speech_activity_prob', 'max')}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions else ''}
        
        {f'''
        <div class="faces-list">
            <h2>Tracked Faces</h2>
            <p>Total tracks: {len(faces)}</p>
            <ul>
                {''.join([f'<li>Track {f.get("tracking_id", 0)}: {f.get("frames_count", 0)} frames</li>' for f in faces[:10]])}
                {f'<li>... and {len(faces) - 10} more tracks</li>' if len(faces) > 10 else ''}
            </ul>
        </div>
        ''' if faces else ''}
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
                            text: 'Face Count'
                        }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'Metrics (0-1)'
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
    
    logger.info(f"Saved Detalize Face HTML render to {output_path}")
    return output_path


__all__ = ["render_detalize_face", "render_detalize_face_html"]

