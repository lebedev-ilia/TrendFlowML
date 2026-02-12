"""
Renderer для story_structure: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_story_structure(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для story_structure."""
    render = {
        "component": "story_structure",
        "summary": {},
        "timeline": [],
        "distributions": {},
        "markers": {},
        "peaks": {},
    }
    
    # Extract data
    story_energy_curve = npz_data.get("story_energy_curve")
    embedding_change_rate_per_sec = npz_data.get("embedding_change_rate_per_sec")
    motion_norm_per_sec_mean = npz_data.get("motion_norm_per_sec_mean")
    topic_shift_curve = npz_data.get("topic_shift_curve")
    any_face_present = npz_data.get("any_face_present")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    features = npz_data.get("features")
    
    # Convert to numpy arrays if needed
    if story_energy_curve is not None:
        if isinstance(story_energy_curve, list):
            story_energy_curve = np.array(story_energy_curve, dtype=np.float32)
        elif isinstance(story_energy_curve, np.ndarray):
            story_energy_curve = np.asarray(story_energy_curve, dtype=np.float32)
        else:
            story_energy_curve = None
    
    if embedding_change_rate_per_sec is not None:
        if isinstance(embedding_change_rate_per_sec, list):
            embedding_change_rate_per_sec = np.array(embedding_change_rate_per_sec, dtype=np.float32)
        elif isinstance(embedding_change_rate_per_sec, np.ndarray):
            embedding_change_rate_per_sec = np.asarray(embedding_change_rate_per_sec, dtype=np.float32)
        else:
            embedding_change_rate_per_sec = None
    
    if motion_norm_per_sec_mean is not None:
        if isinstance(motion_norm_per_sec_mean, list):
            motion_norm_per_sec_mean = np.array(motion_norm_per_sec_mean, dtype=np.float32)
        elif isinstance(motion_norm_per_sec_mean, np.ndarray):
            motion_norm_per_sec_mean = np.asarray(motion_norm_per_sec_mean, dtype=np.float32)
        else:
            motion_norm_per_sec_mean = None
    
    if topic_shift_curve is not None:
        if isinstance(topic_shift_curve, list):
            topic_shift_curve = np.array(topic_shift_curve, dtype=np.float32)
        elif isinstance(topic_shift_curve, np.ndarray):
            topic_shift_curve = np.asarray(topic_shift_curve, dtype=np.float32)
        else:
            topic_shift_curve = None
    
    if any_face_present is not None:
        if isinstance(any_face_present, list):
            any_face_present = np.array(any_face_present, dtype=bool)
        elif isinstance(any_face_present, np.ndarray):
            any_face_present = np.asarray(any_face_present, dtype=bool)
        else:
            any_face_present = None
    
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
    if story_energy_curve is not None and story_energy_curve.size > 0:
        valid_energy = story_energy_curve[np.isfinite(story_energy_curve)]
        
        render["summary"] = {
            "frames_count": int(story_energy_curve.shape[0]) if story_energy_curve.ndim >= 1 else 0,
            "video_length_seconds": float(features.get("video_length_seconds", 0.0)),
            "story_energy_mean": float(np.mean(valid_energy)) if valid_energy.size > 0 else 0.0,
            "story_energy_std": float(np.std(valid_energy)) if valid_energy.size > 0 else 0.0,
            "story_energy_min": float(np.min(valid_energy)) if valid_energy.size > 0 else 0.0,
            "story_energy_max": float(np.max(valid_energy)) if valid_energy.size > 0 else 0.0,
            "story_energy_median": float(np.median(valid_energy)) if valid_energy.size > 0 else 0.0,
            # Hook metrics
            "hook_visual_surprise_score": float(features.get("hook_visual_surprise_score", 0.0)),
            "hook_motion_intensity": float(features.get("hook_motion_intensity", 0.0)),
            "hook_face_presence": float(features.get("hook_face_presence", 0.0)),
            # Climax metrics
            "climax_time_sec": float(features.get("climax_time_sec", 0.0)),
            "climax_position_normalized": float(features.get("climax_position_normalized", 0.0)),
            "climax_strength_normalized": float(features.get("climax_strength_normalized", 0.0)),
            "number_of_peaks": int(features.get("number_of_peaks", 0)),
            # Character proxies
            "main_character_screen_time": float(features.get("main_character_screen_time", 0.0)),
            "speaker_switches_per_minute": float(features.get("speaker_switches_per_minute", 0.0)),
            # Text
            "topic_shift_curve_present": bool(features.get("topic_shift_curve_present", False)),
            "topic_shift_peaks_count": int(features.get("topic_shift_peaks_count", 0)),
        }
    
    # Timeline data
    if times_s is not None and story_energy_curve is not None:
        n = min(len(times_s), len(story_energy_curve))
        timeline = []
        
        for i in range(n):
            frame_idx = int(frame_indices[i]) if frame_indices is not None and i < len(frame_indices) else i
            time_sec = float(times_s[i]) if i < len(times_s) else 0.0
            energy = float(story_energy_curve[i]) if np.isfinite(story_energy_curve[i]) else None
            emb_rate = float(embedding_change_rate_per_sec[i]) if embedding_change_rate_per_sec is not None and i < len(embedding_change_rate_per_sec) and np.isfinite(embedding_change_rate_per_sec[i]) else None
            motion = float(motion_norm_per_sec_mean[i]) if motion_norm_per_sec_mean is not None and i < len(motion_norm_per_sec_mean) and np.isfinite(motion_norm_per_sec_mean[i]) else None
            topic_shift = float(topic_shift_curve[i]) if topic_shift_curve is not None and i < len(topic_shift_curve) and np.isfinite(topic_shift_curve[i]) else None
            face_present = bool(any_face_present[i]) if any_face_present is not None and i < len(any_face_present) else None
            
            timeline.append({
                "frame_index": frame_idx,
                "time_sec": time_sec,
                "story_energy": energy,
                "embedding_change_rate_per_sec": emb_rate,
                "motion_norm_per_sec_mean": motion,
                "topic_shift_curve": topic_shift,
                "any_face_present": face_present,
            })
        
        render["timeline"] = timeline
    
    # Markers (hook window, climax)
    if features:
        hook_end_t = float(features.get("hook_end_time", 0.0))
        if hook_end_t == 0.0 and times_s is not None and times_s.size > 0:
            # Estimate hook window from video length
            video_len = float(features.get("video_length_seconds", 0.0))
            hook_len_s = min(5.0, 0.15 * video_len) if video_len > 0 else 5.0
            hook_end_t = float(times_s[0] + hook_len_s) if times_s.size > 0 else 0.0
        
        render["markers"] = {
            "hook_window": {
                "t_start_s": float(times_s[0]) if times_s is not None and times_s.size > 0 else 0.0,
                "t_end_s": hook_end_t,
            },
            "climax": {
                "t_s": float(features.get("climax_time_sec", 0.0)),
                "frame_index": int(features.get("climax_timestamp", -1)),
                "strength_z": float(features.get("climax_strength_normalized", 0.0)),
            },
        }
    
    # Peaks
    story_energy_peaks_idx = npz_data.get("story_energy_peaks_idx")
    story_energy_peaks_times_s = npz_data.get("story_energy_peaks_times_s")
    story_energy_peaks_values_z = npz_data.get("story_energy_peaks_values_z")
    
    if story_energy_peaks_idx is not None:
        if isinstance(story_energy_peaks_idx, list):
            story_energy_peaks_idx = np.array(story_energy_peaks_idx, dtype=np.int32)
        elif isinstance(story_energy_peaks_idx, np.ndarray):
            story_energy_peaks_idx = np.asarray(story_energy_peaks_idx, dtype=np.int32)
        
        if isinstance(story_energy_peaks_times_s, list):
            story_energy_peaks_times_s = np.array(story_energy_peaks_times_s, dtype=np.float32)
        elif isinstance(story_energy_peaks_times_s, np.ndarray):
            story_energy_peaks_times_s = np.asarray(story_energy_peaks_times_s, dtype=np.float32)
        
        if isinstance(story_energy_peaks_values_z, list):
            story_energy_peaks_values_z = np.array(story_energy_peaks_values_z, dtype=np.float32)
        elif isinstance(story_energy_peaks_values_z, np.ndarray):
            story_energy_peaks_values_z = np.asarray(story_energy_peaks_values_z, dtype=np.float32)
        
        peaks = []
        n_peaks = len(story_energy_peaks_idx) if story_energy_peaks_idx is not None else 0
        for i in range(n_peaks):
            peak_idx = int(story_energy_peaks_idx[i]) if i < len(story_energy_peaks_idx) else -1
            peak_time = float(story_energy_peaks_times_s[i]) if story_energy_peaks_times_s is not None and i < len(story_energy_peaks_times_s) else 0.0
            peak_value = float(story_energy_peaks_values_z[i]) if story_energy_peaks_values_z is not None and i < len(story_energy_peaks_values_z) else 0.0
            peaks.append({
                "index": peak_idx,
                "time_sec": peak_time,
                "value_z": peak_value,
            })
        
        render["peaks"] = {
            "energy": peaks,
        }
    
    # Distributions
    distributions = {}
    
    if story_energy_curve is not None:
        valid_energy = story_energy_curve[np.isfinite(story_energy_curve)]
        if valid_energy.size > 0:
            distributions["story_energy_curve"] = {
                "min": float(np.min(valid_energy)),
                "max": float(np.max(valid_energy)),
                "mean": float(np.mean(valid_energy)),
                "std": float(np.std(valid_energy)),
                "median": float(np.median(valid_energy)),
                "p25": float(np.percentile(valid_energy, 25)),
                "p75": float(np.percentile(valid_energy, 75)),
                "p05": float(np.percentile(valid_energy, 5)),
                "p95": float(np.percentile(valid_energy, 95)),
            }
    
    if embedding_change_rate_per_sec is not None:
        valid_emb = embedding_change_rate_per_sec[np.isfinite(embedding_change_rate_per_sec)]
        if valid_emb.size > 0:
            distributions["embedding_change_rate_per_sec"] = {
                "min": float(np.min(valid_emb)),
                "max": float(np.max(valid_emb)),
                "mean": float(np.mean(valid_emb)),
                "std": float(np.std(valid_emb)),
                "median": float(np.median(valid_emb)),
                "p25": float(np.percentile(valid_emb, 25)),
                "p75": float(np.percentile(valid_emb, 75)),
            }
    
    if motion_norm_per_sec_mean is not None:
        valid_motion = motion_norm_per_sec_mean[np.isfinite(motion_norm_per_sec_mean)]
        if valid_motion.size > 0:
            distributions["motion_norm_per_sec_mean"] = {
                "min": float(np.min(valid_motion)),
                "max": float(np.max(valid_motion)),
                "mean": float(np.mean(valid_motion)),
                "std": float(np.std(valid_motion)),
                "median": float(np.median(valid_motion)),
                "p25": float(np.percentile(valid_motion, 25)),
                "p75": float(np.percentile(valid_motion, 75)),
            }
    
    if topic_shift_curve is not None:
        valid_topic = topic_shift_curve[np.isfinite(topic_shift_curve)]
        if valid_topic.size > 0:
            distributions["topic_shift_curve"] = {
                "min": float(np.min(valid_topic)),
                "max": float(np.max(valid_topic)),
                "mean": float(np.mean(valid_topic)),
                "std": float(np.std(valid_topic)),
                "median": float(np.median(valid_topic)),
                "p25": float(np.percentile(valid_topic, 25)),
                "p75": float(np.percentile(valid_topic, 75)),
            }
    
    render["distributions"] = distributions
    
    return render


def render_story_structure_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага story_structure результатов.
    
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
    render = render_story_structure(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    markers = render.get("markers", {})
    peaks = render.get("peaks", {})
    
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
        energies = [t.get("story_energy") for t in timeline if t.get("story_energy") is not None]
        motions = [t.get("motion_norm_per_sec_mean") for t in timeline if t.get("motion_norm_per_sec_mean") is not None]
        emb_rates = [t.get("embedding_change_rate_per_sec") for t in timeline if t.get("embedding_change_rate_per_sec") is not None]
        
        # Build datasets array
        datasets = []
        
        if energies:
            datasets.append({
                "label": "Story Energy (z-score)",
                "data": energies,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "tension": 0.1,
                "yAxisID": "y"
            })
        
        if motions:
            datasets.append({
                "label": "Motion (per-sec mean)",
                "data": motions,
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
                            text: 'Motion'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }"""
        
        if emb_rates:
            datasets.append({
                "label": "Embedding Change Rate (/s)",
                "data": emb_rates,
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
                            text: 'Embedding Change Rate'
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
    
    # Prepare peaks data
    peaks_js = ""
    if peaks and "energy" in peaks:
        energy_peaks = peaks["energy"]
        if energy_peaks:
            peaks_js = f"""
        const peaksData = {json.dumps(energy_peaks)};
        """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Story Structure Debug Render</title>
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
        .markers {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; border: 1px solid #ffc107; }}
        .peaks {{ background-color: #d1ecf1; padding: 15px; border-radius: 5px; margin: 20px 0; border: 1px solid #0dcaf0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Story Structure Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Video Length</strong>
                    <span class="metric-value">{summary.get('video_length_seconds', 0.0):.2f}s</span>
                </div>
                <div class="metric-card">
                    <strong>Hook Visual Surprise</strong>
                    <span class="metric-value">{summary.get('hook_visual_surprise_score', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Climax Time</strong>
                    <span class="metric-value">{summary.get('climax_time_sec', 0.0):.2f}s</span>
                </div>
                <div class="metric-card">
                    <strong>Climax Position</strong>
                    <span class="metric-value">{summary.get('climax_position_normalized', 0.0):.2%}</span>
                </div>
                <div class="metric-card">
                    <strong>Number of Peaks</strong>
                    <span class="metric-value">{summary.get('number_of_peaks', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Main Character Screen Time</strong>
                    <span class="metric-value">{summary.get('main_character_screen_time', 0.0):.2%}</span>
                </div>
                <div class="metric-card">
                    <strong>Topic Shift Present</strong>
                    <span class="metric-value">{'Yes' if summary.get('topic_shift_curve_present', False) else 'No'}</span>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="markers">
            <h2>Markers</h2>
            <p><strong>Hook Window:</strong> {markers.get('hook_window', {}).get('t_start_s', 0.0):.2f}s - {markers.get('hook_window', {}).get('t_end_s', 0.0):.2f}s</p>
            <p><strong>Climax:</strong> {markers.get('climax', {}).get('t_s', 0.0):.2f}s (frame {markers.get('climax', {}).get('frame_index', -1)}, strength: {markers.get('climax', {}).get('strength_z', 0.0):.4f})</p>
        </div>
        ''' if markers else ''}
        
        {f'''
        <div class="peaks">
            <h2>Energy Peaks</h2>
            <p>Found {len(peaks.get('energy', []))} peaks above 90th percentile</p>
        </div>
        ''' if peaks and peaks.get('energy') else ''}
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Story Energy and Motion Over Time</h2>
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
                        <th>Story Energy</th>
                        <th>Embedding Change Rate</th>
                        <th>Motion</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('story_energy_curve', 'min')}</td>
                        <td>{format_dist_value('embedding_change_rate_per_sec', 'min')}</td>
                        <td>{format_dist_value('motion_norm_per_sec_mean', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('story_energy_curve', 'max')}</td>
                        <td>{format_dist_value('embedding_change_rate_per_sec', 'max')}</td>
                        <td>{format_dist_value('motion_norm_per_sec_mean', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('story_energy_curve', 'mean')}</td>
                        <td>{format_dist_value('embedding_change_rate_per_sec', 'mean')}</td>
                        <td>{format_dist_value('motion_norm_per_sec_mean', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('story_energy_curve', 'std')}</td>
                        <td>{format_dist_value('embedding_change_rate_per_sec', 'std')}</td>
                        <td>{format_dist_value('motion_norm_per_sec_mean', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('story_energy_curve', 'median')}</td>
                        <td>{format_dist_value('embedding_change_rate_per_sec', 'median')}</td>
                        <td>{format_dist_value('motion_norm_per_sec_mean', 'median')}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions else ''}
    </div>
    
    {f'''
    <script>
        {timeline_js}
        {peaks_js}
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
                            text: 'Story Energy (z-score)'
                        }}
                    }}{y1_scale_js}{y2_scale_js}
                }},
                plugins: {{
                    annotation: {{
                        annotations: {{
                            hookWindow: {{
                                type: 'box',
                                xMin: {markers.get('hook_window', {}).get('t_start_s', 0.0)},
                                xMax: {markers.get('hook_window', {}).get('t_end_s', 0.0)},
                                backgroundColor: 'rgba(0, 255, 0, 0.1)',
                                borderColor: 'rgba(0, 255, 0, 0.5)',
                            }},
                            climax: {{
                                type: 'line',
                                xMin: {markers.get('climax', {}).get('t_s', 0.0)},
                                xMax: {markers.get('climax', {}).get('t_s', 0.0)},
                                borderColor: 'rgba(255, 0, 0, 0.8)',
                                borderWidth: 2,
                            }}
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
    logger.info(f"Saved Story Structure HTML render to {rel_output_path}")
    return output_path


__all__ = ["render_story_structure", "render_story_structure_html"]

