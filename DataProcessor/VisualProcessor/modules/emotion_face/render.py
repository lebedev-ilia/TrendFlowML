"""
Renderer для emotion_face: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_emotion_face(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для emotion_face."""
    render = {
        "component": "emotion_face",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract emotion data
    sequence_features = npz_data.get("sequence_features", {})
    if not isinstance(sequence_features, dict):
        sequence_features = {}
    
    frame_indices = sequence_features.get("frame_indices")
    times_s = sequence_features.get("times_s")
    valence = sequence_features.get("valence")
    arousal = sequence_features.get("arousal")
    intensity = sequence_features.get("intensity")
    emotion_confidence = sequence_features.get("emotion_confidence")
    emotion_probs = sequence_features.get("emotion_probs")
    dominant_emotion_id = sequence_features.get("dominant_emotion_id")
    face_count = sequence_features.get("face_count")
    
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
    
    if valence is not None:
        if isinstance(valence, list):
            valence = np.array(valence, dtype=np.float32)
        elif isinstance(valence, np.ndarray):
            valence = np.asarray(valence, dtype=np.float32)
        else:
            valence = None
    
    if arousal is not None:
        if isinstance(arousal, list):
            arousal = np.array(arousal, dtype=np.float32)
        elif isinstance(arousal, np.ndarray):
            arousal = np.asarray(arousal, dtype=np.float32)
        else:
            arousal = None
    
    if intensity is not None:
        if isinstance(intensity, list):
            intensity = np.array(intensity, dtype=np.float32)
        elif isinstance(intensity, np.ndarray):
            intensity = np.asarray(intensity, dtype=np.float32)
        else:
            intensity = None
    
    if emotion_confidence is not None:
        if isinstance(emotion_confidence, list):
            emotion_confidence = np.array(emotion_confidence, dtype=np.float32)
        elif isinstance(emotion_confidence, np.ndarray):
            emotion_confidence = np.asarray(emotion_confidence, dtype=np.float32)
        else:
            emotion_confidence = None
    
    # Summary statistics
    if frame_indices is not None and frame_indices.size > 0:
        n_frames = int(frame_indices.shape[0])
        
        render["summary"] = {
            "frames_count": n_frames,
        }
        
        if valence is not None and valence.size > 0:
            valid_valence = valence[np.isfinite(valence)]
            if valid_valence.size > 0:
                render["summary"]["valence_mean"] = float(np.mean(valid_valence))
                render["summary"]["valence_std"] = float(np.std(valid_valence))
                render["summary"]["valence_min"] = float(np.min(valid_valence))
                render["summary"]["valence_max"] = float(np.max(valid_valence))
        
        if arousal is not None and arousal.size > 0:
            valid_arousal = arousal[np.isfinite(arousal)]
            if valid_arousal.size > 0:
                render["summary"]["arousal_mean"] = float(np.mean(valid_arousal))
                render["summary"]["arousal_std"] = float(np.std(valid_arousal))
                render["summary"]["arousal_min"] = float(np.min(valid_arousal))
                render["summary"]["arousal_max"] = float(np.max(valid_arousal))
        
        if intensity is not None and intensity.size > 0:
            valid_intensity = intensity[np.isfinite(intensity)]
            if valid_intensity.size > 0:
                render["summary"]["intensity_mean"] = float(np.mean(valid_intensity))
                render["summary"]["intensity_std"] = float(np.std(valid_intensity))
                render["summary"]["intensity_min"] = float(np.min(valid_intensity))
                render["summary"]["intensity_max"] = float(np.max(valid_intensity))
        
        if emotion_confidence is not None and emotion_confidence.size > 0:
            valid_conf = emotion_confidence[np.isfinite(emotion_confidence)]
            if valid_conf.size > 0:
                render["summary"]["emotion_confidence_mean"] = float(np.mean(valid_conf))
                render["summary"]["emotion_confidence_std"] = float(np.std(valid_conf))
        
        if face_count is not None and face_count.size > 0:
            valid_face_count = face_count[face_count >= 0]
            if valid_face_count.size > 0:
                render["summary"]["face_count_mean"] = float(np.mean(valid_face_count))
                render["summary"]["face_count_max"] = int(np.max(valid_face_count))
                render["summary"]["faces_found_frames"] = int(np.sum(valid_face_count > 0))
        
        # Dominant emotion distribution
        if dominant_emotion_id is not None and dominant_emotion_id.size > 0:
            emotion_classes = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
            valid_emotion_ids = dominant_emotion_id[dominant_emotion_id >= 0]
            if valid_emotion_ids.size > 0:
                unique, counts = np.unique(valid_emotion_ids, return_counts=True)
                emotion_dist = {}
                for emo_id, count in zip(unique, counts):
                    if 0 <= int(emo_id) < len(emotion_classes):
                        emotion_dist[emotion_classes[int(emo_id)]] = int(count)
                render["summary"]["dominant_emotion_distribution"] = emotion_dist
    
    # Timeline data (per-frame statistics)
    if frame_indices is not None and times_s is not None and valence is not None and arousal is not None:
        n = min(frame_indices.size, times_s.size, valence.size, arousal.size)
        if n > 0:
            timeline = []
            for i in range(n):
                if not (np.isfinite(times_s[i]) and np.isfinite(valence[i]) and np.isfinite(arousal[i])):
                    continue
                timeline.append({
                    "frame_index": int(frame_indices[i]),
                    "time_s": float(times_s[i]),
                    "valence": float(valence[i]),
                    "arousal": float(arousal[i]),
                    "intensity": float(intensity[i]) if intensity is not None and i < intensity.size and np.isfinite(intensity[i]) else None,
                    "emotion_confidence": float(emotion_confidence[i]) if emotion_confidence is not None and i < emotion_confidence.size and np.isfinite(emotion_confidence[i]) else None,
                    "face_count": int(face_count[i]) if face_count is not None and i < face_count.size else 0,
                })
            render["timeline"] = timeline
    
    # Distributions
    distributions = {}
    
    if valence is not None and valence.size > 0:
        valid_valence = valence[np.isfinite(valence)]
        if valid_valence.size > 0:
            distributions["valence"] = {
                "mean": float(np.mean(valid_valence)),
                "std": float(np.std(valid_valence)),
                "min": float(np.min(valid_valence)),
                "max": float(np.max(valid_valence)),
                "median": float(np.median(valid_valence)),
                "p25": float(np.percentile(valid_valence, 25)),
                "p75": float(np.percentile(valid_valence, 75)),
            }
    
    if arousal is not None and arousal.size > 0:
        valid_arousal = arousal[np.isfinite(arousal)]
        if valid_arousal.size > 0:
            distributions["arousal"] = {
                "mean": float(np.mean(valid_arousal)),
                "std": float(np.std(valid_arousal)),
                "min": float(np.min(valid_arousal)),
                "max": float(np.max(valid_arousal)),
                "median": float(np.median(valid_arousal)),
                "p25": float(np.percentile(valid_arousal, 25)),
                "p75": float(np.percentile(valid_arousal, 75)),
            }
    
    if intensity is not None and intensity.size > 0:
        valid_intensity = intensity[np.isfinite(intensity)]
        if valid_intensity.size > 0:
            distributions["intensity"] = {
                "mean": float(np.mean(valid_intensity)),
                "std": float(np.std(valid_intensity)),
                "min": float(np.min(valid_intensity)),
                "max": float(np.max(valid_intensity)),
                "median": float(np.median(valid_intensity)),
                "p25": float(np.percentile(valid_intensity, 25)),
                "p75": float(np.percentile(valid_intensity, 75)),
            }
    
    render["distributions"] = distributions
    
    return render


def render_emotion_face_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага emotion_face результатов.
    
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
    render = render_emotion_face(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    
    # Prepare timeline data for Chart.js
    timeline_js = ""
    if timeline:
        times = [t["time_s"] for t in timeline]
        valence_data = [t["valence"] for t in timeline]
        arousal_data = [t["arousal"] for t in timeline]
        intensity_data = [t.get("intensity") or 0.0 for t in timeline]
        
        timeline_js = f"""
        const timelineData = {{
            labels: {json.dumps(times)},
            datasets: [
                {{
                    label: 'Valence',
                    data: {json.dumps(valence_data)},
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    yAxisID: 'y',
                }},
                {{
                    label: 'Arousal',
                    data: {json.dumps(arousal_data)},
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    yAxisID: 'y',
                }},
                {{
                    label: 'Intensity',
                    data: {json.dumps(intensity_data)},
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    yAxisID: 'y1',
                }}
            ]
        }};
        """
    
    # Format distribution values
    def format_dist_value(category: str, stat: str) -> str:
        if category not in distributions:
            return "-"
        val = distributions[category].get(stat)
        if val is None:
            return "-"
        return f"{val:.4f}"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Face Debug Render</title>
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
        <h1>Emotion Face Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Valence Mean</strong>
                    <span class="metric-value">{summary.get('valence_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Arousal Mean</strong>
                    <span class="metric-value">{summary.get('arousal_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Intensity Mean</strong>
                    <span class="metric-value">{summary.get('intensity_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Emotion Confidence Mean</strong>
                    <span class="metric-value">{summary.get('emotion_confidence_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Faces Found Frames</strong>
                    <span class="metric-value">{summary.get('faces_found_frames', 0)}</span>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Emotion Metrics Over Time</h2>
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
                        <th>Valence</th>
                        <th>Arousal</th>
                        <th>Intensity</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('valence', 'mean')}</td>
                        <td>{format_dist_value('arousal', 'mean')}</td>
                        <td>{format_dist_value('intensity', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('valence', 'std')}</td>
                        <td>{format_dist_value('arousal', 'std')}</td>
                        <td>{format_dist_value('intensity', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('valence', 'min')}</td>
                        <td>{format_dist_value('arousal', 'min')}</td>
                        <td>{format_dist_value('intensity', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('valence', 'max')}</td>
                        <td>{format_dist_value('arousal', 'max')}</td>
                        <td>{format_dist_value('intensity', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('valence', 'median')}</td>
                        <td>{format_dist_value('arousal', 'median')}</td>
                        <td>{format_dist_value('intensity', 'median')}</td>
                    </tr>
                    <tr>
                        <td><strong>P25</strong></td>
                        <td>{format_dist_value('valence', 'p25')}</td>
                        <td>{format_dist_value('arousal', 'p25')}</td>
                        <td>{format_dist_value('intensity', 'p25')}</td>
                    </tr>
                    <tr>
                        <td><strong>P75</strong></td>
                        <td>{format_dist_value('valence', 'p75')}</td>
                        <td>{format_dist_value('arousal', 'p75')}</td>
                        <td>{format_dist_value('intensity', 'p75')}</td>
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
                            text: 'Valence / Arousal'
                        }}
                    }},
                    y1: {{
                        type: 'linear',
                        position: 'right',
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Intensity'
                        }},
                        grid: {{
                            drawOnChartArea: false,
                        }},
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
    logger.info(f"Saved Emotion Face HTML render to {rel_output_path}")
    return output_path


__all__ = ["render_emotion_face", "render_emotion_face_html"]

