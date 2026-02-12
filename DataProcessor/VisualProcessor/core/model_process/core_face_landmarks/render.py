"""
Renderer для core_face_landmarks: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_core_face_landmarks(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для core_face_landmarks."""
    render = {
        "component": "core_face_landmarks",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract landmarks data
    face_landmarks = npz_data.get("face_landmarks")
    face_present = npz_data.get("face_present")
    pose_landmarks = npz_data.get("pose_landmarks")
    pose_present = npz_data.get("pose_present")
    hands_landmarks = npz_data.get("hands_landmarks")
    hands_present = npz_data.get("hands_present")
    has_any_face = npz_data.get("has_any_face")
    has_any_pose = npz_data.get("has_any_pose")
    has_any_hands = npz_data.get("has_any_hands")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    
    # Convert to numpy arrays if needed
    if face_landmarks is not None:
        if isinstance(face_landmarks, list):
            face_landmarks = np.array(face_landmarks, dtype=np.float32)
        elif isinstance(face_landmarks, np.ndarray):
            face_landmarks = np.asarray(face_landmarks, dtype=np.float32)
        else:
            face_landmarks = None
    
    if face_present is not None:
        if isinstance(face_present, list):
            face_present = np.array(face_present, dtype=bool)
        elif isinstance(face_present, np.ndarray):
            face_present = np.asarray(face_present, dtype=bool)
        else:
            face_present = None
    
    if pose_landmarks is not None:
        if isinstance(pose_landmarks, list):
            pose_landmarks = np.array(pose_landmarks, dtype=np.float32)
        elif isinstance(pose_landmarks, np.ndarray):
            pose_landmarks = np.asarray(pose_landmarks, dtype=np.float32)
        else:
            pose_landmarks = None
    
    if pose_present is not None:
        if isinstance(pose_present, list):
            pose_present = np.array(pose_present, dtype=bool)
        elif isinstance(pose_present, np.ndarray):
            pose_present = np.asarray(pose_present, dtype=bool)
        else:
            pose_present = None
    
    if hands_landmarks is not None:
        if isinstance(hands_landmarks, list):
            hands_landmarks = np.array(hands_landmarks, dtype=np.float32)
        elif isinstance(hands_landmarks, np.ndarray):
            hands_landmarks = np.asarray(hands_landmarks, dtype=np.float32)
        else:
            hands_landmarks = None
    
    if hands_present is not None:
        if isinstance(hands_present, list):
            hands_present = np.array(hands_present, dtype=bool)
        elif isinstance(hands_present, np.ndarray):
            hands_present = np.asarray(hands_present, dtype=bool)
        else:
            hands_present = None
    
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
    n_frames = 0
    if frame_indices is not None:
        n_frames = len(frame_indices)
    elif face_landmarks is not None:
        n_frames = face_landmarks.shape[0] if face_landmarks.ndim >= 1 else 0
    elif pose_landmarks is not None:
        n_frames = pose_landmarks.shape[0] if pose_landmarks.ndim >= 1 else 0
    
    render["summary"] = {
        "frames_count": int(n_frames),
        "has_any_face": bool(has_any_face) if has_any_face is not None else False,
        "has_any_pose": bool(has_any_pose) if has_any_pose is not None else False,
        "has_any_hands": bool(has_any_hands) if has_any_hands is not None else False,
    }
    
    # Face statistics
    if face_present is not None and face_present.size > 0:
        if face_present.ndim == 1:
            # Single face per frame
            face_count_per_frame = face_present.astype(int)
        elif face_present.ndim == 2:
            # Multiple faces per frame
            face_count_per_frame = np.sum(face_present.astype(int), axis=1)
        else:
            face_count_per_frame = np.array([0])
        
        frames_with_face = np.sum(face_count_per_frame > 0)
        render["summary"]["face_frames_count"] = int(frames_with_face)
        render["summary"]["face_frames_percentage"] = float(frames_with_face / n_frames * 100) if n_frames > 0 else 0.0
        render["summary"]["face_count_mean"] = float(np.mean(face_count_per_frame))
        render["summary"]["face_count_max"] = int(np.max(face_count_per_frame)) if face_count_per_frame.size > 0 else 0
    else:
        render["summary"]["face_frames_count"] = 0
        render["summary"]["face_frames_percentage"] = 0.0
        render["summary"]["face_count_mean"] = 0.0
        render["summary"]["face_count_max"] = 0
    
    # Pose statistics
    if pose_present is not None and pose_present.size > 0:
        frames_with_pose = np.sum(pose_present.astype(int))
        render["summary"]["pose_frames_count"] = int(frames_with_pose)
        render["summary"]["pose_frames_percentage"] = float(frames_with_pose / n_frames * 100) if n_frames > 0 else 0.0
    else:
        render["summary"]["pose_frames_count"] = 0
        render["summary"]["pose_frames_percentage"] = 0.0
    
    # Hands statistics
    if hands_present is not None and hands_present.size > 0:
        if hands_present.ndim == 1:
            hands_count_per_frame = hands_present.astype(int)
        elif hands_present.ndim == 2:
            hands_count_per_frame = np.sum(hands_present.astype(int), axis=1)
        else:
            hands_count_per_frame = np.array([0])
        
        frames_with_hands = np.sum(hands_count_per_frame > 0)
        render["summary"]["hands_frames_count"] = int(frames_with_hands)
        render["summary"]["hands_frames_percentage"] = float(frames_with_hands / n_frames * 100) if n_frames > 0 else 0.0
        render["summary"]["hands_count_mean"] = float(np.mean(hands_count_per_frame))
        render["summary"]["hands_count_max"] = int(np.max(hands_count_per_frame)) if hands_count_per_frame.size > 0 else 0
    else:
        render["summary"]["hands_frames_count"] = 0
        render["summary"]["hands_frames_percentage"] = 0.0
        render["summary"]["hands_count_mean"] = 0.0
        render["summary"]["hands_count_max"] = 0
    
    # Timeline data
    if times_s is not None and frame_indices is not None and n_frames > 0:
        timeline = []
        
        for i in range(min(n_frames, len(times_s), len(frame_indices))):
            frame_idx = int(frame_indices[i]) if i < len(frame_indices) else i
            time_sec = float(times_s[i]) if i < len(times_s) else 0.0
            
            # Face presence
            has_face = False
            face_count = 0
            if face_present is not None:
                if face_present.ndim == 1 and i < len(face_present):
                    has_face = bool(face_present[i])
                    face_count = 1 if has_face else 0
                elif face_present.ndim == 2 and i < face_present.shape[0]:
                    has_face = bool(np.any(face_present[i]))
                    face_count = int(np.sum(face_present[i].astype(int)))
            
            # Pose presence
            has_pose = False
            if pose_present is not None and i < len(pose_present):
                has_pose = bool(pose_present[i])
            
            # Hands presence
            has_hands = False
            hands_count = 0
            if hands_present is not None:
                if hands_present.ndim == 1 and i < len(hands_present):
                    has_hands = bool(hands_present[i])
                    hands_count = 1 if has_hands else 0
                elif hands_present.ndim == 2 and i < hands_present.shape[0]:
                    has_hands = bool(np.any(hands_present[i]))
                    hands_count = int(np.sum(hands_present[i].astype(int)))
            
            timeline.append({
                "frame_index": frame_idx,
                "time_sec": time_sec,
                "has_face": has_face,
                "face_count": face_count,
                "has_pose": has_pose,
                "has_hands": has_hands,
                "hands_count": hands_count,
            })
        
        render["timeline"] = timeline
    
    # Distributions
    distributions = {}
    
    if face_present is not None and face_present.size > 0:
        if face_present.ndim == 1:
            face_counts = face_present.astype(int)
        elif face_present.ndim == 2:
            face_counts = np.sum(face_present.astype(int), axis=1)
        else:
            face_counts = np.array([0])
        
        if face_counts.size > 0:
            distributions["face_count_per_frame"] = {
                "min": int(np.min(face_counts)),
                "max": int(np.max(face_counts)),
                "mean": float(np.mean(face_counts)),
                "std": float(np.std(face_counts)),
                "median": float(np.median(face_counts)),
            }
    
    if hands_present is not None and hands_present.size > 0:
        if hands_present.ndim == 1:
            hands_counts = hands_present.astype(int)
        elif hands_present.ndim == 2:
            hands_counts = np.sum(hands_present.astype(int), axis=1)
        else:
            hands_counts = np.array([0])
        
        if hands_counts.size > 0:
            distributions["hands_count_per_frame"] = {
                "min": int(np.min(hands_counts)),
                "max": int(np.max(hands_counts)),
                "mean": float(np.mean(hands_counts)),
                "std": float(np.std(hands_counts)),
                "median": float(np.median(hands_counts)),
            }
    
    render["distributions"] = distributions
    
    return render


def render_core_face_landmarks_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага core_face_landmarks результатов.
    
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
    render = render_core_face_landmarks(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    
    # Prepare timeline data for chart
    timeline_js = ""
    if timeline:
        times = [t.get("time_sec", 0.0) for t in timeline]
        face_counts = [t.get("face_count", 0) for t in timeline]
        hands_counts = [t.get("hands_count", 0) for t in timeline]
        
        # Build datasets array
        datasets = []
        
        if any(face_counts):
            datasets.append({
                "label": "Face Count",
                "data": face_counts,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "tension": 0.1,
                "yAxisID": "y"
            })
        
        if any(hands_counts):
            datasets.append({
                "label": "Hands Count",
                "data": hands_counts,
                "borderColor": "rgb(255, 99, 132)",
                "backgroundColor": "rgba(255, 99, 132, 0.2)",
                "tension": 0.1,
                "yAxisID": "y"
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
    <title>Core Face Landmarks Debug Render</title>
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
        <h1>Core Face Landmarks Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Has Any Face</strong>
                    <span class="metric-value">{'Yes' if summary.get('has_any_face', False) else 'No'}</span>
                </div>
                <div class="metric-card">
                    <strong>Face Frames</strong>
                    <span class="metric-value">{summary.get('face_frames_count', 0)} ({summary.get('face_frames_percentage', 0.0):.1f}%)</span>
                </div>
                <div class="metric-card">
                    <strong>Face Count Mean</strong>
                    <span class="metric-value">{summary.get('face_count_mean', 0.0):.2f}</span>
                </div>
                <div class="metric-card">
                    <strong>Face Count Max</strong>
                    <span class="metric-value">{summary.get('face_count_max', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Has Any Pose</strong>
                    <span class="metric-value">{'Yes' if summary.get('has_any_pose', False) else 'No'}</span>
                </div>
                <div class="metric-card">
                    <strong>Pose Frames</strong>
                    <span class="metric-value">{summary.get('pose_frames_count', 0)} ({summary.get('pose_frames_percentage', 0.0):.1f}%)</span>
                </div>
                <div class="metric-card">
                    <strong>Has Any Hands</strong>
                    <span class="metric-value">{'Yes' if summary.get('has_any_hands', False) else 'No'}</span>
                </div>
                <div class="metric-card">
                    <strong>Hands Frames</strong>
                    <span class="metric-value">{summary.get('hands_frames_count', 0)} ({summary.get('hands_frames_percentage', 0.0):.1f}%)</span>
                </div>
                <div class="metric-card">
                    <strong>Hands Count Mean</strong>
                    <span class="metric-value">{summary.get('hands_count_mean', 0.0):.2f}</span>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Face and Hands Count Over Time</h2>
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
                        <th>Hands Count</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{distributions.get('face_count_per_frame', {}).get('min', 0)}</td>
                        <td>{distributions.get('hands_count_per_frame', {}).get('min', 0)}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{distributions.get('face_count_per_frame', {}).get('max', 0)}</td>
                        <td>{distributions.get('hands_count_per_frame', {}).get('max', 0)}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{distributions.get('face_count_per_frame', {}).get('mean', 0.0):.2f}</td>
                        <td>{distributions.get('hands_count_per_frame', {}).get('mean', 0.0):.2f}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{distributions.get('face_count_per_frame', {}).get('std', 0.0):.2f}</td>
                        <td>{distributions.get('hands_count_per_frame', {}).get('std', 0.0):.2f}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{distributions.get('face_count_per_frame', {}).get('median', 0.0):.2f}</td>
                        <td>{distributions.get('hands_count_per_frame', {}).get('median', 0.0):.2f}</td>
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
                            text: 'Count'
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
    
    logger.info(f"Saved Core Face Landmarks HTML render to {output_path}")
    return output_path


__all__ = ["render_core_face_landmarks", "render_core_face_landmarks_html"]

