"""
Renderer для core_object_detections: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_core_object_detections(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для core_object_detections."""
    render = {
        "component": "core_object_detections",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract detection data
    boxes = npz_data.get("boxes")
    scores = npz_data.get("scores")
    class_ids = npz_data.get("class_ids")
    valid_mask = npz_data.get("valid_mask")
    class_names = npz_data.get("class_names")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    
    # Convert to numpy arrays if needed
    if boxes is not None:
        if isinstance(boxes, list):
            boxes = np.array(boxes, dtype=np.float32)
        elif isinstance(boxes, np.ndarray):
            boxes = np.asarray(boxes, dtype=np.float32)
        else:
            boxes = None
    
    if scores is not None:
        if isinstance(scores, list):
            scores = np.array(scores, dtype=np.float32)
        elif isinstance(scores, np.ndarray):
            scores = np.asarray(scores, dtype=np.float32)
        else:
            scores = None
    
    if class_ids is not None:
        if isinstance(class_ids, list):
            class_ids = np.array(class_ids, dtype=np.int32)
        elif isinstance(class_ids, np.ndarray):
            class_ids = np.asarray(class_ids, dtype=np.int32)
        else:
            class_ids = None
    
    if valid_mask is not None:
        if isinstance(valid_mask, list):
            valid_mask = np.array(valid_mask, dtype=bool)
        elif isinstance(valid_mask, np.ndarray):
            valid_mask = np.asarray(valid_mask, dtype=bool)
        else:
            valid_mask = None
    
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
    
    # Parse class names
    class_names_dict = {}
    if class_names is not None:
        if isinstance(class_names, np.ndarray):
            for item in class_names:
                if isinstance(item, str):
                    parts = item.split(":", 1)
                    if len(parts) == 2:
                        try:
                            cls_id = int(parts[0])
                            cls_name = parts[1]
                            class_names_dict[cls_id] = cls_name
                        except ValueError:
                            pass
    
    # Summary statistics
    if boxes is not None and valid_mask is not None and boxes.size > 0:
        n_frames = boxes.shape[0] if boxes.ndim >= 2 else 1
        
        # Count detections per frame
        detections_per_frame = np.sum(valid_mask, axis=1) if valid_mask.ndim == 2 else np.array([np.sum(valid_mask)])
        
        # Extract valid scores
        valid_scores = []
        if scores is not None and valid_mask is not None:
            for i in range(n_frames):
                if i < len(valid_mask):
                    frame_valid = valid_mask[i]
                    if i < len(scores):
                        frame_scores = scores[i]
                        valid_scores.extend(frame_scores[frame_valid].tolist())
        
        # Extract valid boxes areas
        valid_areas = []
        if boxes is not None and valid_mask is not None:
            for i in range(n_frames):
                if i < len(valid_mask):
                    frame_valid = valid_mask[i]
                    if i < len(boxes):
                        frame_boxes = boxes[i]
                        for j in range(len(frame_boxes)):
                            if j < len(frame_valid) and frame_valid[j]:
                                box = frame_boxes[j]
                                if len(box) >= 4:
                                    area = float((box[2] - box[0]) * (box[3] - box[1]))
                                    if area > 0:
                                        valid_areas.append(area)
        
        # Class distribution
        class_distribution = {}
        if class_ids is not None and valid_mask is not None:
            for i in range(n_frames):
                if i < len(valid_mask) and i < len(class_ids):
                    frame_valid = valid_mask[i]
                    frame_class_ids = class_ids[i]
                    for j in range(len(frame_class_ids)):
                        if j < len(frame_valid) and frame_valid[j]:
                            cls_id = int(frame_class_ids[j])
                            class_distribution[cls_id] = class_distribution.get(cls_id, 0) + 1
        
        render["summary"] = {
            "frames_count": int(n_frames),
            "total_detections": int(np.sum(valid_mask)) if valid_mask is not None else 0,
            "detections_per_frame_mean": float(np.mean(detections_per_frame)) if detections_per_frame.size > 0 else 0.0,
            "detections_per_frame_std": float(np.std(detections_per_frame)) if detections_per_frame.size > 0 else 0.0,
            "detections_per_frame_min": int(np.min(detections_per_frame)) if detections_per_frame.size > 0 else 0,
            "detections_per_frame_max": int(np.max(detections_per_frame)) if detections_per_frame.size > 0 else 0,
            "unique_classes_count": len(class_distribution),
        }
        
        if valid_scores:
            render["summary"]["score_mean"] = float(np.mean(valid_scores))
            render["summary"]["score_std"] = float(np.std(valid_scores))
            render["summary"]["score_min"] = float(np.min(valid_scores))
            render["summary"]["score_max"] = float(np.max(valid_scores))
            render["summary"]["score_median"] = float(np.median(valid_scores))
        
        if valid_areas:
            render["summary"]["box_area_mean"] = float(np.mean(valid_areas))
            render["summary"]["box_area_std"] = float(np.std(valid_areas))
            render["summary"]["box_area_min"] = float(np.min(valid_areas))
            render["summary"]["box_area_max"] = float(np.max(valid_areas))
            render["summary"]["box_area_median"] = float(np.median(valid_areas))
        
        # Top classes
        if class_distribution:
            top_classes = sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
            render["summary"]["top_classes"] = [
                {
                    "class_id": int(cls_id),
                    "class_name": class_names_dict.get(cls_id, f"class_{cls_id}"),
                    "count": int(count),
                }
                for cls_id, count in top_classes
            ]
        
        # Timeline data
        if times_s is not None and len(times_s) == n_frames:
            timeline = []
            for i in range(n_frames):
                time_sec = float(times_s[i]) if i < len(times_s) else 0.0
                frame_idx = int(frame_indices[i]) if frame_indices is not None and i < len(frame_indices) else i
                det_count = int(detections_per_frame[i]) if i < len(detections_per_frame) else 0
                
                # Average score for this frame
                avg_score = None
                if scores is not None and valid_mask is not None and i < len(scores) and i < len(valid_mask):
                    frame_valid = valid_mask[i]
                    frame_scores = scores[i]
                    if np.any(frame_valid):
                        avg_score = float(np.mean(frame_scores[frame_valid]))
                
                timeline.append({
                    "frame_index": frame_idx,
                    "time_sec": time_sec,
                    "detections_count": det_count,
                    "average_score": avg_score,
                })
            
            render["timeline"] = timeline
        
        # Distribution statistics
        distributions = {}
        
        if detections_per_frame.size > 0:
            distributions["detections_per_frame"] = {
                "min": int(np.min(detections_per_frame)),
                "max": int(np.max(detections_per_frame)),
                "mean": float(np.mean(detections_per_frame)),
                "std": float(np.std(detections_per_frame)),
                "median": float(np.median(detections_per_frame)),
                "p25": float(np.percentile(detections_per_frame, 25)),
                "p75": float(np.percentile(detections_per_frame, 75)),
            }
        
        if valid_scores:
            distributions["scores"] = {
                "min": float(np.min(valid_scores)),
                "max": float(np.max(valid_scores)),
                "mean": float(np.mean(valid_scores)),
                "std": float(np.std(valid_scores)),
                "median": float(np.median(valid_scores)),
                "p25": float(np.percentile(valid_scores, 25)),
                "p75": float(np.percentile(valid_scores, 75)),
            }
        
        if valid_areas:
            distributions["box_areas"] = {
                "min": float(np.min(valid_areas)),
                "max": float(np.max(valid_areas)),
                "mean": float(np.mean(valid_areas)),
                "std": float(np.std(valid_areas)),
                "median": float(np.median(valid_areas)),
                "p25": float(np.percentile(valid_areas, 25)),
                "p75": float(np.percentile(valid_areas, 75)),
            }
        
        render["distributions"] = distributions
    
    return render


def render_core_object_detections_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага core_object_detections результатов.
    
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
    render = render_core_object_detections(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    
    # Helper function to format distribution values
    def format_dist_value(dist_key, stat_key):
        dist = distributions.get(dist_key)
        if dist and stat_key in dist:
            return f"{dist[stat_key]:.4f}" if isinstance(dist[stat_key], float) else str(dist[stat_key])
        return "N/A"
    
    # Prepare timeline data for chart
    timeline_js = ""
    if timeline:
        times = [t.get("time_sec", 0.0) for t in timeline]
        det_counts = [t.get("detections_count", 0) for t in timeline]
        avg_scores = [t.get("average_score") for t in timeline if t.get("average_score") is not None]
        
        # Build datasets array
        datasets = [{
            "label": "Detections Count",
            "data": det_counts,
            "borderColor": "rgb(75, 192, 192)",
            "backgroundColor": "rgba(75, 192, 192, 0.2)",
            "tension": 0.1,
            "yAxisID": "y"
        }]
        
        if avg_scores:
            datasets.append({
                "label": "Average Score",
                "data": avg_scores,
                "borderColor": "rgb(255, 99, 132)",
                "backgroundColor": "rgba(255, 99, 132, 0.2)",
                "tension": 0.1,
                "yAxisID": "y1"
            })
        
        timeline_js = f"""
        const timelineData = {{
            labels: {json.dumps([f"{t:.2f}s" for t in times])},
            datasets: {json.dumps(datasets)}
        }};
        """
    
    # Top classes table
    top_classes_html = ""
    if summary.get("top_classes"):
        top_classes_rows = ""
        for cls_info in summary["top_classes"]:
            top_classes_rows += f"""
                    <tr>
                        <td>{cls_info.get('class_id', 'N/A')}</td>
                        <td>{cls_info.get('class_name', 'N/A')}</td>
                        <td>{cls_info.get('count', 0)}</td>
                    </tr>
            """
        top_classes_html = f"""
        <div class="top-classes">
            <h2>Top Classes</h2>
            <table>
                <thead>
                    <tr>
                        <th>Class ID</th>
                        <th>Class Name</th>
                        <th>Count</th>
                    </tr>
                </thead>
                <tbody>
                    {top_classes_rows}
                </tbody>
            </table>
        </div>
        """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Core Object Detections Debug Render</title>
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
        .distributions table, .top-classes table {{ width: 100%; border-collapse: collapse; }}
        .distributions th, .distributions td, .top-classes th, .top-classes td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .distributions th, .top-classes th {{ background-color: #0056b3; color: white; }}
        .top-classes {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Core Object Detections Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Total Detections</strong>
                    <span class="metric-value">{summary.get('total_detections', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Detections/Frame (Mean)</strong>
                    <span class="metric-value">{summary.get('detections_per_frame_mean', 0.0):.2f}</span>
                </div>
                <div class="metric-card">
                    <strong>Unique Classes</strong>
                    <span class="metric-value">{summary.get('unique_classes_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Score Mean</strong>
                    <span class="metric-value">{summary.get('score_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Score Median</strong>
                    <span class="metric-value">{summary.get('score_median', 0.0):.4f}</span>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Detections Over Time</h2>
            <canvas id="timelineChart"></canvas>
        </div>
        ''' if timeline else '<p>No timeline data available</p>'}
        
        {top_classes_html}
        
        {f'''
        <div class="distributions">
            <h2>Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Detections/Frame</th>
                        <th>Scores</th>
                        <th>Box Areas</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('detections_per_frame', 'min')}</td>
                        <td>{format_dist_value('scores', 'min')}</td>
                        <td>{format_dist_value('box_areas', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('detections_per_frame', 'max')}</td>
                        <td>{format_dist_value('scores', 'max')}</td>
                        <td>{format_dist_value('box_areas', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('detections_per_frame', 'mean')}</td>
                        <td>{format_dist_value('scores', 'mean')}</td>
                        <td>{format_dist_value('box_areas', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('detections_per_frame', 'std')}</td>
                        <td>{format_dist_value('scores', 'std')}</td>
                        <td>{format_dist_value('box_areas', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('detections_per_frame', 'median')}</td>
                        <td>{format_dist_value('scores', 'median')}</td>
                        <td>{format_dist_value('box_areas', 'median')}</td>
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
                            text: 'Detections Count'
                        }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'Average Score'
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
    
    logger.info(f"Saved Core Object Detections HTML render to {output_path}")
    return output_path


__all__ = ["render_core_object_detections", "render_core_object_detections_html"]

