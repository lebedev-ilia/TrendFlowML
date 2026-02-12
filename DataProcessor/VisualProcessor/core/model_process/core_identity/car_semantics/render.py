"""
Renderer для car_semantics: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_car_semantics(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для car_semantics."""
    render = {
        "component": "car_semantics",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract car recognition data
    track_ids = npz_data.get("track_ids")
    track_topk_ids = npz_data.get("track_topk_ids")
    track_topk_scores = npz_data.get("track_topk_scores")
    track_topk_makes = npz_data.get("track_topk_makes")
    track_topk_models = npz_data.get("track_topk_models")
    track_topk_segments = npz_data.get("track_topk_segments")
    frame_topk_ids = npz_data.get("frame_topk_ids")
    frame_topk_scores = npz_data.get("frame_topk_scores")
    semantic_label_names = npz_data.get("semantic_label_names")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    
    # Convert to numpy arrays if needed
    if track_ids is not None:
        if isinstance(track_ids, list):
            track_ids = np.array(track_ids, dtype=np.int32)
        elif isinstance(track_ids, np.ndarray):
            track_ids = np.asarray(track_ids, dtype=np.int32)
        else:
            track_ids = None
    
    if track_topk_ids is not None:
        if isinstance(track_topk_ids, list):
            track_topk_ids = np.array(track_topk_ids, dtype=np.int32)
        elif isinstance(track_topk_ids, np.ndarray):
            track_topk_ids = np.asarray(track_topk_ids, dtype=np.int32)
        else:
            track_topk_ids = None
    
    if track_topk_scores is not None:
        if isinstance(track_topk_scores, list):
            track_topk_scores = np.array(track_topk_scores, dtype=np.float32)
        elif isinstance(track_topk_scores, np.ndarray):
            track_topk_scores = np.asarray(track_topk_scores, dtype=np.float32)
        else:
            track_topk_scores = None
    
    if track_topk_makes is not None:
        if isinstance(track_topk_makes, list):
            track_topk_makes = np.array(track_topk_makes, dtype="U256")
        elif isinstance(track_topk_makes, np.ndarray):
            track_topk_makes = np.asarray(track_topk_makes, dtype="U256")
        else:
            track_topk_makes = None
    
    if track_topk_models is not None:
        if isinstance(track_topk_models, list):
            track_topk_models = np.array(track_topk_models, dtype="U256")
        elif isinstance(track_topk_models, np.ndarray):
            track_topk_models = np.asarray(track_topk_models, dtype="U256")
        else:
            track_topk_models = None
    
    if track_topk_segments is not None:
        if isinstance(track_topk_segments, list):
            track_topk_segments = np.array(track_topk_segments, dtype="U256")
        elif isinstance(track_topk_segments, np.ndarray):
            track_topk_segments = np.asarray(track_topk_segments, dtype="U256")
        else:
            track_topk_segments = None
    
    if frame_topk_ids is not None:
        if isinstance(frame_topk_ids, list):
            frame_topk_ids = np.array(frame_topk_ids, dtype=np.int32)
        elif isinstance(frame_topk_ids, np.ndarray):
            frame_topk_ids = np.asarray(frame_topk_ids, dtype=np.int32)
        else:
            frame_topk_ids = None
    
    if frame_topk_scores is not None:
        if isinstance(frame_topk_scores, list):
            frame_topk_scores = np.array(frame_topk_scores, dtype=np.float32)
        elif isinstance(frame_topk_scores, np.ndarray):
            frame_topk_scores = np.asarray(frame_topk_scores, dtype=np.float32)
        else:
            frame_topk_scores = None
    
    if semantic_label_names is not None:
        if isinstance(semantic_label_names, list):
            semantic_label_names = np.array(semantic_label_names, dtype="U")
        elif isinstance(semantic_label_names, np.ndarray):
            semantic_label_names = np.asarray(semantic_label_names, dtype="U")
        else:
            semantic_label_names = None
    
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
    
    # Build car ID to name mapping
    car_id_to_name: Dict[int, str] = {}
    if semantic_label_names is not None:
        for item in semantic_label_names:
            item_str = str(item)
            if ":" in item_str:
                try:
                    car_id_str, car_name = item_str.split(":", 1)
                    car_id_to_name[int(car_id_str)] = car_name
                except Exception:
                    continue
    
    # Summary statistics
    if frame_topk_scores is not None and frame_topk_scores.size > 0:
        # Filter valid scores (> 0)
        valid_scores = frame_topk_scores[frame_topk_scores > 0]
        top1_scores = frame_topk_scores[:, 0] if frame_topk_scores.ndim == 2 else frame_topk_scores
        valid_top1_scores = top1_scores[top1_scores > 0]
        
        # Count unique cars per frame
        unique_cars_per_frame = []
        if frame_topk_ids is not None and frame_topk_ids.ndim == 2:
            for i in range(frame_topk_ids.shape[0]):
                unique_cars = set()
                for j in range(frame_topk_ids.shape[1]):
                    car_id = int(frame_topk_ids[i, j])
                    if car_id > 0 and car_id in car_id_to_name:
                        unique_cars.add(car_id)
                unique_cars_per_frame.append(len(unique_cars))
        
        render["summary"] = {
            "frames_count": int(frame_topk_scores.shape[0]) if frame_topk_scores.ndim >= 1 else 0,
            "tracks_count": int(track_ids.shape[0]) if track_ids is not None and track_ids.size > 0 else 0,
            "unique_cars_count": len(car_id_to_name),
            "top1_score_mean": float(np.mean(valid_top1_scores)) if valid_top1_scores.size > 0 else 0.0,
            "top1_score_std": float(np.std(valid_top1_scores)) if valid_top1_scores.size > 0 else 0.0,
            "top1_score_min": float(np.min(valid_top1_scores)) if valid_top1_scores.size > 0 else 0.0,
            "top1_score_max": float(np.max(valid_top1_scores)) if valid_top1_scores.size > 0 else 0.0,
            "top1_score_median": float(np.median(valid_top1_scores)) if valid_top1_scores.size > 0 else 0.0,
            "unique_cars_per_frame_mean": float(np.mean(unique_cars_per_frame)) if unique_cars_per_frame else 0.0,
            "confident_predictions_count": int(np.sum(valid_top1_scores > 0.5)) if valid_top1_scores.size > 0 else 0,
            "confident_predictions_ratio": float(np.mean(valid_top1_scores > 0.5)) if valid_top1_scores.size > 0 else 0.0,
        }
        
        # Timeline data
        if times_s is not None and frame_indices is not None and len(times_s) == len(frame_indices):
            timeline = []
            for i in range(len(times_s)):
                frame_idx = int(frame_indices[i]) if i < len(frame_indices) else i
                time_sec = float(times_s[i]) if i < len(times_s) else 0.0
                
                # Get top-1 car
                top1_car_id = int(frame_topk_ids[i, 0]) if frame_topk_ids is not None and i < frame_topk_ids.shape[0] else -1
                top1_car_name = car_id_to_name.get(top1_car_id, "unknown") if top1_car_id > 0 else None
                top1_score = float(frame_topk_scores[i, 0]) if frame_topk_scores is not None and i < frame_topk_scores.shape[0] else 0.0
                is_confident = top1_score > 0.5
                
                # Get top-K scores
                topk_scores = []
                if frame_topk_scores is not None and i < frame_topk_scores.shape[0]:
                    for k in range(min(3, frame_topk_scores.shape[1])):
                        score = float(frame_topk_scores[i, k])
                        if score > 0:
                            topk_scores.append(score)
                
                timeline.append({
                    "frame_index": frame_idx,
                    "time_sec": time_sec,
                    "top1_car_id": top1_car_id if top1_car_id > 0 else None,
                    "top1_car_name": top1_car_name,
                    "top1_score": top1_score if top1_score > 0 else None,
                    "is_confident": is_confident,
                    "topk_scores": topk_scores,
                    "unique_cars_count": unique_cars_per_frame[i] if i < len(unique_cars_per_frame) else 0,
                })
            
            render["timeline"] = timeline
        
        # Distribution statistics
        distributions = {}
        
        if valid_top1_scores.size > 0:
            distributions["top1_scores"] = {
                "min": float(np.min(valid_top1_scores)),
                "max": float(np.max(valid_top1_scores)),
                "mean": float(np.mean(valid_top1_scores)),
                "std": float(np.std(valid_top1_scores)),
                "median": float(np.median(valid_top1_scores)),
                "p25": float(np.percentile(valid_top1_scores, 25)),
                "p75": float(np.percentile(valid_top1_scores, 75)),
                "p05": float(np.percentile(valid_top1_scores, 5)),
                "p95": float(np.percentile(valid_top1_scores, 95)),
            }
        
        if valid_scores.size > 0:
            distributions["topk_scores"] = {
                "min": float(np.min(valid_scores)),
                "max": float(np.max(valid_scores)),
                "mean": float(np.mean(valid_scores)),
                "std": float(np.std(valid_scores)),
                "median": float(np.median(valid_scores)),
                "p25": float(np.percentile(valid_scores, 25)),
                "p75": float(np.percentile(valid_scores, 75)),
                "p05": float(np.percentile(valid_scores, 5)),
                "p95": float(np.percentile(valid_scores, 95)),
            }
        
        render["distributions"] = distributions
        
        # Top cars (by track count) with make/model/segment
        if track_topk_ids is not None and track_topk_ids.size > 0:
            car_track_counts: Dict[int, int] = {}
            car_total_scores: Dict[int, float] = {}
            car_makes: Dict[int, str] = {}
            car_models: Dict[int, str] = {}
            car_segments: Dict[int, str] = {}
            
            for track_idx in range(track_topk_ids.shape[0]):
                for k in range(track_topk_ids.shape[1]):
                    car_id = int(track_topk_ids[track_idx, k])
                    if car_id > 0 and car_id in car_id_to_name:
                        if car_id not in car_track_counts:
                            car_track_counts[car_id] = 0
                            car_total_scores[car_id] = 0.0
                        car_track_counts[car_id] += 1
                        if track_topk_scores is not None and track_idx < track_topk_scores.shape[0] and k < track_topk_scores.shape[1]:
                            car_total_scores[car_id] += float(track_topk_scores[track_idx, k])
                        
                        # Store make/model/segment from first occurrence
                        if car_id not in car_makes and track_topk_makes is not None:
                            make = str(track_topk_makes[track_idx, k]).strip()
                            if make:
                                car_makes[car_id] = make
                        if car_id not in car_models and track_topk_models is not None:
                            model = str(track_topk_models[track_idx, k]).strip()
                            if model:
                                car_models[car_id] = model
                        if car_id not in car_segments and track_topk_segments is not None:
                            segment = str(track_topk_segments[track_idx, k]).strip()
                            if segment:
                                car_segments[car_id] = segment
            
            # Sort by track count
            top_cars = sorted(
                car_track_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            render["top_cars"] = [
                {
                    "car_id": car_id,
                    "car_name": car_id_to_name.get(car_id, "unknown"),
                    "make": car_makes.get(car_id, ""),
                    "model": car_models.get(car_id, ""),
                    "segment": car_segments.get(car_id, ""),
                    "track_count": count,
                    "average_score": float(car_total_scores[car_id] / count) if car_id in car_total_scores and count > 0 else 0.0,
                }
                for car_id, count in top_cars
            ]
    
    return render


def render_car_semantics_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага car_semantics результатов.
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    vp_root = Path(__file__).resolve().parent.parent.parent.parent.parent
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
    render = render_car_semantics(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    top_cars = render.get("top_cars", [])
    
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
        top1_scores = [t.get("top1_score") for t in timeline if t.get("top1_score") is not None]
        
        # Build datasets array
        datasets = [{
            "label": "Top-1 Score",
            "data": top1_scores,
            "borderColor": "rgb(75, 192, 192)",
            "backgroundColor": "rgba(75, 192, 192, 0.2)",
            "tension": 0.1,
        }]
        
        timeline_js = f"""
        const timelineData = {{
            labels: {json.dumps([f"{t:.2f}s" for t in times])},
            datasets: {json.dumps(datasets)}
        }};
        """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Car Semantics Debug Render</title>
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
        .top-cars {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .top-cars table {{ width: 100%; border-collapse: collapse; }}
        .top-cars th, .top-cars td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .top-cars th {{ background-color: #0056b3; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Car Semantics Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Tracks Count</strong>
                    <span class="metric-value">{summary.get('tracks_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Unique Cars</strong>
                    <span class="metric-value">{summary.get('unique_cars_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Top-1 Score Mean</strong>
                    <span class="metric-value">{summary.get('top1_score_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Confident Predictions</strong>
                    <span class="metric-value">{summary.get('confident_predictions_count', 0)} ({summary.get('confident_predictions_ratio', 0.0):.1%})</span>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Top-1 Car Score Over Time</h2>
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
                        <th>Top-1 Scores</th>
                        <th>Top-K Scores</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('top1_scores', 'min')}</td>
                        <td>{format_dist_value('topk_scores', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('top1_scores', 'max')}</td>
                        <td>{format_dist_value('topk_scores', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('top1_scores', 'mean')}</td>
                        <td>{format_dist_value('topk_scores', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('top1_scores', 'std')}</td>
                        <td>{format_dist_value('topk_scores', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('top1_scores', 'median')}</td>
                        <td>{format_dist_value('topk_scores', 'median')}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions else ''}
        
        {f'''
        <div class="top-cars">
            <h2>Top Cars</h2>
            <table>
                <thead>
                    <tr>
                        <th>Car Name</th>
                        <th>Make</th>
                        <th>Model</th>
                        <th>Segment</th>
                        <th>Track Count</th>
                        <th>Average Score</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join([
                        f'<tr><td>{c["car_name"]}</td><td>{c["make"]}</td><td>{c["model"]}</td><td>{c["segment"]}</td><td>{c["track_count"]}</td><td>{c["average_score"]:.4f}</td></tr>'
                        for c in top_cars[:10]
                    ])}
                </tbody>
            </table>
        </div>
        ''' if top_cars else ''}
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
                        max: 1.0,
                        title: {{
                            display: true,
                            text: 'Similarity Score'
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
    
    logger.info(f"Saved Car Semantics HTML render to {output_path}")
    return output_path


__all__ = ["render_car_semantics", "render_car_semantics_html"]

