"""
Renderer для place_semantics: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_place_semantics(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для place_semantics."""
    render = {
        "component": "place_semantics",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract place recognition data
    frame_topk_ids = npz_data.get("frame_topk_ids")
    frame_topk_scores = npz_data.get("frame_topk_scores")
    frame_is_confident_top1 = npz_data.get("frame_is_confident_top1")
    track_topk_ids = npz_data.get("track_topk_ids")
    track_topk_scores = npz_data.get("track_topk_scores")
    track_is_confident_top1 = npz_data.get("track_is_confident_top1")
    semantic_label_names = npz_data.get("semantic_label_names")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    track_ids = npz_data.get("track_ids")
    
    # Convert to numpy arrays if needed
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
    
    if frame_is_confident_top1 is not None:
        if isinstance(frame_is_confident_top1, list):
            frame_is_confident_top1 = np.array(frame_is_confident_top1, dtype=np.bool_)
        elif isinstance(frame_is_confident_top1, np.ndarray):
            frame_is_confident_top1 = np.asarray(frame_is_confident_top1, dtype=np.bool_)
        else:
            frame_is_confident_top1 = None
    
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
    
    if track_is_confident_top1 is not None:
        if isinstance(track_is_confident_top1, list):
            track_is_confident_top1 = np.array(track_is_confident_top1, dtype=np.bool_)
        elif isinstance(track_is_confident_top1, np.ndarray):
            track_is_confident_top1 = np.asarray(track_is_confident_top1, dtype=np.bool_)
        else:
            track_is_confident_top1 = None
    
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
    
    if track_ids is not None:
        if isinstance(track_ids, list):
            track_ids = np.array(track_ids, dtype=np.int32)
        elif isinstance(track_ids, np.ndarray):
            track_ids = np.asarray(track_ids, dtype=np.int32)
        else:
            track_ids = None
    
    # Build label_id -> place_name mapping
    label_to_name: Dict[int, str] = {}
    if semantic_label_names is not None:
        for label_str in semantic_label_names:
            if ":" in label_str:
                label_id_str, place_name = label_str.split(":", 1)
                try:
                    label_id = int(label_id_str)
                    label_to_name[label_id] = place_name
                except ValueError:
                    pass
    
    # Summary statistics
    if frame_topk_scores is not None and frame_topk_scores.size > 0:
        # Get top-1 scores (first column)
        top1_scores = frame_topk_scores[:, 0] if frame_topk_scores.ndim == 2 else frame_topk_scores
        valid_top1_scores = top1_scores[np.isfinite(top1_scores)]
        
        # Count unique places
        unique_places = set()
        if frame_topk_ids is not None:
            for frame_idx in range(frame_topk_ids.shape[0]):
                for k in range(frame_topk_ids.shape[1]):
                    place_id = int(frame_topk_ids[frame_idx, k])
                    if place_id >= 0 and place_id in label_to_name:
                        unique_places.add(place_id)
        
        # Count confident predictions
        confident_count = 0
        if frame_is_confident_top1 is not None:
            confident_count = int(np.sum(frame_is_confident_top1))
        
        render["summary"] = {
            "frames_count": int(frame_topk_ids.shape[0]) if frame_topk_ids is not None and frame_topk_ids.ndim >= 1 else 0,
            "tracks_count": int(track_ids.shape[0]) if track_ids is not None and track_ids.ndim >= 1 else 0,
            "unique_places_count": len(unique_places),
            "confident_predictions_count": confident_count,
            "confident_predictions_ratio": float(confident_count / frame_topk_ids.shape[0]) if frame_topk_ids is not None and frame_topk_ids.shape[0] > 0 else 0.0,
        }
        
        if valid_top1_scores.size > 0:
            render["summary"]["top1_score_mean"] = float(np.mean(valid_top1_scores))
            render["summary"]["top1_score_std"] = float(np.std(valid_top1_scores))
            render["summary"]["top1_score_min"] = float(np.min(valid_top1_scores))
            render["summary"]["top1_score_max"] = float(np.max(valid_top1_scores))
            render["summary"]["top1_score_median"] = float(np.median(valid_top1_scores))
    
    # Timeline data (per-frame)
    if frame_topk_ids is not None and times_s is not None and frame_indices is not None:
        n = len(frame_topk_ids)
        timeline = []
        
        for i in range(n):
            frame_idx = int(frame_indices[i]) if i < len(frame_indices) else i
            time_sec = float(times_s[i]) if i < len(times_s) else 0.0
            
            # Get top-1 place
            top1_place_id = int(frame_topk_ids[i, 0]) if frame_topk_ids.ndim == 2 and frame_topk_ids.shape[1] > 0 else -1
            top1_place_name = label_to_name.get(top1_place_id, "unknown")
            top1_score = float(frame_topk_scores[i, 0]) if frame_topk_scores is not None and frame_topk_scores.ndim == 2 and frame_topk_scores.shape[1] > 0 and np.isfinite(frame_topk_scores[i, 0]) else None
            is_confident = bool(frame_is_confident_top1[i]) if frame_is_confident_top1 is not None and i < len(frame_is_confident_top1) else False
            
            # Count unique places in this frame
            unique_places_in_frame = set()
            for k in range(frame_topk_ids.shape[1]):
                place_id = int(frame_topk_ids[i, k])
                if place_id >= 0:
                    unique_places_in_frame.add(place_id)
            
            # Get topk scores
            topk_scores = []
            if frame_topk_scores is not None and frame_topk_scores.ndim == 2:
                for k in range(frame_topk_scores.shape[1]):
                    score = frame_topk_scores[i, k]
                    if np.isfinite(score):
                        topk_scores.append(float(score))
            
            timeline.append({
                "frame_index": frame_idx,
                "time_sec": time_sec,
                "top1_place_id": top1_place_id,
                "top1_place_name": top1_place_name,
                "top1_score": top1_score,
                "is_confident": is_confident,
                "unique_places_count": len(unique_places_in_frame),
                "topk_scores": topk_scores,
            })
        
        render["timeline"] = timeline
    
    # Distributions
    distributions = {}
    
    if frame_topk_scores is not None and frame_topk_scores.size > 0:
        # Top-1 scores distribution
        top1_scores = frame_topk_scores[:, 0] if frame_topk_scores.ndim == 2 else frame_topk_scores
        valid_top1_scores = top1_scores[np.isfinite(top1_scores)]
        
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
        
        # All topk scores distribution
        all_scores = frame_topk_scores[np.isfinite(frame_topk_scores)]
        if all_scores.size > 0:
            distributions["topk_scores"] = {
                "min": float(np.min(all_scores)),
                "max": float(np.max(all_scores)),
                "mean": float(np.mean(all_scores)),
                "std": float(np.std(all_scores)),
                "median": float(np.median(all_scores)),
                "p25": float(np.percentile(all_scores, 25)),
                "p75": float(np.percentile(all_scores, 75)),
                "p05": float(np.percentile(all_scores, 5)),
                "p95": float(np.percentile(all_scores, 95)),
            }
    
    # Top places (by frequency)
    if frame_topk_ids is not None and label_to_name:
        place_counts: Dict[int, int] = {}
        place_total_scores: Dict[int, float] = {}
        
        for frame_idx in range(frame_topk_ids.shape[0]):
            for k in range(frame_topk_ids.shape[1]):
                place_id = int(frame_topk_ids[frame_idx, k])
                if place_id >= 0 and place_id in label_to_name:
                    place_counts[place_id] = place_counts.get(place_id, 0) + 1
                    if frame_topk_scores is not None and frame_topk_scores.ndim == 2:
                        score = frame_topk_scores[frame_idx, k]
                        if np.isfinite(score):
                            place_total_scores[place_id] = place_total_scores.get(place_id, 0.0) + float(score)
        
        # Sort by count
        top_places = []
        for place_id, count in sorted(place_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            avg_score = place_total_scores.get(place_id, 0.0) / count if count > 0 else 0.0
            top_places.append({
                "place_id": place_id,
                "place_name": label_to_name.get(place_id, "unknown"),
                "frames_count": count,
                "avg_score": avg_score,
            })
        
        render["top_places"] = top_places
    
    render["distributions"] = distributions
    
    return render


def render_place_semantics_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага place_semantics результатов.
    
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
    render = render_place_semantics(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    top_places = render.get("top_places", [])
    
    # Helper function to format distribution values
    def format_dist_value(dist_key, stat_key):
        dist = distributions.get(dist_key)
        if dist and stat_key in dist:
            return f"{dist[stat_key]:.4f}"
        return "N/A"
    
    # Pre-format top places table rows to avoid nested f-string issues
    top_places_rows = ""
    if top_places:
        rows = []
        for place in top_places:
            place_name = place.get('place_name', 'unknown')
            frames_count = place.get('frames_count', 0)
            avg_score = place.get('avg_score', 0.0)
            try:
                avg_score_str = f"{float(avg_score or 0.0):.4f}"
            except (ValueError, TypeError):
                avg_score_str = "0.0000"
            rows.append(f"""
                    <tr>
                        <td>{place_name}</td>
                        <td>{frames_count}</td>
                        <td>{avg_score_str}</td>
                    </tr>""")
        top_places_rows = ''.join(rows)
    
    # Prepare timeline data for chart
    timeline_js = ""
    if timeline:
        times = [t.get("time_sec", 0.0) for t in timeline]
        top1_scores = [t.get("top1_score") for t in timeline if t.get("top1_score") is not None]
        
        # Build datasets array
        datasets = []
        
        if top1_scores:
            datasets.append({
                "label": "Top-1 Place Score",
                "data": top1_scores,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "tension": 0.1,
                "yAxisID": "y"
            })
        
        if datasets:
            # Format time labels
            time_labels = [f"{t:.2f}s" for t in times[:len(top1_scores)]]
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
    <title>Place Semantics Debug Render</title>
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
        .top-places {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .top-places table {{ width: 100%; border-collapse: collapse; }}
        .top-places th, .top-places td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .top-places th {{ background-color: #0056b3; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Place Semantics Debug Render</h1>
        
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
                    <strong>Unique Places</strong>
                    <span class="metric-value">{summary.get('unique_places_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Confident Predictions</strong>
                    <span class="metric-value">{summary.get('confident_predictions_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Confident Ratio</strong>
                    <span class="metric-value">{summary.get('confident_predictions_ratio', 0.0):.2%}</span>
                </div>
                <div class="metric-card">
                    <strong>Top-1 Score Mean</strong>
                    <span class="metric-value">{summary.get('top1_score_mean', 0.0):.4f}</span>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Top-1 Place Scores Over Time</h2>
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
        <div class="top-places">
            <h2>Top Places</h2>
            <table>
                <thead>
                    <tr>
                        <th>Place Name</th>
                        <th>Frames Count</th>
                        <th>Avg Score</th>
                    </tr>
                </thead>
                <tbody>
                    {top_places_rows}
                </tbody>
            </table>
        </div>
        ''' if top_places else ''}
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
                            text: 'Top-1 Place Score'
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
    logger.info(f"Saved Place Semantics HTML render to {rel_output_path}")
    return output_path


__all__ = ["render_place_semantics", "render_place_semantics_html"]

