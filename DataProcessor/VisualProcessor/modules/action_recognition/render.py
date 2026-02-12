"""
Renderer для action_recognition: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_action_recognition(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для action_recognition."""
    render = {
        "component": "action_recognition",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract track data
    tracks = npz_data.get("tracks")
    embeddings = npz_data.get("embeddings")
    results_json = npz_data.get("results_json")
    
    # Convert tracks to numpy array if needed
    if tracks is not None:
        if isinstance(tracks, list):
            tracks = np.array(tracks, dtype=np.int32)
        elif isinstance(tracks, np.ndarray):
            tracks = np.asarray(tracks, dtype=np.int32)
        else:
            tracks = None
    
    # Extract embeddings (object array)
    track_embeddings = {}
    if embeddings is not None:
        if isinstance(embeddings, np.ndarray) and embeddings.dtype == object:
            if tracks is not None and len(tracks) == len(embeddings):
                for i, track_id in enumerate(tracks):
                    emb = embeddings[i]
                    if isinstance(emb, np.ndarray):
                        track_embeddings[int(track_id)] = np.asarray(emb, dtype=np.float32)
                    elif isinstance(emb, (list, tuple)):
                        track_embeddings[int(track_id)] = np.array(emb, dtype=np.float32)
    
    # Extract results_json (object array)
    track_results = {}
    if results_json is not None:
        if isinstance(results_json, np.ndarray) and results_json.dtype == object:
            if tracks is not None and len(tracks) == len(results_json):
                for i, track_id in enumerate(tracks):
                    rj = results_json[i]
                    if isinstance(rj, dict):
                        track_results[int(track_id)] = rj
                    elif isinstance(rj, np.ndarray) and rj.dtype == object and rj.size == 1:
                        track_results[int(track_id)] = rj.item() if isinstance(rj.item(), dict) else {}
    
    # Summary statistics
    num_tracks = len(track_embeddings)
    total_clips = 0
    stability_vals = []
    max_jump_vals = []
    mean_jump_vals = []
    norm_mean_vals = []
    norm_std_vals = []
    
    for track_id, emb in track_embeddings.items():
        if emb.size > 0 and emb.ndim == 2:
            total_clips += emb.shape[0]
            # Calculate norms
            norms = np.linalg.norm(emb, axis=1)
            norm_mean_vals.append(float(np.mean(norms)))
            norm_std_vals.append(float(np.std(norms)))
        
        # Extract metrics from results_json
        if track_id in track_results:
            rj = track_results[track_id]
            stability = rj.get("stability")
            if stability is not None and not np.isnan(stability):
                stability_vals.append(float(stability))
            max_jump = rj.get("max_temporal_jump")
            if max_jump is not None and not np.isnan(max_jump):
                max_jump_vals.append(float(max_jump))
            mean_jump = rj.get("mean_temporal_jump")
            if mean_jump is not None and not np.isnan(mean_jump):
                mean_jump_vals.append(float(mean_jump))
    
    render["summary"] = {
        "num_tracks": int(num_tracks),
        "num_clips_total": int(total_clips),
        "avg_stability": float(np.mean(stability_vals)) if stability_vals else None,
        "avg_max_temporal_jump": float(np.mean(max_jump_vals)) if max_jump_vals else None,
        "avg_mean_temporal_jump": float(np.mean(mean_jump_vals)) if mean_jump_vals else None,
        "avg_embedding_norm_mean": float(np.mean(norm_mean_vals)) if norm_mean_vals else None,
        "avg_embedding_norm_std": float(np.mean(norm_std_vals)) if norm_std_vals else None,
    }
    
    # Timeline data (per-track, per-clip)
    timeline = []
    for track_id, emb in track_embeddings.items():
        if emb.size == 0 or emb.ndim != 2:
            continue
        
        rj = track_results.get(track_id, {})
        clip_center_times = rj.get("clip_center_times_s", [])
        temporal_jumps = rj.get("temporal_jumps", [])
        
        # Calculate norms
        norms = np.linalg.norm(emb, axis=1)
        
        for clip_idx in range(emb.shape[0]):
            time_sec = float(clip_center_times[clip_idx]) if clip_idx < len(clip_center_times) else None
            jump = float(temporal_jumps[clip_idx]) if clip_idx < len(temporal_jumps) else None
            
            timeline.append({
                "track_id": int(track_id),
                "clip_index": int(clip_idx),
                "time_sec": time_sec,
                "embedding_norm": float(norms[clip_idx]),
                "temporal_jump": jump,
            })
    
    # Sort timeline by time_sec if available
    timeline.sort(key=lambda x: x.get("time_sec") or 0.0)
    render["timeline"] = timeline
    
    # Distributions
    distributions = {}
    
    if norm_mean_vals:
        distributions["embedding_norm_mean"] = {
            "min": float(np.min(norm_mean_vals)),
            "max": float(np.max(norm_mean_vals)),
            "mean": float(np.mean(norm_mean_vals)),
            "std": float(np.std(norm_mean_vals)),
            "median": float(np.median(norm_mean_vals)),
            "p25": float(np.percentile(norm_mean_vals, 25)),
            "p75": float(np.percentile(norm_mean_vals, 75)),
        }
    
    if stability_vals:
        distributions["stability"] = {
            "min": float(np.min(stability_vals)),
            "max": float(np.max(stability_vals)),
            "mean": float(np.mean(stability_vals)),
            "std": float(np.std(stability_vals)),
            "median": float(np.median(stability_vals)),
            "p25": float(np.percentile(stability_vals, 25)),
            "p75": float(np.percentile(stability_vals, 75)),
        }
    
    if max_jump_vals:
        distributions["max_temporal_jump"] = {
            "min": float(np.min(max_jump_vals)),
            "max": float(np.max(max_jump_vals)),
            "mean": float(np.mean(max_jump_vals)),
            "std": float(np.std(max_jump_vals)),
            "median": float(np.median(max_jump_vals)),
            "p25": float(np.percentile(max_jump_vals, 25)),
            "p75": float(np.percentile(max_jump_vals, 75)),
        }
    
    render["distributions"] = distributions
    
    return render


def render_action_recognition_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага action_recognition результатов.
    
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
    if str(vp_root / "utils") not in sys.path:
        sys.path.insert(0, str(vp_root / "utils"))
    
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
    render = render_action_recognition(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    
    # Helper function to format distribution values
    def format_dist_value(dist_key, stat_key):
        dist = distributions.get(dist_key)
        if dist and stat_key in dist:
            return f"{dist[stat_key]:.4f}"
        return "N/A"
    
    # Prepare timeline data for chart (group by track_id)
    timeline_js = ""
    if timeline:
        # Group by track_id
        tracks_data = {}
        for entry in timeline:
            track_id = entry.get("track_id")
            if track_id not in tracks_data:
                tracks_data[track_id] = {
                    "times": [],
                    "norms": [],
                    "jumps": [],
                }
            if entry.get("time_sec") is not None:
                tracks_data[track_id]["times"].append(entry["time_sec"])
                tracks_data[track_id]["norms"].append(entry.get("embedding_norm", 0.0))
                if entry.get("temporal_jump") is not None:
                    tracks_data[track_id]["jumps"].append(entry["temporal_jump"])
        
        # Build datasets for Chart.js
        datasets = []
        colors = [
            "rgb(75, 192, 192)",
            "rgb(255, 99, 132)",
            "rgb(54, 162, 235)",
            "rgb(255, 206, 86)",
            "rgb(153, 102, 255)",
        ]
        
        for idx, (track_id, data) in enumerate(tracks_data.items()):
            if data["times"]:
                color = colors[idx % len(colors)]
                datasets.append({
                    "label": f"Track {track_id} (Norm)",
                    "data": data["norms"],
                    "borderColor": color,
                    "backgroundColor": color.replace("rgb", "rgba").replace(")", ", 0.2)"),
                    "tension": 0.1,
                    "yAxisID": "y"
                })
        
        if datasets:
            # Use first track's times as labels
            first_track_data = list(tracks_data.values())[0]
            time_labels = [f"{t:.2f}s" for t in first_track_data["times"]]
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
    <title>Action Recognition Debug Render</title>
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
        <h1>Action Recognition Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Number of Tracks</strong>
                    <span class="metric-value">{summary.get('num_tracks', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Total Clips</strong>
                    <span class="metric-value">{summary.get('num_clips_total', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Avg Stability</strong>
                    <span class="metric-value">{summary.get('avg_stability', 0.0) or 0.0:.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Avg Max Temporal Jump</strong>
                    <span class="metric-value">{summary.get('avg_max_temporal_jump', 0.0) or 0.0:.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Avg Mean Temporal Jump</strong>
                    <span class="metric-value">{summary.get('avg_mean_temporal_jump', 0.0) or 0.0:.4f}</span>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Embedding Norm Over Time (by Track)</h2>
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
                        <th>Embedding Norm Mean</th>
                        <th>Stability</th>
                        <th>Max Temporal Jump</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('embedding_norm_mean', 'min')}</td>
                        <td>{format_dist_value('stability', 'min')}</td>
                        <td>{format_dist_value('max_temporal_jump', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('embedding_norm_mean', 'max')}</td>
                        <td>{format_dist_value('stability', 'max')}</td>
                        <td>{format_dist_value('max_temporal_jump', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('embedding_norm_mean', 'mean')}</td>
                        <td>{format_dist_value('stability', 'mean')}</td>
                        <td>{format_dist_value('max_temporal_jump', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('embedding_norm_mean', 'std')}</td>
                        <td>{format_dist_value('stability', 'std')}</td>
                        <td>{format_dist_value('max_temporal_jump', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('embedding_norm_mean', 'median')}</td>
                        <td>{format_dist_value('stability', 'median')}</td>
                        <td>{format_dist_value('max_temporal_jump', 'median')}</td>
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
                            text: 'Embedding Norm'
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
    
    logger.info(f"Saved Action Recognition HTML render to {output_path}")
    return output_path


__all__ = ["render_action_recognition", "render_action_recognition_html"]

