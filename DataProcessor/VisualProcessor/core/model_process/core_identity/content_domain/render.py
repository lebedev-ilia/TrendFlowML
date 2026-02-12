"""
Renderer для content_domain: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_content_domain(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для content_domain."""
    render = {
        "component": "content_domain",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract content domain data
    frame_topk_ids = npz_data.get("frame_topk_ids")
    frame_topk_scores = npz_data.get("frame_topk_scores")
    frame_is_confident_top1 = npz_data.get("frame_is_confident_top1")
    semantic_label_names = npz_data.get("semantic_label_names")
    track_topk_ids = npz_data.get("track_topk_ids")
    track_topk_scores = npz_data.get("track_topk_scores")
    track_is_confident_top1 = npz_data.get("track_is_confident_top1")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    
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
    
    if semantic_label_names is not None:
        if isinstance(semantic_label_names, list):
            semantic_label_names = np.array(semantic_label_names, dtype="U")
        elif isinstance(semantic_label_names, np.ndarray):
            semantic_label_names = np.asarray(semantic_label_names, dtype="U")
        else:
            semantic_label_names = None
    
    # Ensure all arrays are numpy arrays before using .size
    if frame_topk_ids is not None and not isinstance(frame_topk_ids, np.ndarray):
        frame_topk_ids = np.asarray(frame_topk_ids)
    if frame_topk_scores is not None and not isinstance(frame_topk_scores, np.ndarray):
        frame_topk_scores = np.asarray(frame_topk_scores)
    if frame_is_confident_top1 is not None and not isinstance(frame_is_confident_top1, np.ndarray):
        frame_is_confident_top1 = np.asarray(frame_is_confident_top1)
    if track_topk_ids is not None and not isinstance(track_topk_ids, np.ndarray):
        track_topk_ids = np.asarray(track_topk_ids)
    if track_topk_scores is not None and not isinstance(track_topk_scores, np.ndarray):
        track_topk_scores = np.asarray(track_topk_scores)
    if track_is_confident_top1 is not None and not isinstance(track_is_confident_top1, np.ndarray):
        track_is_confident_top1 = np.asarray(track_is_confident_top1)
    if times_s is not None and not isinstance(times_s, np.ndarray):
        times_s = np.asarray(times_s)
    if frame_indices is not None and not isinstance(frame_indices, np.ndarray):
        frame_indices = np.asarray(frame_indices)
    
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
    if frame_topk_ids is not None and frame_topk_ids.size > 0:
        n_frames = frame_topk_ids.shape[0] if frame_topk_ids.ndim >= 2 else 1
        topk = frame_topk_ids.shape[1] if frame_topk_ids.ndim >= 2 else 1
        
        # Extract top-1 domain for each frame
        top1_ids = frame_topk_ids[:, 0] if frame_topk_ids.ndim >= 2 else frame_topk_ids
        top1_scores = frame_topk_scores[:, 0] if frame_topk_scores is not None and frame_topk_scores.ndim >= 2 else None
        
        # Count unique domains
        unique_domains = np.unique(top1_ids)
        unique_domains = unique_domains[unique_domains >= 0]  # Filter out -1 (invalid)
        
        # Count confident predictions
        confident_count = int(np.sum(frame_is_confident_top1)) if frame_is_confident_top1 is not None else 0
        
        render["summary"] = {
            "frames_count": int(n_frames),
            "topk": int(topk),
            "unique_domains_count": int(len(unique_domains)),
            "confident_predictions_count": confident_count,
            "confident_predictions_ratio": float(confident_count / n_frames) if n_frames > 0 else 0.0,
        }
        
        if top1_scores is not None and top1_scores.size > 0:
            valid_scores = top1_scores[np.isfinite(top1_scores)]
            if valid_scores.size > 0:
                render["summary"]["top1_score_mean"] = float(np.mean(valid_scores))
                render["summary"]["top1_score_std"] = float(np.std(valid_scores))
                render["summary"]["top1_score_min"] = float(np.min(valid_scores))
                render["summary"]["top1_score_max"] = float(np.max(valid_scores))
                render["summary"]["top1_score_median"] = float(np.median(valid_scores))
        
        # Video-level aggregate (track)
        if track_topk_ids is not None and track_topk_ids.size > 0:
            track_top1_id = int(track_topk_ids[0, 0]) if track_topk_ids.ndim >= 2 else int(track_topk_ids[0])
            track_top1_score = float(track_topk_scores[0, 0]) if track_topk_scores is not None and track_topk_scores.ndim >= 2 else None
            track_confident = bool(track_is_confident_top1[0]) if track_is_confident_top1 is not None and track_is_confident_top1.size > 0 else False
            
            # Get domain name
            domain_name = "unknown"
            if semantic_label_names is not None:
                for label_str in semantic_label_names:
                    if isinstance(label_str, str) and ":" in label_str:
                        label_id_str, label_name = label_str.split(":", 1)
                        try:
                            if int(label_id_str) == track_top1_id:
                                domain_name = label_name
                                break
                        except (ValueError, TypeError):
                            continue
            
            render["summary"]["video_domain"] = {
                "domain_id": track_top1_id,
                "domain_name": domain_name,
                "score": track_top1_score,
                "is_confident": track_confident,
            }
        
        # Top domains by frequency
        if len(unique_domains) > 0:
            domain_counts = {}
            for domain_id in top1_ids:
                if domain_id >= 0:
                    domain_counts[int(domain_id)] = domain_counts.get(int(domain_id), 0) + 1
            
            # Sort by frequency
            sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            top_domains = []
            for domain_id, count in sorted_domains:
                domain_name = "unknown"
                if semantic_label_names is not None:
                    for label_str in semantic_label_names:
                        if isinstance(label_str, str) and ":" in label_str:
                            label_id_str, label_name = label_str.split(":", 1)
                            try:
                                if int(label_id_str) == domain_id:
                                    domain_name = label_name
                                    break
                            except (ValueError, TypeError):
                                continue
                
                top_domains.append({
                    "domain_id": int(domain_id),
                    "domain_name": domain_name,
                    "count": int(count),
                    "ratio": float(count / n_frames) if n_frames > 0 else 0.0,
                })
            
            render["summary"]["top_domains"] = top_domains
    
    # Timeline data (per-frame domain predictions)
    if frame_topk_ids is not None and times_s is not None and frame_indices is not None:
        n = len(frame_topk_ids)
        timeline = []
        
        for i in range(n):
            frame_idx = int(frame_indices[i]) if i < len(frame_indices) else i
            time_sec = float(times_s[i]) if i < len(times_s) else 0.0
            
            # Top-1 domain
            top1_id = int(frame_topk_ids[i, 0]) if frame_topk_ids.ndim >= 2 else int(frame_topk_ids[i])
            top1_score = float(frame_topk_scores[i, 0]) if frame_topk_scores is not None and frame_topk_scores.ndim >= 2 else None
            is_confident = bool(frame_is_confident_top1[i]) if frame_is_confident_top1 is not None and i < len(frame_is_confident_top1) else False
            
            # Get domain name
            domain_name = "unknown"
            if semantic_label_names is not None:
                for label_str in semantic_label_names:
                    if isinstance(label_str, str) and ":" in label_str:
                        label_id_str, label_name = label_str.split(":", 1)
                        try:
                            if int(label_id_str) == top1_id:
                                domain_name = label_name
                                break
                        except (ValueError, TypeError):
                            continue
            
            # Top-K scores
            topk_scores = []
            if frame_topk_scores is not None and frame_topk_scores.ndim >= 2:
                for k in range(min(5, frame_topk_scores.shape[1])):
                    topk_scores.append(float(frame_topk_scores[i, k]) if np.isfinite(frame_topk_scores[i, k]) else None)
            
            timeline.append({
                "frame_index": frame_idx,
                "time_sec": time_sec,
                "top1_domain_id": top1_id if top1_id >= 0 else None,
                "top1_domain_name": domain_name if top1_id >= 0 else None,
                "top1_score": top1_score,
                "is_confident": is_confident,
                "topk_scores": topk_scores,
            })
        
        render["timeline"] = timeline
    
    # Distributions
    distributions = {}
    
    if frame_topk_scores is not None and frame_topk_scores.size > 0:
        # Top-1 scores distribution
        top1_scores = frame_topk_scores[:, 0] if frame_topk_scores.ndim >= 2 else frame_topk_scores
        valid_scores = top1_scores[np.isfinite(top1_scores)]
        if valid_scores.size > 0:
            distributions["top1_scores"] = {
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
    
    return render


def render_content_domain_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага content_domain результатов.
    
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
    render = render_content_domain(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    
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
        datasets = []
        
        if top1_scores:
            datasets.append({
                "label": "Top-1 Score",
                "data": top1_scores,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
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
    
    # Video domain info
    video_domain = summary.get("video_domain", {})
    video_domain_html = ""
    if video_domain:
        video_domain_html = f"""
        <div class="metric-card">
            <strong>Video Domain</strong>
            <span class="metric-value">{video_domain.get('domain_name', 'unknown')}</span>
        </div>
        <div class="metric-card">
            <strong>Domain Score</strong>
            <span class="metric-value">{video_domain.get('score', 0.0):.4f}</span>
        </div>
        <div class="metric-card">
            <strong>Is Confident</strong>
            <span class="metric-value">{'Yes' if video_domain.get('is_confident', False) else 'No'}</span>
        </div>
        """
    
    # Top domains table
    top_domains = summary.get("top_domains", [])
    top_domains_html = ""
    if top_domains:
        top_domains_rows = ""
        for domain in top_domains[:10]:
            top_domains_rows += f"""
            <tr>
                <td>{domain.get('domain_name', 'unknown')}</td>
                <td>{domain.get('count', 0)}</td>
                <td>{domain.get('ratio', 0.0):.2%}</td>
            </tr>
            """
        top_domains_html = f"""
        <div class="distributions">
            <h2>Top Domains by Frequency</h2>
            <table>
                <thead>
                    <tr>
                        <th>Domain Name</th>
                        <th>Count</th>
                        <th>Ratio</th>
                    </tr>
                </thead>
                <tbody>
                    {top_domains_rows}
                </tbody>
            </table>
        </div>
        """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Content Domain Debug Render</title>
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
        <h1>Content Domain Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Unique Domains</strong>
                    <span class="metric-value">{summary.get('unique_domains_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Confident Predictions</strong>
                    <span class="metric-value">{summary.get('confident_predictions_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Confident Ratio</strong>
                    <span class="metric-value">{summary.get('confident_predictions_ratio', 0.0):.2%}</span>
                </div>
                {video_domain_html}
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Top-1 Domain Score Over Time</h2>
            <canvas id="timelineChart"></canvas>
        </div>
        ''' if timeline else '<p>No timeline data available</p>'}
        
        {top_domains_html}
        
        {f'''
        <div class="distributions">
            <h2>Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Top-1 Score</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('top1_scores', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('top1_scores', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('top1_scores', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('top1_scores', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('top1_scores', 'median')}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions.get('top1_scores') else ''}
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
                            text: 'Top-1 Score'
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
    logger.info(f"Saved Content Domain HTML render to {rel_output_path}")
    return output_path


__all__ = ["render_content_domain", "render_content_domain_html"]

