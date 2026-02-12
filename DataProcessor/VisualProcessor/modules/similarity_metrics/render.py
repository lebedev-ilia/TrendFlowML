"""
Renderer для similarity_metrics: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_similarity_metrics(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для similarity_metrics."""
    render = {
        "component": "similarity_metrics",
        "summary": {},
        "timeline": [],
        "distributions": {},
        "reference_similarities": {},
    }
    
    # Extract data
    features = npz_data.get("features")
    centroid_sims = npz_data.get("centroid_sims")
    temporal_sim_next = npz_data.get("temporal_sim_next")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    reference_present = npz_data.get("reference_present")
    ui_payload = npz_data.get("ui_payload")
    
    # Convert to numpy arrays if needed
    if centroid_sims is not None:
        if isinstance(centroid_sims, list):
            centroid_sims = np.array(centroid_sims, dtype=np.float32)
        elif isinstance(centroid_sims, np.ndarray):
            centroid_sims = np.asarray(centroid_sims, dtype=np.float32)
        else:
            centroid_sims = None
    
    if temporal_sim_next is not None:
        if isinstance(temporal_sim_next, list):
            temporal_sim_next = np.array(temporal_sim_next, dtype=np.float32)
        elif isinstance(temporal_sim_next, np.ndarray):
            temporal_sim_next = np.asarray(temporal_sim_next, dtype=np.float32)
        else:
            temporal_sim_next = None
    
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
    features_dict = {}
    if features is not None:
        if isinstance(features, np.ndarray) and features.dtype == object:
            if features.size == 1:
                features_dict = features.item() if isinstance(features.item(), dict) else {}
            else:
                features_dict = {}
        elif isinstance(features, dict):
            features_dict = features
        elif isinstance(features, list) and len(features) > 0:
            # Try to convert list to dict
            try:
                features_dict = dict(features) if isinstance(features[0], (list, tuple)) else {}
            except:
                features_dict = {}
    
    # Summary statistics
    summary = {
        "frames_count": int(centroid_sims.shape[0]) if centroid_sims is not None and centroid_sims.size > 0 else 0,
        "reference_present": bool(reference_present.item() if isinstance(reference_present, np.ndarray) else reference_present) if reference_present is not None else False,
    }
    
    if centroid_sims is not None and centroid_sims.size > 0:
        valid_centroid = centroid_sims[np.isfinite(centroid_sims)]
        if valid_centroid.size > 0:
            summary.update({
                "centroid_sim_mean": float(np.mean(valid_centroid)),
                "centroid_sim_std": float(np.std(valid_centroid)),
                "centroid_sim_min": float(np.min(valid_centroid)),
                "centroid_sim_max": float(np.max(valid_centroid)),
                "centroid_sim_median": float(np.median(valid_centroid)),
            })
    
    if temporal_sim_next is not None and temporal_sim_next.size > 0:
        valid_temporal = temporal_sim_next[np.isfinite(temporal_sim_next)]
        if valid_temporal.size > 0:
            summary.update({
                "temporal_sim_mean": float(np.mean(valid_temporal)),
                "temporal_sim_std": float(np.std(valid_temporal)),
                "temporal_sim_min": float(np.min(valid_temporal)),
                "temporal_sim_max": float(np.max(valid_temporal)),
            })
    
    # Add features to summary
    if features_dict:
        summary.update({
            "n_frames": int(features_dict.get("n_frames", 0)),
            "centroid_sim_p10": float(features_dict.get("centroid_sim_p10", float("nan"))) if "centroid_sim_p10" in features_dict else None,
            "centroid_sim_p90": float(features_dict.get("centroid_sim_p90", float("nan"))) if "centroid_sim_p90" in features_dict else None,
            "reference_similarity_mean_topn": float(features_dict.get("reference_similarity_mean_topn", float("nan"))) if "reference_similarity_mean_topn" in features_dict else None,
            "reference_similarity_max": float(features_dict.get("reference_similarity_max", float("nan"))) if "reference_similarity_max" in features_dict else None,
            "reference_similarity_p10": float(features_dict.get("reference_similarity_p10", float("nan"))) if "reference_similarity_p10" in features_dict else None,
        })
    
    render["summary"] = summary
    
    # Timeline data
    if centroid_sims is not None and times_s is not None and frame_indices is not None:
        n = min(len(centroid_sims), len(times_s), len(frame_indices))
        timeline = []
        
        for i in range(n):
            frame_idx = int(frame_indices[i]) if i < len(frame_indices) else i
            time_sec = float(times_s[i]) if i < len(times_s) else 0.0
            centroid_sim = float(centroid_sims[i]) if i < len(centroid_sims) and np.isfinite(centroid_sims[i]) else None
            temporal_sim = float(temporal_sim_next[i]) if temporal_sim_next is not None and i < len(temporal_sim_next) and np.isfinite(temporal_sim_next[i]) else None
            
            timeline.append({
                "frame_index": frame_idx,
                "time_sec": time_sec,
                "centroid_similarity": centroid_sim,
                "temporal_similarity": temporal_sim,
            })
        
        render["timeline"] = timeline
    
    # Distributions
    distributions = {}
    
    if centroid_sims is not None:
        valid_centroid = centroid_sims[np.isfinite(centroid_sims)]
        if valid_centroid.size > 0:
            distributions["centroid_similarity"] = {
                "min": float(np.min(valid_centroid)),
                "max": float(np.max(valid_centroid)),
                "mean": float(np.mean(valid_centroid)),
                "std": float(np.std(valid_centroid)),
                "median": float(np.median(valid_centroid)),
                "p25": float(np.percentile(valid_centroid, 25)),
                "p75": float(np.percentile(valid_centroid, 75)),
                "p10": float(np.percentile(valid_centroid, 10)),
                "p90": float(np.percentile(valid_centroid, 90)),
            }
    
    if temporal_sim_next is not None:
        valid_temporal = temporal_sim_next[np.isfinite(temporal_sim_next)]
        if valid_temporal.size > 0:
            distributions["temporal_similarity"] = {
                "min": float(np.min(valid_temporal)),
                "max": float(np.max(valid_temporal)),
                "mean": float(np.mean(valid_temporal)),
                "std": float(np.std(valid_temporal)),
                "median": float(np.median(valid_temporal)),
                "p25": float(np.percentile(valid_temporal, 25)),
                "p75": float(np.percentile(valid_temporal, 75)),
            }
    
    render["distributions"] = distributions
    
    # Reference similarities (from UI payload)
    if ui_payload is not None:
        if isinstance(ui_payload, np.ndarray) and ui_payload.dtype == object:
            if ui_payload.size == 1:
                ui_payload = ui_payload.item() if isinstance(ui_payload.item(), dict) else {}
            else:
                ui_payload = {}
        elif not isinstance(ui_payload, dict):
            ui_payload = {}
        
        if isinstance(ui_payload, dict):
            topk_refs = ui_payload.get("topk_refs", [])
            if topk_refs:
                render["reference_similarities"] = {
                    "reference_set_id": ui_payload.get("reference_set_id"),
                    "topk_count": len(topk_refs),
                    "topk_refs": topk_refs[:10],  # Limit to top 10 for render
                }
    
    return render


def render_similarity_metrics_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага similarity_metrics результатов.
    
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
    render = render_similarity_metrics(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    reference_similarities = render.get("reference_similarities", {})
    
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
        centroid_sims = [t.get("centroid_similarity") for t in timeline if t.get("centroid_similarity") is not None]
        temporal_sims = [t.get("temporal_similarity") for t in timeline if t.get("temporal_similarity") is not None]
        
        # Build datasets array
        datasets = []
        
        if centroid_sims:
            datasets.append({
                "label": "Centroid Similarity",
                "data": centroid_sims,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "tension": 0.1,
                "yAxisID": "y"
            })
        
        if temporal_sims:
            datasets.append({
                "label": "Temporal Similarity",
                "data": temporal_sims,
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
                            text: 'Temporal Similarity'
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
    
    # Reference similarities table
    ref_table_html = ""
    if reference_similarities and reference_similarities.get("topk_refs"):
        ref_table_html = f"""
        <div class="distributions">
            <h2>Top-K Reference Similarities</h2>
            <p><strong>Reference Set ID:</strong> {reference_similarities.get('reference_set_id', 'N/A')}</p>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Reference Video ID</th>
                        <th>Overall Score</th>
                        <th>CLIP</th>
                        <th>Audio (CLAP)</th>
                        <th>Text</th>
                        <th>Pacing</th>
                        <th>Quality</th>
                        <th>Emotion</th>
                    </tr>
                </thead>
                <tbody>
        """
        for idx, ref in enumerate(reference_similarities.get("topk_refs", []), 1):
            scores = ref.get("scores_by_modality", {})
            ref_table_html += f"""
                    <tr>
                        <td>{idx}</td>
                        <td>{ref.get('reference_video_id', 'N/A')}</td>
                        <td>{ref.get('score', 0.0):.4f if ref.get('score') is not None else 'N/A'}</td>
                        <td>{scores.get('clip', 0.0):.4f if scores.get('clip') is not None else 'N/A'}</td>
                        <td>{scores.get('audio_clap', 0.0):.4f if scores.get('audio_clap') is not None else 'N/A'}</td>
                        <td>{scores.get('text', 0.0):.4f if scores.get('text') is not None else 'N/A'}</td>
                        <td>{scores.get('pacing', 0.0):.4f if scores.get('pacing') is not None else 'N/A'}</td>
                        <td>{scores.get('quality', 0.0):.4f if scores.get('quality') is not None else 'N/A'}</td>
                        <td>{scores.get('emotion', 0.0):.4f if scores.get('emotion') is not None else 'N/A'}</td>
                    </tr>
            """
        ref_table_html += """
                </tbody>
            </table>
        </div>
        """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Similarity Metrics Debug Render</title>
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
        <h1>Similarity Metrics Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Reference Present</strong>
                    <span class="metric-value">{'Yes' if summary.get('reference_present', False) else 'No'}</span>
                </div>
                <div class="metric-card">
                    <strong>Centroid Sim Mean</strong>
                    <span class="metric-value">{summary.get('centroid_sim_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Centroid Sim Std</strong>
                    <span class="metric-value">{summary.get('centroid_sim_std', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Temporal Sim Mean</strong>
                    <span class="metric-value">{summary.get('temporal_sim_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Reference Sim Max</strong>
                    <span class="metric-value">{summary.get('reference_similarity_max', 0.0):.4f if summary.get('reference_similarity_max') is not None else 'N/A'}</span>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Similarity Metrics Over Time</h2>
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
                        <th>Centroid Similarity</th>
                        <th>Temporal Similarity</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('centroid_similarity', 'min')}</td>
                        <td>{format_dist_value('temporal_similarity', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('centroid_similarity', 'max')}</td>
                        <td>{format_dist_value('temporal_similarity', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('centroid_similarity', 'mean')}</td>
                        <td>{format_dist_value('temporal_similarity', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('centroid_similarity', 'std')}</td>
                        <td>{format_dist_value('temporal_similarity', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('centroid_similarity', 'median')}</td>
                        <td>{format_dist_value('temporal_similarity', 'median')}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions else ''}
        
        {ref_table_html}
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
                            text: 'Centroid Similarity'
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
    
    # Show relative path for cleaner output
    rel_output_path = os.path.relpath(output_path, os.getcwd()) if os.path.exists(output_path) else output_path
    logger.info(f"Saved Similarity Metrics HTML render to {rel_output_path}")
    return output_path


__all__ = ["render_similarity_metrics", "render_similarity_metrics_html"]

