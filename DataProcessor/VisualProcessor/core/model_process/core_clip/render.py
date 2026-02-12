"""
Renderer для core_clip: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_core_clip(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для core_clip."""
    render = {
        "component": "core_clip",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract frame embeddings
    frame_embeddings = npz_data.get("frame_embeddings")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    
    if frame_embeddings is not None:
        if isinstance(frame_embeddings, list):
            frame_embeddings = np.array(frame_embeddings, dtype=np.float32)
        elif isinstance(frame_embeddings, np.ndarray):
            frame_embeddings = np.asarray(frame_embeddings, dtype=np.float32)
        else:
            frame_embeddings = None
    
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
    if frame_embeddings is not None and frame_embeddings.size > 0:
        # Calculate norms for each frame
        norms = np.linalg.norm(frame_embeddings, axis=1) if frame_embeddings.ndim == 2 else np.array([np.linalg.norm(frame_embeddings)])
        
        # Calculate cosine similarities between consecutive frames
        cosine_similarities = []
        if frame_embeddings.ndim == 2 and frame_embeddings.shape[0] > 1:
            # Normalize embeddings for cosine similarity
            normalized = frame_embeddings / (norms[:, np.newaxis] + 1e-8)
            for i in range(len(normalized) - 1):
                cos_sim = float(np.dot(normalized[i], normalized[i + 1]))
                cosine_similarities.append(cos_sim)
        
        render["summary"] = {
            "frames_count": int(frame_embeddings.shape[0]) if frame_embeddings.ndim == 2 else 1,
            "embedding_dim": int(frame_embeddings.shape[1]) if frame_embeddings.ndim == 2 else int(frame_embeddings.size),
            "embedding_norm_mean": float(np.mean(norms)),
            "embedding_norm_std": float(np.std(norms)),
            "embedding_norm_min": float(np.min(norms)),
            "embedding_norm_max": float(np.max(norms)),
            "embedding_norm_median": float(np.median(norms)),
            "cosine_similarity_mean": float(np.mean(cosine_similarities)) if cosine_similarities else 0.0,
            "cosine_similarity_std": float(np.std(cosine_similarities)) if cosine_similarities else 0.0,
        }
        
        # Timeline data
        if times_s is not None and len(times_s) == len(norms):
            timeline = []
            for i, (time_sec, norm) in enumerate(zip(times_s, norms)):
                timeline.append({
                    "time_sec": float(time_sec),
                    "frame_index": int(frame_indices[i]) if frame_indices is not None and i < len(frame_indices) else i,
                    "embedding_norm": float(norm),
                    "cosine_similarity": float(cosine_similarities[i]) if i < len(cosine_similarities) else None,
                })
            render["timeline"] = timeline
        
        # Distribution statistics
        render["distributions"] = {
            "embedding_norm": {
                "min": float(np.min(norms)),
                "max": float(np.max(norms)),
                "mean": float(np.mean(norms)),
                "std": float(np.std(norms)),
                "median": float(np.median(norms)),
                "q25": float(np.percentile(norms, 25)),
                "q75": float(np.percentile(norms, 75)),
            },
        }
        
        if cosine_similarities:
            render["distributions"]["cosine_similarity"] = {
                "min": float(np.min(cosine_similarities)),
                "max": float(np.max(cosine_similarities)),
                "mean": float(np.mean(cosine_similarities)),
                "std": float(np.std(cosine_similarities)),
                "median": float(np.median(cosine_similarities)),
            }
    
    # Extract text embeddings info
    text_embedding_keys = [
        "shot_quality_text_embeddings",
        "scene_aesthetic_text_embeddings",
        "scene_luxury_text_embeddings",
        "scene_atmosphere_text_embeddings",
        "cut_detection_transition_text_embeddings",
        "popularity_topic_text_embeddings",
    ]
    
    text_embeddings_info = {}
    for key in text_embedding_keys:
        text_emb = npz_data.get(key)
        if text_emb is not None:
            if isinstance(text_emb, list):
                text_emb = np.array(text_emb, dtype=np.float32)
            elif isinstance(text_emb, np.ndarray):
                text_emb = np.asarray(text_emb, dtype=np.float32)
            
            if text_emb.size > 0:
                text_norms = np.linalg.norm(text_emb, axis=1) if text_emb.ndim == 2 else np.array([np.linalg.norm(text_emb)])
                text_embeddings_info[key.replace("_text_embeddings", "")] = {
                    "count": int(text_emb.shape[0]) if text_emb.ndim == 2 else 1,
                    "dim": int(text_emb.shape[1]) if text_emb.ndim == 2 else int(text_emb.size),
                    "norm_mean": float(np.mean(text_norms)),
                    "norm_std": float(np.std(text_norms)),
                }
    
    if text_embeddings_info:
        render["summary"]["text_embeddings"] = text_embeddings_info
    
    return render


def render_core_clip_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага core_clip результатов.
    
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
        
        def extract_meta(npz_data):
            meta = npz_data.get("meta")
            if meta is None:
                return {}
            if isinstance(meta, np.ndarray) and meta.dtype == object:
                if meta.size == 1:
                    return meta.item() if isinstance(meta.item(), dict) else {}
                return meta.item() if hasattr(meta, 'item') else {}
            if isinstance(meta, dict):
                return meta
            return {}
    
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_core_clip(npz_data, meta)
    
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    
    # Prepare timeline data for chart
    timeline_js = ""
    y1_scale_js = ""
    if timeline:
        times = [t.get("time_sec", 0.0) for t in timeline]
        norms = [t.get("embedding_norm", 0.0) for t in timeline]
        cosine_sims = [t.get("cosine_similarity") for t in timeline if t.get("cosine_similarity") is not None]
        
        # Build datasets array
        datasets = [{
            "label": "Embedding Norm",
            "data": norms,
            "borderColor": "rgb(75, 192, 192)",
            "backgroundColor": "rgba(75, 192, 192, 0.2)",
            "tension": 0.1,
            "yAxisID": "y"
        }]
        
        if cosine_sims:
            datasets.append({
                "label": "Cosine Similarity",
                "data": cosine_sims,
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
                            text: 'Cosine Similarity'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }"""
        
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
    <title>Core CLIP Debug Render</title>
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
        <h1>Core CLIP Debug Render</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames Count</strong>
                    <span class="metric-value">{summary.get('frames_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Embedding Dim</strong>
                    <span class="metric-value">{summary.get('embedding_dim', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Norm Mean</strong>
                    <span class="metric-value">{summary.get('embedding_norm_mean', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Norm Std</strong>
                    <span class="metric-value">{summary.get('embedding_norm_std', 0.0):.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Cosine Similarity Mean</strong>
                    <span class="metric-value">{summary.get('cosine_similarity_mean', 0.0):.4f}</span>
                </div>
            </div>
        </div>
        
        {f'''
        <div class="chart-container">
            <h2>Timeline: Embedding Norm Over Time</h2>
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
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{distributions.get('embedding_norm', {}).get('min', 0.0):.4f}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{distributions.get('embedding_norm', {}).get('max', 0.0):.4f}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{distributions.get('embedding_norm', {}).get('mean', 0.0):.4f}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{distributions.get('embedding_norm', {}).get('std', 0.0):.4f}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{distributions.get('embedding_norm', {}).get('median', 0.0):.4f}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions.get('embedding_norm') else ''}
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
    
    logger.info(f"Saved Core CLIP HTML render to {output_path}")
    return output_path


__all__ = ["render_core_clip", "render_core_clip_html"]

