"""
Renderer для face_identity: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_face_identity(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для face_identity."""
    render = {
        "component": "face_identity",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract face recognition data
    face_ids = npz_data.get("face_ids")
    face_names = npz_data.get("face_names")
    face_similarities = npz_data.get("face_similarities")
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    
    # Convert to numpy arrays if needed
    if face_ids is not None:
        if isinstance(face_ids, list):
            face_ids = np.array(face_ids, dtype=np.int32)
        elif isinstance(face_ids, np.ndarray):
            face_ids = np.asarray(face_ids, dtype=np.int32)
        else:
            face_ids = None
    
    if face_names is not None:
        if isinstance(face_names, list):
            face_names = np.array(face_names, dtype="U256")
        elif isinstance(face_names, np.ndarray):
            face_names = np.asarray(face_names, dtype="U256")
        else:
            face_names = None
    
    if face_similarities is not None:
        if isinstance(face_similarities, list):
            face_similarities = np.array(face_similarities, dtype=np.float32)
        elif isinstance(face_similarities, np.ndarray):
            face_similarities = np.asarray(face_similarities, dtype=np.float32)
        else:
            face_similarities = None
    
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
    if face_ids is not None and face_ids.size > 0:
        n_frames = face_ids.shape[0]
        topk = face_ids.shape[1] if face_ids.ndim > 1 else 1
        
        # Extract valid results (face_id != -1)
        valid_mask = face_ids != -1
        valid_similarities = face_similarities[valid_mask] if face_similarities is not None else np.array([])
        
        # Count unique faces
        unique_face_ids = set()
        unique_face_names = set()
        if face_ids is not None:
            for frame_idx in range(n_frames):
                for k in range(topk):
                    face_id = int(face_ids[frame_idx, k])
                    if face_id != -1:
                        unique_face_ids.add(face_id)
                        if face_names is not None:
                            face_name = str(face_names[frame_idx, k]).strip()
                            if face_name:
                                unique_face_names.add(face_name)
        
        # Count confident predictions (similarity > 0.7)
        confident_count = 0
        if valid_similarities.size > 0:
            confident_count = int(np.sum(valid_similarities > 0.7))
        
        # Top-1 statistics
        top1_similarities = []
        if face_similarities is not None and face_similarities.shape[0] > 0:
            top1_similarities = face_similarities[:, 0].tolist()
            top1_similarities = [s for s in top1_similarities if s > 0.0]
        
        render["summary"] = {
            "frames_count": int(n_frames),
            "topk": int(topk),
            "unique_faces_count": len(unique_face_ids),
            "unique_face_names_count": len(unique_face_names),
            "total_identifications": int(np.sum(valid_mask)),
            "confident_predictions_count": confident_count,
            "confident_predictions_ratio": float(confident_count / max(1, np.sum(valid_mask))),
            "top1_score_mean": float(np.mean(top1_similarities)) if top1_similarities else 0.0,
            "top1_score_std": float(np.std(top1_similarities)) if top1_similarities else 0.0,
            "top1_score_min": float(np.min(top1_similarities)) if top1_similarities else 0.0,
            "top1_score_max": float(np.max(top1_similarities)) if top1_similarities else 0.0,
            "top1_score_median": float(np.median(top1_similarities)) if top1_similarities else 0.0,
        }
        
        # Timeline data
        if times_s is not None and len(times_s) == n_frames:
            timeline = []
            for i in range(n_frames):
                frame_data = {
                    "time_sec": float(times_s[i]),
                    "frame_index": int(frame_indices[i]) if frame_indices is not None and i < len(frame_indices) else i,
                }
                
                # Top-1 face
                if face_ids is not None and i < face_ids.shape[0]:
                    top1_id = int(face_ids[i, 0])
                    top1_name = str(face_names[i, 0]).strip() if face_names is not None and i < face_names.shape[0] else ""
                    top1_score = float(face_similarities[i, 0]) if face_similarities is not None and i < face_similarities.shape[0] else 0.0
                    
                    frame_data["top1_face_id"] = top1_id if top1_id != -1 else None
                    frame_data["top1_face_name"] = top1_name if top1_name else None
                    frame_data["top1_score"] = top1_score if top1_score > 0.0 else None
                    frame_data["is_confident"] = top1_score > 0.7
                    
                    # Count unique faces in this frame
                    frame_face_ids = set()
                    frame_face_names = set()
                    for k in range(topk):
                        face_id = int(face_ids[i, k])
                        if face_id != -1:
                            frame_face_ids.add(face_id)
                            if face_names is not None:
                                face_name = str(face_names[i, k]).strip()
                                if face_name:
                                    frame_face_names.add(face_name)
                    
                    frame_data["unique_faces_count"] = len(frame_face_ids)
                    
                    # Top-K scores
                    if face_similarities is not None and i < face_similarities.shape[0]:
                        topk_scores = [float(s) for s in face_similarities[i, :] if s > 0.0]
                        frame_data["topk_scores"] = topk_scores
                
                timeline.append(frame_data)
            
            render["timeline"] = timeline
        
        # Distribution statistics
        if valid_similarities.size > 0:
            render["distributions"] = {
                "top1_scores": {
                    "min": float(np.min(top1_similarities)) if top1_similarities else 0.0,
                    "max": float(np.max(top1_similarities)) if top1_similarities else 0.0,
                    "mean": float(np.mean(top1_similarities)) if top1_similarities else 0.0,
                    "std": float(np.std(top1_similarities)) if top1_similarities else 0.0,
                    "median": float(np.median(top1_similarities)) if top1_similarities else 0.0,
                    "q25": float(np.percentile(top1_similarities, 25)) if top1_similarities else 0.0,
                    "q75": float(np.percentile(top1_similarities, 75)) if top1_similarities else 0.0,
                },
                "all_scores": {
                    "min": float(np.min(valid_similarities)),
                    "max": float(np.max(valid_similarities)),
                    "mean": float(np.mean(valid_similarities)),
                    "std": float(np.std(valid_similarities)),
                    "median": float(np.median(valid_similarities)),
                    "q25": float(np.percentile(valid_similarities, 25)),
                    "q75": float(np.percentile(valid_similarities, 75)),
                },
            }
        
        # Top faces (by count and average score)
        if face_names is not None and face_ids is not None:
            face_stats: Dict[str, Dict[str, Any]] = {}
            
            for frame_idx in range(n_frames):
                for k in range(topk):
                    face_id = int(face_ids[frame_idx, k])
                    if face_id == -1:
                        continue
                    
                    face_name = str(face_names[frame_idx, k]).strip() if face_names is not None else ""
                    similarity = float(face_similarities[frame_idx, k]) if face_similarities is not None else 0.0
                    
                    if not face_name:
                        face_name = f"face_{face_id}"
                    
                    if face_name not in face_stats:
                        face_stats[face_name] = {
                            "face_id": face_id,
                            "count": 0,
                            "total_score": 0.0,
                            "max_score": 0.0,
                        }
                    
                    face_stats[face_name]["count"] += 1
                    face_stats[face_name]["total_score"] += similarity
                    face_stats[face_name]["max_score"] = max(face_stats[face_name]["max_score"], similarity)
            
            # Calculate averages and sort
            top_faces = []
            for face_name, stats in face_stats.items():
                top_faces.append({
                    "face_name": face_name,
                    "face_id": stats["face_id"],
                    "count": stats["count"],
                    "avg_score": stats["total_score"] / stats["count"],
                    "max_score": stats["max_score"],
                })
            
            top_faces.sort(key=lambda x: (x["count"], x["avg_score"]), reverse=True)
            render["top_faces"] = top_faces[:20]  # Top 20
    
    return render


def render_face_identity_html(render_context: Dict[str, Any], output_path: str) -> None:
    """Генерировать HTML debug страницу для face_identity."""
    summary = render_context.get("summary", {})
    timeline = render_context.get("timeline", [])
    distributions = render_context.get("distributions", {})
    top_faces = render_context.get("top_faces", [])
    
    # Prepare data for charts
    timeline_times = [d["time_sec"] for d in timeline]
    timeline_scores = [d.get("top1_score", 0.0) or 0.0 for d in timeline]
    timeline_names = [d.get("top1_face_name", "") or "" for d in timeline]
    
    # Unique colors for faces
    unique_names = list(set([n for n in timeline_names if n]))
    color_map = {name: f"hsl({(hash(name) % 360)}, 70%, 50%)" for name in unique_names}
    timeline_colors = [color_map.get(name, "#999") if name else "#999" for name in timeline_names]
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Identity - Debug Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .summary-item {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .summary-item label {{
            display: block;
            font-weight: bold;
            color: #666;
            margin-bottom: 5px;
        }}
        .summary-item value {{
            display: block;
            font-size: 24px;
            color: #333;
        }}
        .chart-container {{
            margin: 30px 0;
            position: relative;
            height: 400px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Face Identity Recognition - Debug Visualization</h1>
        
        <h2>Summary</h2>
        <div class="summary">
            <div class="summary-item">
                <label>Frames Count</label>
                <value>{summary.get('frames_count', 0)}</value>
            </div>
            <div class="summary-item">
                <label>Unique Faces</label>
                <value>{summary.get('unique_faces_count', 0)}</value>
            </div>
            <div class="summary-item">
                <label>Total Identifications</label>
                <value>{summary.get('total_identifications', 0)}</value>
            </div>
            <div class="summary-item">
                <label>Confident Predictions</label>
                <value>{summary.get('confident_predictions_count', 0)} ({summary.get('confident_predictions_ratio', 0.0):.1%})</value>
            </div>
            <div class="summary-item">
                <label>Top-1 Score Mean</label>
                <value>{summary.get('top1_score_mean', 0.0):.3f}</value>
            </div>
            <div class="summary-item">
                <label>Top-1 Score Median</label>
                <value>{summary.get('top1_score_median', 0.0):.3f}</value>
            </div>
        </div>
        
        <h2>Timeline: Top-1 Face Scores</h2>
        <div class="chart-container">
            <canvas id="timelineChart"></canvas>
        </div>
        
        <h2>Distribution: Top-1 Scores</h2>
        <div class="chart-container">
            <canvas id="distributionChart"></canvas>
        </div>
        
        <h2>Top Faces</h2>
        <table>
            <thead>
                <tr>
                    <th>Face Name</th>
                    <th>Face ID</th>
                    <th>Count</th>
                    <th>Avg Score</th>
                    <th>Max Score</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for face in top_faces[:20]:
        html_content += f"""
                <tr>
                    <td>{face['face_name']}</td>
                    <td>{face['face_id']}</td>
                    <td>{face['count']}</td>
                    <td>{face['avg_score']:.3f}</td>
                    <td>{face['max_score']:.3f}</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
    </div>
    
    <script>
        // Timeline chart
        const timelineCtx = document.getElementById('timelineChart').getContext('2d');
        new Chart(timelineCtx, {
            type: 'line',
            data: {
                labels: """ + json.dumps([f"{t:.1f}s" for t in timeline_times]) + """,
                datasets: [{
                    label: 'Top-1 Similarity Score',
                    data: """ + json.dumps(timeline_scores) + """,
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0
                    }
                }
            }
        });
        
        // Distribution chart
        const distCtx = document.getElementById('distributionChart').getContext('2d');
        const distData = """ + json.dumps(timeline_scores) + """;
        new Chart(distCtx, {
            type: 'histogram',
            data: {
                datasets: [{
                    label: 'Top-1 Score Distribution',
                    data: distData,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'Similarity Score'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Frequency'
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>
"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Generated HTML render for face_identity: {output_path}")

