"""
Renderer для emotion_diarization_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ...core.renderer import load_npz, extract_meta

def render_emotion_diarization_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для emotion_diarization_extractor."""
    logger.info(f"emotion_diarization | render: starting render-context generation")
    
    render = {
        "component": "emotion_diarization_extractor",
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract scalar features
    feature_names = npz_data.get("feature_names", [])
    feature_values = npz_data.get("feature_values", [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    if isinstance(feature_values, np.ndarray):
        feature_values = feature_values.tolist()
    
    features = {}
    for i, name in enumerate(feature_names):
        if i < len(feature_values):
            features[name] = feature_values[i]
    
    # Log available data for debugging
    logger.debug(f"emotion_diarization | render: feature_names={feature_names}")
    logger.debug(f"emotion_diarization | render: npz_data keys={list(npz_data.keys())}")
    logger.debug(f"emotion_diarization | render: meta features_enabled={meta.get('features_enabled', [])}")
    
    # Check if emotion_id and emotion_confidence are available
    emotion_id_available = "emotion_id" in npz_data
    emotion_confidence_available = "emotion_confidence" in npz_data
    logger.info(f"emotion_diarization | render: emotion_id available={emotion_id_available}, emotion_confidence available={emotion_confidence_available}")
    
    if not emotion_id_available and not emotion_confidence_available:
        logger.warning(f"emotion_diarization | render: WARNING - emotion_id and emotion_confidence are not available in NPZ. This likely means feature flags (enable_ids, enable_confidence) are disabled in config.")
    
    # Helper function to safely convert to int (handles NaN)
    def safe_int(value, default=0):
        try:
            if value is None:
                return default
            v = float(value)
            if np.isnan(v) or np.isinf(v):
                return default
            return int(v)
        except (ValueError, TypeError):
            return default
    
    # Helper function to safely convert to float (handles NaN)
    def safe_float(value, default=0.0):
        try:
            if value is None:
                return default
            v = float(value)
            if np.isnan(v) or np.isinf(v):
                return default
            return v
        except (ValueError, TypeError):
            return default
    
    # Summary
    segments_count = safe_int(features.get("segments_count", 0))
    render["summary"] = {
        "segments_count": segments_count,
        "sample_rate": safe_int(features.get("sample_rate", 16000)),
    }
    
    # Emotion labels
    emotion_labels = npz_data.get("emotion_labels")
    if emotion_labels is not None:
        if isinstance(emotion_labels, np.ndarray) and emotion_labels.dtype == object:
            emotion_labels = emotion_labels.item() if emotion_labels.size == 1 else emotion_labels.tolist()
        if isinstance(emotion_labels, list):
            render["summary"]["emotion_labels"] = emotion_labels
    
    # Feature-gated aggregates
    if "emotion_entropy" in features:
        render["summary"]["emotion_entropy"] = safe_float(features["emotion_entropy"])
    if "dominant_emotion_id" in features:
        render["summary"]["dominant_emotion_id"] = safe_int(features["dominant_emotion_id"], default=-1)
    if "dominant_emotion_prob" in features:
        render["summary"]["dominant_emotion_prob"] = safe_float(features["dominant_emotion_prob"])
    if "emotion_transitions_count" in features:
        render["summary"]["emotion_transitions_count"] = safe_int(features["emotion_transitions_count"])
    if "emotion_stability_score" in features:
        render["summary"]["emotion_stability_score"] = safe_float(features["emotion_stability_score"])
    if "emotion_diversity_score" in features:
        render["summary"]["emotion_diversity_score"] = safe_float(features["emotion_diversity_score"])
    
    # Emotion distribution
    emotion_distribution = npz_data.get("emotion_distribution")
    if emotion_distribution is not None:
        if isinstance(emotion_distribution, np.ndarray) and emotion_distribution.dtype == object:
            emotion_distribution = emotion_distribution.item() if emotion_distribution.size == 1 else {}
        if isinstance(emotion_distribution, dict):
            render["summary"]["emotion_distribution"] = {str(k): safe_float(v) for k, v in emotion_distribution.items()}
    
    # Emotion segments per emotion
    emotion_segments_per_emotion = npz_data.get("emotion_segments_per_emotion")
    if emotion_segments_per_emotion is not None:
        if isinstance(emotion_segments_per_emotion, np.ndarray) and emotion_segments_per_emotion.dtype == object:
            emotion_segments_per_emotion = emotion_segments_per_emotion.item() if emotion_segments_per_emotion.size == 1 else {}
        if isinstance(emotion_segments_per_emotion, dict):
            render["summary"]["emotion_segments_per_emotion"] = {str(k): safe_int(v) for k, v in emotion_segments_per_emotion.items()}
    
    # Quality metrics
    emotion_quality_metrics = npz_data.get("emotion_quality_metrics")
    if emotion_quality_metrics is not None:
        if isinstance(emotion_quality_metrics, np.ndarray) and emotion_quality_metrics.dtype == object:
            emotion_quality_metrics = emotion_quality_metrics.item() if emotion_quality_metrics.size == 1 else {}
        if isinstance(emotion_quality_metrics, dict):
            render["summary"]["emotion_quality_metrics"] = {
                k: safe_float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in emotion_quality_metrics.items()
            }
    
    # Timeline (emotion segments)
    segment_start_sec = npz_data.get("segment_start_sec")
    segment_end_sec = npz_data.get("segment_end_sec")
    segment_center_sec = npz_data.get("segment_center_sec")
    emotion_id = npz_data.get("emotion_id")
    emotion_confidence = npz_data.get("emotion_confidence")
    
    if segment_center_sec is not None:
        if isinstance(segment_center_sec, np.ndarray):
            segment_center_sec = segment_center_sec.tolist()
        if isinstance(segment_start_sec, np.ndarray):
            segment_start_sec = segment_start_sec.tolist()
        if isinstance(segment_end_sec, np.ndarray):
            segment_end_sec = segment_end_sec.tolist()
        if isinstance(emotion_id, np.ndarray):
            emotion_id = emotion_id.tolist()
        if isinstance(emotion_confidence, np.ndarray):
            emotion_confidence = emotion_confidence.tolist()
        
        timeline = []
        for i, center_sec in enumerate(segment_center_sec):
            entry = {
                "center_sec": safe_float(center_sec),
                "segment_index": i,
            }
            if segment_start_sec and i < len(segment_start_sec):
                entry["start_sec"] = safe_float(segment_start_sec[i])
            if segment_end_sec and i < len(segment_end_sec):
                entry["end_sec"] = safe_float(segment_end_sec[i])
            if emotion_id and i < len(emotion_id):
                entry["emotion_id"] = safe_int(emotion_id[i], default=-1)
            if emotion_confidence and i < len(emotion_confidence):
                entry["emotion_confidence"] = safe_float(emotion_confidence[i])
            timeline.append(entry)
        render["timeline"] = timeline
    
    # Distribution of emotion IDs
    if emotion_id is not None:
        if isinstance(emotion_id, np.ndarray):
            emotion_id = emotion_id.tolist()
        if emotion_id:
            unique_emotions = sorted(set(emotion_id))
            emotion_counts = {}
            for emo_id in unique_emotions:
                emotion_counts[str(emo_id)] = safe_int(emotion_id.count(emo_id))
            render["distributions"]["emotion_ids"] = emotion_counts
            
            # Distribution of confidence
            if emotion_confidence:
                if isinstance(emotion_confidence, np.ndarray):
                    emotion_confidence = emotion_confidence.tolist()
                conf_arr = np.asarray(emotion_confidence, dtype=np.float32)
                # Filter out NaN/inf values for statistics
                conf_arr_clean = conf_arr[np.isfinite(conf_arr)]
                if len(conf_arr_clean) > 0:
                    render["distributions"]["emotion_confidence"] = {
                        "min": safe_float(np.min(conf_arr_clean)),
                        "max": safe_float(np.max(conf_arr_clean)),
                        "mean": safe_float(np.mean(conf_arr_clean)),
                        "std": safe_float(np.std(conf_arr_clean)),
                        "median": safe_float(np.median(conf_arr_clean)),
                    }
                else:
                    render["distributions"]["emotion_confidence"] = {
                        "min": 0.0,
                        "max": 0.0,
                        "mean": 0.0,
                        "std": 0.0,
                        "median": 0.0,
                    }
    
    return render


def render_emotion_diarization_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML render для emotion_diarization_extractor (debug mode).
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_emotion_diarization_extractor(npz_data, meta)
    
    # Helper function to safely convert to int (handles NaN)
    def safe_int(value, default=0):
        try:
            if value is None:
                return default
            v = float(value)
            if np.isnan(v) or np.isinf(v):
                return default
            return int(v)
        except (ValueError, TypeError):
            return default
    
    # Extract data for visualization
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    emotion_labels = summary.get("emotion_labels", [])
    emotion_distribution = summary.get("emotion_distribution", {})
    quality_metrics = summary.get("emotion_quality_metrics", {})
    
    # Generate color palette for emotions
    emotion_colors = {}
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336", "#00BCD4", "#FFEB3B", "#795548", "#E91E63", "#3F51B5"]
    unique_emotions = sorted(set(seg.get("emotion_id", -1) for seg in timeline if "emotion_id" in seg))
    for i, emo_id in enumerate(unique_emotions):
        emotion_colors[emo_id] = colors[i % len(colors)]
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Diarization Extractor Debug View</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
        .stat-label {{ font-size: 0.9em; color: #666; }}
        .stat-value {{ font-size: 1.5em; font-weight: bold; color: #333; }}
        .timeline {{ margin: 20px 0; }}
        .segment {{ background: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 3px solid #4CAF50; border-radius: 3px; }}
        .segment-header {{ font-weight: bold; color: #333; }}
        .emotion-badge {{ display: inline-block; padding: 3px 8px; border-radius: 3px; color: white; font-weight: bold; font-size: 0.85em; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #e3f2fd; padding: 15px; border-radius: 5px; }}
        .metric-label {{ font-size: 0.9em; color: #1976d2; }}
        .metric-value {{ font-size: 1.3em; font-weight: bold; color: #0d47a1; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Emotion Diarization Extractor Debug View</h1>
        <p><strong>Status:</strong> {meta.get('status', 'unknown')}</p>
        <p><strong>Producer:</strong> {meta.get('producer', 'unknown')} v{meta.get('producer_version', 'unknown')}</p>
        <p><strong>Contract Version:</strong> {meta.get('emotion_contract_version', 'unknown')}</p>
        
        <h2>Summary Statistics</h2>
        <div class="summary">
            <div class="stat-card">
                <div class="stat-label">Segments Count</div>
                <div class="stat-value">{summary.get('segments_count', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Emotion Entropy</div>
                <div class="stat-value">{summary.get('emotion_entropy', 0.0):.3f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Dominant Emotion ID</div>
                <div class="stat-value">{summary.get('dominant_emotion_id', -1)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Dominant Emotion Prob</div>
                <div class="stat-value">{summary.get('dominant_emotion_prob', 0.0):.3f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Transitions Count</div>
                <div class="stat-value">{summary.get('emotion_transitions_count', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Stability Score</div>
                <div class="stat-value">{summary.get('emotion_stability_score', 0.0):.3f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Diversity Score</div>
                <div class="stat-value">{summary.get('emotion_diversity_score', 0.0):.3f}</div>
            </div>
        </div>
        
        <h2>Emotion Distribution</h2>
        <div class="summary">
"""
    
    for emo_id, ratio in emotion_distribution.items():
        emo_id_int = safe_int(emo_id, default=-1)
        color = emotion_colors.get(emo_id_int, "#666")
        emo_name = emotion_labels[emo_id_int] if emo_id_int >= 0 and emo_id_int < len(emotion_labels) else f"emotion_{emo_id_int}"
        html_content += f"""
            <div class="stat-card">
                <div class="stat-label">{emo_name}</div>
                <div class="stat-value" style="color: {color};">{ratio:.1%}</div>
            </div>
"""
    
    html_content += """
        </div>
"""
    
    # Quality metrics
    if quality_metrics:
        html_content += """
        <h2>Quality Metrics</h2>
        <div class="metrics">
"""
        for metric_name, metric_value in quality_metrics.items():
            if isinstance(metric_value, (int, float)) and not np.isinf(metric_value):
                html_content += f"""
            <div class="metric-card">
                <div class="metric-label">{metric_name.replace('_', ' ').title()}</div>
                <div class="metric-value">{metric_value:.4f}</div>
            </div>
"""
        html_content += """
        </div>
"""
    
    # Timeline visualization
    html_content += """
        <h2>Timeline (Emotion Segments)</h2>
        <div id="timeline-plot" style="width: 100%; height: 400px;"></div>
        <div class="timeline">
"""
    
    for seg in timeline[:50]:  # Show first 50 segments
        emo_id = seg.get("emotion_id", -1)
        color = emotion_colors.get(emo_id, "#666")
        emo_name = emotion_labels[emo_id] if emo_id >= 0 and emo_id < len(emotion_labels) else f"emotion_{emo_id}"
        start_sec = seg.get("start_sec", 0.0)
        end_sec = seg.get("end_sec", 0.0)
        confidence = seg.get("emotion_confidence", 0.0)
        html_content += f"""
            <div class="segment">
                <div class="segment-header">
                    Segment {seg.get('segment_index', -1)}: 
                    <span class="emotion-badge" style="background-color: {color};">{emo_name}</span>
                    [{start_sec:.2f}s - {end_sec:.2f}s] (confidence: {confidence:.3f})
                </div>
            </div>
"""
    
    if len(timeline) > 50:
        html_content += f"""
            <p><em>... and {len(timeline) - 50} more segments</em></p>
"""
    
    html_content += """
        </div>
    </div>
    
    <script>
        // Timeline plot
        var timelineData = [
"""
    
    # Generate plotly data for timeline
    for emo_id in unique_emotions:
        emotion_segments = [seg for seg in timeline if seg.get("emotion_id") == emo_id]
        if emotion_segments:
            color = emotion_colors.get(emo_id, "#666")
            emo_name = emotion_labels[emo_id] if emo_id >= 0 and emo_id < len(emotion_labels) else f"emotion_{emo_id}"
            x_data = [seg.get('start_sec', 0.0) for seg in emotion_segments]
            y_data = [emo_id] * len(emotion_segments)
            html_content += f"""
            {{
                x: {json.dumps(x_data)},
                y: {json.dumps(y_data)},
                mode: 'markers',
                type: 'scatter',
                name: '{emo_name}',
                marker: {{ color: '{color}', size: 10 }},
            }},
"""
    
    html_content += """
        ];
        
        var layout = {
            title: 'Emotion Timeline',
            xaxis: { title: 'Time (seconds)' },
            yaxis: { title: 'Emotion ID' },
            height: 400,
        };
        
        Plotly.newPlot('timeline-plot', timelineData, layout);
    </script>
</body>
</html>
"""
    
    # Save HTML
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Saved emotion_diarization HTML render to {output_path}")
    return output_path

__all__ = ["render_emotion_diarization_extractor", "render_emotion_diarization_extractor_html"]
