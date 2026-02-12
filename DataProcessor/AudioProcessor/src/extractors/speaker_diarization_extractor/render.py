"""
Renderer для speaker_diarization_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ...core.renderer import load_npz, extract_meta

def render_speaker_diarization_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для speaker_diarization_extractor."""
    render = {
        "component": "speaker_diarization_extractor",
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
            if np.isnan(v):
                return default
            if np.isinf(v):
                return default
            return v
        except (ValueError, TypeError):
            return default
    
    # Summary
    speaker_count = safe_int(features.get("speaker_count", 0))
    segments_count = safe_int(features.get("segments_count", 0))
    duration_sec = safe_float(features.get("duration_sec", 0.0))
    
    render["summary"] = {
        "speaker_count": speaker_count,
        "segments_count": segments_count,
        "duration_sec": duration_sec,
    }
    
    # Additional aggregates (if available)
    if "speaker_balance_score" in features:
        render["summary"]["speaker_balance_score"] = safe_float(features["speaker_balance_score"])
    if "speaker_transitions_count" in features:
        render["summary"]["speaker_transitions_count"] = safe_int(features["speaker_transitions_count"])
    if "speaker_segments_density" in features:
        render["summary"]["speaker_segments_density"] = safe_float(features["speaker_segments_density"])
    if "dominant_speaker_id" in features:
        render["summary"]["dominant_speaker_id"] = safe_int(features["dominant_speaker_id"])
    
    # Speaker time ratios (if available)
    speaker_time_ratios = npz_data.get("speaker_time_ratios")
    if speaker_time_ratios is not None:
        if isinstance(speaker_time_ratios, np.ndarray) and speaker_time_ratios.dtype == object:
            speaker_time_ratios = speaker_time_ratios.item() if speaker_time_ratios.size == 1 else {}
        if isinstance(speaker_time_ratios, dict):
            render["summary"]["speaker_time_ratios"] = {str(k): float(v) for k, v in speaker_time_ratios.items()}
    
    # Clustering metrics (if available)
    clustering_metrics = npz_data.get("clustering_metrics")
    if clustering_metrics is not None:
        if isinstance(clustering_metrics, np.ndarray) and clustering_metrics.dtype == object:
            clustering_metrics = clustering_metrics.item() if clustering_metrics.size == 1 else {}
        if isinstance(clustering_metrics, dict):
            render["summary"]["clustering_metrics"] = {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in clustering_metrics.items()
            }
    
    # Timeline (speaker segments)
    segment_start_sec = npz_data.get("segment_start_sec")
    segment_end_sec = npz_data.get("segment_end_sec")
    segment_center_sec = npz_data.get("segment_center_sec")
    speaker_ids = npz_data.get("speaker_ids")
    speaker_segments = npz_data.get("speaker_segments")
    
    # Use speaker_segments if available (more detailed), otherwise reconstruct from arrays
    if speaker_segments is not None:
        if isinstance(speaker_segments, np.ndarray) and speaker_segments.dtype == object:
            speaker_segments = speaker_segments.tolist()
        if isinstance(speaker_segments, list):
            timeline = []
            for seg in speaker_segments:
                if isinstance(seg, dict):
                    timeline.append({
                        "start_sec": safe_float(seg.get("start", 0.0)),
                        "end_sec": safe_float(seg.get("end", 0.0)),
                        "center_sec": safe_float((seg.get("start", 0.0) + seg.get("end", 0.0)) / 2.0),
                        "speaker_id": safe_int(seg.get("speaker_id", -1)),
                        "segment_index": safe_int(seg.get("segment_index", -1)),
                        "duration": safe_float(seg.get("duration", 0.0)),
                    })
            render["timeline"] = timeline
    elif segment_center_sec is not None:
        if isinstance(segment_center_sec, np.ndarray):
            segment_center_sec = segment_center_sec.tolist()
        if isinstance(segment_start_sec, np.ndarray):
            segment_start_sec = segment_start_sec.tolist()
        if isinstance(segment_end_sec, np.ndarray):
            segment_end_sec = segment_end_sec.tolist()
        if isinstance(speaker_ids, np.ndarray):
            speaker_ids = speaker_ids.tolist()
        
        timeline = []
        for i, center_sec in enumerate(segment_center_sec):
            entry = {
                "center_sec": float(center_sec),
                "segment_index": i,
            }
            if segment_start_sec and i < len(segment_start_sec):
                entry["start_sec"] = float(segment_start_sec[i])
            if segment_end_sec and i < len(segment_end_sec):
                entry["end_sec"] = float(segment_end_sec[i])
            if speaker_ids and i < len(speaker_ids):
                entry["speaker_id"] = safe_int(speaker_ids[i] if i < len(speaker_ids) else -1)
            timeline.append(entry)
        render["timeline"] = timeline
    
    # Speaker statistics distribution
    if speaker_ids is not None:
        if isinstance(speaker_ids, np.ndarray):
            speaker_ids = speaker_ids.tolist()
        if speaker_ids:
            unique_speakers = sorted(set(speaker_ids))
            speaker_stats = {}
            for speaker_id in unique_speakers:
                count = speaker_ids.count(speaker_id)
                speaker_stats[str(speaker_id)] = {
                    "segments_count": count,
                    "time_ratio": count / len(speaker_ids) if speaker_ids else 0.0,
                }
            render["summary"]["speaker_stats"] = speaker_stats
            
            # Distribution of speaker segments
            render["distributions"]["speaker_segments"] = {
                "min": safe_int(min(speaker_ids) if speaker_ids else 0),
                "max": safe_int(max(speaker_ids) if speaker_ids else 0),
                "unique_count": len(unique_speakers),
            }
    
    return render


def render_speaker_diarization_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML render для speaker_diarization_extractor (debug mode).
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_speaker_diarization_extractor(npz_data, meta)
    
    # Extract data for visualization
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    speaker_stats = summary.get("speaker_stats", {})
    speaker_time_ratios = summary.get("speaker_time_ratios", {})
    clustering_metrics = summary.get("clustering_metrics", {})
    
    # Generate color palette for speakers
    speaker_colors = {}
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336", "#00BCD4", "#FFEB3B", "#795548"]
    unique_speakers = sorted(set(seg.get("speaker_id", -1) for seg in timeline))
    for i, speaker_id in enumerate(unique_speakers):
        speaker_colors[speaker_id] = colors[i % len(colors)]
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speaker Diarization Extractor Debug View</title>
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
        .speaker-badge {{ display: inline-block; padding: 3px 8px; border-radius: 3px; color: white; font-weight: bold; font-size: 0.85em; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #e3f2fd; padding: 15px; border-radius: 5px; }}
        .metric-label {{ font-size: 0.9em; color: #1976d2; }}
        .metric-value {{ font-size: 1.3em; font-weight: bold; color: #0d47a1; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Speaker Diarization Extractor Debug View</h1>
        <p><strong>Status:</strong> {meta.get('status', 'unknown')}</p>
        <p><strong>Producer:</strong> {meta.get('producer', 'unknown')} v{meta.get('producer_version', 'unknown')}</p>
        <p><strong>Contract Version:</strong> {meta.get('diarization_contract_version', 'unknown')}</p>
        
        <h2>Summary Statistics</h2>
        <div class="summary">
            <div class="stat-card">
                <div class="stat-label">Speaker Count</div>
                <div class="stat-value">{summary.get('speaker_count', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Segments Count</div>
                <div class="stat-value">{summary.get('segments_count', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Duration (sec)</div>
                <div class="stat-value">{summary.get('duration_sec', 0.0):.1f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Speaker Balance Score</div>
                <div class="stat-value">{summary.get('speaker_balance_score', 0.0):.3f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Transitions Count</div>
                <div class="stat-value">{summary.get('speaker_transitions_count', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Segments Density (seg/sec)</div>
                <div class="stat-value">{summary.get('speaker_segments_density', 0.0):.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Dominant Speaker ID</div>
                <div class="stat-value">{summary.get('dominant_speaker_id', -1)}</div>
            </div>
        </div>
        
        <h2>Speaker Statistics</h2>
        <div class="summary">
"""
    
    for speaker_id, stats in speaker_stats.items():
        color = speaker_colors.get(safe_int(speaker_id), "#666")
        time_ratio = speaker_time_ratios.get(str(speaker_id), stats.get("time_ratio", 0.0))
        html_content += f"""
            <div class="stat-card">
                <div class="stat-label">Speaker {speaker_id}</div>
                <div class="stat-value" style="color: {color};">{stats.get('segments_count', 0)} segments</div>
                <div class="stat-label">Time Ratio: {time_ratio:.1%}</div>
            </div>
"""
    
    html_content += """
        </div>
"""
    
    # Clustering metrics
    if clustering_metrics:
        html_content += """
        <h2>Clustering Quality Metrics</h2>
        <div class="metrics">
"""
        for metric_name, metric_value in clustering_metrics.items():
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
        <h2>Timeline (Speaker Segments)</h2>
        <div id="timeline-plot" style="width: 100%; height: 400px;"></div>
        <div class="timeline">
"""
    
    for seg in timeline[:50]:  # Show first 50 segments
        speaker_id = seg.get("speaker_id", -1)
        color = speaker_colors.get(speaker_id, "#666")
        start_sec = seg.get("start_sec", 0.0)
        end_sec = seg.get("end_sec", 0.0)
        duration = seg.get("duration", end_sec - start_sec)
        html_content += f"""
            <div class="segment">
                <div class="segment-header">
                    Segment {seg.get('segment_index', -1)}: 
                    <span class="speaker-badge" style="background-color: {color};">Speaker {speaker_id}</span>
                    [{start_sec:.2f}s - {end_sec:.2f}s] ({duration:.2f}s)
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
    for speaker_id in unique_speakers:
        speaker_segments = [seg for seg in timeline if seg.get("speaker_id") == speaker_id]
        if speaker_segments:
            x_data = []
            y_data = []
            for seg in speaker_segments:
                start_sec = seg.get("start_sec", 0.0)
                end_sec = seg.get("end_sec", 0.0)
                x_data.push([start_sec, end_sec])
                y_data.push(speaker_id)
            
            color = speaker_colors.get(speaker_id, "#666")
            x_data = [seg.get('start_sec', 0.0) for seg in speaker_segments]
            y_data = [speaker_id] * len(speaker_segments)
            html_content += f"""
            {{
                x: {json.dumps(x_data)},
                y: {json.dumps(y_data)},
                mode: 'markers',
                type: 'scatter',
                name: 'Speaker {speaker_id}',
                marker: {{ color: '{color}', size: 10 }},
            }},
"""
    
    html_content += """
        ];
        
        var layout = {
            title: 'Speaker Timeline',
            xaxis: { title: 'Time (seconds)' },
            yaxis: { title: 'Speaker ID' },
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
    
    logger.info(f"Saved speaker_diarization HTML render to {output_path}")
    return output_path

__all__ = ["render_speaker_diarization_extractor", "render_speaker_diarization_extractor_html"]
