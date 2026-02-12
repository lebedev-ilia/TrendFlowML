"""
Renderer для rhythmic_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ...core.renderer import load_npz, extract_meta


def render_rhythmic_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для rhythmic_extractor."""
    render = {
        "component": "rhythmic_extractor",
        "summary": {},
        "basic_metrics": {},
        "interval_stats": {},
        "regularity_metrics": {},
        "tempo_metrics": {},
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

    # Extract payload
    payload = npz_data.get("payload")
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        payload = payload.item() if payload.size == 1 else {}
    if not isinstance(payload, dict):
        payload = {}

    # Summary
    render["summary"] = {
        "sample_rate": int(payload.get("sample_rate", 22050)),
        "hop_length": int(payload.get("hop_length", 512)),
        "backend": str(payload.get("backend", "unknown")),
        "duration": float(payload.get("duration", 0.0)),
        "segments_count": int(payload.get("segments_count", 0)) if "segments_count" in payload else None,
    }

    # Basic metrics (feature-gated)
    if "rhythm_tempo_bpm" in payload:
        render["basic_metrics"] = {
            "tempo_bpm": float(payload.get("rhythm_tempo_bpm", 0.0)),
            "beats_count": int(payload.get("rhythm_beats_count", 0)),
            "beat_density": float(payload.get("rhythm_beat_density", 0.0)),
        }

    # Interval stats (feature-gated)
    if "rhythm_avg_period_sec" in payload:
        render["interval_stats"] = {
            "avg_period_sec": float(payload.get("rhythm_avg_period_sec", 0.0)),
            "period_std_sec": float(payload.get("rhythm_period_std_sec", 0.0)),
            "median_period_sec": float(payload.get("rhythm_median_period_sec", 0.0)),
            "min_period_sec": float(payload.get("rhythm_min_period_sec", 0.0)),
            "max_period_sec": float(payload.get("rhythm_max_period_sec", 0.0)),
        }

    # Regularity metrics (feature-gated)
    if "rhythm_regularity" in payload:
        render["regularity_metrics"] = {
            "regularity": float(payload.get("rhythm_regularity", 0.0)),
        }
        if "rhythm_syncopation_score" in payload:
            render["regularity_metrics"]["syncopation_score"] = float(payload.get("rhythm_syncopation_score", 0.0))
        if "rhythm_polyrhythm_score" in payload:
            render["regularity_metrics"]["polyrhythm_score"] = float(payload.get("rhythm_polyrhythm_score", 0.0))
        if "rhythm_beat_strength_mean" in payload:
            render["regularity_metrics"]["beat_strength_mean"] = float(payload.get("rhythm_beat_strength_mean", 0.0))
        if "rhythm_beat_strength_std" in payload:
            render["regularity_metrics"]["beat_strength_std"] = float(payload.get("rhythm_beat_strength_std", 0.0))
        if "rhythm_metrical_stability" in payload:
            render["regularity_metrics"]["metrical_stability"] = float(payload.get("rhythm_metrical_stability", 0.0))

    # Tempo metrics (feature-gated)
    if "rhythm_median_bpm" in payload:
        render["tempo_metrics"] = {
            "median_bpm": float(payload.get("rhythm_median_bpm", 0.0)),
        }
        if "rhythm_tempo_variation" in payload:
            render["tempo_metrics"]["tempo_variation"] = float(payload.get("rhythm_tempo_variation", 0.0))
        if "rhythm_beat_consistency" in payload:
            render["tempo_metrics"]["beat_consistency"] = float(payload.get("rhythm_beat_consistency", 0.0))
        if "rhythm_tempo_mean" in payload:
            render["tempo_metrics"]["tempo_mean"] = float(payload.get("rhythm_tempo_mean", 0.0))
        if "rhythm_tempo_std" in payload:
            render["tempo_metrics"]["tempo_std"] = float(payload.get("rhythm_tempo_std", 0.0))
        if "rhythm_tempo_min" in payload:
            render["tempo_metrics"]["tempo_min"] = float(payload.get("rhythm_tempo_min", 0.0))
        if "rhythm_tempo_max" in payload:
            render["tempo_metrics"]["tempo_max"] = float(payload.get("rhythm_tempo_max", 0.0))

    # Timeline (beat_times if available)
    beat_times = payload.get("beat_times")
    beat_times_npy = payload.get("beat_times_npy")
    segment_beat_times = payload.get("segment_beat_times")
    segment_centers_sec = payload.get("segment_centers_sec")

    if beat_times is not None:
        if isinstance(beat_times, np.ndarray):
            beat_times = beat_times.tolist()
        render["timeline"] = [
            {"time_sec": float(bt), "beat_index": i} for i, bt in enumerate(beat_times)
        ]
    elif segment_beat_times is not None and segment_centers_sec is not None:
        if isinstance(segment_centers_sec, np.ndarray):
            segment_centers_sec = segment_centers_sec.tolist()
        timeline = []
        for seg_idx, (center_sec, seg_beats) in enumerate(zip(segment_centers_sec, segment_beat_times)):
            if isinstance(seg_beats, list):
                for beat_time in seg_beats:
                    timeline.append({
                        "time_sec": float(beat_time),
                        "segment_index": seg_idx,
                        "center_sec": float(center_sec),
                    })
        render["timeline"] = timeline
    elif beat_times_npy is not None:
        # Beat times saved to .npy file
        render["timeline"] = []  # Would need to load from file for full timeline

    # Distributions
    if render["timeline"]:
        beat_times_list = [t["time_sec"] for t in render["timeline"]]
        if beat_times_list:
            intervals = np.diff(np.array(beat_times_list))
            if intervals.size > 0:
                render["distributions"]["intervals"] = {
                    "min": float(np.min(intervals)),
                    "max": float(np.max(intervals)),
                    "mean": float(np.mean(intervals)),
                    "std": float(np.std(intervals)),
                    "median": float(np.median(intervals)),
                }

    return render


def render_rhythmic_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага rhythmic_extractor результатов.

    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML файла

    Returns:
        Путь к сохраненному HTML файлу
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_rhythmic_extractor(npz_data, meta)

    summary = render.get("summary", {})
    basic_metrics = render.get("basic_metrics", {})
    interval_stats = render.get("interval_stats", {})
    regularity_metrics = render.get("regularity_metrics", {})
    tempo_metrics = render.get("tempo_metrics", {})
    timeline = render.get("timeline", [])

    # Helper function to safely format values
    def safe_str(value, default='N/A'):
        if value is None:
            return default
        try:
            return str(value)
        except:
            return default

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

    # Build HTML sections
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Rhythmic Extractor Debug</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .metric { margin: 10px 0; }
        .metric-label { font-weight: bold; }
        .metric-value { color: #333; }
        .timeline { margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Rhythmic Extractor Debug</h1>

    <div class="section">
        <h2>Summary</h2>
        <div class="metric">
            <span class="metric-label">Sample Rate:</span>
            <span class="metric-value">""" + safe_str(summary.get('sample_rate', 'N/A')) + """ Hz</span>
        </div>
        <div class="metric">
            <span class="metric-label">Hop Length:</span>
            <span class="metric-value">""" + safe_str(summary.get('hop_length', 'N/A')) + """</span>
        </div>
        <div class="metric">
            <span class="metric-label">Backend:</span>
            <span class="metric-value">""" + safe_str(summary.get('backend', 'N/A')) + """</span>
        </div>
        <div class="metric">
            <span class="metric-label">Duration:</span>
            <span class="metric-value">""" + f"{safe_float(summary.get('duration', 0.0)):.2f}" + """ sec</span>
        </div>"""
    
    if summary.get('segments_count') is not None:
        html_content += """
        <div class="metric">
            <span class="metric-label">Segments Count:</span>
            <span class="metric-value">""" + safe_str(summary.get('segments_count', 'N/A')) + """</span>
        </div>"""
    
    html_content += """
    </div>"""

    # Basic metrics section
    if basic_metrics:
        html_content += """
    <div class="section">
        <h2>Basic Metrics</h2>
        <div class="metric">
            <span class="metric-label">Tempo (BPM):</span>
            <span class="metric-value">""" + safe_str(basic_metrics.get('tempo_bpm', 'N/A')) + """</span>
        </div>
        <div class="metric">
            <span class="metric-label">Beats Count:</span>
            <span class="metric-value">""" + safe_str(basic_metrics.get('beats_count', 'N/A')) + """</span>
        </div>
        <div class="metric">
            <span class="metric-label">Beat Density (beats/sec):</span>
            <span class="metric-value">""" + safe_str(basic_metrics.get('beat_density', 'N/A')) + """</span>
        </div>
    </div>"""

    # Interval stats section
    if interval_stats:
        html_content += """
    <div class="section">
        <h2>Interval Statistics</h2>
        <div class="metric">
            <span class="metric-label">Avg Period (sec):</span>
            <span class="metric-value">""" + safe_str(interval_stats.get('avg_period_sec', 'N/A')) + """</span>
        </div>
        <div class="metric">
            <span class="metric-label">Period Std (sec):</span>
            <span class="metric-value">""" + safe_str(interval_stats.get('period_std_sec', 'N/A')) + """</span>
        </div>
        <div class="metric">
            <span class="metric-label">Median Period (sec):</span>
            <span class="metric-value">""" + safe_str(interval_stats.get('median_period_sec', 'N/A')) + """</span>
        </div>
        <div class="metric">
            <span class="metric-label">Min Period (sec):</span>
            <span class="metric-value">""" + safe_str(interval_stats.get('min_period_sec', 'N/A')) + """</span>
        </div>
        <div class="metric">
            <span class="metric-label">Max Period (sec):</span>
            <span class="metric-value">""" + safe_str(interval_stats.get('max_period_sec', 'N/A')) + """</span>
        </div>
    </div>"""

    # Regularity metrics section
    if regularity_metrics:
        html_content += """
    <div class="section">
        <h2>Regularity Metrics</h2>
        <div class="metric">
            <span class="metric-label">Regularity:</span>
            <span class="metric-value">""" + safe_str(regularity_metrics.get('regularity', 'N/A')) + """</span>
        </div>"""
        if 'syncopation_score' in regularity_metrics:
            html_content += """
        <div class="metric">
            <span class="metric-label">Syncopation Score:</span>
            <span class="metric-value">""" + safe_str(regularity_metrics.get('syncopation_score', 'N/A')) + """</span>
        </div>"""
        if 'polyrhythm_score' in regularity_metrics:
            html_content += """
        <div class="metric">
            <span class="metric-label">Polyrhythm Score:</span>
            <span class="metric-value">""" + safe_str(regularity_metrics.get('polyrhythm_score', 'N/A')) + """</span>
        </div>"""
        if 'beat_strength_mean' in regularity_metrics:
            html_content += """
        <div class="metric">
            <span class="metric-label">Beat Strength Mean:</span>
            <span class="metric-value">""" + safe_str(regularity_metrics.get('beat_strength_mean', 'N/A')) + """</span>
        </div>"""
        if 'beat_strength_std' in regularity_metrics:
            html_content += """
        <div class="metric">
            <span class="metric-label">Beat Strength Std:</span>
            <span class="metric-value">""" + safe_str(regularity_metrics.get('beat_strength_std', 'N/A')) + """</span>
        </div>"""
        if 'metrical_stability' in regularity_metrics:
            html_content += """
        <div class="metric">
            <span class="metric-label">Metrical Stability:</span>
            <span class="metric-value">""" + safe_str(regularity_metrics.get('metrical_stability', 'N/A')) + """</span>
        </div>"""
        html_content += """
    </div>"""

    # Tempo metrics section
    if tempo_metrics:
        html_content += """
    <div class="section">
        <h2>Tempo Metrics</h2>
        <div class="metric">
            <span class="metric-label">Median BPM:</span>
            <span class="metric-value">""" + safe_str(tempo_metrics.get('median_bpm', 'N/A')) + """</span>
        </div>"""
        if 'tempo_variation' in tempo_metrics:
            html_content += """
        <div class="metric">
            <span class="metric-label">Tempo Variation:</span>
            <span class="metric-value">""" + safe_str(tempo_metrics.get('tempo_variation', 'N/A')) + """</span>
        </div>"""
        if 'beat_consistency' in tempo_metrics:
            html_content += """
        <div class="metric">
            <span class="metric-label">Beat Consistency:</span>
            <span class="metric-value">""" + safe_str(tempo_metrics.get('beat_consistency', 'N/A')) + """</span>
        </div>"""
        if 'tempo_mean' in tempo_metrics:
            html_content += """
        <div class="metric">
            <span class="metric-label">Tempo Mean:</span>
            <span class="metric-value">""" + safe_str(tempo_metrics.get('tempo_mean', 'N/A')) + """</span>
        </div>"""
        if 'tempo_std' in tempo_metrics:
            html_content += """
        <div class="metric">
            <span class="metric-label">Tempo Std:</span>
            <span class="metric-value">""" + safe_str(tempo_metrics.get('tempo_std', 'N/A')) + """</span>
        </div>"""
        if 'tempo_min' in tempo_metrics:
            html_content += """
        <div class="metric">
            <span class="metric-label">Tempo Min:</span>
            <span class="metric-value">""" + safe_str(tempo_metrics.get('tempo_min', 'N/A')) + """</span>
        </div>"""
        if 'tempo_max' in tempo_metrics:
            html_content += """
        <div class="metric">
            <span class="metric-label">Tempo Max:</span>
            <span class="metric-value">""" + safe_str(tempo_metrics.get('tempo_max', 'N/A')) + """</span>
        </div>"""
        html_content += """
    </div>"""

    # Timeline section
    html_content += """
    <div class="section">
        <h2>Beat Timeline</h2>
        <div id="timeline-plot"></div>
    </div>

    <script>
        var timeline = """ + json.dumps(timeline) + """;
        var duration = """ + str(safe_float(summary.get('duration', 0.0))) + """;

        if (timeline.length > 0) {
            var beatTimes = timeline.map(function(t) { return t.time_sec; });
            var trace = {
                x: beatTimes,
                y: Array(beatTimes.length).fill(1),
                mode: 'markers',
                type: 'scatter',
                marker: { size: 10, color: 'red' },
                name: 'Beats'
            };

            var layout = {
                title: 'Beat Timeline',
                xaxis: { title: 'Time (seconds)', range: [0, duration] },
                yaxis: { title: 'Beat Events', range: [0, 2] },
                height: 400
            };

            Plotly.newPlot('timeline-plot', [trace], layout);
        } else {
            document.getElementById('timeline-plot').innerHTML = '<p>Beat timeline not available (feature not enabled or saved to .npy file)</p>';
        }
    </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


__all__ = ["render_rhythmic_extractor", "render_rhythmic_extractor_html"]

