"""
Renderer для key_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ...core.renderer import load_npz, extract_meta


def render_key_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для key_extractor."""
    render = {
        "component": "key_extractor",
        "summary": {},
        "key_info": {},
        "confidence_info": {},
        "top_k_keys": {},
        "time_series": {},
        "key_changes": {},
        "stability_metrics": {},
    }

    features_enabled = meta.get("features_enabled", [])
    logger.info(f"key_renderer | features_enabled from meta: {features_enabled}")
    logger.info(f"key_renderer | meta keys: {list(meta.keys())}")

    # Helper function to safely convert values (handles NaN, None, etc.)
    def safe_value(value, default=None):
        """Safely convert value, handling NaN, None, and type conversions."""
        if value is None:
            return default
        # Handle numpy NaN
        if isinstance(value, (float, np.floating)):
            if np.isnan(value) or np.isinf(value):
                return default
            return value
        # Handle numpy string arrays (may contain 'nan' string)
        if isinstance(value, (str, np.str_)):
            if value == 'nan' or value == 'NaN' or (isinstance(value, str) and value.lower() == 'nan'):
                return default
            return value
        # Handle numpy arrays with single element
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            if value.size == 1:
                val = value.item()
                return safe_value(val, default)
            return value
        return value

    # Extract scalar features from feature_names/feature_values
    feature_names = npz_data.get("feature_names", [])
    feature_values = npz_data.get("feature_values", [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    if isinstance(feature_values, np.ndarray):
        feature_values = feature_values.tolist()

    features = {}
    for i, name in enumerate(feature_names):
        if i < len(feature_values):
            value = feature_values[i]
            # Convert NaN/None to appropriate defaults based on name
            if name in ["key_name", "key_mode", "method", "key_confidence_category", "key_confidence_reason"]:
                # String fields - default to "unknown"
                features[name] = safe_value(value, "unknown")
            elif name in ["key_confidence", "duration", "sample_rate", "hop_length"]:
                # Numeric fields - default to 0.0 or 0
                default_val = 0 if name in ["sample_rate", "hop_length"] else 0.0
                features[name] = safe_value(value, default_val)
            else:
                features[name] = safe_value(value, None)

    # Extract payload (if exists)
    payload = npz_data.get("payload")
    if payload is not None:
        if isinstance(payload, np.ndarray):
            if payload.dtype == object:
                # Object array - extract dict
                if payload.size == 1:
                    payload = payload.item()
                elif payload.size > 1:
                    # Multi-element array - try to extract first element
                    try:
                        payload = payload.item() if payload.ndim == 0 else payload[0].item() if payload.size > 0 else {}
                    except (ValueError, IndexError):
                        payload = {}
                else:
                    payload = {}
            else:
                # Non-object array - not a dict
                payload = {}
        if not isinstance(payload, dict):
            payload = {}
    else:
        payload = {}
    
    # Merge features into payload (payload takes precedence for string values, features for numeric)
    # String values should come from payload (not from feature_values where they become NaN)
    for key in ["key_name", "key_mode", "method", "key_confidence_category", "key_confidence_reason"]:
        if key in payload and payload[key] not in [None, "unknown", "nan", "NaN"]:
            features[key] = payload[key]
    
    # Merge: payload takes precedence for all fields
    payload = {**features, **payload}

    # Helper to safely get values with NaN handling
    def safe_get(key: str, default: Any = None, convert_type: type = None) -> Any:
        """Safely get value from payload, handling NaN."""
        value = payload.get(key, default)
        if value is None:
            return default
        # Handle NaN
        if isinstance(value, (float, np.floating)):
            if np.isnan(value) or np.isinf(value):
                return default
        # Handle string 'nan'
        if isinstance(value, (str, np.str_)):
            if value == 'nan' or value == 'NaN' or (isinstance(value, str) and value.lower() == 'nan'):
                return default
        # Type conversion
        if convert_type:
            try:
                if convert_type == int:
                    return int(float(value)) if not (np.isnan(float(value)) if isinstance(value, (float, np.floating)) else False) else default
                elif convert_type == float:
                    return float(value) if not (np.isnan(value) if isinstance(value, (float, np.floating)) else False) else default
                elif convert_type == str:
                    return str(value) if value != 'nan' else default
                elif convert_type == bool:
                    return bool(value)
            except (ValueError, TypeError):
                return default
        return value

    # Summary
    render["summary"] = {
        "key_name": safe_get("key_name", "unknown", str),
        "key_mode": safe_get("key_mode", "unknown", str),
        "key_confidence": safe_get("key_confidence", 0.0, float),
        "method": safe_get("method", "unknown", str),
        "sample_rate": safe_get("sample_rate", 0, int),
        "hop_length": safe_get("hop_length", 0, int),
        "duration": safe_get("duration", 0.0, float),
    }

    # Key info
    key_name = safe_get("key_name", "unknown", str)
    key_mode = safe_get("key_mode", "unknown", str)
    render["key_info"] = {
        "key": f"{key_name} {key_mode}",
        "confidence": safe_get("key_confidence", 0.0, float),
        "confidence_category": safe_get("key_confidence_category", "unknown", str),
        "low_confidence_warning": safe_get("key_low_confidence_warning", False, bool),
        "confidence_reason": safe_get("key_confidence_reason", "unknown", str),
    }

    # Confidence info (detailed confidence statistics)
    confidence = safe_get("key_confidence", 0.0, float)
    confidence_category = safe_get("key_confidence_category", "unknown", str)
    confidence_reason = safe_get("key_confidence_reason", "unknown", str)
    low_confidence_warning = safe_get("key_low_confidence_warning", False, bool)
    
    # Get confidence statistics from meta if available (from stability_metrics)
    confidence_mean = float(meta.get("key_confidence_mean", 0.0)) if "stability_metrics" in features_enabled else confidence
    confidence_std = float(meta.get("key_confidence_std", 0.0)) if "stability_metrics" in features_enabled else 0.0
    confidence_min = float(meta.get("key_confidence_min", 0.0)) if "stability_metrics" in features_enabled else confidence
    confidence_max = float(meta.get("key_confidence_max", 0.0)) if "stability_metrics" in features_enabled else confidence
    
    render["confidence_info"] = {
        "confidence": confidence,
        "confidence_category": confidence_category,
        "confidence_reason": confidence_reason,
        "low_confidence_warning": low_confidence_warning,
        "confidence_mean": confidence_mean,
        "confidence_std": confidence_std,
        "confidence_min": confidence_min,
        "confidence_max": confidence_max,
        "confidence_range": [confidence_min, confidence_max],
    }

    # Detailed scores if enabled
    if "detailed_scores" in features_enabled:
        # Extract from NPZ array (not from payload)
        key_scores = npz_data.get("key_scores")
        if key_scores is not None:
            if isinstance(key_scores, np.ndarray):
                scores = key_scores.tolist() if key_scores.size > 0 else []
            elif isinstance(key_scores, list):
                scores = key_scores
            else:
                scores = []
            
            if scores:
                render["key_info"]["key_scores"] = scores

        # Find best major and minor
        scores_arr = np.array(scores)
        major_scores = scores_arr[::2]
        minor_scores = scores_arr[1::2]
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        best_major_idx = int(np.argmax(major_scores))
        best_minor_idx = int(np.argmax(minor_scores))
        render["key_info"]["best_major"] = {
            "key": keys[best_major_idx],
            "score": float(major_scores[best_major_idx]),
        }
        render["key_info"]["best_minor"] = {
            "key": keys[best_minor_idx],
            "score": float(minor_scores[best_minor_idx]),
        }

    # Top-K keys if enabled (from meta)
    if "top_k" in features_enabled:
        top_k = meta.get("key_top_k", [])
        if isinstance(top_k, list) and len(top_k) > 0:
            render["top_k_keys"] = {
                "keys": top_k,
                "count": len(top_k),
            }
        else:
            render["top_k_keys"] = {
                "keys": [],
                "count": 0,
            }
    else:
        render["top_k_keys"] = {
            "keys": [],
            "count": 0,
        }

    # Time series if enabled
    logger.info(f"key_renderer | Checking time_series: 'time_series' in features_enabled = {'time_series' in features_enabled}")
    logger.info(f"key_renderer | NPZ data keys: {list(npz_data.keys())}")
    if "time_series" in features_enabled:
        # Extract from NPZ arrays (not from payload)
        segment_centers = npz_data.get("segment_centers_sec")
        key_names_seq = npz_data.get("key_names_sequence")
        key_modes_seq = npz_data.get("key_modes_sequence")
        key_confidences_seq = npz_data.get("key_confidences_sequence")
        segment_durations = npz_data.get("segment_durations")
        
        logger.info(f"key_renderer | time_series data from NPZ:")
        logger.info(f"  segment_centers: {type(segment_centers)}, {segment_centers.shape if hasattr(segment_centers, 'shape') else 'no shape'}")
        logger.info(f"  key_names_sequence: {type(key_names_seq)}, {key_names_seq.shape if hasattr(key_names_seq, 'shape') else 'no shape'}")
        logger.info(f"  key_modes_sequence: {type(key_modes_seq)}, {key_modes_seq.shape if hasattr(key_modes_seq, 'shape') else 'no shape'}")
        logger.info(f"  key_confidences_sequence: {type(key_confidences_seq)}, {key_confidences_seq.shape if hasattr(key_confidences_seq, 'shape') else 'no shape'}")
        logger.info(f"  segment_durations: {type(segment_durations)}, {segment_durations.shape if hasattr(segment_durations, 'shape') else 'no shape'}")

        # Convert to lists
        if segment_centers is not None:
            if isinstance(segment_centers, np.ndarray):
                segment_centers = segment_centers.tolist() if segment_centers.size > 0 else []
            elif isinstance(segment_centers, list):
                segment_centers = segment_centers
            else:
                segment_centers = []
        else:
            segment_centers = []

        if key_names_seq is not None:
            if isinstance(key_names_seq, np.ndarray):
                key_names_seq = key_names_seq.tolist() if key_names_seq.size > 0 else []
            elif isinstance(key_names_seq, list):
                key_names_seq = key_names_seq
            else:
                key_names_seq = []
        else:
            key_names_seq = []

        if key_modes_seq is not None:
            if isinstance(key_modes_seq, np.ndarray):
                key_modes_seq = key_modes_seq.tolist() if key_modes_seq.size > 0 else []
            elif isinstance(key_modes_seq, list):
                key_modes_seq = key_modes_seq
            else:
                key_modes_seq = []
        else:
            key_modes_seq = []

        if key_confidences_seq is not None:
            if isinstance(key_confidences_seq, np.ndarray):
                key_confidences_seq = key_confidences_seq.tolist() if key_confidences_seq.size > 0 else []
            elif isinstance(key_confidences_seq, list):
                key_confidences_seq = key_confidences_seq
            else:
                key_confidences_seq = []
        else:
            key_confidences_seq = []

        if segment_durations is not None:
            if isinstance(segment_durations, np.ndarray):
                segment_durations = segment_durations.tolist() if segment_durations.size > 0 else []
            elif isinstance(segment_durations, list):
                segment_durations = segment_durations
            else:
                segment_durations = []
        else:
            segment_durations = []

        render["time_series"] = {
            "segment_centers_sec": segment_centers,
            "key_names": key_names_seq,
            "key_modes": key_modes_seq,
            "key_confidences": key_confidences_seq,
            "segment_durations": segment_durations,
            "count": len(segment_centers),
        }
    else:
        render["time_series"] = {
            "segment_centers_sec": [],
            "key_names": [],
            "key_modes": [],
            "key_confidences": [],
            "segment_durations": [],
            "count": 0,
        }

    # Key changes if enabled (from meta)
    if "key_changes" in features_enabled:
        transitions = meta.get("key_transitions", [])
        if not isinstance(transitions, list):
            transitions = []
        render["key_changes"] = {
            "transitions": transitions,
            "transitions_count": int(meta.get("key_transitions_count", 0)),
            "transitions_rate": float(meta.get("key_transitions_rate", 0.0)),
        }
    else:
        render["key_changes"] = {
            "transitions": [],
            "transitions_count": 0,
            "transitions_rate": 0.0,
        }

    # Stability metrics if enabled (from meta)
    if "stability_metrics" in features_enabled:
        key_distribution = meta.get("key_distribution", {})
        if not isinstance(key_distribution, dict):
            key_distribution = {}
        render["stability_metrics"] = {
            "key_stability_score": float(meta.get("key_stability_score", 0.0)),
            "key_confidence_mean": float(meta.get("key_confidence_mean", 0.0)),
            "key_confidence_std": float(meta.get("key_confidence_std", 0.0)),
            "key_confidence_min": float(meta.get("key_confidence_min", 0.0)),
            "key_confidence_max": float(meta.get("key_confidence_max", 0.0)),
            "key_distribution": key_distribution,
            "key_diversity": int(meta.get("key_diversity", 0)),
            "key_detection_quality": float(meta.get("key_detection_quality", 0.0)),
        }
    else:
        render["stability_metrics"] = {
            "key_stability_score": 0.0,
            "key_confidence_mean": 0.0,
            "key_confidence_std": 0.0,
            "key_confidence_min": 0.0,
            "key_confidence_max": 0.0,
            "key_distribution": {},
            "key_diversity": 0,
            "key_detection_quality": 0.0,
        }

    return render


def render_key_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага key_extractor результатов.

    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML файла

    Returns:
        Путь к сохраненному HTML файлу
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_key_extractor(npz_data, meta)

    summary = render.get("summary", {})
    key_info = render.get("key_info", {})
    confidence_info = render.get("confidence_info", {})
    top_k_keys = render.get("top_k_keys", {})
    if isinstance(top_k_keys, list):
        # Legacy format - convert to new format
        top_k_keys = {"keys": top_k_keys, "count": len(top_k_keys)}
    time_series = render.get("time_series", {})
    key_changes = render.get("key_changes", {})
    stability_metrics = render.get("stability_metrics", {})

    # Prepare data for visualization
    segment_centers = time_series.get("segment_centers_sec", [])
    key_names = time_series.get("key_names", [])
    key_confidences = time_series.get("key_confidences", [])

    # Key distribution for pie chart
    key_distribution = stability_metrics.get("key_distribution", {})
    distribution_labels = list(key_distribution.keys())
    distribution_values = list(key_distribution.values())

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Key Extractor Debug</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .metric-label {{ font-weight: bold; }}
        .metric-value {{ color: #333; }}
        .warning {{ color: #ff6600; font-weight: bold; }}
        .high-confidence {{ color: #00aa00; }}
        .medium-confidence {{ color: #ffaa00; }}
        .low-confidence {{ color: #ff6600; }}
        .very-low-confidence {{ color: #ff0000; }}
    </style>
</head>
<body>
    <h1>Key Extractor Debug</h1>

    <div class="section">
        <h2>Summary</h2>
        <div class="metric">
            <span class="metric-label">Key:</span>
            <span class="metric-value">{summary.get("key_name", "unknown")} {summary.get("key_mode", "unknown")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Confidence:</span>
            <span class="metric-value {str(key_info.get('confidence_category', 'unknown')).replace('_', '-')}-confidence">{summary.get("key_confidence", 0.0):.3f}</span>
            <span class="metric-value">({key_info.get("confidence_category", "unknown")})</span>
        </div>
        {f'<div class="metric warning">⚠ Low Confidence Warning: {key_info.get("confidence_reason", "unknown")}</div>' if key_info.get("low_confidence_warning") else ''}
        <div class="metric">
            <span class="metric-label">Method:</span>
            <span class="metric-value">{summary.get("method", "unknown")}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Duration:</span>
            <span class="metric-value">{summary.get("duration", 0.0):.2f}s</span>
        </div>
        <div class="metric">
            <span class="metric-label">Sample Rate:</span>
            <span class="metric-value">{summary.get("sample_rate", 0)} Hz</span>
        </div>
    </div>

    {f'''
    <div class="section">
        <h2>Top-K Alternative Keys</h2>
        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>Rank</th>
                <th>Key</th>
                <th>Mode</th>
                <th>Score</th>
            </tr>
            {''.join([f'<tr><td>{i+1}</td><td>{entry.get("key", "unknown")}</td><td>{entry.get("mode", "unknown")}</td><td>{entry.get("score", 0.0):.3f}</td></tr>' for i, entry in enumerate(top_k_keys.get("keys", []))])}
        </table>
    </div>
    ''' if top_k_keys.get("keys") else ''}

    {f'''
    <div class="section">
        <h2>Key Distribution</h2>
        <div id="key-distribution-plot"></div>
    </div>
    ''' if distribution_labels else ''}

    {f'''
    <div class="section">
        <h2>Key Changes Over Time</h2>
        <div class="metric">
            <span class="metric-label">Transitions Count:</span>
            <span class="metric-value">{key_changes.get("transitions_count", 0)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Transitions Rate:</span>
            <span class="metric-value">{key_changes.get("transitions_rate", 0.0):.4f} transitions/sec</span>
        </div>
        <div id="key-timeline-plot"></div>
    </div>
    ''' if segment_centers else ''}

    {f'''
    <div class="section">
        <h2>Confidence Over Time</h2>
        <div id="confidence-timeline-plot"></div>
    </div>
    ''' if key_confidences else ''}

    {f'''
    <div class="section">
        <h2>Stability Metrics</h2>
        <div class="metric">
            <span class="metric-label">Key Stability Score:</span>
            <span class="metric-value">{stability_metrics.get("key_stability_score", 0.0):.3f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Confidence Mean:</span>
            <span class="metric-value">{stability_metrics.get("key_confidence_mean", 0.0):.3f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Confidence Std:</span>
            <span class="metric-value">{stability_metrics.get("key_confidence_std", 0.0):.3f}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Confidence Range:</span>
            <span class="metric-value">[{stability_metrics.get("key_confidence_min", 0.0):.3f}, {stability_metrics.get("key_confidence_max", 0.0):.3f}]</span>
        </div>
        <div class="metric">
            <span class="metric-label">Key Diversity:</span>
            <span class="metric-value">{stability_metrics.get("key_diversity", 0)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Detection Quality:</span>
            <span class="metric-value">{stability_metrics.get("key_detection_quality", 0.0):.3f}</span>
        </div>
    </div>
    ''' if stability_metrics else ''}

    <script>
        var segmentCenters = {json.dumps(segment_centers)};
        var keyNames = {json.dumps(key_names)};
        var keyConfidences = {json.dumps(key_confidences)};
        var distributionLabels = {json.dumps(distribution_labels)};
        var distributionValues = {json.dumps(distribution_values)};
        var keyTransitions = {json.dumps(key_changes.get("transitions", []))};

        // Key distribution pie chart
        {f'''
        if (distributionLabels.length > 0) {{
            var traceDistribution = {{
                labels: distributionLabels,
                values: distributionValues,
                type: 'pie',
                textinfo: 'label+percent',
                textposition: 'outside',
            }};
            var layoutDistribution = {{
                title: 'Key Distribution (Time Proportion)',
                height: 400
            }};
            Plotly.newPlot('key-distribution-plot', [traceDistribution], layoutDistribution);
        }}
        ''' if distribution_labels else ''}

        // Key timeline plot
        {f'''
        if (segmentCenters.length > 0 && keyNames.length > 0) {{
            // Create color map for keys
            var uniqueKeys = [...new Set(keyNames)];
            var colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'];
            var keyColorMap = {{}};
            uniqueKeys.forEach(function(key, i) {{
                keyColorMap[key] = colors[i % colors.length];
            }});

            var traces = uniqueKeys.map(function(key) {{
                var indices = [];
                var times = [];
                for (var i = 0; i < keyNames.length; i++) {{
                    if (keyNames[i] === key) {{
                        indices.push(i);
                        times.push(segmentCenters[i]);
                    }}
                }}
                return {{
                    x: times,
                    y: Array(times.length).fill(key),
                    mode: 'markers',
                    type: 'scatter',
                    name: key,
                    marker: {{
                        color: keyColorMap[key],
                        size: 10
                    }}
                }};
            }});

            var layoutTimeline = {{
                title: 'Key Changes Over Time',
                xaxis: {{ title: 'Time (seconds)' }},
                yaxis: {{ title: 'Key', type: 'category' }},
                height: 400,
                showlegend: true
            }};

            Plotly.newPlot('key-timeline-plot', traces, layoutTimeline);
        }}
        ''' if segment_centers and key_names else ''}

        // Confidence timeline plot
        {f'''
        if (segmentCenters.length > 0 && keyConfidences.length > 0) {{
            var traceConfidence = {{
                x: segmentCenters,
                y: keyConfidences,
                mode: 'lines+markers',
                type: 'scatter',
                name: 'Confidence',
                line: {{ color: 'blue' }},
                marker: {{ size: 5 }}
            }};

            var layoutConfidence = {{
                title: 'Key Detection Confidence Over Time',
                xaxis: {{ title: 'Time (seconds)' }},
                yaxis: {{ title: 'Confidence', range: [0, 1] }},
                height: 300
            }};

            Plotly.newPlot('confidence-timeline-plot', [traceConfidence], layoutConfidence);
        }}
        ''' if segment_centers and key_confidences else ''}
    </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


__all__ = ["render_key_extractor", "render_key_extractor_html"]

