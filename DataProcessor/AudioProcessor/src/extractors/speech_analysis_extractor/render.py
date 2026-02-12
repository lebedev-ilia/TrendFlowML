"""
Renderer для speech_analysis_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ...core.renderer import load_npz, extract_meta

def render_speech_analysis_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для speech_analysis_extractor."""
    
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
    
    render = {
        "component": "speech_analysis_extractor",
        "summary": {},
        # Initialize as empty dicts - will be populated if features are enabled
        "asr_metrics": {},
        "diarization_metrics": {},
        "pitch_metrics": {},
    }
    
    # Extract features_enabled from meta (source of truth)
    # features_enabled is stored in meta.extra.features_enabled
    features_enabled = []
    
    # Try to get from meta.extra first (most common case)
    extra = meta.get("extra", {})
    if isinstance(extra, dict):
        features_enabled = extra.get("features_enabled", [])
    elif isinstance(extra, np.ndarray) and extra.dtype == object:
        if extra.size == 1:
            extra_dict = extra.item()
            if isinstance(extra_dict, dict):
                features_enabled = extra_dict.get("features_enabled", [])
        elif extra.size > 0:
            # Multiple entries - try first one
            extra_dict = extra[0] if hasattr(extra, '__getitem__') else {}
            if isinstance(extra_dict, dict):
                features_enabled = extra_dict.get("features_enabled", [])
    
    # Fallback: try top level
    if not features_enabled:
        features_enabled = meta.get("features_enabled", [])
    
    # Normalize to list
    if isinstance(features_enabled, np.ndarray):
        features_enabled = features_enabled.tolist()
    if not isinstance(features_enabled, list):
        features_enabled = []
    
    # Extract scalar features (must be before debug logging)
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
    
    # Debug logging (use info level for visibility)
    logger.info(f"speech_analysis render | features_enabled: {features_enabled}, type: {type(features_enabled)}")
    logger.info(f"speech_analysis render | feature_names count: {len(feature_names)}, sample: {feature_names[:10] if len(feature_names) > 10 else feature_names}")
    logger.info(f"speech_analysis render | features dict keys: {list(features.keys())[:20] if len(features) > 20 else list(features.keys())}")
    logger.info(f"speech_analysis render | has asr_segments_count: {'asr_segments_count' in features}, value: {features.get('asr_segments_count', 'NOT_FOUND')}")
    
    # Summary (always present)
    render["summary"] = {
        "duration_sec": safe_float(features.get("duration_sec", 0.0)),
        "sample_rate": safe_int(features.get("sample_rate", 16000)),
    }
    
    # ASR metrics (feature-gated) - проверяем и по features_enabled, и по наличию полей
    # Всегда создаем словарь, если feature включен, даже если значения 0
    should_render_asr = "asr_metrics" in features_enabled or "asr_segments_count" in features
    
    logger.info(f"speech_analysis render | should_render_asr: {should_render_asr} (asr_metrics in enabled: {'asr_metrics' in features_enabled}, asr_segments_count in features: {'asr_segments_count' in features})")
    
    if should_render_asr:
        asr_metrics_dict = {
            "segments_count": safe_int(features.get("asr_segments_count", 0)),
            "token_total": safe_float(features.get("asr_token_total", 0.0)),
            "token_mean": safe_float(features.get("asr_token_mean", 0.0)),
            "token_std": safe_float(features.get("asr_token_std", 0.0)),
            "token_density_per_sec": safe_float(features.get("asr_token_density_per_sec", 0.0)),
            "speech_rate_wpm": safe_float(features.get("asr_speech_rate_wpm", 0.0)),
        }
        
        # Language distribution
        asr_lang_distribution = npz_data.get("asr_lang_distribution")
        if asr_lang_distribution is not None:
            if isinstance(asr_lang_distribution, np.ndarray) and asr_lang_distribution.dtype == object:
                asr_lang_distribution = asr_lang_distribution.item() if asr_lang_distribution.size == 1 else {}
            if isinstance(asr_lang_distribution, dict) and asr_lang_distribution:
                asr_metrics_dict["lang_distribution"] = {str(k): safe_float(v) for k, v in asr_lang_distribution.items()}
        
        # Language IDs by segment
        asr_lang_id_by_segment = npz_data.get("asr_lang_id_by_segment")
        if asr_lang_id_by_segment is not None:
            if isinstance(asr_lang_id_by_segment, np.ndarray):
                asr_lang_id_by_segment = asr_lang_id_by_segment.tolist()
            if asr_lang_id_by_segment:
                asr_metrics_dict["lang_id_by_segment"] = asr_lang_id_by_segment
        
        # Всегда сохраняем метрики, если feature включен или поля есть
        render["asr_metrics"] = asr_metrics_dict
        logger.info(f"speech_analysis render | Created asr_metrics with {len(asr_metrics_dict)} keys: {list(asr_metrics_dict.keys())}")
    else:
        logger.warning(f"speech_analysis render | NOT creating asr_metrics: features_enabled={features_enabled}, has asr_segments_count={'asr_segments_count' in features}")
    
    # Diarization metrics (feature-gated) - проверяем и по features_enabled, и по наличию полей
    if "diarization_metrics" in features_enabled or "speaker_count" in features:
        render["diarization_metrics"] = {
            "segments_count": safe_int(features.get("diar_segments_count", 0)),
            "speaker_count": safe_int(features.get("speaker_count", 0)),
            "dominant_speaker_share": safe_float(features.get("dominant_speaker_share", 0.0)),
            "speaker_balance_score": safe_float(features.get("speaker_balance_score", 0.0)),
            "speaker_transitions_count": safe_int(features.get("speaker_transitions_count", 0)),
        }
        
        # Speaker IDs
        speaker_ids = npz_data.get("speaker_ids")
        if speaker_ids is not None:
            if isinstance(speaker_ids, np.ndarray):
                speaker_ids = speaker_ids.tolist()
            render["diarization_metrics"]["speaker_ids"] = speaker_ids
    
    # Pitch metrics (feature-gated)
    if "pitch_enabled" in features and features.get("pitch_enabled"):
        render["pitch_metrics"] = {
            "enabled": True,
            "f0_mean": safe_float(features.get("pitch_f0_mean", 0.0)),
            "f0_std": safe_float(features.get("pitch_f0_std", 0.0)),
            "f0_min": safe_float(features.get("pitch_f0_min", 0.0)),
            "f0_max": safe_float(features.get("pitch_f0_max", 0.0)),
            "f0_range": safe_float(features.get("pitch_f0_range", 0.0)),
            "stability": safe_float(features.get("pitch_stability", 0.0)),
        }
        
        # Pitch distribution
        pitch_distribution = npz_data.get("pitch_distribution")
        if pitch_distribution is not None:
            if isinstance(pitch_distribution, np.ndarray) and pitch_distribution.dtype == object:
                pitch_distribution = pitch_distribution.item() if pitch_distribution.size == 1 else {}
            if isinstance(pitch_distribution, dict):
                render["pitch_metrics"]["distribution"] = {str(k): safe_float(v) for k, v in pitch_distribution.items()}
    else:
        render["pitch_metrics"] = {"enabled": False}
    
    return render


def render_speech_analysis_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML render для speech_analysis_extractor (debug mode).
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_speech_analysis_extractor(npz_data, meta)
    
    # Extract data for visualization
    summary = render.get("summary", {})
    asr_metrics = render.get("asr_metrics", {})
    diarization_metrics = render.get("diarization_metrics", {})
    pitch_metrics = render.get("pitch_metrics", {})
    
    # Debug logging
    logger.info(f"speech_analysis HTML render | summary: {summary}")
    logger.info(f"speech_analysis HTML render | asr_metrics keys: {list(asr_metrics.keys()) if asr_metrics else 'empty'}")
    logger.info(f"speech_analysis HTML render | diarization_metrics keys: {list(diarization_metrics.keys()) if diarization_metrics else 'empty'}")
    logger.info(f"speech_analysis HTML render | pitch_metrics: {pitch_metrics}")
    
    # Prepare data for charts
    lang_distribution_data = asr_metrics.get("lang_distribution", {}) if asr_metrics else {}
    lang_id_by_segment = asr_metrics.get("lang_id_by_segment", []) if asr_metrics else []
    speaker_ids = diarization_metrics.get("speaker_ids", []) if diarization_metrics else []
    pitch_distribution_data = pitch_metrics.get("distribution", {}) if pitch_metrics.get("enabled") else {}
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speech Analysis Extractor Debug View</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }}
        h1 {{ color: #333; margin-bottom: 10px; font-size: 2em; }}
        h2 {{ color: #555; margin-top: 30px; margin-bottom: 15px; font-size: 1.5em; border-bottom: 2px solid #667eea; padding-bottom: 5px; }}
        h3 {{ color: #666; margin-top: 20px; margin-bottom: 10px; font-size: 1.2em; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .stat-label {{ font-size: 0.9em; opacity: 0.9; margin-bottom: 5px; }}
        .stat-value {{ font-size: 2em; font-weight: bold; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .metric-label {{ font-size: 0.9em; opacity: 0.9; margin-bottom: 5px; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; }}
        .section {{ margin: 30px 0; padding: 25px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #667eea; }}
        .chart-container {{ margin: 20px 0; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .meta-info {{ background: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .meta-info p {{ margin: 5px 0; color: #555; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Speech Analysis Extractor</h1>
        <div class="meta-info">
            <p><strong>Status:</strong> <span style="color: {'green' if meta.get('status') == 'ok' else 'orange' if meta.get('status') == 'empty' else 'red'}">{meta.get('status', 'unknown')}</span></p>
            <p><strong>Producer:</strong> {meta.get('producer', 'unknown')} v{meta.get('producer_version', 'unknown')}</p>
            <p><strong>Contract Version:</strong> {meta.get('speech_analysis_contract_version', 'unknown')}</p>
        </div>
        
        <h2>📊 Summary</h2>
        <div class="summary">
            <div class="stat-card">
                <div class="stat-label">Duration</div>
                <div class="stat-value">{summary.get('duration_sec', 0.0):.2f}s</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Sample Rate</div>
                <div class="stat-value">{summary.get('sample_rate', 16000)} Hz</div>
            </div>
        </div>
"""
    
    # ASR metrics
    if asr_metrics and len(asr_metrics) > 0:
        html_content += """
        <div class="section">
            <h2>🎙️ ASR Metrics</h2>
            <div class="metrics">
"""
        for key, value in asr_metrics.items():
            if key not in ["lang_distribution", "lang_id_by_segment"]:
                # Format value based on type
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                else:
                    formatted_value = str(value)
                html_content += f"""
                <div class="metric-card">
                    <div class="metric-label">{key.replace('_', ' ').title()}</div>
                    <div class="metric-value">{formatted_value}</div>
                </div>
"""
        html_content += """
            </div>
"""
        
        # Language Distribution Pie Chart
        if lang_distribution_data:
            lang_labels = list(lang_distribution_data.keys())
            lang_values = list(lang_distribution_data.values())
            html_content += f"""
            <h3>Language Distribution</h3>
            <div class="chart-container">
                <div id="lang-pie-chart" style="height: 400px;"></div>
            </div>
            <script>
                var langData = [{{
                    labels: {json.dumps(lang_labels)},
                    values: {json.dumps(lang_values)},
                    type: 'pie',
                    hole: 0.4,
                    marker: {{
                        colors: ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7']
                    }},
                    textinfo: 'label+percent',
                    textposition: 'outside'
                }}];
                var langLayout = {{
                    title: {{
                        text: 'Language Distribution',
                        font: {{ size: 18 }}
                    }},
                    showlegend: true,
                    height: 400
                }};
                Plotly.newPlot('lang-pie-chart', langData, langLayout);
            </script>
"""
        
        # Language Timeline
        if lang_id_by_segment and len(lang_id_by_segment) > 0:
            # Create timeline data
            segment_indices = list(range(len(lang_id_by_segment)))
            unique_langs = sorted(set(lang_id_by_segment))
            lang_colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
            lang_color_map = {lang: lang_colors[i % len(lang_colors)] for i, lang in enumerate(unique_langs)}
            
            html_content += f"""
            <h3>Language Timeline</h3>
            <div class="chart-container">
                <div id="lang-timeline-chart" style="height: 300px;"></div>
            </div>
            <script>
                var langTimelineData = [];
                var uniqueLangs = {json.dumps(unique_langs)};
                var langColorMap = {json.dumps(lang_color_map)};
                var segmentIndices = {json.dumps(segment_indices)};
                var langIds = {json.dumps(lang_id_by_segment)};
                
                uniqueLangs.forEach(function(lang) {{
                    var yValues = langIds.map(function(id, idx) {{
                        return id === lang ? lang : null;
                    }});
                    langTimelineData.push({{
                        x: segmentIndices,
                        y: yValues,
                        type: 'scatter',
                        mode: 'markers',
                        name: 'Lang ' + lang,
                        marker: {{
                            color: langColorMap[lang],
                            size: 8
                        }}
                    }});
                }});
                
                var langTimelineLayout = {{
                    title: {{
                        text: 'Language ID by Segment',
                        font: {{ size: 16 }}
                    }},
                    xaxis: {{
                        title: 'Segment Index'
                    }},
                    yaxis: {{
                        title: 'Language ID',
                        tickmode: 'linear',
                        tickvals: uniqueLangs
                    }},
                    height: 300,
                    showlegend: true
                }};
                Plotly.newPlot('lang-timeline-chart', langTimelineData, langTimelineLayout);
            </script>
"""
        
        html_content += """
        </div>
"""
    
    # Diarization metrics
    if diarization_metrics and len(diarization_metrics) > 0:
        html_content += """
        <div class="section">
            <h2>👥 Diarization Metrics</h2>
            <div class="metrics">
"""
        for key, value in diarization_metrics.items():
            if key not in ["speaker_ids"]:
                # Format value based on type
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                else:
                    formatted_value = str(value)
                html_content += f"""
                <div class="metric-card">
                    <div class="metric-label">{key.replace('_', ' ').title()}</div>
                    <div class="metric-value">{formatted_value}</div>
                </div>
"""
        html_content += """
            </div>
"""
        
        # Speaker Distribution
        if speaker_ids and len(speaker_ids) > 0:
            # Count speaker occurrences
            speaker_counts = {}
            for speaker_id in speaker_ids:
                speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
            
            speaker_labels = [f"Speaker {sid}" for sid in sorted(speaker_counts.keys())]
            speaker_values = [speaker_counts[sid] for sid in sorted(speaker_counts.keys())]
            
            html_content += f"""
            <h3>Speaker Distribution</h3>
            <div class="chart-container">
                <div id="speaker-bar-chart" style="height: 400px;"></div>
            </div>
            <script>
                var speakerData = [{{
                    x: {json.dumps(speaker_labels)},
                    y: {json.dumps(speaker_values)},
                    type: 'bar',
                    marker: {{
                        color: ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'],
                        line: {{
                            color: 'rgb(0,0,0)',
                            width: 1
                        }}
                    }},
                    text: {json.dumps(speaker_values)},
                    textposition: 'auto'
                }}];
                var speakerLayout = {{
                    title: {{
                        text: 'Speaker Distribution',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{
                        title: 'Speaker'
                    }},
                    yaxis: {{
                        title: 'Occurrences'
                    }},
                    height: 400
                }};
                Plotly.newPlot('speaker-bar-chart', speakerData, speakerLayout);
            </script>
            <p><strong>Speaker IDs:</strong> {', '.join(map(str, speaker_ids))}</p>
"""
        html_content += """
        </div>
"""
    
    # Pitch metrics
    if pitch_metrics.get("enabled"):
        html_content += """
        <div class="section">
            <h2>🎵 Pitch Metrics</h2>
            <div class="metrics">
"""
        for key, value in pitch_metrics.items():
            if key not in ["enabled", "distribution"]:
                html_content += f"""
                <div class="metric-card">
                    <div class="metric-label">{key.replace('_', ' ').title()}</div>
                    <div class="metric-value">{value:.2f if isinstance(value, float) else value}</div>
                </div>
"""
        html_content += """
            </div>
"""
        
        # Pitch Distribution Bar Chart
        if pitch_distribution_data:
            octave_labels = list(pitch_distribution_data.keys())
            octave_values = [float(v) for v in pitch_distribution_data.values()]
            octave_text = [f"{v:.1%}" for v in octave_values]
            
            html_content += f"""
            <h3>Pitch Distribution by Octave</h3>
            <div class="chart-container">
                <div id="pitch-bar-chart" style="height: 400px;"></div>
            </div>
            <script>
                var pitchData = [{{
                    x: {json.dumps(octave_labels)},
                    y: {json.dumps(octave_values)},
                    type: 'bar',
                    marker: {{
                        color: 'rgb(102, 126, 234)',
                        line: {{
                            color: 'rgb(0,0,0)',
                            width: 1
                        }}
                    }},
                    text: {json.dumps(octave_text)},
                    textposition: 'auto'
                }}];
                var pitchLayout = {{
                    title: {{
                        text: 'Pitch Distribution by Octave',
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{
                        title: 'Octave'
                    }},
                    yaxis: {{
                        title: 'Ratio',
                        tickformat: '.1%'
                    }},
                    height: 400
                }};
                Plotly.newPlot('pitch-bar-chart', pitchData, pitchLayout);
            </script>
"""
        
        # Pitch Statistics Gauge
        if pitch_metrics.get("f0_mean") and pitch_metrics.get("f0_min") and pitch_metrics.get("f0_max"):
            f0_mean = pitch_metrics.get("f0_mean", 0)
            f0_min = pitch_metrics.get("f0_min", 0)
            f0_max = pitch_metrics.get("f0_max", 0)
            f0_range = f0_max - f0_min if f0_max > f0_min else 1
            normalized_mean = (f0_mean - f0_min) / f0_range if f0_range > 0 else 0.5
            
            html_content += f"""
            <h3>Pitch Statistics</h3>
            <div class="chart-container">
                <div id="pitch-gauge-chart" style="height: 300px;"></div>
            </div>
            <script>
                var pitchGaugeData = [{{
                    type: "indicator",
                    mode: "gauge+number",
                    value: {f0_mean:.2f},
                    title: {{ text: "Mean F0 (Hz)" }},
                    gauge: {{
                        axis: {{ range: [{f0_min:.0f}, {f0_max:.0f}] }},
                        bar: {{ color: "rgb(102, 126, 234)" }},
                        steps: [
                            {{ range: [{f0_min:.0f}, {(f0_min + f0_range * 0.33):.0f}], color: "lightgray" }},
                            {{ range: [{(f0_min + f0_range * 0.33):.0f}, {(f0_min + f0_range * 0.67):.0f}], color: "gray" }}
                        ],
                        threshold: {{
                            line: {{ color: "red", width: 4 }},
                            thickness: 0.75,
                            value: {(f0_min + f0_range * 0.8):.0f}
                        }}
                    }}
                }}];
                var pitchGaugeLayout = {{
                    title: {{
                        text: "Pitch Range: {f0_min:.0f} - {f0_max:.0f} Hz",
                        font: {{ size: 16 }}
                    }},
                    height: 300
                }};
                Plotly.newPlot('pitch-gauge-chart', pitchGaugeData, pitchGaugeLayout);
            </script>
"""
        
        html_content += """
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    # Save HTML (atomic write to ensure we overwrite old file)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    import os
    os.replace(tmp_path, output_path)
    
    logger.info(f"Saved speech_analysis HTML render to {output_path} (status={meta.get('status')}, asr_metrics_count={len(asr_metrics) if asr_metrics else 0})")
    return output_path

__all__ = ["render_speech_analysis_extractor", "render_speech_analysis_extractor_html"]
