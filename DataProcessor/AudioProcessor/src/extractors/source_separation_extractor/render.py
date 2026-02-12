"""
Renderer для source_separation_extractor: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

from ...core.renderer import load_npz, extract_meta

def render_source_separation_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для source_separation_extractor."""
    
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
        "component": "source_separation_extractor",
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
    
    # Summary
    segments_count = safe_int(features.get("segments_count", 0))
    render["summary"] = {
        "segments_count": segments_count,
        "sample_rate": safe_int(features.get("sample_rate", 44100)),
    }
    
    # Source order
    source_order = npz_data.get("source_order")
    if source_order is not None:
        if isinstance(source_order, np.ndarray) and source_order.dtype == object:
            source_order = source_order.item() if source_order.size == 1 else source_order.tolist()
        if isinstance(source_order, list):
            render["summary"]["source_order"] = source_order
    
    # Feature-gated aggregates
    if "dominant_source_id" in features:
        render["summary"]["dominant_source_id"] = safe_int(features["dominant_source_id"])
    if "dominant_source_share" in features:
        render["summary"]["dominant_source_share"] = safe_float(features["dominant_source_share"])
    if "source_balance_score" in features:
        render["summary"]["source_balance_score"] = safe_float(features["source_balance_score"])
    if "source_transitions_count" in features:
        render["summary"]["source_transitions_count"] = safe_int(features["source_transitions_count"])
    if "source_stability_score" in features:
        render["summary"]["source_stability_score"] = safe_float(features["source_stability_score"])
    
    # Source distribution
    source_distribution = npz_data.get("source_distribution")
    if source_distribution is not None:
        if isinstance(source_distribution, np.ndarray) and source_distribution.dtype == object:
            source_distribution = source_distribution.item() if source_distribution.size == 1 else {}
        if isinstance(source_distribution, dict):
            render["summary"]["source_distribution"] = {str(k): safe_float(v) for k, v in source_distribution.items()}
    
    # Source segments per source
    source_segments_per_source = npz_data.get("source_segments_per_source")
    if source_segments_per_source is not None:
        if isinstance(source_segments_per_source, np.ndarray) and source_segments_per_source.dtype == object:
            source_segments_per_source = source_segments_per_source.item() if source_segments_per_source.size == 1 else {}
        if isinstance(source_segments_per_source, dict):
            render["summary"]["source_segments_per_source"] = {str(k): safe_int(v) for k, v in source_segments_per_source.items()}
    
    # Advanced features (transition, stability, distribution, energy balance, musical heuristics)
    # These are automatically computed if share_sequence is enabled
    # Extract directly from npz_data (not from feature_names/feature_values)
    source_order_list = source_order if isinstance(source_order, list) else ["vocals", "drums", "bass", "other"]
    advanced_features = {}
    
    # Helper function to safely get value from npz_data
    def get_npz_value(key, default=0.0):
        value = npz_data.get(key)
        if value is None:
            return default
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return safe_float(value.item(), default)
            return safe_float(value[0], default) if len(value) > 0 else default
        return safe_float(value, default) if isinstance(value, (int, float, np.number)) else default
    
    # Transition features
    for source_name in source_order_list:
        delta_mean_key = f"{source_name}_delta_mean"
        delta_std_key = f"{source_name}_delta_std"
        delta_max_key = f"{source_name}_delta_max"
        delta_mean = get_npz_value(delta_mean_key)
        if delta_mean != 0.0 or delta_mean_key in npz_data:  # Include even if 0.0 if key exists
            if "transition_features" not in advanced_features:
                advanced_features["transition_features"] = {}
            advanced_features["transition_features"][source_name] = {
                "delta_mean": get_npz_value(delta_mean_key, 0.0),
                "delta_std": get_npz_value(delta_std_key, 0.0),
                "delta_max": get_npz_value(delta_max_key, 0.0),
            }
    
    # Stability features
    for source_name in source_order_list:
        stability_key = f"{source_name}_stability"
        stability = get_npz_value(stability_key)
        if stability != 0.0 or stability_key in npz_data:
            if "stability_features" not in advanced_features:
                advanced_features["stability_features"] = {}
            advanced_features["stability_features"][source_name] = get_npz_value(stability_key, 0.0)
    
    # Distribution features
    for source_name in source_order_list:
        mean_share_key = f"{source_name}_mean_share"
        dominance_ratio_key = f"{source_name}_dominance_ratio"
        mean_share = get_npz_value(mean_share_key)
        dominance_ratio = get_npz_value(dominance_ratio_key)
        if mean_share != 0.0 or dominance_ratio != 0.0 or mean_share_key in npz_data or dominance_ratio_key in npz_data:
            if "distribution_features" not in advanced_features:
                advanced_features["distribution_features"] = {}
            advanced_features["distribution_features"][source_name] = {
                "mean_share": get_npz_value(mean_share_key, 0.0),
                "dominance_ratio": get_npz_value(dominance_ratio_key, 0.0),
            }
    
    # Energy balance
    if "source_entropy_mean" in npz_data or "energy_balance_mean" in npz_data:
        advanced_features["energy_balance"] = {
            "source_entropy_mean": get_npz_value("source_entropy_mean", 0.0),
            "source_entropy_std": get_npz_value("source_entropy_std", 0.0),
            "energy_balance_mean": get_npz_value("energy_balance_mean", 0.0),
        }
    
    # Musical heuristics
    if "vocals_presence_ratio" in npz_data or "drums_flux" in npz_data or "bass_floor_p20" in npz_data:
        advanced_features["musical_heuristics"] = {
            "vocals_presence_ratio": get_npz_value("vocals_presence_ratio", 0.0),
            "drums_flux": get_npz_value("drums_flux", 0.0),
            "bass_floor_p20": get_npz_value("bass_floor_p20", 0.0),
        }
    
    if advanced_features:
        render["summary"]["advanced_features"] = advanced_features
    
    # Quality metrics
    source_quality_metrics = npz_data.get("source_quality_metrics")
    if source_quality_metrics is not None:
        if isinstance(source_quality_metrics, np.ndarray) and source_quality_metrics.dtype == object:
            source_quality_metrics = source_quality_metrics.item() if source_quality_metrics.size == 1 else {}
        if isinstance(source_quality_metrics, dict):
            render["summary"]["source_quality_metrics"] = {
                k: safe_float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in source_quality_metrics.items()
            }
    
    # Timeline (source segments)
    segment_start_sec = npz_data.get("segment_start_sec")
    segment_end_sec = npz_data.get("segment_end_sec")
    segment_center_sec = npz_data.get("segment_center_sec")
    share_sequence = npz_data.get("share_sequence")
    
    if segment_center_sec is not None:
        if isinstance(segment_center_sec, np.ndarray):
            segment_center_sec = segment_center_sec.tolist()
        if isinstance(segment_start_sec, np.ndarray):
            segment_start_sec = segment_start_sec.tolist()
        if isinstance(segment_end_sec, np.ndarray):
            segment_end_sec = segment_end_sec.tolist()
        
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
            if share_sequence is not None:
                if isinstance(share_sequence, np.ndarray):
                    share_sequence = share_sequence.tolist()
                if share_sequence and i < len(share_sequence):
                    shares = share_sequence[i]
                    if isinstance(shares, list) and len(shares) >= 4:
                        entry["share_vocals"] = safe_float(shares[0])
                        entry["share_drums"] = safe_float(shares[1])
                        entry["share_bass"] = safe_float(shares[2])
                        entry["share_other"] = safe_float(shares[3])
                        # Dominant source
                        dominant_idx = safe_int(np.argmax(shares))
                        entry["dominant_source_id"] = dominant_idx
                        entry["dominant_source_share"] = safe_float(shares[dominant_idx])
            timeline.append(entry)
        render["timeline"] = timeline
    
    # Distribution of source shares
    share_mean = npz_data.get("share_mean")
    if share_mean is not None:
        if isinstance(share_mean, np.ndarray):
            share_mean = share_mean.tolist()
        if share_mean and len(share_mean) >= 4:
            render["distributions"]["share_mean"] = {
                "vocals": safe_float(share_mean[0]),
                "drums": safe_float(share_mean[1]),
                "bass": safe_float(share_mean[2]),
                "other": safe_float(share_mean[3]),
            }
    
    share_std = npz_data.get("share_std")
    if share_std is not None:
        if isinstance(share_std, np.ndarray):
            share_std = share_std.tolist()
        if share_std and len(share_std) >= 4:
            render["distributions"]["share_std"] = {
                "vocals": safe_float(share_std[0]),
                "drums": safe_float(share_std[1]),
                "bass": safe_float(share_std[2]),
                "other": safe_float(share_std[3]),
            }
    
    return render


def render_source_separation_extractor_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML render для source_separation_extractor (debug mode).
    
    Args:
        npz_path: Путь к NPZ файлу
        output_path: Путь для сохранения HTML
    
    Returns:
        Путь к сохранённому HTML файлу
    """
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_source_separation_extractor(npz_data, meta)
    
    # Extract data for visualization
    timeline = render.get("timeline", [])
    summary = render.get("summary", {})
    source_order = summary.get("source_order", ["vocals", "drums", "bass", "other"])
    source_distribution = summary.get("source_distribution", {})
    quality_metrics = summary.get("source_quality_metrics", {})
    advanced_features = summary.get("advanced_features", {})
    
    # Generate color palette for sources
    source_colors = {
        0: "#E91E63",  # vocals - pink
        1: "#FF9800",  # drums - orange
        2: "#2196F3",  # bass - blue
        3: "#4CAF50",  # other - green
    }
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Source Separation Extractor Debug View</title>
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
        .source-badge {{ display: inline-block; padding: 3px 8px; border-radius: 3px; color: white; font-weight: bold; font-size: 0.85em; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #e3f2fd; padding: 15px; border-radius: 5px; }}
        .metric-label {{ font-size: 0.9em; color: #1976d2; }}
        .metric-value {{ font-size: 1.3em; font-weight: bold; color: #0d47a1; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Source Separation Extractor Debug View</h1>
        <p><strong>Status:</strong> {meta.get('status', 'unknown')}</p>
        <p><strong>Producer:</strong> {meta.get('producer', 'unknown')} v{meta.get('producer_version', 'unknown')}</p>
        <p><strong>Contract Version:</strong> {meta.get('source_separation_contract_version', 'unknown')}</p>
        
        <h2>Summary Statistics</h2>
        <div class="summary">
            <div class="stat-card">
                <div class="stat-label">Segments Count</div>
                <div class="stat-value">{summary.get('segments_count', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Dominant Source ID</div>
                <div class="stat-value">{summary.get('dominant_source_id', -1)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Dominant Source Share</div>
                <div class="stat-value">{summary.get('dominant_source_share', 0.0):.3f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Balance Score</div>
                <div class="stat-value">{summary.get('source_balance_score', 0.0):.3f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Transitions Count</div>
                <div class="stat-value">{summary.get('source_transitions_count', 0)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Stability Score</div>
                <div class="stat-value">{summary.get('source_stability_score', 0.0):.3f}</div>
            </div>
        </div>
        
        <h2>Source Distribution</h2>
        <div class="summary">
"""
    
    for src_id, ratio in source_distribution.items():
        src_id_int = int(src_id)
        color = source_colors.get(src_id_int, "#666")
        src_name = source_order[src_id_int] if src_id_int < len(source_order) else f"source_{src_id_int}"
        html_content += f"""
            <div class="stat-card">
                <div class="stat-label">{src_name}</div>
                <div class="stat-value" style="color: {color};">{ratio:.1%}</div>
            </div>
"""
    
    html_content += """
        </div>
"""
    
    # Advanced features
    if advanced_features:
        # Transition features
        transition_features = advanced_features.get("transition_features", {})
        if transition_features:
            html_content += """
        <h2>Transition Features (Delta)</h2>
        <div class="metrics">
"""
            for source_name, metrics in transition_features.items():
                html_content += f"""
            <div class="metric-card">
                <div class="metric-label">{source_name.title()} Delta Mean</div>
                <div class="metric-value">{metrics.get('delta_mean', 0.0):.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">{source_name.title()} Delta Std</div>
                <div class="metric-value">{metrics.get('delta_std', 0.0):.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">{source_name.title()} Delta Max</div>
                <div class="metric-value">{metrics.get('delta_max', 0.0):.4f}</div>
            </div>
"""
            html_content += """
        </div>
"""
        
        # Stability features
        stability_features = advanced_features.get("stability_features", {})
        if stability_features:
            html_content += """
        <h2>Stability Features</h2>
        <div class="metrics">
"""
            for source_name, stability in stability_features.items():
                html_content += f"""
            <div class="metric-card">
                <div class="metric-label">{source_name.title()} Stability</div>
                <div class="metric-value">{stability:.4f}</div>
            </div>
"""
            html_content += """
        </div>
"""
        
        # Distribution features
        distribution_features = advanced_features.get("distribution_features", {})
        if distribution_features:
            html_content += """
        <h2>Distribution Features</h2>
        <div class="metrics">
"""
            for source_name, metrics in distribution_features.items():
                html_content += f"""
            <div class="metric-card">
                <div class="metric-label">{source_name.title()} Mean Share</div>
                <div class="metric-value">{metrics.get('mean_share', 0.0):.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">{source_name.title()} Dominance Ratio</div>
                <div class="metric-value">{metrics.get('dominance_ratio', 0.0):.4f}</div>
            </div>
"""
            html_content += """
        </div>
"""
        
        # Energy balance
        energy_balance = advanced_features.get("energy_balance", {})
        if energy_balance:
            html_content += """
        <h2>Energy Balance</h2>
        <div class="metrics">
"""
            html_content += f"""
            <div class="metric-card">
                <div class="metric-label">Source Entropy Mean</div>
                <div class="metric-value">{energy_balance.get('source_entropy_mean', 0.0):.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Source Entropy Std</div>
                <div class="metric-value">{energy_balance.get('source_entropy_std', 0.0):.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Energy Balance Mean</div>
                <div class="metric-value">{energy_balance.get('energy_balance_mean', 0.0):.4f}</div>
            </div>
"""
            html_content += """
        </div>
"""
        
        # Musical heuristics
        musical_heuristics = advanced_features.get("musical_heuristics", {})
        if musical_heuristics:
            html_content += """
        <h2>Musical Heuristics</h2>
        <div class="metrics">
"""
            html_content += f"""
            <div class="metric-card">
                <div class="metric-label">Vocals Presence Ratio</div>
                <div class="metric-value">{musical_heuristics.get('vocals_presence_ratio', 0.0):.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Drums Flux</div>
                <div class="metric-value">{musical_heuristics.get('drums_flux', 0.0):.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Bass Floor (P20)</div>
                <div class="metric-value">{musical_heuristics.get('bass_floor_p20', 0.0):.4f}</div>
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
        <h2>Timeline (Source Segments)</h2>
        <div id="timeline-plot" style="width: 100%; height: 400px;"></div>
        <div class="timeline">
"""
    
    for seg in timeline[:50]:  # Show first 50 segments
        src_id = seg.get("dominant_source_id", -1)
        color = source_colors.get(src_id, "#666")
        src_name = source_order[src_id] if src_id >= 0 and src_id < len(source_order) else f"source_{src_id}"
        start_sec = seg.get("start_sec", 0.0)
        end_sec = seg.get("end_sec", 0.0)
        share = seg.get("dominant_source_share", 0.0)
        html_content += f"""
            <div class="segment">
                <div class="segment-header">
                    Segment {seg.get('segment_index', -1)}: 
                    <span class="source-badge" style="background-color: {color};">{src_name}</span>
                    [{start_sec:.2f}s - {end_sec:.2f}s] (share: {share:.3f})
                </div>
"""
        if "share_vocals" in seg:
            html_content += f"""
                <div>Vocals: {seg.get('share_vocals', 0.0):.3f} | Drums: {seg.get('share_drums', 0.0):.3f} | Bass: {seg.get('share_bass', 0.0):.3f} | Other: {seg.get('share_other', 0.0):.3f}</div>
"""
        html_content += """
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
    for src_id in range(4):
        source_segments = [seg for seg in timeline if seg.get("dominant_source_id") == src_id]
        if source_segments:
            color = source_colors.get(src_id, "#666")
            src_name = source_order[src_id] if src_id < len(source_order) else f"source_{src_id}"
            x_data = [seg.get('start_sec', 0.0) for seg in source_segments]
            y_data = [src_id] * len(source_segments)
            html_content += f"""
            {{
                x: {json.dumps(x_data)},
                y: {json.dumps(y_data)},
                mode: 'markers',
                type: 'scatter',
                name: '{src_name}',
                marker: {{ color: '{color}', size: 10 }},
            }},
"""
    
    html_content += """
        ];
        
        var layout = {
            title: 'Source Timeline',
            xaxis: { title: 'Time (seconds)' },
            yaxis: { title: 'Source ID' },
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
    
    logger.info(f"Saved source_separation HTML render to {output_path}")
    return output_path

__all__ = ["render_source_separation_extractor", "render_source_separation_extractor_html"]
