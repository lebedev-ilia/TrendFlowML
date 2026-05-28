"""
Renderer для color_light: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _esc(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _svg_line_chart(
    *,
    times_s: np.ndarray,
    values: np.ndarray,
    title: str,
    stroke: str = "#2563eb",
    width: int = 960,
    height: int = 220,
    pad: int = 18,
) -> str:
    if times_s.size == 0 or values.size == 0:
        return ""

    m = np.isfinite(times_s) & np.isfinite(values)
    if not np.any(m):
        return ""

    x = times_s[m].astype(np.float64, copy=False)
    y = values[m].astype(np.float64, copy=False)
    if x.size < 2:
        return ""

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if xmax <= xmin:
        xmax = xmin + 1e-6
    if ymax <= ymin:
        ymax = ymin + 1e-6

    def sx(v: float) -> float:
        return pad + (v - xmin) / (xmax - xmin) * (width - 2 * pad)

    def sy(v: float) -> float:
        return height - pad - (v - ymin) / (ymax - ymin) * (height - 2 * pad)

    pts = " ".join(f"{sx(float(xx)):.2f},{sy(float(yy)):.2f}" for xx, yy in zip(x, y))
    title_esc = _esc(title)
    return f"""
<div class="chart">
  <div class="chart-title">{title_esc}</div>
  <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="{title_esc}">
    <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#e5e7eb"/>
    <polyline fill="none" stroke="{stroke}" stroke-width="2" points="{pts}"/>
    <text x="{pad}" y="{pad}" font-size="12" fill="#6b7280">{_esc(f'[{ymin:.3f} .. {ymax:.3f}]')}</text>
  </svg>
</div>
""".strip()


def render_color_light(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для color_light."""
    render = {
        "component": "color_light",
        "summary": {},
        "timeline": [],
        "distributions": {},
        "key_facts": [],
        "top_examples": {},
        "anti_top_examples": {},
        "config_highlights": {},
    }
    
    # Extract data
    video_features = npz_data.get("video_features", {})
    frames = npz_data.get("frames", {})
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    sequence_frame_indices = npz_data.get("sequence_frame_indices")
    sequence_times_s = npz_data.get("sequence_times_s")
    frame_compact_features = npz_data.get("frame_compact_features")
    frame_compact_feature_names = npz_data.get("frame_compact_feature_names")
    frame_compact_frame_indices = npz_data.get("frame_compact_frame_indices")
    
    # Convert to numpy arrays if needed
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
    
    if sequence_times_s is not None:
        if isinstance(sequence_times_s, list):
            sequence_times_s = np.array(sequence_times_s, dtype=np.float32)
        elif isinstance(sequence_times_s, np.ndarray):
            sequence_times_s = np.asarray(sequence_times_s, dtype=np.float32)
        else:
            sequence_times_s = None

    if frame_compact_features is not None:
        try:
            frame_compact_features = np.asarray(frame_compact_features, dtype=np.float32)
        except Exception:
            frame_compact_features = None

    if frame_compact_feature_names is not None:
        try:
            frame_compact_feature_names = np.asarray(frame_compact_feature_names, dtype=object)
        except Exception:
            frame_compact_feature_names = None

    if frame_compact_frame_indices is not None:
        try:
            frame_compact_frame_indices = np.asarray(frame_compact_frame_indices, dtype=np.int32)
        except Exception:
            frame_compact_frame_indices = None
    
    # Summary statistics from video_features
    if video_features:
        render["summary"] = {
            "frames_count": int(len(frame_indices)) if frame_indices is not None else 0,
            "scenes_count": int(len(frames)) if isinstance(frames, dict) else 0,
            "color_distribution_entropy": float(video_features.get("color_distribution_entropy", 0.0)) if isinstance(video_features.get("color_distribution_entropy"), (int, float)) else None,
            "color_distribution_gini": float(video_features.get("color_distribution_gini", 0.0)) if isinstance(video_features.get("color_distribution_gini"), (int, float)) else None,
            "global_brightness_change_speed": float(video_features.get("global_brightness_change_speed", 0.0)) if isinstance(video_features.get("global_brightness_change_speed"), (int, float)) else None,
            "global_color_change_speed": float(video_features.get("global_color_change_speed", 0.0)) if isinstance(video_features.get("global_color_change_speed"), (int, float)) else None,
            "strobe_transition_frequency": float(video_features.get("strobe_transition_frequency", 0.0)) if isinstance(video_features.get("strobe_transition_frequency"), (int, float)) else None,
        }
        
        # Style probabilities
        style_probs = {k: float(v) for k, v in video_features.items() if k.startswith("style_") and isinstance(v, (int, float))}
        if style_probs:
            render["summary"]["style_probabilities"] = style_probs
    
    # Timeline data (per-frame features)
    timeline = []
    # Prefer fixed compact arrays (stable and available even when store_debug_objects=0).
    if (
        frame_compact_features is not None
        and frame_compact_feature_names is not None
        and frame_compact_frame_indices is not None
        and sequence_times_s is not None
        and frame_compact_features.ndim == 2
        and frame_compact_feature_names.ndim == 1
        and frame_compact_features.shape[1] == frame_compact_feature_names.size
    ):
        name_to_col = {str(n): int(i) for i, n in enumerate(frame_compact_feature_names.tolist())}
        def _col(name: str) -> np.ndarray:
            j = name_to_col.get(name)
            if j is None:
                return np.full((int(frame_compact_features.shape[0]),), np.nan, dtype=np.float32)
            return frame_compact_features[:, j]

        hue_vals = _col("hue_mean_norm")
        col_vals = _col("colorfulness_norm")
        br_vals = np.full((int(frame_compact_features.shape[0]),), np.nan, dtype=np.float32)
        # brightness_mean_norm is not part of compact vector; best-effort from legacy frames below.
        ct_vals = _col("global_contrast_norm")
        n = min(int(frame_compact_features.shape[0]), int(len(sequence_times_s)))
        for i in range(n):
            timeline.append(
                {
                    "frame_index": int(frame_compact_frame_indices[i]) if i < int(frame_compact_frame_indices.size) else None,
                    "time_sec": float(sequence_times_s[i]),
                    "hue_mean_norm": float(hue_vals[i]) if np.isfinite(hue_vals[i]) else None,
                    "colorfulness_norm": float(col_vals[i]) if np.isfinite(col_vals[i]) else None,
                    "brightness_mean_norm": float(br_vals[i]) if np.isfinite(br_vals[i]) else None,
                    "global_contrast_norm": float(ct_vals[i]) if np.isfinite(ct_vals[i]) else None,
                    "saturation_mean_norm": None,
                    "value_mean_norm": None,
                }
            )
    elif frames and isinstance(frames, dict) and sequence_times_s is not None:
        # Collect frame features from all scenes
        frame_feat_map: Dict[int, Dict[str, Any]] = {}
        for scene_dict in frames.values():
            if isinstance(scene_dict, dict):
                for frame_idx, frame_obj in scene_dict.items():
                    if isinstance(frame_obj, dict):
                        feat = frame_obj.get("features", {})
                        try:
                            frame_feat_map[int(frame_idx)] = feat
                        except Exception:
                            continue
        
        # Build timeline from sequence
        if sequence_frame_indices is not None:
            if isinstance(sequence_frame_indices, list):
                sequence_frame_indices = np.array(sequence_frame_indices, dtype=np.int32)
            elif isinstance(sequence_frame_indices, np.ndarray):
                sequence_frame_indices = np.asarray(sequence_frame_indices, dtype=np.int32)
            
            n = len(sequence_times_s)
            for i in range(n):
                if i >= len(sequence_frame_indices):
                    break
                frame_idx = int(sequence_frame_indices[i])
                time_sec = float(sequence_times_s[i])
                feat = frame_feat_map.get(frame_idx, {})
                
                timeline.append({
                    "frame_index": frame_idx,
                    "time_sec": time_sec,
                    "hue_mean_norm": float(feat.get("hue_mean_norm", 0.0)) if isinstance(feat.get("hue_mean_norm"), (int, float)) else None,
                    "colorfulness_norm": float(feat.get("colorfulness_norm", 0.0)) if isinstance(feat.get("colorfulness_norm"), (int, float)) else None,
                    "brightness_mean_norm": float(feat.get("brightness_mean", 0.0) / 255.0) if isinstance(feat.get("brightness_mean"), (int, float)) else None,
                    "global_contrast_norm": float(feat.get("global_contrast_norm", 0.0)) if isinstance(feat.get("global_contrast_norm"), (int, float)) else None,
                    "saturation_mean_norm": float(feat.get("sat_mean_norm", 0.0)) if isinstance(feat.get("sat_mean_norm"), (int, float)) else None,
                    "value_mean_norm": float(feat.get("val_mean_norm", 0.0)) if isinstance(feat.get("val_mean_norm"), (int, float)) else None,
                })
    
    render["timeline"] = timeline
    
    # Distribution statistics
    distributions = {}
    
    if timeline:
        hue_vals = [t.get("hue_mean_norm") for t in timeline if t.get("hue_mean_norm") is not None]
        colorfulness_vals = [t.get("colorfulness_norm") for t in timeline if t.get("colorfulness_norm") is not None]
        brightness_vals = [t.get("brightness_mean_norm") for t in timeline if t.get("brightness_mean_norm") is not None]
        contrast_vals = [t.get("global_contrast_norm") for t in timeline if t.get("global_contrast_norm") is not None]
        
        if hue_vals:
            hue_arr = np.array(hue_vals, dtype=np.float32)
            distributions["hue_mean_norm"] = {
                "min": float(np.min(hue_arr)),
                "max": float(np.max(hue_arr)),
                "mean": float(np.mean(hue_arr)),
                "std": float(np.std(hue_arr)),
                "median": float(np.median(hue_arr)),
                "p25": float(np.percentile(hue_arr, 25)),
                "p75": float(np.percentile(hue_arr, 75)),
            }
        
        if colorfulness_vals:
            colorfulness_arr = np.array(colorfulness_vals, dtype=np.float32)
            distributions["colorfulness_norm"] = {
                "min": float(np.min(colorfulness_arr)),
                "max": float(np.max(colorfulness_arr)),
                "mean": float(np.mean(colorfulness_arr)),
                "std": float(np.std(colorfulness_arr)),
                "median": float(np.median(colorfulness_arr)),
                "p25": float(np.percentile(colorfulness_arr, 25)),
                "p75": float(np.percentile(colorfulness_arr, 75)),
            }
        
        if brightness_vals:
            brightness_arr = np.array(brightness_vals, dtype=np.float32)
            distributions["brightness_mean_norm"] = {
                "min": float(np.min(brightness_arr)),
                "max": float(np.max(brightness_arr)),
                "mean": float(np.mean(brightness_arr)),
                "std": float(np.std(brightness_arr)),
                "median": float(np.median(brightness_arr)),
                "p25": float(np.percentile(brightness_arr, 25)),
                "p75": float(np.percentile(brightness_arr, 75)),
            }
        
        if contrast_vals:
            contrast_arr = np.array(contrast_vals, dtype=np.float32)
            distributions["global_contrast_norm"] = {
                "min": float(np.min(contrast_arr)),
                "max": float(np.max(contrast_arr)),
                "mean": float(np.mean(contrast_arr)),
                "std": float(np.std(contrast_arr)),
                "median": float(np.median(contrast_arr)),
                "p25": float(np.percentile(contrast_arr, 25)),
                "p75": float(np.percentile(contrast_arr, 75)),
            }
    
    render["distributions"] = distributions
    
    return render


def render_color_light_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага color_light результатов.
    
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
    render = render_color_light(npz_data, meta)
    
    timeline = render.get("timeline", []) or []
    summary = render.get("summary", {}) or {}
    distributions = render.get("distributions", {}) or {}

    # Key facts (Audit v3)
    key_facts: List[str] = []
    status = str(meta.get("status") or "unknown")
    empty_reason = meta.get("empty_reason")
    key_facts.append(f"status: {status}" + (f" ({empty_reason})" if empty_reason else ""))
    if meta.get("producer_version"):
        key_facts.append(f"producer_version: {meta.get('producer_version')}")
    if meta.get("schema_version"):
        key_facts.append(f"schema_version: {meta.get('schema_version')}")
    if meta.get("sampling_policy_version"):
        key_facts.append(f"sampling_policy_version: {meta.get('sampling_policy_version')}")
    if meta.get("module_sampling_policy_version"):
        key_facts.append(f"module_sampling_policy_version: {meta.get('module_sampling_policy_version')}")
    if isinstance(summary.get("frames_count"), int):
        key_facts.append(f"frames_count: {summary.get('frames_count')}")
    if isinstance(summary.get("scenes_count"), int):
        key_facts.append(f"scenes_count: {summary.get('scenes_count')}")

    config_highlights = {
        k: meta.get(k)
        for k in [
            "store_debug_objects",
            "hue_hist_bins",
            "palette_sample_size",
            "palette_kmeans_max_colors",
            "palette_kmeans_random_state",
            "palette_kmeans_n_init",
            "max_frames_per_scene",
            "stride",
        ]
        if k in meta
    }

    stage_timings = meta.get("stage_timings_ms") if isinstance(meta.get("stage_timings_ms"), dict) else {}

    def _topk(metric: str, k: int = 5, reverse: bool = True) -> List[Dict[str, Any]]:
        rows: List[Tuple[float, Dict[str, Any]]] = []
        for t in timeline:
            v = t.get(metric)
            if isinstance(v, (int, float)) and np.isfinite(float(v)):
                rows.append((float(v), t))
        rows.sort(key=lambda x: x[0], reverse=reverse)
        out: List[Dict[str, Any]] = []
        for v, t in rows[:k]:
            out.append({"time_sec": t.get("time_sec"), "frame_index": t.get("frame_index"), metric: v})
        return out

    top_examples = {
        "colorfulness_norm_top": _topk("colorfulness_norm", k=5, reverse=True),
        "brightness_mean_norm_top": _topk("brightness_mean_norm", k=5, reverse=True),
    }
    anti_top_examples = {
        "brightness_mean_norm_low": _topk("brightness_mean_norm", k=5, reverse=False),
        "global_contrast_norm_low": _topk("global_contrast_norm", k=5, reverse=False),
    }
    
    # Helper function to format distribution values
    def format_dist_value(dist_key, stat_key):
        dist = distributions.get(dist_key)
        if dist and stat_key in dist:
            return f"{dist[stat_key]:.4f}"
        return "N/A"
    
    # Offline SVG charts (no CDN / no JS)
    charts_html = ""
    if timeline:
        t_arr = np.asarray([t.get("time_sec", np.nan) for t in timeline], dtype=np.float64)
        hue_arr = np.asarray([t.get("hue_mean_norm", np.nan) for t in timeline], dtype=np.float64)
        col_arr = np.asarray([t.get("colorfulness_norm", np.nan) for t in timeline], dtype=np.float64)
        br_arr = np.asarray([t.get("brightness_mean_norm", np.nan) for t in timeline], dtype=np.float64)
        ct_arr = np.asarray([t.get("global_contrast_norm", np.nan) for t in timeline], dtype=np.float64)
        charts_html = "\n".join(
            [
                _svg_line_chart(times_s=t_arr, values=hue_arr, title="hue_mean_norm", stroke="#14b8a6"),
                _svg_line_chart(times_s=t_arr, values=col_arr, title="colorfulness_norm", stroke="#ef4444"),
                _svg_line_chart(times_s=t_arr, values=br_arr, title="brightness_mean_norm", stroke="#a855f7"),
                _svg_line_chart(times_s=t_arr, values=ct_arr, title="global_contrast_norm", stroke="#f59e0b"),
            ]
        )
    
    # Style probabilities display
    style_probs_html = ""
    style_probs = summary.get("style_probabilities", {})
    if style_probs:
        style_probs_html = "<div class='style-probs'><h3>Style Probabilities</h3><ul>"
        for style_name, prob in sorted(style_probs.items(), key=lambda x: x[1], reverse=True):
            style_probs_html += f"<li><strong>{style_name.replace('style_', '').replace('_', ' ').title()}</strong>: {prob:.4f}</li>"
        style_probs_html += "</ul></div>"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>color_light — debug render (offline)</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 16px; background: #f9fafb; color: #111827; }}
    .container {{ background: #fff; padding: 16px; border-radius: 10px; border: 1px solid #e5e7eb; max-width: 1100px; margin: 0 auto; }}
    h1 {{ margin: 0 0 10px 0; font-size: 20px; }}
    h2 {{ margin: 18px 0 10px 0; font-size: 16px; }}
    .muted {{ color: #6b7280; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
    .card {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }}
    .kv {{ font-size: 13px; }}
    .kv .k {{ color: #6b7280; }}
    .charts {{ display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 12px; }}
    .chart {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }}
    .chart-title {{ font-weight: 600; margin-bottom: 6px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }}
    th {{ color: #374151; background: #f9fafb; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; }}
  </style>
</head>
<body>
  <div class="container">
    <h1><code>color_light</code> — debug render (offline)</h1>
    <div class="muted">NPZ: {_esc(os.path.basename(npz_path))}</div>

    <h2>Key facts</h2>
    <div class="card kv">
      <ul>
        {''.join(f'<li>{_esc(x)}</li>' for x in key_facts)}
      </ul>
    </div>

    <h2>Summary</h2>
    <div class="grid">
      <div class="card kv"><div class="k">frames_count</div><div><strong>{_esc(summary.get('frames_count', 0))}</strong></div></div>
      <div class="card kv"><div class="k">scenes_count</div><div><strong>{_esc(summary.get('scenes_count', 0))}</strong></div></div>
      <div class="card kv"><div class="k">color_distribution_entropy</div><div><strong>{_esc(summary.get('color_distribution_entropy'))}</strong></div></div>
      <div class="card kv"><div class="k">color_distribution_gini</div><div><strong>{_esc(summary.get('color_distribution_gini'))}</strong></div></div>
      <div class="card kv"><div class="k">global_brightness_change_speed</div><div><strong>{_esc(summary.get('global_brightness_change_speed'))}</strong></div></div>
      <div class="card kv"><div class="k">global_color_change_speed</div><div><strong>{_esc(summary.get('global_color_change_speed'))}</strong></div></div>
    </div>

    {style_probs_html}

    <h2>Timeline charts (offline SVG)</h2>
    {f'<div class="charts">{charts_html}</div>' if charts_html else '<p class="muted">No timeline data available</p>'}
        
        {f'''
        <div class="distributions">
            <h2>Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Hue Mean</th>
                        <th>Colorfulness</th>
                        <th>Brightness</th>
                        <th>Contrast</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('hue_mean_norm', 'min')}</td>
                        <td>{format_dist_value('colorfulness_norm', 'min')}</td>
                        <td>{format_dist_value('brightness_mean_norm', 'min')}</td>
                        <td>{format_dist_value('global_contrast_norm', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('hue_mean_norm', 'max')}</td>
                        <td>{format_dist_value('colorfulness_norm', 'max')}</td>
                        <td>{format_dist_value('brightness_mean_norm', 'max')}</td>
                        <td>{format_dist_value('global_contrast_norm', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('hue_mean_norm', 'mean')}</td>
                        <td>{format_dist_value('colorfulness_norm', 'mean')}</td>
                        <td>{format_dist_value('brightness_mean_norm', 'mean')}</td>
                        <td>{format_dist_value('global_contrast_norm', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('hue_mean_norm', 'std')}</td>
                        <td>{format_dist_value('colorfulness_norm', 'std')}</td>
                        <td>{format_dist_value('brightness_mean_norm', 'std')}</td>
                        <td>{format_dist_value('global_contrast_norm', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('hue_mean_norm', 'median')}</td>
                        <td>{format_dist_value('colorfulness_norm', 'median')}</td>
                        <td>{format_dist_value('brightness_mean_norm', 'median')}</td>
                        <td>{format_dist_value('global_contrast_norm', 'median')}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions else ''}

    <h2>Top / anti-top frames (from sequence)</h2>
    <div class="grid">
      <div class="card"><div class="chart-title">Top colorfulness_norm</div><pre class="muted">{_esc(json.dumps(top_examples.get('colorfulness_norm_top', []), ensure_ascii=False, indent=2))}</pre></div>
      <div class="card"><div class="chart-title">Top brightness_mean_norm</div><pre class="muted">{_esc(json.dumps(top_examples.get('brightness_mean_norm_top', []), ensure_ascii=False, indent=2))}</pre></div>
      <div class="card"><div class="chart-title">Low brightness_mean_norm</div><pre class="muted">{_esc(json.dumps(anti_top_examples.get('brightness_mean_norm_low', []), ensure_ascii=False, indent=2))}</pre></div>
      <div class="card"><div class="chart-title">Low global_contrast_norm</div><pre class="muted">{_esc(json.dumps(anti_top_examples.get('global_contrast_norm_low', []), ensure_ascii=False, indent=2))}</pre></div>
    </div>

    <h2>Config highlights</h2>
    <div class="card"><pre class="muted">{_esc(json.dumps(config_highlights, ensure_ascii=False, indent=2))}</pre></div>

    <h2>Stage timings (ms)</h2>
    <div class="card"><pre class="muted">{_esc(json.dumps(stage_timings, ensure_ascii=False, indent=2))}</pre></div>
  </div>
</body>
</html>"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Show relative path for cleaner output
    rel_output_path = os.path.relpath(output_path, os.getcwd()) if os.path.exists(output_path) else output_path
    logger.info(f"Saved Color & Light HTML render to {rel_output_path}")
    return output_path


__all__ = ["render_color_light", "render_color_light_html"]

