"""
Renderer для high_level_semantic: генерация render-context JSON и HTML debug страницы.

Audit v3: fully-offline HTML (no CDN), includes key facts / top examples / config highlights / stage timings.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

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


def _as_float_array(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=np.float32)
    return None


def _as_int_array(x: Any, dtype=np.int32) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, list):
        return np.asarray(x, dtype=dtype)
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=dtype)
    return None


def _stats(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


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


def render_high_level_semantic(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для high_level_semantic."""
    render: Dict[str, Any] = {
        "component": "high_level_semantic",
        "summary": {},
        "timeline": [],
        "distributions": {},
        "key_facts": [],
        "top_examples": {},
        "anti_top_examples": {},
        "config_highlights": {},
    }

    frame_indices = _as_int_array(npz_data.get("frame_indices"), dtype=np.int32)
    times_s = _as_float_array(npz_data.get("times_s"))
    frame_feature_names_raw = npz_data.get("frame_feature_names")
    frame_features = npz_data.get("frame_features")
    event_times_s = _as_float_array(npz_data.get("event_times_s"))
    event_type_id = _as_int_array(npz_data.get("event_type_id"), dtype=np.int16)
    event_strength = _as_float_array(npz_data.get("event_strength"))
    event_frame_pos = _as_int_array(npz_data.get("event_frame_pos"), dtype=np.int32)
    features = npz_data.get("features") or {}
    ui = npz_data.get("ui") or {}

    # Decode frame_feature_names
    frame_feature_names: List[str] = []
    if isinstance(frame_feature_names_raw, np.ndarray) and frame_feature_names_raw.dtype == object:
        frame_feature_names = [str(x) for x in frame_feature_names_raw.flatten()]
    elif isinstance(frame_feature_names_raw, list):
        frame_feature_names = [str(x) for x in frame_feature_names_raw]

    if isinstance(frame_features, list):
        frame_features = np.asarray(frame_features, dtype=np.float32)
    elif isinstance(frame_features, np.ndarray):
        frame_features = np.asarray(frame_features, dtype=np.float32)
    else:
        frame_features = None

    # Summary
    n_frames = int(features.get("n_frames") or (len(frame_indices) if frame_indices is not None else 0))
    n_scenes = int(features.get("n_scenes") or 0)
    summary = {
        "n_frames": n_frames,
        "n_scenes": n_scenes,
        "clip_sim_prev_mean": float(features.get("clip_sim_prev_mean", np.nan)),
        "clip_novelty_prev_mean": float(features.get("clip_novelty_prev_mean", np.nan)),
        "hard_cuts_count": int(features.get("hard_cuts_count", 0)),
        "semantic_jump_events_count": int(features.get("semantic_jump_events_count", 0)),
    }

    # Event counts by type
    event_type_map = (ui.get("event_type_map") or {}) if isinstance(ui, dict) else {}
    if event_type_id is not None:
        unique_types, counts = np.unique(event_type_id, return_counts=True)
        by_type: Dict[str, int] = {}
        for t, c in zip(unique_types.tolist(), counts.tolist()):
            name = event_type_map.get(int(t)) if isinstance(event_type_map, dict) else None
            key = f"{int(t)}:{name}" if name else str(int(t))
            by_type[key] = int(c)
        summary["event_counts_by_type"] = by_type

    render["summary"] = summary

    # Timeline (lightweight)
    timeline: List[Dict[str, Any]] = []
    if (
        frame_indices is not None
        and times_s is not None
        and frame_features is not None
        and frame_features.ndim == 2
        and len(frame_feature_names) == frame_features.shape[1]
    ):
        name_to_idx = {name: i for i, name in enumerate(frame_feature_names)}

        def col(name: str) -> Optional[np.ndarray]:
            idx = name_to_idx.get(name)
            if idx is None or idx < 0 or idx >= frame_features.shape[1]:
                return None
            return frame_features[:, idx].astype(np.float32)

        sim_prev = col("clip_sim_prev")
        novelty_prev = col("clip_novelty_prev")
        loud_dbfs = col("loudness_dbfs")
        tempo_bpm = col("tempo_bpm")
        emo_intensity = col("emo_intensity")

        N = min(len(frame_indices), len(times_s))
        if N > 0:
            events_by_pos: Dict[int, List[Dict[str, Any]]] = {}
            if (
                event_frame_pos is not None
                and event_times_s is not None
                and event_type_id is not None
                and event_strength is not None
            ):
                M = min(len(event_frame_pos), len(event_times_s), len(event_type_id), len(event_strength))
                for j in range(M):
                    pos = int(event_frame_pos[j])
                    ev = {
                        "time_sec": float(event_times_s[j]),
                        "type_id": int(event_type_id[j]),
                        "type_name": event_type_map.get(int(event_type_id[j])) if isinstance(event_type_map, dict) else None,
                        "strength": float(event_strength[j]),
                    }
                    events_by_pos.setdefault(pos, []).append(ev)

            for i in range(N):
                entry: Dict[str, Any] = {"frame_index": int(frame_indices[i]), "time_sec": float(times_s[i])}
                if sim_prev is not None:
                    entry["clip_sim_prev"] = float(sim_prev[i])
                if novelty_prev is not None:
                    entry["clip_novelty_prev"] = float(novelty_prev[i])
                if loud_dbfs is not None:
                    entry["loudness_dbfs"] = float(loud_dbfs[i])
                if tempo_bpm is not None:
                    entry["tempo_bpm"] = float(tempo_bpm[i])
                if emo_intensity is not None:
                    entry["emo_intensity"] = float(emo_intensity[i])
                evs_here = events_by_pos.get(i)
                if evs_here:
                    entry["events"] = evs_here
                timeline.append(entry)

    render["timeline"] = timeline

    # Distributions
    distributions: Dict[str, Any] = {}
    if frame_features is not None and frame_features.ndim == 2 and len(frame_feature_names) == frame_features.shape[1]:
        name_to_idx = {name: i for i, name in enumerate(frame_feature_names)}
        for key in ["clip_novelty_prev", "emo_intensity", "loudness_dbfs", "tempo_bpm"]:
            idx = name_to_idx.get(key)
            if idx is None:
                continue
            col_vals = frame_features[:, idx]
            col_vals = col_vals[np.isfinite(col_vals)]
            if col_vals.size == 0:
                continue
            distributions[key] = _stats(col_vals)
    render["distributions"] = distributions

    # Key facts
    key_facts: List[Dict[str, Any]] = [
        {"key": "frames", "value": int(summary.get("n_frames", 0))},
        {"key": "scenes", "value": int(summary.get("n_scenes", 0))},
        {"key": "hard_cuts", "value": int(summary.get("hard_cuts_count", 0))},
        {"key": "semantic_jumps", "value": int(summary.get("semantic_jump_events_count", 0))},
    ]
    if isinstance(meta, dict) and "feature_groups" in meta:
        key_facts.append({"key": "feature_groups", "value": str(meta.get("feature_groups"))})
    render["key_facts"] = key_facts

    # Top / anti-top semantic_jump events
    top: List[Dict[str, Any]] = []
    anti: List[Dict[str, Any]] = []
    try:
        if event_type_id is not None and event_strength is not None and event_times_s is not None:
            m = (event_type_id.astype(np.int32) == 200) & np.isfinite(event_strength) & np.isfinite(event_times_s)
            if np.any(m):
                idx = np.where(m)[0]
                order = np.argsort(event_strength[idx])[::-1]
                for j in idx[order[:10]].tolist():
                    top.append({"time_s": float(event_times_s[j]), "strength": float(event_strength[j])})
                for j in idx[order[-10:]].tolist():
                    anti.append({"time_s": float(event_times_s[j]), "strength": float(event_strength[j])})
    except Exception:
        pass
    render["top_examples"] = {"semantic_jump": top}
    render["anti_top_examples"] = {"semantic_jump": anti}

    # Config highlights
    cfg = {}
    if isinstance(meta, dict):
        for k in [
            "feature_groups",
            "require_cut_detection_model_facing",
            "require_text_processor",
            "require_audio_loudness",
            "require_audio_tempo",
            "require_audio_clap",
            "progress_every_frames",
            "semantic_jump_topk_events",
            "semantic_jump_min_strength",
            "semantic_jump_min_distance_frames",
        ]:
            if k in meta:
                cfg[k] = meta.get(k)
        if "stage_timings_ms" in meta:
            cfg["stage_timings_ms"] = meta.get("stage_timings_ms")
    render["config_highlights"] = cfg

    return render


def render_high_level_semantic_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага high_level_semantic результатов.

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

    # Try to import shared helpers
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
    render = render_high_level_semantic(npz_data, meta)

    summary = render.get("summary", {})
    distributions = render.get("distributions", {})
    key_facts = render.get("key_facts", [])
    top_examples = render.get("top_examples", {})
    anti_top_examples = render.get("anti_top_examples", {})
    config_highlights = render.get("config_highlights", {})

    # Helper for table formatting
    def format_dist_value(dist_key, stat_key):
        dist = distributions.get(dist_key)
        if dist and stat_key in dist:
            return f"{dist[stat_key]:.4f}"
        return "N/A"

    # Offline SVG charts from NPZ arrays (no CDN / no JS)
    def _col(name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            ff = npz_data.get("frame_features")
            fn = npz_data.get("frame_feature_names")
            ts = npz_data.get("times_s")
            if ff is None or fn is None or ts is None:
                return None, None
            ff = np.asarray(ff, dtype=np.float32)
            fn = np.asarray(fn, dtype=object).reshape(-1)
            ts = np.asarray(ts, dtype=np.float32).reshape(-1)
            if ff.ndim != 2 or fn.size != ff.shape[1]:
                return None, None
            name_to_idx = {str(x): i for i, x in enumerate(fn.tolist())}
            idx = name_to_idx.get(name)
            if idx is None:
                return None, None
            return ts, ff[:, int(idx)].astype(np.float32)
        except Exception:
            return None, None

    charts_html = ""
    for (nm, title, stroke) in [
        ("clip_novelty_prev", "Semantic Novelty (1 - cos)", "#ef4444"),
        ("emo_intensity", "Emotion intensity", "#3b82f6"),
        ("loudness_dbfs", "Loudness (dBFS)", "#f59e0b"),
        ("tempo_bpm", "Tempo (BPM)", "#8b5cf6"),
    ]:
        ts, vv = _col(nm)
        if ts is None or vv is None:
            continue
        svg = _svg_line_chart(times_s=ts, values=vv, title=title, stroke=stroke)
        if svg:
            charts_html += svg + "\n"

    # Event counts HTML
    event_counts = summary.get("event_counts_by_type", {})
    events_html = ""
    if event_counts:
        rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(event_counts.items(), key=lambda x: x[0])
        )
        events_html = f"""
        <div class="events">
            <h3>Event Counts by Type</h3>
            <table>
                <thead><tr><th>Type</th><th>Count</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>High Level Semantic Debug Render</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 1200px; margin: 0 auto; }}
        h1, h2, h3 {{ color: #0056b3; }}
        .summary {{ background-color: #eaf4ff; padding: 15px; border-radius: 5px; margin: 20px 0; border: 1px solid #cce0ff; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .metric-card {{ background-color: #f8f9fa; padding: 12px; border-radius: 5px; border: 1px solid #dee2e6; }}
        .metric-card strong {{ color: #0056b3; display: block; margin-bottom: 5px; }}
        .metric-value {{ font-size: 1.2em; color: #333; }}
        .chart {{ margin: 14px 0; }}
        .chart-title {{ font-size: 14px; color: #111827; margin-bottom: 6px; }}
        .distributions {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .distributions table {{ width: 100%; border-collapse: collapse; }}
        .distributions th, .distributions td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .distributions th {{ background-color: #0056b3; color: white; }}
        .events table {{ width: 100%; border-collapse: collapse; }}
        .events th, .events td {{ padding: 8px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        .events th {{ background-color: #0056b3; color: white; }}
        .kv {{ display: grid; grid-template-columns: 260px 1fr; gap: 8px 14px; }}
        .kv div {{ padding: 6px 0; border-bottom: 1px solid #eef2f7; }}
        .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>High Level Semantic Debug Render</h1>

        <div class="summary">
            <h2>Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <strong>Frames</strong>
                    <span class="metric-value">{summary.get('n_frames', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Scenes</strong>
                    <span class="metric-value">{summary.get('n_scenes', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Mean Semantic Novelty</strong>
                    <span class="metric-value">{summary.get('clip_novelty_prev_mean', 0.0) or 0.0:.4f}</span>
                </div>
                <div class="metric-card">
                    <strong>Hard Cuts</strong>
                    <span class="metric-value">{summary.get('hard_cuts_count', 0)}</span>
                </div>
                <div class="metric-card">
                    <strong>Semantic Jumps</strong>
                    <span class="metric-value">{summary.get('semantic_jump_events_count', 0)}</span>
                </div>
            </div>
        </div>

        {events_html}

        <h2>Key facts</h2>
        <div class="kv">
          {''.join([f"<div><strong>{_esc(k.get('key'))}</strong></div><div class='mono'>{_esc(k.get('value'))}</div>" for k in (key_facts or [])])}
        </div>

        <h2>Charts (offline)</h2>
        {charts_html if charts_html.strip() else '<p>No chartable signals available</p>'}

        <h2>Top / Anti-top examples</h2>
        <div class="kv">
          <div><strong>Top semantic_jump</strong></div><div class="mono">{_esc(json.dumps((top_examples or {{}}).get('semantic_jump', []), ensure_ascii=False)[:2000])}</div>
          <div><strong>Anti-top semantic_jump</strong></div><div class="mono">{_esc(json.dumps((anti_top_examples or {{}}).get('semantic_jump', []), ensure_ascii=False)[:2000])}</div>
        </div>

        <h2>Config highlights</h2>
        <div class="kv">
          {''.join([f"<div><strong>{_esc(k)}</strong></div><div class='mono'>{_esc(v)}</div>" for k,v in (config_highlights or {{}}).items() if k != 'stage_timings_ms'])}
        </div>

        <h2>Stage timings (ms)</h2>
        <pre class="mono">{_esc(json.dumps((config_highlights or {{}}).get('stage_timings_ms') or meta.get('stage_timings_ms') or {{}}, ensure_ascii=False, indent=2))}</pre>

        {f'''
        <div class="distributions">
            <h2>Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Semantic Novelty</th>
                        <th>Emotion Intensity</th>
                        <th>Loudness (dBFS)</th>
                        <th>Tempo (BPM)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Min</strong></td>
                        <td>{format_dist_value('clip_novelty_prev', 'min')}</td>
                        <td>{format_dist_value('emo_intensity', 'min')}</td>
                        <td>{format_dist_value('loudness_dbfs', 'min')}</td>
                        <td>{format_dist_value('tempo_bpm', 'min')}</td>
                    </tr>
                    <tr>
                        <td><strong>Max</strong></td>
                        <td>{format_dist_value('clip_novelty_prev', 'max')}</td>
                        <td>{format_dist_value('emo_intensity', 'max')}</td>
                        <td>{format_dist_value('loudness_dbfs', 'max')}</td>
                        <td>{format_dist_value('tempo_bpm', 'max')}</td>
                    </tr>
                    <tr>
                        <td><strong>Mean</strong></td>
                        <td>{format_dist_value('clip_novelty_prev', 'mean')}</td>
                        <td>{format_dist_value('emo_intensity', 'mean')}</td>
                        <td>{format_dist_value('loudness_dbfs', 'mean')}</td>
                        <td>{format_dist_value('tempo_bpm', 'mean')}</td>
                    </tr>
                    <tr>
                        <td><strong>Std</strong></td>
                        <td>{format_dist_value('clip_novelty_prev', 'std')}</td>
                        <td>{format_dist_value('emo_intensity', 'std')}</td>
                        <td>{format_dist_value('loudness_dbfs', 'std')}</td>
                        <td>{format_dist_value('tempo_bpm', 'std')}</td>
                    </tr>
                    <tr>
                        <td><strong>Median</strong></td>
                        <td>{format_dist_value('clip_novelty_prev', 'median')}</td>
                        <td>{format_dist_value('emo_intensity', 'median')}</td>
                        <td>{format_dist_value('loudness_dbfs', 'median')}</td>
                        <td>{format_dist_value('tempo_bpm', 'median')}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions else ''}
    </div>

</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Saved High Level Semantic HTML render to {output_path}")
    return output_path


__all__ = ["render_high_level_semantic", "render_high_level_semantic_html"]


