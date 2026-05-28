"""
Renderer для behavioral: генерация render-context JSON и HTML debug страницы.
"""

import os
import json
import logging
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def _to_1d_float(arr: Any) -> np.ndarray:
    """Best-effort convert to 1D float32 numpy array (NaNs preserved if possible)."""
    try:
        if arr is None:
            return np.asarray([], dtype=np.float32)
        if isinstance(arr, np.ndarray):
            return np.asarray(arr, dtype=np.float32).reshape(-1)
        if isinstance(arr, list):
            return np.asarray(arr, dtype=np.float32).reshape(-1)
    except Exception:
        pass
    return np.asarray([], dtype=np.float32)


def _svg_line_chart(
    *,
    times_s: Any,
    values: Any,
    width: int = 1000,
    height: int = 240,
    stroke: str = "#2563eb",
    title: str = "",
) -> str:
    """
    Minimal offline-friendly SVG line chart.
    - Handles NaNs by breaking the polyline into segments.
    - Downsamples to ~500 points for size.
    """
    t = _to_1d_float(times_s)
    v = _to_1d_float(values)
    n = int(min(t.size, v.size))
    if n <= 1:
        return "<div class='muted'>No data</div>"

    t = t[:n]
    v = v[:n]

    max_points = 500
    step = max(1, int(np.ceil(float(n) / float(max_points))))
    t = t[::step]
    v = v[::step]

    finite = np.isfinite(t) & np.isfinite(v)
    if not np.any(finite):
        return "<div class='muted'>No finite data</div>"

    t_f = t[finite]
    v_f = v[finite]

    t_min = float(np.min(t_f))
    t_max = float(np.max(t_f))
    v_min = float(np.min(v_f))
    v_max = float(np.max(v_f))

    if abs(t_max - t_min) < 1e-9:
        t_max = t_min + 1e-6
    if abs(v_max - v_min) < 1e-9:
        v_max = v_min + 1e-6

    pad_x = 40
    pad_y = 24
    w = max(240, int(width))
    h = max(120, int(height))
    x0, y0 = pad_x, pad_y
    x1, y1 = w - pad_x, h - pad_y

    def _sx(tt: float) -> float:
        return x0 + (tt - t_min) / (t_max - t_min) * (x1 - x0)

    def _sy(vv: float) -> float:
        return y1 - (vv - v_min) / (v_max - v_min) * (y1 - y0)

    path_parts = []
    pen_down = False
    for tt, vv, ok in zip(t.tolist(), v.tolist(), finite.tolist()):
        if not ok:
            pen_down = False
            continue
        x = _sx(float(tt))
        y = _sy(float(vv))
        if not pen_down:
            path_parts.append(f"M {x:.2f} {y:.2f}")
            pen_down = True
        else:
            path_parts.append(f"L {x:.2f} {y:.2f}")

    path_d = " ".join(path_parts)
    esc_title = (title or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    return f"""
    <div class="chart">
      <div class="chart-title">{esc_title}</div>
      <svg viewBox="0 0 {w} {h}" width="100%" height="{h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{esc_title}">
        <rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff" stroke="#e5e7eb"/>
        <line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#e5e7eb"/>
        <line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#e5e7eb"/>
        <path d="{path_d}" fill="none" stroke="{stroke}" stroke-width="2"/>
        <text x="{x0}" y="{y0 - 8}" font-size="12" fill="#6b7280">{v_max:.3f}</text>
        <text x="{x0}" y="{y1 + 16}" font-size="12" fill="#6b7280">{v_min:.3f}</text>
        <text x="{x0}" y="{h - 4}" font-size="12" fill="#6b7280">{t_min:.2f}s</text>
        <text x="{x1 - 40}" y="{h - 4}" font-size="12" fill="#6b7280">{t_max:.2f}s</text>
      </svg>
    </div>
    """.strip()


def render_behavioral(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Генерировать render-context для behavioral."""
    render = {
        "component": "behavioral",
        "key_facts": {},
        "summary": {},
        "timeline": [],
        "distributions": {},
    }
    
    # Extract behavioral data
    times_s = npz_data.get("times_s")
    frame_indices = npz_data.get("frame_indices")
    landmarks_present = npz_data.get("landmarks_present")
    
    # Sequence features
    seq_keys = [
        "num_hands", "hands_visibility", "hand_motion_energy",
        "arm_openness", "pose_expansion", "body_lean_angle", "balance_offset",
        "shoulder_angle", "shoulder_angle_velocity",
        "head_position_x_norm", "head_position_y_norm", "head_motion_energy", "head_stability",
        "mouth_width_norm", "mouth_height_norm", "mouth_area_norm",
        "mouth_velocity", "mouth_open_ratio", "speech_activity_proxy",
        "blink_flag", "blink_rate_short", "self_touch_flag", "fidgeting_energy",
        "timestamp_norm",
    ]
    
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
    
    if landmarks_present is not None:
        if isinstance(landmarks_present, list):
            landmarks_present = np.array(landmarks_present, dtype=bool)
        elif isinstance(landmarks_present, np.ndarray):
            landmarks_present = np.asarray(landmarks_present, dtype=bool)
        else:
            landmarks_present = None
    
    # Extract sequence arrays
    seq_arrays = {}
    for key in seq_keys:
        arr_key = f"seq_{key}"
        arr = npz_data.get(arr_key)
        if arr is not None:
            if isinstance(arr, list):
                arr = np.array(arr, dtype=np.float32)
            elif isinstance(arr, np.ndarray):
                arr = np.asarray(arr, dtype=np.float32)
            seq_arrays[key] = arr
    
    # Extract aggregated results
    aggregated = npz_data.get("aggregated")
    if aggregated is not None:
        if isinstance(aggregated, np.ndarray) and aggregated.dtype == object:
            try:
                aggregated = aggregated.item()
            except Exception:
                aggregated = {}
        if not isinstance(aggregated, dict):
            aggregated = {}
    else:
        aggregated = {}
    
    # Summary statistics
    n_frames = len(times_s) if times_s is not None else 0
    landmarks_ratio = float(np.mean(landmarks_present)) if landmarks_present is not None and landmarks_present.size > 0 else 0.0

    # Key facts (mini-dashboard header)
    if isinstance(meta, dict):
        render["key_facts"] = {
            "schema_version": meta.get("schema_version", "unknown"),
            "producer_version": meta.get("producer_version", "unknown"),
            "status": meta.get("status", "unknown"),
            "empty_reason": meta.get("empty_reason"),
            "frames_count": int(n_frames),
            "landmarks_present_ratio": landmarks_ratio,
        }

    render["summary"] = {
        "frames_count": int(n_frames),
        "landmarks_present_ratio": landmarks_ratio,
    }
    
    # Add aggregated metrics to summary
    if aggregated:
        render["summary"].update({
            "avg_engagement": aggregated.get("avg_engagement"),
            "avg_confidence": aggregated.get("avg_confidence"),
            "avg_stress": aggregated.get("avg_stress"),
            "gesture_rate_per_sec": aggregated.get("gesture_rate_per_sec"),
            "hands_visibility_ratio": aggregated.get("hands_visibility_ratio"),
            "face_visibility_ratio": aggregated.get("face_visibility_ratio"),
        })
    
    # Timeline data (per-frame features)
    if times_s is not None and frame_indices is not None and len(times_s) > 0:
        timeline = []
        n = len(times_s)
        
        for i in range(n):
            frame_idx = int(frame_indices[i]) if i < len(frame_indices) else i
            time_sec = float(times_s[i]) if np.isfinite(times_s[i]) else 0.0
            
            timeline_entry = {
                "frame_index": frame_idx,
                "time_s": time_sec,
                "landmarks_present": bool(landmarks_present[i]) if landmarks_present is not None and i < len(landmarks_present) else False,
            }
            
            # Add sequence features to timeline
            for key in seq_keys:
                arr = seq_arrays.get(key)
                if arr is not None and i < len(arr):
                    val = arr[i]
                    if np.isfinite(val):
                        timeline_entry[key] = float(val)
            
            timeline.append(timeline_entry)
        
        render["timeline"] = timeline
    
    # Distribution statistics
    distributions = {}
    
    # Key behavioral features for distributions
    key_features = [
        "speech_activity_proxy",
        "arm_openness",
        "body_lean_angle",
        "hand_motion_energy",
        "head_motion_energy",
        "blink_rate_short",
        "fidgeting_energy",
    ]
    
    for key in key_features:
        arr = seq_arrays.get(key)
        if arr is not None:
            valid_vals = arr[np.isfinite(arr)]
            if valid_vals.size > 0:
                distributions[key] = {
                    "min": float(np.min(valid_vals)),
                    "max": float(np.max(valid_vals)),
                    "mean": float(np.mean(valid_vals)),
                    "std": float(np.std(valid_vals)),
                    "median": float(np.median(valid_vals)),
                    "p25": float(np.percentile(valid_vals, 25)),
                    "p75": float(np.percentile(valid_vals, 75)),
                    "p05": float(np.percentile(valid_vals, 5)),
                    "p95": float(np.percentile(valid_vals, 95)),
                }
    
    # Gesture distribution
    gesture_counts = aggregated.get("gesture_counts", {})
    if gesture_counts:
        distributions["gesture_distribution"] = gesture_counts
    
    render["distributions"] = distributions

    # Top / anti-top (QA helpers, offline)
    try:
        t_arr = _to_1d_float(npz_data.get("times_s"))
        fi_arr = np.asarray(npz_data.get("frame_indices") or [], dtype=np.int32).reshape(-1)
        blink = _to_1d_float(npz_data.get("seq_blink_rate_short"))
        self_touch = _to_1d_float(npz_data.get("seq_self_touch_flag"))
        fidget = _to_1d_float(npz_data.get("seq_fidgeting_energy"))
        speech = _to_1d_float(npz_data.get("seq_speech_activity_proxy"))
        arm = _to_1d_float(npz_data.get("seq_arm_openness"))

        n = int(min(t_arr.size, fi_arr.size, blink.size, self_touch.size, fidget.size))
        if n > 0:
            stress_proxy = 0.4 * blink[:n] + 0.3 * self_touch[:n] + 0.3 * (1.0 / (1.0 + np.exp(-fidget[:n] * 10.0)))
            ok = np.isfinite(stress_proxy) & np.isfinite(t_arr[:n])
            idxs = np.where(ok)[0]
            if idxs.size > 0:
                top_order = idxs[np.argsort(stress_proxy[idxs])[::-1]]
                anti_order = idxs[np.argsort(stress_proxy[idxs])]

                def _mk(i: int) -> Dict[str, Any]:
                    return {
                        "frame_index": int(fi_arr[i]) if i < fi_arr.size else int(i),
                        "time_s": float(t_arr[i]),
                        "stress_proxy": float(stress_proxy[i]),
                        "speech_activity_proxy": float(speech[i]) if i < speech.size and np.isfinite(speech[i]) else None,
                        "arm_openness": float(arm[i]) if i < arm.size and np.isfinite(arm[i]) else None,
                    }

                render["top"] = {
                    "stress_proxy": [_mk(int(i)) for i in top_order[:10].tolist()],
                    "stress_proxy_anti": [_mk(int(i)) for i in anti_order[:10].tolist()],
                }
    except Exception:
        pass
    
    return render


def render_behavioral_html(npz_path: str, output_path: str) -> str:
    """
    Генерировать HTML страницу для дебага behavioral результатов.
    
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
    render = render_behavioral(npz_data, meta)

    summary = render.get("summary", {}) if isinstance(render, dict) else {}
    distributions = render.get("distributions", {}) if isinstance(render, dict) else {}
    key_facts = render.get("key_facts", {}) if isinstance(render, dict) else {}
    top = render.get("top", {}) if isinstance(render, dict) else {}
    top_stress = top.get("stress_proxy") or []
    anti_stress = top.get("stress_proxy_anti") or []

    # Helper function to format distribution values
    def format_dist_value(dist_key, stat_key):
        dist = distributions.get(dist_key)
        if dist and stat_key in dist:
            try:
                return f"{float(dist[stat_key]):.4f}"
            except Exception:
                return str(dist[stat_key])
        return "N/A"

    # Compute stress_proxy series for charts (same as UI proxy)
    times_arr = _to_1d_float(npz_data.get("times_s"))
    speech_arr = _to_1d_float(npz_data.get("seq_speech_activity_proxy"))
    arm_arr = _to_1d_float(npz_data.get("seq_arm_openness"))
    blink_arr = _to_1d_float(npz_data.get("seq_blink_rate_short"))
    self_touch_arr = _to_1d_float(npz_data.get("seq_self_touch_flag"))
    fidget_arr = _to_1d_float(npz_data.get("seq_fidgeting_energy"))
    n_local = int(min(times_arr.size, blink_arr.size, self_touch_arr.size, fidget_arr.size))
    stress_arr = np.asarray([], dtype=np.float32)
    if n_local > 0:
        stress_arr = 0.4 * blink_arr[:n_local] + 0.3 * self_touch_arr[:n_local] + 0.3 * (1.0 / (1.0 + np.exp(-fidget_arr[:n_local] * 10.0)))

    def _rows(items):
        out = []
        for it in items:
            if not isinstance(it, dict):
                continue
            frame_index = it.get("frame_index", "")
            time_s = it.get("time_s", "")
            stress = it.get("stress_proxy", "")
            speech = it.get("speech_activity_proxy", "")
            arm = it.get("arm_openness", "")
            def _fmt(x, nd=4):
                if x is None:
                    return ""
                if isinstance(x, (int, float)):
                    return f"{float(x):.{nd}f}"
                return str(x)
            out.append(
                "<tr>"
                f"<td>{frame_index}</td>"
                f"<td>{_fmt(time_s, nd=3)}</td>"
                f"<td>{_fmt(stress, nd=4)}</td>"
                f"<td>{_fmt(speech, nd=4)}</td>"
                f"<td>{_fmt(arm, nd=4)}</td>"
                "</tr>"
            )
        return "\n".join(out) if out else "<tr><td colspan='5' class='muted'>No data</td></tr>"

    html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>behavioral — render (dev-only)</title>
    <style>
        body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; background: #f6f7fb; color: #111827; }}
        .container {{ background: #fff; padding: 24px; border-radius: 12px; box-shadow: 0 8px 24px rgba(17,24,39,0.08); max-width: 1180px; margin: 24px auto; }}
        h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
        h2 {{ margin: 22px 0 10px 0; font-size: 16px; }}
        .muted {{ color: #6b7280; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
        .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px 14px; background: #fafafa; }}
        .kv {{ display: grid; grid-template-columns: 160px 1fr; gap: 8px 12px; }}
        .k {{ color: #6b7280; font-size: 12px; }}
        .v {{ font-weight: 600; }}
        .chart {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; background: #fff; margin-bottom: 12px; }}
        .chart-title {{ font-size: 13px; color: #374151; margin-bottom: 8px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; }}
        th {{ background: #f3f4f6; font-weight: 700; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>behavioral — render (dev-only)</h1>
        <div class="muted">Offline HTML (no CDN). NPZ is source-of-truth.</div>

        <h2>Key facts</h2>
        <div class="card">
          <div class="kv">
            <div class="k">schema_version</div><div class="v">{key_facts.get('schema_version','unknown')}</div>
            <div class="k">producer_version</div><div class="v">{key_facts.get('producer_version','unknown')}</div>
            <div class="k">status</div><div class="v">{key_facts.get('status','unknown')}</div>
            <div class="k">empty_reason</div><div class="v">{key_facts.get('empty_reason')}</div>
            <div class="k">frames_count</div><div class="v">{summary.get('frames_count', 0)}</div>
            <div class="k">landmarks_present_ratio</div><div class="v">{summary.get('landmarks_present_ratio', 0.0):.2%}</div>
          </div>
        </div>

        <h2>Summary</h2>
        <div class="grid">
          <div class="card"><div class="k">avg_engagement</div><div class="v">{summary.get('avg_engagement', 0.0):.4f}</div></div>
          <div class="card"><div class="k">avg_confidence</div><div class="v">{summary.get('avg_confidence', 0.0):.4f}</div></div>
          <div class="card"><div class="k">avg_stress</div><div class="v">{summary.get('avg_stress', 0.0):.4f}</div></div>
          <div class="card"><div class="k">gesture_rate_per_sec</div><div class="v">{summary.get('gesture_rate_per_sec', 0.0):.3f}</div></div>
        </div>

        <h2>Timeline (offline charts)</h2>
        {_svg_line_chart(times_s=times_arr, values=speech_arr, stroke="#14b8a6", title="speech_activity_proxy")}
        {_svg_line_chart(times_s=times_arr, values=arm_arr, stroke="#ef4444", title="arm_openness")}
        {_svg_line_chart(times_s=times_arr[:n_local], values=stress_arr, stroke="#7c3aed", title="stress_proxy (UI proxy)")}
        
        {f'''
        <div class="distributions">
            <h2>Distribution Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Mean</th>
                        <th>Std</th>
                        <th>Median</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Speech Activity</strong></td>
                        <td>{format_dist_value('speech_activity_proxy', 'min')}</td>
                        <td>{format_dist_value('speech_activity_proxy', 'max')}</td>
                        <td>{format_dist_value('speech_activity_proxy', 'mean')}</td>
                        <td>{format_dist_value('speech_activity_proxy', 'std')}</td>
                        <td>{format_dist_value('speech_activity_proxy', 'median')}</td>
                    </tr>
                    <tr>
                        <td><strong>Arm Openness</strong></td>
                        <td>{format_dist_value('arm_openness', 'min')}</td>
                        <td>{format_dist_value('arm_openness', 'max')}</td>
                        <td>{format_dist_value('arm_openness', 'mean')}</td>
                        <td>{format_dist_value('arm_openness', 'std')}</td>
                        <td>{format_dist_value('arm_openness', 'median')}</td>
                    </tr>
                    <tr>
                        <td><strong>Body Lean Angle</strong></td>
                        <td>{format_dist_value('body_lean_angle', 'min')}</td>
                        <td>{format_dist_value('body_lean_angle', 'max')}</td>
                        <td>{format_dist_value('body_lean_angle', 'mean')}</td>
                        <td>{format_dist_value('body_lean_angle', 'std')}</td>
                        <td>{format_dist_value('body_lean_angle', 'median')}</td>
                    </tr>
                    <tr>
                        <td><strong>Hand Motion Energy</strong></td>
                        <td>{format_dist_value('hand_motion_energy', 'min')}</td>
                        <td>{format_dist_value('hand_motion_energy', 'max')}</td>
                        <td>{format_dist_value('hand_motion_energy', 'mean')}</td>
                        <td>{format_dist_value('hand_motion_energy', 'std')}</td>
                        <td>{format_dist_value('hand_motion_energy', 'median')}</td>
                    </tr>
                    <tr>
                        <td><strong>Blink Rate</strong></td>
                        <td>{format_dist_value('blink_rate_short', 'min')}</td>
                        <td>{format_dist_value('blink_rate_short', 'max')}</td>
                        <td>{format_dist_value('blink_rate_short', 'mean')}</td>
                        <td>{format_dist_value('blink_rate_short', 'std')}</td>
                        <td>{format_dist_value('blink_rate_short', 'median')}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        ''' if distributions else ''}

        <h2>Top / Anti-top (stress_proxy)</h2>
        <div class="grid">
          <div class="card">
            <div class="k">Top-10 (highest)</div>
            <table>
              <thead><tr><th>frame</th><th>time_s</th><th>stress</th><th>speech</th><th>arm_open</th></tr></thead>
              <tbody>{_rows(top_stress)}</tbody>
            </table>
          </div>
          <div class="card">
            <div class="k">Anti-top-10 (lowest)</div>
            <table>
              <thead><tr><th>frame</th><th>time_s</th><th>stress</th><th>speech</th><th>arm_open</th></tr></thead>
              <tbody>{_rows(anti_stress)}</tbody>
            </table>
          </div>
        </div>
    </div>
</body>
</html>"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Saved Behavioral HTML render to {output_path}")
    return output_path


__all__ = ["render_behavioral", "render_behavioral_html"]

