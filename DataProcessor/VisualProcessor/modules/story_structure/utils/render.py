"""
Renderer for story_structure: render-context JSON + fully-offline HTML (no CDN).

Audit v3:
- Prefer offline SVG charts (no Chart.js).
- meta.ui_payload is the source for UI extras (markers/peaks pointers), not top-level NPZ keys.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, List, Tuple

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
        return np.asarray(x, dtype=np.float32).reshape(-1)
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=np.float32).reshape(-1)
    return None


def _as_int_array(x: Any, dtype=np.int32) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, list):
        return np.asarray(x, dtype=dtype).reshape(-1)
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=dtype).reshape(-1)
    return None


def _as_bool_array(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, list):
        return np.asarray(x, dtype=bool).reshape(-1)
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=bool).reshape(-1)
    return None


def _stats(arr: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(arr, dtype=np.float32)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}
    return {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "median": float(np.median(a)),
        "p05": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
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


def _tabular_features(npz_data: Dict[str, Any]) -> List[Tuple[str, float]]:
    fn = npz_data.get("feature_names")
    fv = npz_data.get("feature_values")
    if fn is None or fv is None:
        return []
    try:
        names = [str(x) for x in np.asarray(fn, dtype=object).reshape(-1).tolist()]
        vals = np.asarray(fv, dtype=np.float32).reshape(-1)
    except Exception:
        return []
    out: List[Tuple[str, float]] = []
    n = min(len(names), int(vals.size))
    for i in range(n):
        out.append((names[i], float(vals[i])))
    return out


def render_story_structure(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    times_s = _as_float_array(npz_data.get("times_s")) or np.asarray([], dtype=np.float32)
    fi = _as_int_array(npz_data.get("frame_indices")) or np.asarray([], dtype=np.int32)
    story_energy = _as_float_array(npz_data.get("story_energy_curve"))
    motion = _as_float_array(npz_data.get("motion_norm_per_sec_mean"))
    emb_rate = _as_float_array(npz_data.get("embedding_change_rate_per_sec"))
    topic = _as_float_array(npz_data.get("topic_shift_curve"))
    any_face = _as_bool_array(npz_data.get("any_face_present"))

    ui_payload = meta.get("ui_payload") if isinstance(meta, dict) else None
    if not isinstance(ui_payload, dict):
        ui_payload = {}

    # Distributions
    dist: Dict[str, Any] = {}
    if story_energy is not None:
        dist["story_energy_curve"] = _stats(story_energy)
    if motion is not None:
        dist["motion_norm_per_sec_mean"] = _stats(motion)
    if emb_rate is not None:
        dist["embedding_change_rate_per_sec"] = _stats(emb_rate)
    if topic is not None:
        dist["topic_shift_curve"] = _stats(topic)

    # Key facts from tabular features (fallback to meta)
    feats = dict(_tabular_features(npz_data))
    key_facts = {
        "n_frames": int(feats.get("n_frames", float(len(fi)))),
        "video_length_seconds": float(feats.get("video_length_seconds", float("nan"))),
        "hook_visual_surprise_score": float(feats.get("hook_visual_surprise_score", float("nan"))),
        "climax_time_sec": float(feats.get("climax_time_sec", float("nan"))),
        "climax_position_normalized": float(feats.get("climax_position_normalized", float("nan"))),
        "number_of_peaks": float(feats.get("number_of_peaks", float("nan"))),
        "main_character_screen_time": float(feats.get("main_character_screen_time", float("nan"))),
        "topic_shift_curve_present": bool(feats.get("topic_shift_curve_present", 0.0) >= 0.5),
    }

    return {
        "component": "story_structure",
        "schema_version": str(meta.get("schema_version") or ""),
        "producer_version": str(meta.get("producer_version") or ""),
        "status": str(meta.get("status") or ""),
        "key_facts": key_facts,
        "config_highlights": {
            "min_frames": meta.get("min_frames"),
            "max_frames": meta.get("max_frames"),
            "energy_smoothing_sigma": meta.get("energy_smoothing_sigma"),
            "text_mode": meta.get("text_mode"),
            "clip_text_model_spec": meta.get("clip_text_model_spec"),
            "clip_text_batch_size": meta.get("clip_text_batch_size"),
            "ocr_max_chars_per_frame": meta.get("ocr_max_chars_per_frame"),
        },
        "stage_timings_ms": meta.get("stage_timings_ms") if isinstance(meta.get("stage_timings_ms"), dict) else {},
        "ui_payload": ui_payload,
        "distributions": dist,
        "timeline": {
            "has_times": bool(times_s.size > 0),
            "n": int(min(times_s.size, story_energy.size if story_energy is not None else times_s.size)),
        },
        # For HTML: pass raw arrays (small)
        "_arrays": {
            "times_s": times_s,
            "frame_indices": fi,
            "story_energy_curve": story_energy,
            "motion_norm_per_sec_mean": motion,
            "embedding_change_rate_per_sec": emb_rate,
            "topic_shift_curve": topic,
            "any_face_present": any_face,
            "story_energy_peaks_times_s": _as_float_array(npz_data.get("story_energy_peaks_times_s")),
            "story_energy_peaks_values_z": _as_float_array(npz_data.get("story_energy_peaks_values_z")),
        },
    }


def render_story_structure_html(npz_path: str, output_path: str) -> str:
    """
    Generate fully-offline HTML debug page for story_structure.
    """
    # Import here to avoid circular imports
    try:
        from utils.renderer import load_npz, extract_meta  # type: ignore
    except Exception:
        # Fallback: direct load
        def load_npz(path: str):
            data = np.load(path, allow_pickle=True)
            result: Dict[str, Any] = {}
            for key in data.files:
                arr = data[key]
                if isinstance(arr, np.ndarray) and arr.dtype == object:
                    result[key] = arr.item() if arr.size == 1 else arr.tolist()
                else:
                    result[key] = arr
            return result

        def extract_meta(npz_data: Dict[str, Any]) -> Dict[str, Any]:
            meta2 = npz_data.get("meta")
            if isinstance(meta2, np.ndarray) and meta2.dtype == object:
                return meta2.item() if meta2.size == 1 else {}
            return meta2 if isinstance(meta2, dict) else {}

    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    ctx = render_story_structure(npz_data, meta)

    arrays = ctx.get("_arrays") or {}
    times_s = arrays.get("times_s")
    if not isinstance(times_s, np.ndarray):
        times_s = np.asarray([], dtype=np.float32)

    charts: List[str] = []
    for key, title, color in [
        ("story_energy_curve", "Story energy (z)", "#0ea5e9"),
        ("motion_norm_per_sec_mean", "Motion (per-sec mean)", "#22c55e"),
        ("embedding_change_rate_per_sec", "Embedding change rate (/s)", "#a855f7"),
        ("topic_shift_curve", "Topic shift (/s)", "#f59e0b"),
    ]:
        v = arrays.get(key)
        if isinstance(v, np.ndarray):
            charts.append(_svg_line_chart(times_s=times_s, values=np.asarray(v, dtype=np.float32).reshape(-1), title=title, stroke=color))

    # Peaks (small table)
    peak_t = arrays.get("story_energy_peaks_times_s")
    peak_v = arrays.get("story_energy_peaks_values_z")
    peaks_rows = ""
    if isinstance(peak_t, np.ndarray) and isinstance(peak_v, np.ndarray):
        t = np.asarray(peak_t, dtype=np.float32).reshape(-1)
        v = np.asarray(peak_v, dtype=np.float32).reshape(-1)
        n = int(min(t.size, v.size, 30))
        rows = []
        for i in range(n):
            rows.append(f"<tr><td>{i}</td><td>{t[i]:.2f}</td><td>{v[i]:.3f}</td></tr>")
        peaks_rows = "\n".join(rows)

    key_facts = ctx.get("key_facts") or {}
    stage_timings = ctx.get("stage_timings_ms") or {}
    cfg = ctx.get("config_highlights") or {}
    status = _esc(ctx.get("status") or "")
    schema_version = _esc(ctx.get("schema_version") or "")
    producer_version = _esc(ctx.get("producer_version") or "")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>story_structure — render</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 20px; background: #f8fafc; color: #0f172a; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; }}
    .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin: 12px 0; }}
    .muted {{ color: #64748b; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; }}
    .kv {{ display: grid; grid-template-columns: 1fr auto; gap: 8px 12px; }}
    .kv div {{ padding: 2px 0; border-bottom: 1px dashed #e2e8f0; }}
    .chart-title {{ font-size: 13px; font-weight: 600; margin: 0 0 6px 0; color: #334155; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #e2e8f0; font-size: 13px; }}
    th {{ background: #f1f5f9; }}
    code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div style="display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap;">
        <div>
          <div style="font-size:18px; font-weight:700;">story_structure</div>
          <div class="muted">schema: <code>{schema_version}</code> · producer_version: <code>{producer_version}</code> · status: <code>{status}</code></div>
        </div>
      </div>
    </div>

    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Key facts</div>
      <div class="kv">
        {"".join([f"<div>{_esc(k)}</div><div>{_esc(v)}</div>" for k,v in key_facts.items()])}
      </div>
    </div>

    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Config highlights</div>
      <div class="kv">
        {"".join([f"<div>{_esc(k)}</div><div>{_esc(v)}</div>" for k,v in cfg.items() if v is not None])}
      </div>
    </div>

    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Stage timings (ms)</div>
      <div class="kv">
        {"".join([f"<div>{_esc(k)}</div><div>{float(v):.2f}</div>" for k,v in stage_timings.items()])}
      </div>
    </div>

    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Curves</div>
      {"".join([c for c in charts if c]) or "<div class='muted'>No curve data</div>"}
    </div>

    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Energy peaks (top 30)</div>
      {"<table><thead><tr><th>#</th><th>t (s)</th><th>value (z)</th></tr></thead><tbody>"+peaks_rows+"</tbody></table>" if peaks_rows else "<div class='muted'>No peaks</div>"}
    </div>
  </div>
</body>
</html>
"""

    from pathlib import Path

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)


