"""
Renderer for video_pacing: render-context JSON + fully-offline HTML (no CDN).

Audit v3:
- feature_names/feature_values is the model-facing scalar source-of-truth (no object-dict `features`).
- meta.ui_payload contains UI hints (shot boundaries + curve pointers).
"""

from __future__ import annotations

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


def render_video_pacing(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    times_s = _as_float_array(npz_data.get("times_s")) or np.asarray([], dtype=np.float32)
    motion = _as_float_array(npz_data.get("motion_norm_per_sec_mean"))
    sem = _as_float_array(npz_data.get("semantic_change_rate_per_sec"))
    col = _as_float_array(npz_data.get("color_change_rate_per_sec"))
    shot_b = _as_int_array(npz_data.get("shot_boundary_frame_indices"))

    feats = dict(_tabular_features(npz_data))
    ui_payload = meta.get("ui_payload") if isinstance(meta, dict) else None
    if not isinstance(ui_payload, dict):
        ui_payload = {}

    key_facts = {
        "shots_count": float(feats.get("shots_count", float("nan"))),
        "shot_duration_mean": float(feats.get("shot_duration_mean", float("nan"))),
        "cuts_per_10s": float(feats.get("cuts_per_10s", float("nan"))),
        "mean_motion_speed_per_shot": float(feats.get("mean_motion_speed_per_shot", float("nan"))),
        "frame_embedding_diff_mean": float(feats.get("frame_embedding_diff_mean", float("nan"))),
        "color_change_rate_mean": float(feats.get("color_change_rate_mean", float("nan"))),
    }

    return {
        "component": "video_pacing",
        "schema_version": str(meta.get("schema_version") or ""),
        "producer_version": str(meta.get("producer_version") or ""),
        "status": str(meta.get("status") or ""),
        "key_facts": key_facts,
        "config_highlights": {
            "downscale_factor": meta.get("downscale_factor"),
            "min_shot_length_seconds": meta.get("min_shot_length_seconds"),
            "shot_detect_k": meta.get("shot_detect_k"),
            "min_frames": meta.get("min_frames"),
            "enable_entropy_features": meta.get("enable_entropy_features"),
            "enable_histograms": meta.get("enable_histograms"),
            "enable_pace_curve_peaks": meta.get("enable_pace_curve_peaks"),
            "enable_periodicity": meta.get("enable_periodicity"),
            "enable_bursts": meta.get("enable_bursts"),
        },
        "stage_timings_ms": meta.get("stage_timings_ms") if isinstance(meta.get("stage_timings_ms"), dict) else {},
        "ui_payload": ui_payload,
        "distributions": {
            "motion_norm_per_sec_mean": _stats(motion) if motion is not None else {},
            "semantic_change_rate_per_sec": _stats(sem) if sem is not None else {},
            "color_change_rate_per_sec": _stats(col) if col is not None else {},
        },
        "_arrays": {
            "times_s": times_s,
            "motion_norm_per_sec_mean": motion,
            "semantic_change_rate_per_sec": sem,
            "color_change_rate_per_sec": col,
            "shot_boundary_frame_indices": shot_b,
        },
    }


def render_video_pacing_html(npz_path: str, output_path: str) -> str:
    """
    Generate fully-offline HTML debug page for video_pacing.
    """
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
    ctx = render_video_pacing(npz_data, meta)

    arrays = ctx.get("_arrays") or {}
    times_s = arrays.get("times_s")
    if not isinstance(times_s, np.ndarray):
        times_s = np.asarray([], dtype=np.float32)

    charts: List[str] = []
    for key, title, color in [
        ("motion_norm_per_sec_mean", "Motion (per-sec mean)", "#22c55e"),
        ("semantic_change_rate_per_sec", "Semantic change (/s)", "#a855f7"),
        ("color_change_rate_per_sec", "Color change (/s)", "#0ea5e9"),
    ]:
        v = arrays.get(key)
        if isinstance(v, np.ndarray):
            charts.append(_svg_line_chart(times_s=times_s[: int(np.asarray(v).reshape(-1).size)], values=np.asarray(v, dtype=np.float32).reshape(-1), title=title, stroke=color))

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
  <title>video_pacing — render</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 20px; background: #f8fafc; color: #0f172a; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; }}
    .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin: 12px 0; }}
    .muted {{ color: #64748b; }}
    .kv {{ display: grid; grid-template-columns: 1fr auto; gap: 8px 12px; }}
    .kv div {{ padding: 2px 0; border-bottom: 1px dashed #e2e8f0; }}
    .chart-title {{ font-size: 13px; font-weight: 600; margin: 0 0 6px 0; color: #334155; }}
    code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div style="font-size:18px; font-weight:700;">video_pacing</div>
      <div class="muted">schema: <code>{schema_version}</code> · producer_version: <code>{producer_version}</code> · status: <code>{status}</code></div>
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
  </div>
</body>
</html>
"""

    from pathlib import Path

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)


