"""
Renderer for shot_quality: render-context JSON + fully-offline HTML debug page (no CDN).

Audit v3: prefer offline SVG charts.
"""

from __future__ import annotations

import json
import os
import logging
from typing import Any, Dict, Optional, List

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
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
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


def render_shot_quality(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    times_s = _as_float_array(npz_data.get("times_s"))
    frame_indices = _as_int_array(npz_data.get("frame_indices"))
    feature_names = npz_data.get("feature_names")
    frame_features = npz_data.get("frame_features")
    quality_probs = npz_data.get("quality_probs")
    shot_ids = _as_int_array(npz_data.get("shot_ids"))
    shot_start = _as_int_array(npz_data.get("shot_start_frame"))
    shot_end = _as_int_array(npz_data.get("shot_end_frame"))

    if isinstance(frame_features, list):
        frame_features = np.asarray(frame_features, dtype=np.float32)
    elif isinstance(frame_features, np.ndarray):
        frame_features = np.asarray(frame_features, dtype=np.float32)
    else:
        frame_features = None

    if isinstance(quality_probs, list):
        quality_probs = np.asarray(quality_probs, dtype=np.float32)
    elif isinstance(quality_probs, np.ndarray):
        quality_probs = np.asarray(quality_probs, dtype=np.float32)
    else:
        quality_probs = None

    feat_list: List[str] = []
    if isinstance(feature_names, list):
        feat_list = [str(x) for x in feature_names]
    elif isinstance(feature_names, np.ndarray):
        feat_list = [str(x) for x in feature_names.tolist()]

    quality_conf = None
    if isinstance(quality_probs, np.ndarray) and quality_probs.ndim == 2 and quality_probs.size > 0:
        quality_conf = np.max(quality_probs.astype(np.float32), axis=1).astype(np.float32)

    sharpness = None
    if isinstance(frame_features, np.ndarray) and frame_features.ndim == 2 and feat_list:
        if "sharpness_tenengrad" in feat_list:
            sharpness = frame_features[:, feat_list.index("sharpness_tenengrad")].astype(np.float32)

    summary = {
        "frames_count": int(frame_indices.size) if isinstance(frame_indices, np.ndarray) else 0,
        "features_count": int(frame_features.shape[1]) if isinstance(frame_features, np.ndarray) and frame_features.ndim == 2 else 0,
        "shots_count": int(shot_start.size) if isinstance(shot_start, np.ndarray) else 0,
        "schema_version": str(meta.get("schema_version")) if isinstance(meta, dict) else None,
    }

    distributions = {}
    if isinstance(quality_conf, np.ndarray):
        distributions["quality_confidence"] = _stats(quality_conf)
    if isinstance(sharpness, np.ndarray):
        distributions["sharpness_tenengrad"] = _stats(sharpness)

    # top/anti-top frames by confidence
    top, anti = [], []
    if isinstance(quality_conf, np.ndarray) and isinstance(times_s, np.ndarray) and isinstance(frame_indices, np.ndarray):
        m = np.isfinite(quality_conf) & np.isfinite(times_s)
        idx = np.where(m)[0]
        if idx.size > 0:
            order = np.argsort(quality_conf[idx])[::-1]
            for j in idx[order[:10]].tolist():
                top.append({"frame_index": int(frame_indices[j]), "time_s": float(times_s[j]), "quality_conf": float(quality_conf[j])})
            for j in idx[order[-10:]].tolist():
                anti.append({"frame_index": int(frame_indices[j]), "time_s": float(times_s[j]), "quality_conf": float(quality_conf[j])})

    # shots table
    shots = []
    if isinstance(shot_start, np.ndarray) and isinstance(shot_end, np.ndarray):
        for sid in range(int(min(shot_start.size, shot_end.size))):
            shots.append({"shot_id": int(sid), "start_frame": int(shot_start[sid]), "end_frame": int(shot_end[sid])})

    return {
        "component": "shot_quality",
        "summary": summary,
        "distributions": distributions,
        "key_facts": [
            {"key": "frames", "value": summary.get("frames_count")},
            {"key": "shots", "value": summary.get("shots_count")},
            {"key": "features", "value": summary.get("features_count")},
            {"key": "schema_version", "value": summary.get("schema_version")},
        ],
        "top_examples": {"quality_confidence": top},
        "anti_top_examples": {"quality_confidence": anti},
        "shots": shots,
        "charts": {
            "times_s": times_s.tolist() if isinstance(times_s, np.ndarray) else [],
            "quality_confidence": quality_conf.tolist() if isinstance(quality_conf, np.ndarray) else [],
            "sharpness_tenengrad": sharpness.tolist() if isinstance(sharpness, np.ndarray) else [],
        },
        "stage_timings_ms": (meta.get("stage_timings_ms") if isinstance(meta, dict) else None),
        "config_highlights": {
            "preset": (meta.get("ui_payload", {}) or {}).get("preset") if isinstance(meta, dict) else None,
            "faces_available": (meta.get("impl_meta", {}) or {}).get("faces_available") if isinstance(meta, dict) else None,
        },
    }


def render_shot_quality_html(
    npz_path: str,
    output_path: str,
    frames_dir: str | None = None,
    assets_dir: str | None = None,
) -> str:
    import sys
    from pathlib import Path

    vp_root = Path(__file__).resolve().parent.parent.parent
    if str(vp_root) not in sys.path:
        sys.path.insert(0, str(vp_root))

    try:
        from utils.renderer import load_npz, extract_meta  # type: ignore
    except ImportError:
        def load_npz(path: str):
            data = np.load(path, allow_pickle=True)
            result = {}
            for key in data.files:
                arr = data[key]
                if isinstance(arr, np.ndarray):
                    if arr.dtype == object:
                        result[key] = arr.item() if arr.shape == () else arr.tolist()
                    else:
                        result[key] = arr.tolist() if arr.size > 0 else []
                else:
                    result[key] = arr
            return result

        def extract_meta(npz_data: Dict[str, Any]) -> Dict[str, Any]:
            meta = npz_data.get("meta")
            if isinstance(meta, np.ndarray) and meta.dtype == object:
                return meta.item() if meta.shape == () else meta.tolist()
            return meta if isinstance(meta, dict) else {}

    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    render = render_shot_quality(npz_data, meta)

    charts = render.get("charts") or {}
    times_s = np.asarray(charts.get("times_s") or [], dtype=np.float32).reshape(-1)
    qconf = np.asarray(charts.get("quality_confidence") or [], dtype=np.float32).reshape(-1)
    sharp = np.asarray(charts.get("sharpness_tenengrad") or [], dtype=np.float32).reshape(-1)

    charts_html = ""
    charts_html += _svg_line_chart(times_s=times_s, values=qconf, title="quality_confidence", stroke="#ef4444") + "\n"
    charts_html += _svg_line_chart(times_s=times_s, values=sharp, title="sharpness_tenengrad", stroke="#3b82f6") + "\n"

    shots_rows = ""
    for sh in (render.get("shots") or [])[:200]:
        shots_rows += f"<tr><td>{_esc(sh.get('shot_id'))}</td><td>{_esc(sh.get('start_frame'))}</td><td>{_esc(sh.get('end_frame'))}</td></tr>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ShotQuality Debug Render</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #111827; }}
    .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 1200px; margin: 0 auto; }}
    h1, h2 {{ color: #0f172a; }}
    .kv {{ display: grid; grid-template-columns: 260px 1fr; gap: 8px 14px; }}
    .kv div {{ padding: 6px 0; border-bottom: 1px solid #eef2f7; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
    .chart {{ margin: 14px 0; }}
    .chart-title {{ font-size: 14px; color: #111827; margin-bottom: 6px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #e5e7eb; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>ShotQuality Debug Render</h1>
    <h2>Key facts</h2>
    <div class="kv">
      {''.join([f"<div><strong>{_esc(k.get('key'))}</strong></div><div class='mono'>{_esc(k.get('value'))}</div>" for k in (render.get('key_facts') or [])])}
    </div>

    <h2>Charts (offline)</h2>
    {charts_html if charts_html.strip() else '<p>No chartable data</p>'}

    <h2>Top / Anti-top frames</h2>
    <div class="kv">
      <div><strong>Top confidence</strong></div><div class="mono">{_esc(json.dumps((render.get('top_examples') or {{}}).get('quality_confidence', []), ensure_ascii=False)[:2000])}</div>
      <div><strong>Anti-top confidence</strong></div><div class="mono">{_esc(json.dumps((render.get('anti_top_examples') or {{}}).get('quality_confidence', []), ensure_ascii=False)[:2000])}</div>
    </div>

    <h2>Shots (first 200)</h2>
    <table>
      <thead><tr><th>shot_id</th><th>start_frame</th><th>end_frame</th></tr></thead>
      <tbody>{shots_rows}</tbody>
    </table>

    <h2>Stage timings (ms)</h2>
    <pre class="mono">{_esc(json.dumps(render.get('stage_timings_ms') or {{}}, ensure_ascii=False, indent=2))}</pre>
  </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info("Saved shot_quality HTML render to %s", output_path)
    return output_path


__all__ = ["render_shot_quality", "render_shot_quality_html"]


