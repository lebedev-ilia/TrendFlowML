"""
Renderer для optical_flow: генерация render-context JSON и HTML debug страницы.

Audit v3: fully-offline HTML (no CDN), includes key facts / top examples / config highlights / stage timings.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, Any, List, Optional

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
    stroke: str = "#10b981",
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


def _as_float_array(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=np.float32)
    return None


def render_optical_flow(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    motion = _as_float_array(npz_data.get("motion_norm_per_sec_mean"))
    times_s = _as_float_array(npz_data.get("times_s"))

    fn = npz_data.get("feature_names")
    fv = npz_data.get("feature_values")
    if isinstance(fn, np.ndarray) and fn.dtype == object:
        feat_names = [str(x) for x in fn.reshape(-1).tolist()]
    elif isinstance(fn, list):
        feat_names = [str(x) for x in fn]
    else:
        feat_names = []
    feat_vals = _as_float_array(fv)

    summary: Dict[str, Any] = {"frames_count": int(motion.size) if isinstance(motion, np.ndarray) else 0}
    if feat_names and isinstance(feat_vals, np.ndarray) and feat_vals.size == len(feat_names):
        for i, name in enumerate(feat_names):
            summary[str(name)] = float(feat_vals[i]) if np.isfinite(feat_vals[i]) else float("nan")

    # Distributions for curve (ignore first element)
    distributions: Dict[str, Any] = {}
    if isinstance(motion, np.ndarray) and motion.size > 0:
        curve_for_stats = motion[1:] if motion.size >= 2 else motion
        valid = curve_for_stats[np.isfinite(curve_for_stats)]
        if valid.size > 0:
            distributions["motion_norm_per_sec_mean"] = {
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
                "median": float(np.median(valid)),
                "p25": float(np.percentile(valid, 25)),
                "p75": float(np.percentile(valid, 75)),
                "p05": float(np.percentile(valid, 5)),
                "p95": float(np.percentile(valid, 95)),
            }

    # Top / anti-top examples
    top: List[Dict[str, Any]] = []
    anti: List[Dict[str, Any]] = []
    try:
        if isinstance(motion, np.ndarray) and isinstance(times_s, np.ndarray):
            m = np.isfinite(motion) & np.isfinite(times_s)
            idx = np.where(m)[0]
            if idx.size > 0:
                order = np.argsort(motion[idx])[::-1]
                for j in idx[order[:10]].tolist():
                    top.append({"time_s": float(times_s[j]), "motion": float(motion[j])})
                for j in idx[order[-10:]].tolist():
                    anti.append({"time_s": float(times_s[j]), "motion": float(motion[j])})
    except Exception:
        pass

    cfg = {}
    if isinstance(meta, dict) and "stage_timings_ms" in meta:
        cfg["stage_timings_ms"] = meta.get("stage_timings_ms")

    return {
        "component": "optical_flow",
        "summary": summary,
        "distributions": distributions,
        "key_facts": [
            {"key": "frames", "value": int(summary.get("frames_count", 0))},
            {"key": "missing_frame_ratio", "value": summary.get("missing_frame_ratio", None)},
        ],
        "top_examples": {"motion": top},
        "anti_top_examples": {"motion": anti},
        "config_highlights": cfg,
    }


def render_optical_flow_html(npz_path: str, output_path: str) -> str:
    # Import here to avoid circular imports
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
    render = render_optical_flow(npz_data, meta)

    times_s = np.asarray(npz_data.get("times_s") or [], dtype=np.float32).reshape(-1)
    motion = np.asarray(npz_data.get("motion_norm_per_sec_mean") or [], dtype=np.float32).reshape(-1)
    charts_html = ""
    charts_html += _svg_line_chart(times_s=times_s, values=motion, title="motion_norm_per_sec_mean (px/sec)", stroke="#ef4444") + "\n"

    # Optional: plot camera-motion curves from frame_feature_values
    try:
        ffn = np.asarray(npz_data.get("frame_feature_names") or [], dtype=object).reshape(-1)
        ffv = np.asarray(npz_data.get("frame_feature_values") or [], dtype=np.float32)
        if ffv.ndim == 2 and ffn.size == ffv.shape[1]:
            name_to_idx = {str(x): i for i, x in enumerate(ffn.tolist())}
            for nm, title, stroke in [
                ("cam_shake_std_norm", "cam_shake_std_norm", "#10b981"),
                ("cam_affine_rotation", "cam_affine_rotation (rad)", "#3b82f6"),
                ("cam_tx_per_sec_norm", "cam_tx_per_sec_norm", "#f59e0b"),
            ]:
                idx = name_to_idx.get(nm)
                if idx is None:
                    continue
                charts_html += _svg_line_chart(
                    times_s=times_s,
                    values=ffv[:, int(idx)].astype(np.float32),
                    title=title,
                    stroke=stroke,
                ) + "\n"
    except Exception:
        pass

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OpticalFlow Debug Render</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #111827; }}
    .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 1200px; margin: 0 auto; }}
    h1, h2, h3 {{ color: #0f172a; }}
    .kv {{ display: grid; grid-template-columns: 260px 1fr; gap: 8px 14px; }}
    .kv div {{ padding: 6px 0; border-bottom: 1px solid #eef2f7; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
    .chart {{ margin: 14px 0; }}
    .chart-title {{ font-size: 14px; color: #111827; margin-bottom: 6px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>OpticalFlow Debug Render</h1>

    <h2>Key facts</h2>
    <div class="kv">
      {''.join([f"<div><strong>{_esc(k.get('key'))}</strong></div><div class='mono'>{_esc(k.get('value'))}</div>" for k in (render.get('key_facts') or [])])}
    </div>

    <h2>Charts (offline)</h2>
    {charts_html if charts_html.strip() else '<p>No chartable data</p>'}

    <h2>Top / Anti-top examples</h2>
    <div class="kv">
      <div><strong>Top motion</strong></div><div class="mono">{_esc(json.dumps((render.get('top_examples') or {{}}).get('motion', []), ensure_ascii=False)[:2000])}</div>
      <div><strong>Anti-top motion</strong></div><div class="mono">{_esc(json.dumps((render.get('anti_top_examples') or {{}}).get('motion', []), ensure_ascii=False)[:2000])}</div>
    </div>

    <h2>Stage timings (ms)</h2>
    <pre class="mono">{_esc(json.dumps((render.get('config_highlights') or {{}}).get('stage_timings_ms') or meta.get('stage_timings_ms') or {{}}, ensure_ascii=False, indent=2))}</pre>
  </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Saved OpticalFlow HTML render to %s", output_path)
    return output_path


__all__ = ["render_optical_flow", "render_optical_flow_html"]


