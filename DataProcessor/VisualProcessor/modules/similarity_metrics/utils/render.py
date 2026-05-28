"""
Renderer for similarity_metrics: render-context JSON + fully-offline HTML (no CDN).

Audit v3: prefer offline SVG charts; meta.ui_payload is the source for UI extras (top-K references).
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


def render_similarity_metrics(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    times_s = _as_float_array(npz_data.get("times_s"))
    frame_indices = _as_int_array(npz_data.get("frame_indices"))
    centroid_sims = _as_float_array(npz_data.get("centroid_sims"))
    temporal_sim_next = _as_float_array(npz_data.get("temporal_sim_next"))
    reference_present = npz_data.get("reference_present")

    # tabular features
    fn = npz_data.get("feature_names")
    fv = npz_data.get("feature_values")
    feat_names: List[str] = []
    feat_vals: List[float] = []
    try:
        if isinstance(fn, np.ndarray):
            feat_names = [str(x) for x in fn.astype(object).reshape(-1).tolist()]
        elif isinstance(fn, list):
            feat_names = [str(x) for x in fn]
        if isinstance(fv, np.ndarray):
            feat_vals = [float(x) for x in np.asarray(fv, dtype=np.float32).reshape(-1).tolist()]
        elif isinstance(fv, list):
            feat_vals = [float(x) for x in np.asarray(fv, dtype=np.float32).reshape(-1).tolist()]
    except Exception:
        feat_names, feat_vals = [], []

    features_dict = {feat_names[i]: feat_vals[i] for i in range(min(len(feat_names), len(feat_vals)))}

    ref_present_bool = False
    try:
        if isinstance(reference_present, np.ndarray):
            ref_present_bool = bool(reference_present.item())
        else:
            ref_present_bool = bool(reference_present)
    except Exception:
        ref_present_bool = False

    ui_payload = {}
    if isinstance(meta, dict):
        ui_payload = meta.get("ui_payload") if isinstance(meta.get("ui_payload"), dict) else {}

    summary = {
        "frames_count": int(frame_indices.size) if isinstance(frame_indices, np.ndarray) else 0,
        "reference_present": bool(ref_present_bool),
        "reference_set_id": ui_payload.get("reference_set_id"),
        "schema_version": str(meta.get("schema_version")) if isinstance(meta, dict) else None,
    }
    if isinstance(centroid_sims, np.ndarray):
        summary["centroid_sim_mean"] = features_dict.get("centroid_sim_mean")
        summary["centroid_sim_std"] = features_dict.get("centroid_sim_std")
    if isinstance(temporal_sim_next, np.ndarray):
        summary["temporal_sim_mean"] = features_dict.get("temporal_sim_mean")
        summary["temporal_sim_std"] = features_dict.get("temporal_sim_std")

    distributions = {}
    if isinstance(centroid_sims, np.ndarray):
        distributions["centroid_sims"] = _stats(centroid_sims)
    if isinstance(temporal_sim_next, np.ndarray):
        distributions["temporal_sim_next"] = _stats(temporal_sim_next)

    # Top/anti-top for centroid sims
    top, anti = [], []
    if isinstance(centroid_sims, np.ndarray) and isinstance(times_s, np.ndarray) and isinstance(frame_indices, np.ndarray):
        m = np.isfinite(centroid_sims) & np.isfinite(times_s)
        idx = np.where(m)[0]
        if idx.size > 0:
            order = np.argsort(centroid_sims[idx])[::-1]
            for j in idx[order[:10]].tolist():
                top.append({"frame_index": int(frame_indices[j]), "time_s": float(times_s[j]), "centroid_sim": float(centroid_sims[j])})
            for j in idx[order[-10:]].tolist():
                anti.append({"frame_index": int(frame_indices[j]), "time_s": float(times_s[j]), "centroid_sim": float(centroid_sims[j])})

    # Limit topk refs for render
    topk_refs = ui_payload.get("topk_refs") if isinstance(ui_payload.get("topk_refs"), list) else []
    topk_refs = topk_refs[:10]

    return {
        "component": "similarity_metrics",
        "summary": summary,
        "distributions": distributions,
        "key_facts": [
            {"key": "frames", "value": summary.get("frames_count")},
            {"key": "reference_present", "value": summary.get("reference_present")},
            {"key": "reference_set_id", "value": summary.get("reference_set_id")},
            {"key": "schema_version", "value": summary.get("schema_version")},
        ],
        "top_examples": {"centroid_sims": top},
        "anti_top_examples": {"centroid_sims": anti},
        "reference": {"topk_refs": topk_refs, "topk_count": int(len(topk_refs))},
        "charts": {
            "times_s": times_s.tolist() if isinstance(times_s, np.ndarray) else [],
            "centroid_sims": centroid_sims.tolist() if isinstance(centroid_sims, np.ndarray) else [],
            "temporal_sim_next": temporal_sim_next.tolist() if isinstance(temporal_sim_next, np.ndarray) else [],
        },
        "config_highlights": {
            "text_present": ui_payload.get("text_present"),
            "audio_required_present": ui_payload.get("audio_required_present"),
        },
        "stage_timings_ms": meta.get("stage_timings_ms") if isinstance(meta, dict) else None,
    }


def render_similarity_metrics_html(npz_path: str, output_path: str) -> str:
    import sys
    from pathlib import Path

    vp_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(vp_root / "modules") not in sys.path:
        sys.path.insert(0, str(vp_root / "modules"))

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
    render = render_similarity_metrics(npz_data, meta)

    charts = render.get("charts") or {}
    times_s = np.asarray(charts.get("times_s") or [], dtype=np.float32).reshape(-1)
    centroid = np.asarray(charts.get("centroid_sims") or [], dtype=np.float32).reshape(-1)
    temporal = np.asarray(charts.get("temporal_sim_next") or [], dtype=np.float32).reshape(-1)

    charts_html = ""
    charts_html += _svg_line_chart(times_s=times_s, values=centroid, title="centroid_sims", stroke="#ef4444") + "\n"
    # temporal has N-1 points; plot against times_s[1:]
    if times_s.size >= 2 and temporal.size == max(0, int(times_s.size) - 1):
        charts_html += _svg_line_chart(times_s=times_s[1:], values=temporal, title="temporal_sim_next", stroke="#3b82f6") + "\n"

    topk_refs = (render.get("reference") or {}).get("topk_refs") or []
    refs_json = _esc(json.dumps(topk_refs, ensure_ascii=False)[:4000])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SimilarityMetrics Debug Render</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #111827; }}
    .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 1200px; margin: 0 auto; }}
    h1, h2 {{ color: #0f172a; }}
    .kv {{ display: grid; grid-template-columns: 260px 1fr; gap: 8px 14px; }}
    .kv div {{ padding: 6px 0; border-bottom: 1px solid #eef2f7; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
    .chart {{ margin: 14px 0; }}
    .chart-title {{ font-size: 14px; color: #111827; margin-bottom: 6px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>SimilarityMetrics Debug Render</h1>

    <h2>Key facts</h2>
    <div class="kv">
      {''.join([f"<div><strong>{_esc(k.get('key'))}</strong></div><div class='mono'>{_esc(k.get('value'))}</div>" for k in (render.get('key_facts') or [])])}
    </div>

    <h2>Charts (offline)</h2>
    {charts_html if charts_html.strip() else '<p>No chartable data</p>'}

    <h2>Top / Anti-top frames</h2>
    <div class="kv">
      <div><strong>Top centroid_sims</strong></div><div class="mono">{_esc(json.dumps((render.get('top_examples') or {{}}).get('centroid_sims', []), ensure_ascii=False)[:2000])}</div>
      <div><strong>Anti-top centroid_sims</strong></div><div class="mono">{_esc(json.dumps((render.get('anti_top_examples') or {{}}).get('centroid_sims', []), ensure_ascii=False)[:2000])}</div>
    </div>

    <h2>Top-K reference matches (debug)</h2>
    <pre class="mono">{refs_json}</pre>

    <h2>Stage timings (ms)</h2>
    <pre class="mono">{_esc(json.dumps(render.get('stage_timings_ms') or {{}}, ensure_ascii=False, indent=2))}</pre>
  </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info("Saved similarity_metrics HTML render to %s", output_path)
    return output_path


__all__ = ["render_similarity_metrics", "render_similarity_metrics_html"]


