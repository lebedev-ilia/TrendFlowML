"""
Renderer for core_object_detections: render-context JSON + fully-offline HTML (no CDN).

Audit v3:
- HTML must work offline (no Chart.js CDN).
- Provide QA-friendly summary + simple curves + top classes.
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


def _as_array(x: Any, dtype=None) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, list):
        return np.asarray(x, dtype=dtype)
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=dtype) if dtype is not None else x
    return None


def _stats(arr: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
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


def _top_classes(class_ids: np.ndarray, valid_mask: np.ndarray, class_names: Optional[np.ndarray], topk: int = 12) -> List[Tuple[str, int]]:
    try:
        cid = np.asarray(class_ids, dtype=np.int32)
        vm = np.asarray(valid_mask, dtype=bool)
        flat = cid[vm]
        if flat.size == 0:
            return []
        uniq, cnt = np.unique(flat, return_counts=True)
        order = np.argsort(-cnt)
        out = []
        for u, c in zip(uniq[order][:topk].tolist(), cnt[order][:topk].tolist()):
            name = str(int(u))
            if isinstance(class_names, np.ndarray) and class_names.size > int(u):
                try:
                    name = str(class_names[int(u)])
                except Exception:
                    name = str(int(u))
            out.append((name, int(c)))
        return out
    except Exception:
        return []


def render_core_object_detections(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    times_s = _as_array(npz_data.get("times_s"), dtype=np.float32)
    det_count = _as_array(npz_data.get("det_count"), dtype=np.int32)
    person_count = _as_array(npz_data.get("person_count"), dtype=np.int32)
    text_region_count = _as_array(npz_data.get("text_region_count"), dtype=np.int32)
    logo_region_count = _as_array(npz_data.get("logo_region_count"), dtype=np.int32)

    class_ids = _as_array(npz_data.get("class_ids"), dtype=np.int32)
    valid_mask = npz_data.get("valid_mask")
    if isinstance(valid_mask, list):
        valid_mask = np.asarray(valid_mask, dtype=bool)
    elif isinstance(valid_mask, np.ndarray):
        valid_mask = np.asarray(valid_mask, dtype=bool)
    else:
        valid_mask = None

    class_names = npz_data.get("class_names")
    if isinstance(class_names, list):
        class_names = np.asarray(class_names)
    elif isinstance(class_names, np.ndarray):
        class_names = np.asarray(class_names)
    else:
        class_names = None

    top_classes = []
    if isinstance(class_ids, np.ndarray) and isinstance(valid_mask, np.ndarray):
        top_classes = _top_classes(class_ids, valid_mask, class_names, topk=12)

    key_facts: Dict[str, Any] = {
        "frames_count": int(times_s.size) if isinstance(times_s, np.ndarray) else 0,
        "total_detections_valid": int(np.sum(valid_mask)) if isinstance(valid_mask, np.ndarray) else 0,
    }

    return {
        "component": "core_object_detections",
        "schema_version": str(meta.get("schema_version") or ""),
        "producer_version": str(meta.get("producer_version") or ""),
        "status": str(meta.get("status") or ""),
        "key_facts": key_facts,
        "stage_timings_ms": meta.get("stage_timings_ms") if isinstance(meta.get("stage_timings_ms"), dict) else {},
        "top_classes": top_classes,
        "distributions": {
            "det_count": _stats(det_count) if isinstance(det_count, np.ndarray) else {},
            "person_count": _stats(person_count) if isinstance(person_count, np.ndarray) else {},
        },
        "_arrays": {
            "times_s": times_s,
            "det_count": det_count,
            "person_count": person_count,
            "text_region_count": text_region_count,
            "logo_region_count": logo_region_count,
        },
    }


def render_core_object_detections_html(npz_path: str, output_path: str) -> str:
    try:
        from utils.renderer import load_npz, extract_meta  # type: ignore
    except Exception:
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
    ctx = render_core_object_detections(npz_data, meta)

    arrays = ctx.get("_arrays") or {}
    times_s = arrays.get("times_s")
    if not isinstance(times_s, np.ndarray):
        times_s = np.asarray([], dtype=np.float32)

    charts: List[str] = []
    for key, title, color in [
        ("det_count", "Detections per frame", "#2563eb"),
        ("person_count", "Person count per frame", "#22c55e"),
        ("text_region_count", "Text regions per frame", "#a855f7"),
        ("logo_region_count", "Logo regions per frame", "#f97316"),
    ]:
        v = arrays.get(key)
        if isinstance(v, np.ndarray) and v.size > 0:
            charts.append(_svg_line_chart(times_s=times_s[: int(v.reshape(-1).size)], values=np.asarray(v, dtype=np.float32).reshape(-1), title=title, stroke=color))

    key_facts = ctx.get("key_facts") or {}
    stage_timings = ctx.get("stage_timings_ms") or {}
    status = _esc(ctx.get("status") or "")
    schema_version = _esc(ctx.get("schema_version") or "")
    producer_version = _esc(ctx.get("producer_version") or "")
    top_classes = ctx.get("top_classes") or []

    top_rows = ""
    if isinstance(top_classes, list) and top_classes:
        rows = []
        for name, cnt in top_classes:
            rows.append(f"<tr><td>{_esc(name)}</td><td>{int(cnt)}</td></tr>")
        top_rows = "".join(rows)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>core_object_detections — render</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 20px; background: #f8fafc; color: #0f172a; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; }}
    .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin: 12px 0; }}
    .muted {{ color: #64748b; }}
    .kv {{ display: grid; grid-template-columns: 1fr auto; gap: 8px 12px; }}
    .kv div {{ padding: 2px 0; border-bottom: 1px dashed #e2e8f0; }}
    .chart-title {{ font-size: 13px; font-weight: 600; margin: 0 0 6px 0; color: #334155; }}
    code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 6px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px; text-align: left; }}
    th {{ color: #334155; font-weight: 700; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div style="font-size:18px; font-weight:700;">core_object_detections</div>
      <div class="muted">schema: <code>{schema_version}</code> · producer_version: <code>{producer_version}</code> · status: <code>{status}</code></div>
    </div>

    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Key facts</div>
      <div class="kv">
        {"".join([f"<div>{_esc(k)}</div><div>{_esc(v)}</div>" for k,v in key_facts.items()])}
      </div>
    </div>

    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Top classes (valid detections)</div>
      {("<table><thead><tr><th>class</th><th>count</th></tr></thead><tbody>"+top_rows+"</tbody></table>") if top_rows else "<div class='muted'>No detections</div>"}
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


__all__ = ["render_core_object_detections", "render_core_object_detections_html"]


