"""
Renderer for uniqueness: render-context JSON + fully-offline HTML (no CDN).

Audit v3:
- feature_names/feature_values is the model-facing scalar source-of-truth (no object-dict `features`).
- meta.ui_payload contains UI hints (top repeats).
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


def render_uniqueness(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    times_s = _as_float_array(npz_data.get("times_s")) or np.asarray([], dtype=np.float32)
    max_sim = _as_float_array(npz_data.get("max_sim_to_other"))
    cos_dist_next = _as_float_array(npz_data.get("cos_dist_next"))

    feats = dict(_tabular_features(npz_data))
    ui_payload = meta.get("ui_payload") if isinstance(meta, dict) else None
    if not isinstance(ui_payload, dict):
        ui_payload = {}

    key_facts = {
        "n_frames": float(feats.get("n_frames", float("nan"))),
        "repetition_ratio": float(feats.get("repetition_ratio", float("nan"))),
        "diversity_score": float(feats.get("diversity_score", float("nan"))),
        "pairwise_sim_mean": float(feats.get("pairwise_sim_mean", float("nan"))),
        "pairwise_sim_p95": float(feats.get("pairwise_sim_p95", float("nan"))),
        "temporal_change_mean": float(feats.get("temporal_change_mean", float("nan"))),
        "repeat_threshold_used": float(feats.get("repeat_threshold_used", float("nan"))),
        "repeat_threshold_is_otsu": bool(feats.get("repeat_threshold_is_otsu", 0.0) >= 0.5),
    }

    return {
        "component": "uniqueness",
        "schema_version": str(meta.get("schema_version") or ""),
        "producer_version": str(meta.get("producer_version") or ""),
        "status": str(meta.get("status") or ""),
        "key_facts": key_facts,
        "config_highlights": {
            "repeat_threshold_mode": meta.get("repeat_threshold_mode"),
            "repeat_threshold": meta.get("repeat_threshold"),
            "repeat_threshold_min": meta.get("repeat_threshold_min"),
            "repeat_threshold_max": meta.get("repeat_threshold_max"),
            "repeat_threshold_bins": meta.get("repeat_threshold_bins"),
            "ui_topk": meta.get("ui_topk"),
            "max_frames": meta.get("max_frames"),
        },
        "stage_timings_ms": meta.get("stage_timings_ms") if isinstance(meta.get("stage_timings_ms"), dict) else {},
        "ui_payload": ui_payload,
        "distributions": {
            "max_sim_to_other": _stats(max_sim) if max_sim is not None else {},
            "cos_dist_next": _stats(cos_dist_next) if cos_dist_next is not None else {},
        },
        "_arrays": {
            "times_s": times_s,
            "max_sim_to_other": max_sim,
            "cos_dist_next": cos_dist_next,
        },
    }


def render_uniqueness_html(npz_path: str, output_path: str) -> str:
    """
    Generate fully-offline HTML debug page for uniqueness.
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
    ctx = render_uniqueness(npz_data, meta)

    arrays = ctx.get("_arrays") or {}
    times_s = arrays.get("times_s")
    if not isinstance(times_s, np.ndarray):
        times_s = np.asarray([], dtype=np.float32)

    charts: List[str] = []
    for key, title, color in [
        ("max_sim_to_other", "Max similarity to any other frame", "#0ea5e9"),
        ("cos_dist_next", "Cosine distance to next frame", "#ef4444"),
    ]:
        v = arrays.get(key)
        if isinstance(v, np.ndarray):
            vv = np.asarray(v, dtype=np.float32).reshape(-1)
            # cos_dist_next aligns to N-1; reuse times[:-1]
            tt = times_s[: vv.size] if vv.size <= times_s.size else times_s
            charts.append(_svg_line_chart(times_s=tt, values=vv, title=title, stroke=color))

    key_facts = ctx.get("key_facts") or {}
    stage_timings = ctx.get("stage_timings_ms") or {}
    cfg = ctx.get("config_highlights") or {}
    ui_payload = ctx.get("ui_payload") if isinstance(ctx.get("ui_payload"), dict) else {}

    status = _esc(ctx.get("status") or "")
    schema_version = _esc(ctx.get("schema_version") or "")
    producer_version = _esc(ctx.get("producer_version") or "")

    top_repeats = ui_payload.get("top_repeats") if isinstance(ui_payload, dict) else None
    top_unique = ui_payload.get("top_unique") if isinstance(ui_payload, dict) else None

    top_rows = ""
    if isinstance(top_repeats, list) and top_repeats:
        rows = []
        for it in top_repeats[:20]:
            if not isinstance(it, dict):
                continue
            rows.append(
                "<tr>"
                f"<td>{_esc(it.get('rank'))}</td>"
                f"<td>{_esc(it.get('t_s'))}</td>"
                f"<td>{_esc(it.get('frame_index'))}</td>"
                f"<td>{_esc(it.get('max_sim_to_other'))}</td>"
                "</tr>"
            )
        top_rows = "\n".join(rows)

    uniq_rows = ""
    if isinstance(top_unique, list) and top_unique:
        rows2 = []
        for it in top_unique[:20]:
            if not isinstance(it, dict):
                continue
            rows2.append(
                "<tr>"
                f"<td>{_esc(it.get('rank'))}</td>"
                f"<td>{_esc(it.get('t_s'))}</td>"
                f"<td>{_esc(it.get('frame_index'))}</td>"
                f"<td>{_esc(it.get('max_sim_to_other'))}</td>"
                "</tr>"
            )
        uniq_rows = "\n".join(rows2)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>uniqueness — render</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 20px; background: #f8fafc; color: #0f172a; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; }}
    .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin: 12px 0; }}
    .muted {{ color: #64748b; }}
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
      <div style="font-size:18px; font-weight:700;">uniqueness</div>
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

    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Top repeats (by max_sim_to_other)</div>
      {"<table><thead><tr><th>#</th><th>t (s)</th><th>frame</th><th>max_sim</th></tr></thead><tbody>"+top_rows+"</tbody></table>" if top_rows else "<div class='muted'>No top repeats</div>"}
    </div>

    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Top unique frames (anti-top by max_sim_to_other)</div>
      {"<table><thead><tr><th>#</th><th>t (s)</th><th>frame</th><th>max_sim</th></tr></thead><tbody>"+uniq_rows+"</tbody></table>" if uniq_rows else "<div class='muted'>No unique frames list</div>"}
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


