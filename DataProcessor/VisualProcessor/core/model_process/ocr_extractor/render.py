"""
Renderer for ocr_extractor: render-context JSON + fully-offline HTML (no CDN).

Audit v3:
- HTML must work offline.
- OCR is privacy-sensitive: render should avoid dumping raw text by default; show counts and small samples.
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


def _unbox_object(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        try:
            return x.item()
        except Exception:
            return x
    return x


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


def _extract_text_len(row: Dict[str, Any]) -> int:
    # Try common keys; if absent, return 0.
    for k in ("text", "raw_text", "value", "ocr_text"):
        v = row.get(k)
        if isinstance(v, str):
            return len(v.strip())
    return 0


def render_ocr_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    frame_indices = _as_array(npz_data.get("frame_indices"), dtype=np.int32)
    times_s = _as_array(npz_data.get("times_s"), dtype=np.float32)
    ocr_raw = _unbox_object(npz_data.get("ocr_raw"))

    # Build per-frame counts best-effort.
    counts = None
    if isinstance(frame_indices, np.ndarray) and isinstance(ocr_raw, (list, tuple)):
        mapping = {int(fi): i for i, fi in enumerate(frame_indices.tolist())}
        c = np.zeros((int(frame_indices.size),), dtype=np.float32)
        for r in ocr_raw:
            if not isinstance(r, dict):
                continue
            fi = r.get("frame_index")
            if fi is None:
                continue
            j = mapping.get(int(fi), None)
            if j is None:
                continue
            c[int(j)] += 1.0
        counts = c

    # Privacy-safe “samples”: only lengths and confidences (no raw text).
    sample_rows: List[Dict[str, Any]] = []
    if isinstance(ocr_raw, (list, tuple)):
        for r in ocr_raw[:50]:
            if not isinstance(r, dict):
                continue
            sample_rows.append(
                {
                    "frame_index": r.get("frame_index"),
                    "t_s": r.get("t_s") or r.get("time_s") or r.get("time_sec"),
                    "text_len": _extract_text_len(r),
                    "conf": r.get("conf") or r.get("score") or r.get("confidence"),
                }
            )

    key_facts: Dict[str, Any] = {
        "frames_count": int(frame_indices.size) if isinstance(frame_indices, np.ndarray) else 0,
        "ocr_rows": int(len(ocr_raw)) if isinstance(ocr_raw, (list, tuple)) else 0,
    }
    if isinstance(counts, np.ndarray):
        key_facts.update({f"rows_per_frame_{k}": v for k, v in _stats(counts).items()})

    return {
        "component": "ocr_extractor",
        "schema_version": str(meta.get("schema_version") or ""),
        "producer_version": str(meta.get("producer_version") or ""),
        "status": str(meta.get("status") or ""),
        "key_facts": key_facts,
        "stage_timings_ms": meta.get("stage_timings_ms") if isinstance(meta.get("stage_timings_ms"), dict) else {},
        "samples": sample_rows,
        "_arrays": {
            "times_s": times_s,
            "rows_per_frame": counts,
        },
    }


def render_ocr_extractor_html(npz_path: str, output_path: str) -> str:
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
    ctx = render_ocr_extractor(npz_data, meta)

    arrays = ctx.get("_arrays") or {}
    times_s = arrays.get("times_s")
    if not isinstance(times_s, np.ndarray):
        times_s = np.asarray([], dtype=np.float32)
    rows_pf = arrays.get("rows_per_frame")

    charts = ""
    if isinstance(rows_pf, np.ndarray) and rows_pf.size > 0:
        charts = _svg_line_chart(times_s=times_s[: rows_pf.size], values=np.asarray(rows_pf, dtype=np.float32).reshape(-1), title="OCR rows per sampled frame", stroke="#2563eb")

    key_facts = ctx.get("key_facts") or {}
    stage_timings = ctx.get("stage_timings_ms") or {}
    status = _esc(ctx.get("status") or "")
    schema_version = _esc(ctx.get("schema_version") or "")
    producer_version = _esc(ctx.get("producer_version") or "")
    samples = ctx.get("samples") or []

    rows_html = ""
    if isinstance(samples, list) and samples:
        rows = []
        for r in samples[:50]:
            if not isinstance(r, dict):
                continue
            rows.append(
                "<tr>"
                f"<td>{_esc(r.get('frame_index'))}</td>"
                f"<td>{_esc(r.get('t_s'))}</td>"
                f"<td>{_esc(r.get('text_len'))}</td>"
                f"<td>{_esc(r.get('conf'))}</td>"
                "</tr>"
            )
        rows_html = "".join(rows)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>ocr_extractor — render</title>
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
    .banner {{ background:#fff7ed; border:1px solid #fed7aa; padding:10px 12px; border-radius:10px; color:#9a3412; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div style="font-size:18px; font-weight:700;">ocr_extractor</div>
      <div class="muted">schema: <code>{schema_version}</code> · producer_version: <code>{producer_version}</code> · status: <code>{status}</code></div>
    </div>

    <div class="card banner">
      <strong>Privacy note</strong>: this render intentionally avoids showing raw OCR text. Use internal tooling if you need raw inspection.
    </div>

    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Key facts</div>
      <div class="kv">
        {"".join([f"<div>{_esc(k)}</div><div>{_esc(v)}</div>" for k,v in key_facts.items()])}
      </div>
    </div>

    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Stage timings (ms)</div>
      <div class="kv">
        {"".join([f"<div>{_esc(k)}</div><div>{float(v):.2f}</div>" for k,v in stage_timings.items()])}
      </div>
    </div>

    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Curve</div>
      {charts or "<div class='muted'>No curve data</div>"}
    </div>

    <div class="card">
      <div style="font-weight:700; margin-bottom:10px;">Samples (privacy-safe)</div>
      {("<table><thead><tr><th>frame</th><th>t (s)</th><th>text_len</th><th>conf</th></tr></thead><tbody>"+rows_html+"</tbody></table>") if rows_html else "<div class='muted'>No samples</div>"}
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


__all__ = ["render_ocr_extractor", "render_ocr_extractor_html"]


