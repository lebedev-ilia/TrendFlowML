"""
Offline renderer for `spectral_entropy_extractor` (Audit v3).

- No external CDNs (offline-only).
- Reads strict NPZ v2 keys (no payload).
"""

import json
import logging
from typing import Any, Dict, List

import numpy as np

from ....core.renderer import extract_meta, load_npz

logger = logging.getLogger(__name__)


def _to_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return x
    return [x]


def render_spectral_entropy_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    feature_names = _to_list(npz_data.get("feature_names"))
    feature_values = _to_list(npz_data.get("feature_values"))
    features: Dict[str, Any] = {}
    for i, n in enumerate(feature_names):
        if i < len(feature_values):
            features[str(n)] = feature_values[i]

    return {
        "component": "spectral_entropy_extractor",
        "meta": {
            "status": meta.get("status"),
            "empty_reason": meta.get("empty_reason"),
            "schema_version": meta.get("schema_version"),
        },
        "summary": {
            "spectral_entropy_mean": float(features.get("spectral_entropy_mean", float("nan"))),
            "spectral_entropy_std": float(features.get("spectral_entropy_std", float("nan"))),
            "segments_count": int(len(_to_list(npz_data.get("segment_start_sec")))),
        },
        "axis": {
            "segment_start_sec": _to_list(npz_data.get("segment_start_sec")),
            "segment_end_sec": _to_list(npz_data.get("segment_end_sec")),
            "segment_center_sec": _to_list(npz_data.get("segment_center_sec")),
            "segment_mask": _to_list(npz_data.get("segment_mask")),
        },
        "per_segment": {
            "entropy_mean_by_segment": _to_list(npz_data.get("entropy_mean_by_segment")),
            "entropy_std_by_segment": _to_list(npz_data.get("entropy_std_by_segment")),
            "entropy_min_by_segment": _to_list(npz_data.get("entropy_min_by_segment")),
            "entropy_max_by_segment": _to_list(npz_data.get("entropy_max_by_segment")),
            "flatness_mean_by_segment": _to_list(npz_data.get("flatness_mean_by_segment")),
            "flatness_std_by_segment": _to_list(npz_data.get("flatness_std_by_segment")),
            "spread_mean_by_segment": _to_list(npz_data.get("spread_mean_by_segment")),
            "spread_std_by_segment": _to_list(npz_data.get("spread_std_by_segment")),
        },
    }


def render_spectral_entropy_extractor_html(npz_path: str, output_path: str) -> str:
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    ctx = render_spectral_entropy_extractor(npz_data, meta)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>spectral_entropy_extractor — debug</title>
  <style>
    :root {{
      --bg: #0b0f14;
      --panel: #101823;
      --text: #e6edf3;
      --muted: #8aa2b2;
    }}
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; background: var(--bg); color: var(--text); }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .card {{ background: var(--panel); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 16px; }}
    h1 {{ margin: 0 0 8px 0; font-size: 18px; }}
    .meta {{ color: var(--muted); font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }}
    .kpi {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px; }}
    .kpi .k {{ color: var(--muted); font-size: 11px; }}
    .kpi .v {{ font-size: 16px; margin-top: 4px; }}
    canvas {{ width: 100%; height: 260px; background: rgba(0,0,0,0.22); border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>spectral_entropy_extractor</h1>
      <div class="meta" id="metaLine"></div>
      <div class="grid" id="kpis"></div>
    </div>

    <div class="card" style="margin-top:16px;">
      <div class="meta">Entropy mean by segment (masked segments skipped)</div>
      <canvas id="plot"></canvas>
    </div>
  </div>

  <script>
    const ctx = {json.dumps(ctx)};
    const meta = ctx.meta || {{}};
    document.getElementById('metaLine').textContent =
      `status=${{meta.status}} schema=${{meta.schema_version}} empty_reason=${{meta.empty_reason}}`;

    const kpis = [
      ['entropy_mean', ctx.summary?.spectral_entropy_mean],
      ['entropy_std', ctx.summary?.spectral_entropy_std],
      ['segments_count', ctx.summary?.segments_count],
    ];
    const kpisEl = document.getElementById('kpis');
    for (const [k,v] of kpis) {{
      const d = document.createElement('div');
      d.className = 'kpi';
      d.innerHTML = `<div class="k">${{k}}</div><div class="v">${{(v ?? '—')}}</div>`;
      kpisEl.appendChild(d);
    }}

    function draw() {{
      const c = document.getElementById('plot');
      const dpr = window.devicePixelRatio || 1;
      const w = Math.floor(c.clientWidth * dpr);
      const h = Math.floor(c.clientHeight * dpr);
      c.width = w; c.height = h;
      const g = c.getContext('2d');
      g.clearRect(0,0,w,h);

      const xs = ctx.axis.segment_center_sec || [];
      const ys = ctx.per_segment.entropy_mean_by_segment || [];
      const mask = ctx.axis.segment_mask || [];
      const pts = [];
      for (let i=0;i<Math.min(xs.length, ys.length); i++) {{
        const ok = mask.length ? !!mask[i] : true;
        const y = ys[i];
        if (!ok) continue;
        if (y === null || y === undefined || Number.isNaN(+y) || !Number.isFinite(+y)) continue;
        pts.push([+xs[i], +y]);
      }}

      if (!pts.length) {{
        g.fillStyle = 'rgba(255,255,255,0.6)';
        g.font = `${{14*dpr}}px ui-sans-serif`;
        g.fillText('no valid segments to plot', 14*dpr, 24*dpr);
        return;
      }}

      let xmin = pts[0][0], xmax = pts[0][0], ymin = pts[0][1], ymax = pts[0][1];
      for (const [x,y] of pts) {{
        xmin = Math.min(xmin, x); xmax = Math.max(xmax, x);
        ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
      }}
      const pad = 18*dpr;
      const xspan = (xmax - xmin) || 1.0;
      const yspan = (ymax - ymin) || 1.0;

      // grid
      g.strokeStyle = 'rgba(255,255,255,0.10)';
      g.lineWidth = 1;
      for (let i=0;i<=10;i++) {{
        const xx = pad + (i/10) * (w - pad*2);
        g.beginPath(); g.moveTo(xx, pad); g.lineTo(xx, h-pad); g.stroke();
      }}
      for (let i=0;i<=6;i++) {{
        const yy = pad + (i/6) * (h - pad*2);
        g.beginPath(); g.moveTo(pad, yy); g.lineTo(w-pad, yy); g.stroke();
      }}

      function X(x) {{ return pad + ((x - xmin) / xspan) * (w - pad*2); }}
      function Y(y) {{ return (h - pad) - ((y - ymin) / yspan) * (h - pad*2); }}

      // line
      g.strokeStyle = '#6ae4ff';
      g.lineWidth = 2*dpr;
      g.beginPath();
      for (let i=0;i<pts.length;i++) {{
        const [x,y] = pts[i];
        const px = X(x), py = Y(y);
        if (i===0) g.moveTo(px, py); else g.lineTo(px, py);
      }}
      g.stroke();

      // points
      g.fillStyle = '#6ae4ff';
      for (const [x,y] of pts) {{
        const px = X(x), py = Y(y);
        g.beginPath(); g.arc(px, py, 3*dpr, 0, Math.PI*2); g.fill();
      }}
    }}

    draw();
    window.addEventListener('resize', draw);
  </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path

