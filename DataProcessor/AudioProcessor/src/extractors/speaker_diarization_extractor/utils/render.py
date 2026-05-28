"""
Offline renderer for `speaker_diarization_extractor` (Audit v3).

- No external CDNs (offline-only).
- Reads strict NPZ v2 keys: feature vector + turn arrays + per-speaker arrays.
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


def render_speaker_diarization_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    feature_names = _to_list(npz_data.get("feature_names"))
    feature_values = _to_list(npz_data.get("feature_values"))
    features: Dict[str, Any] = {}
    for i, n in enumerate(feature_names):
        if i < len(feature_values):
            features[str(n)] = feature_values[i]

    return {
        "component": "speaker_diarization_extractor",
        "meta": {
            "status": meta.get("status"),
            "empty_reason": meta.get("empty_reason"),
            "schema_version": meta.get("schema_version"),
        },
        "summary": {
            "speaker_count": int(float(features.get("speaker_count", 0.0) or 0.0)),
            "duration_sec": float(features.get("duration_sec", 0.0) or 0.0),
            "speaker_balance_score": float(features.get("speaker_balance_score", float("nan"))),
            "dominant_speaker_id": int(float(features.get("dominant_speaker_id", -1.0) or -1.0)),
            "speaker_turns_count": int(float(features.get("speaker_turns_count", 0.0) or 0.0)),
            "speaker_transitions_count": int(float(features.get("speaker_transitions_count", 0.0) or 0.0)),
        },
        "turns": {
            "turn_start_sec": _to_list(npz_data.get("turn_start_sec")),
            "turn_end_sec": _to_list(npz_data.get("turn_end_sec")),
            "turn_speaker_id": _to_list(npz_data.get("turn_speaker_id")),
            "turn_mask": _to_list(npz_data.get("turn_mask")),
        },
        "speakers": {
            "speaker_ids": _to_list(npz_data.get("speaker_ids")),
            "speaker_time_ratio": _to_list(npz_data.get("speaker_time_ratio")),
            "speaker_duration_sec": _to_list(npz_data.get("speaker_duration_sec")),
            "speaker_turns_count_by_speaker": _to_list(npz_data.get("speaker_turns_count_by_speaker")),
        },
        "segment_axis": {
            "segment_start_sec": _to_list(npz_data.get("segment_start_sec")),
            "segment_end_sec": _to_list(npz_data.get("segment_end_sec")),
            "segment_center_sec": _to_list(npz_data.get("segment_center_sec")),
            "segment_mask": _to_list(npz_data.get("segment_mask")),
        },
    }


def render_speaker_diarization_extractor_html(npz_path: str, output_path: str) -> str:
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    ctx = render_speaker_diarization_extractor(npz_data, meta)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>speaker_diarization_extractor — debug</title>
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
    canvas {{ width: 100%; height: 220px; background: rgba(0,0,0,0.22); border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); }}
    .legend {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }}
    .tag {{ font-size: 12px; color: var(--muted); padding: 4px 8px; border: 1px solid rgba(255,255,255,0.12); border-radius: 999px; }}
    .row {{ display: grid; grid-template-columns: 1fr; gap: 16px; margin-top: 16px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>speaker_diarization_extractor</h1>
      <div class="meta" id="metaLine"></div>
      <div class="grid" id="kpis"></div>
    </div>

    <div class="row">
      <div class="card">
        <div class="meta">Speaker turns timeline</div>
        <canvas id="timeline"></canvas>
        <div class="legend" id="legend"></div>
      </div>
      <div class="card">
        <div class="meta">Per-speaker time ratio</div>
        <canvas id="ratios"></canvas>
      </div>
    </div>
  </div>

  <script>
    const ctx = {json.dumps(ctx)};
    const meta = ctx.meta || {{}};
    document.getElementById('metaLine').textContent =
      `status=${{meta.status}} schema=${{meta.schema_version}} empty_reason=${{meta.empty_reason}}`;

    const kpis = [
      ['speaker_count', ctx.summary?.speaker_count],
      ['duration_sec', ctx.summary?.duration_sec],
      ['balance_score', ctx.summary?.speaker_balance_score],
      ['dominant_id', ctx.summary?.dominant_speaker_id],
      ['turns_count', ctx.summary?.speaker_turns_count],
      ['transitions', ctx.summary?.speaker_transitions_count],
    ];
    const kpisEl = document.getElementById('kpis');
    for (const [k,v] of kpis) {{
      const d = document.createElement('div');
      d.className = 'kpi';
      d.innerHTML = `<div class="k">${{k}}</div><div class="v">${{(v ?? '—')}}</div>`;
      kpisEl.appendChild(d);
    }}

    function getColor(i) {{
      const palette = ['#6ae4ff','#ff6ad5','#ffd36a','#6aff9d','#a36aff','#ff6a6a','#6a9dff','#6afff3'];
      return palette[i % palette.length];
    }}

    function drawTimeline() {{
      const c = document.getElementById('timeline');
      const dpr = window.devicePixelRatio || 1;
      const w = Math.floor(c.clientWidth * dpr);
      const h = Math.floor(c.clientHeight * dpr);
      c.width = w; c.height = h;
      const g = c.getContext('2d');
      g.clearRect(0,0,w,h);

      const start = ctx.turns.turn_start_sec || [];
      const end = ctx.turns.turn_end_sec || [];
      const sid = ctx.turns.turn_speaker_id || [];
      const mask = ctx.turns.turn_mask || [];
      const dur = (ctx.summary?.duration_sec || 0) + 1e-9;

      g.strokeStyle = 'rgba(255,255,255,0.10)';
      g.lineWidth = 1;
      for (let i=0;i<=10;i++) {{
        const x = Math.round((i/10)*w);
        g.beginPath(); g.moveTo(x,0); g.lineTo(x,h); g.stroke();
      }}

      const barH = Math.max(10, Math.floor(h / 18));
      const padY = 16 * dpr;
      const y0 = padY;
      for (let i=0;i<start.length;i++) {{
        if (mask.length && !mask[i]) continue;
        const s = start[i], e = end[i];
        const sp = sid[i] ?? 0;
        const x1 = Math.max(0, Math.min(w, Math.floor((s / dur) * w)));
        const x2 = Math.max(0, Math.min(w, Math.floor((e / dur) * w)));
        const y = y0 + (sp * (barH + 6*dpr));
        g.fillStyle = getColor(sp);
        g.globalAlpha = 0.85;
        g.fillRect(x1, y, Math.max(1, x2-x1), barH);
      }}
      g.globalAlpha = 1.0;

      const speakerIds = ctx.speakers.speaker_ids || [];
      const leg = document.getElementById('legend');
      leg.innerHTML = '';
      for (const sp of speakerIds) {{
        const t = document.createElement('div');
        t.className = 'tag';
        t.style.borderColor = getColor(sp);
        t.textContent = `speaker_${{sp}}`;
        leg.appendChild(t);
      }}
    }}

    function drawRatios() {{
      const c = document.getElementById('ratios');
      const dpr = window.devicePixelRatio || 1;
      const w = Math.floor(c.clientWidth * dpr);
      const h = Math.floor(c.clientHeight * dpr);
      c.width = w; c.height = h;
      const g = c.getContext('2d');
      g.clearRect(0,0,w,h);

      const ids = ctx.speakers.speaker_ids || [];
      const ratios = ctx.speakers.speaker_time_ratio || [];
      const n = Math.max(1, ids.length);
      const pad = 18*dpr;
      const barW = (w - pad*2) / n;

      g.strokeStyle = 'rgba(255,255,255,0.10)';
      g.beginPath(); g.moveTo(pad, h-pad); g.lineTo(w-pad, h-pad); g.stroke();

      for (let i=0;i<ids.length;i++) {{
        const sp = ids[i];
        const r = Math.max(0, Math.min(1, ratios[i] ?? 0));
        const x = pad + i * barW + barW*0.1;
        const bw = barW*0.8;
        const bh = (h - pad*2) * r;
        g.fillStyle = getColor(sp);
        g.globalAlpha = 0.85;
        g.fillRect(x, (h-pad) - bh, bw, bh);
      }}
      g.globalAlpha = 1.0;
    }}

    drawTimeline();
    drawRatios();
    window.addEventListener('resize', () => {{ drawTimeline(); drawRatios(); }});
  </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path

