"""
Offline renderer for speech_analysis_extractor (Audit v3).

- No external CDNs (offline-only).
- Vanilla canvas for charts.
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


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return default
        return int(f)
    except (ValueError, TypeError):
        return default


def render_speech_analysis_extractor(npz_data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Render-context from NPZ (Audit v3: flat keys, no payload)."""
    feature_names = _to_list(npz_data.get("feature_names"))
    feature_values = _to_list(npz_data.get("feature_values"))
    features: Dict[str, Any] = {}
    for i, n in enumerate(feature_names):
        if i < len(feature_values):
            v = feature_values[i]
            features[str(n)] = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None

    # Extract features_enabled from meta
    extra = meta.get("extra", {})
    if isinstance(extra, np.ndarray) and extra.dtype == object and extra.size >= 1:
        extra = extra.item() if extra.size == 1 else (extra[0] if hasattr(extra, "__getitem__") else {})
    features_enabled = _to_list(extra.get("features_enabled", [])) if isinstance(extra, dict) else []
    if not features_enabled:
        features_enabled = _to_list(meta.get("features_enabled", []))

    asr_lang_distribution = npz_data.get("asr_lang_distribution")
    if isinstance(asr_lang_distribution, np.ndarray) and asr_lang_distribution.dtype == object:
        asr_lang_distribution = asr_lang_distribution.item() if asr_lang_distribution.size == 1 else {}
    asr_lang_distribution = asr_lang_distribution if isinstance(asr_lang_distribution, dict) else {}

    pitch_distribution = npz_data.get("pitch_distribution")
    if isinstance(pitch_distribution, np.ndarray) and pitch_distribution.dtype == object:
        pitch_distribution = pitch_distribution.item() if pitch_distribution.size == 1 else {}
    pitch_distribution = pitch_distribution if isinstance(pitch_distribution, dict) else {}

    return {
        "component": "speech_analysis_extractor",
        "meta": {
            "status": meta.get("status"),
            "empty_reason": meta.get("empty_reason"),
            "schema_version": meta.get("schema_version"),
        },
        "summary": {
            "duration_sec": _safe_float(features.get("duration_sec")),
            "sample_rate": _safe_int(features.get("sample_rate"), 16000),
        },
        "asr_metrics": {
            "segments_count": _safe_int(features.get("asr_segments_count")),
            "token_total": _safe_float(features.get("asr_token_total")),
            "token_mean": _safe_float(features.get("asr_token_mean")),
            "token_std": _safe_float(features.get("asr_token_std")),
            "token_density_per_sec": _safe_float(features.get("asr_token_density_per_sec")),
            "speech_rate_wpm": _safe_float(features.get("asr_speech_rate_wpm")),
            "lang_distribution": {str(k): _safe_float(v) for k, v in asr_lang_distribution.items()},
            "lang_id_by_segment": _to_list(npz_data.get("asr_lang_id_by_segment")),
        } if "asr_metrics" in features_enabled or "asr_segments_count" in features else {},
        "diarization_metrics": {
            "segments_count": _safe_int(features.get("diar_segments_count")),
            "speaker_count": _safe_int(features.get("speaker_count")),
            "dominant_speaker_share": _safe_float(features.get("dominant_speaker_share")),
            "speaker_balance_score": _safe_float(features.get("speaker_balance_score")),
            "speaker_transitions_count": _safe_int(features.get("speaker_transitions_count")),
            "speaker_ids": _to_list(npz_data.get("speaker_ids")),
        } if "diarization_metrics" in features_enabled or "speaker_count" in features else {},
        "pitch_metrics": {
            "enabled": bool(features.get("pitch_enabled")),
            "f0_mean": _safe_float(features.get("pitch_f0_mean")),
            "f0_std": _safe_float(features.get("pitch_f0_std")),
            "f0_min": _safe_float(features.get("pitch_f0_min")),
            "f0_max": _safe_float(features.get("pitch_f0_max")),
            "f0_range": _safe_float(features.get("pitch_f0_range")),
            "stability": _safe_float(features.get("pitch_stability")),
            "distribution": {str(k): _safe_float(v) for k, v in pitch_distribution.items()},
        },
        "features": features,
    }


def _draw_bar_chart(canvas_id: str, labels: List[str], values: List[float], title: str) -> str:
    """Generate inline script for bar chart."""
    return f"""
    function draw_{canvas_id}() {{
      const c = document.getElementById('{canvas_id}');
      if (!c) return;
      const dpr = window.devicePixelRatio || 1;
      const w = Math.floor(c.clientWidth * dpr);
      const h = Math.floor(c.clientHeight * dpr);
      c.width = w; c.height = h;
      const g = c.getContext('2d');
      g.clearRect(0,0,w,h);

      const labels = {json.dumps(labels)};
      const values = {json.dumps(values)};
      if (!values.length || values.every(v => !Number.isFinite(v))) {{
        g.fillStyle = 'rgba(255,255,255,0.6)';
        g.font = '14px ui-sans-serif';
        g.fillText('no data', 14, 24);
        return;
      }}

      const pad = 40;
      const maxVal = Math.max(...values.filter(v => Number.isFinite(v)), 0.001);
      const barW = Math.max(20, (w - pad * 2) / labels.length - 8);

      g.fillStyle = 'rgba(255,255,255,0.08)';
      g.font = '11px ui-sans-serif';
      g.fillText({json.dumps(title)}, pad, 18);

      for (let i = 0; i < labels.length; i++) {{
        const v = Number.isFinite(values[i]) ? values[i] : 0;
        const hBar = ((v / maxVal) * (h - pad - 30));
        const x = pad + i * (barW + 8);
        const y = h - pad - hBar;

        g.fillStyle = ['#667eea','#764ba2','#f093fb','#f5576c','#4facfe','#00f2fe'][i % 6];
        g.fillRect(x, y, barW, hBar);

        g.fillStyle = 'rgba(255,255,255,0.9)';
        g.textAlign = 'center';
        g.fillText(String(labels[i]).slice(0, 12), x + barW/2, h - 8);
      }}
    }}
"""


def _draw_timeline(canvas_id: str, ys: List[Any], title: str) -> str:
    """Generate inline script for segment timeline (lang_id_by_segment)."""
    return f"""
    function draw_{canvas_id}() {{
      const c = document.getElementById('{canvas_id}');
      if (!c) return;
      const dpr = window.devicePixelRatio || 1;
      const w = Math.floor(c.clientWidth * dpr);
      const h = Math.floor(c.clientHeight * dpr);
      c.width = w; c.height = h;
      const g = c.getContext('2d');
      g.clearRect(0,0,w,h);

      const ys = {json.dumps(ys)};
      if (!ys.length) {{
        g.fillStyle = 'rgba(255,255,255,0.6)';
        g.font = '14px ui-sans-serif';
        g.fillText('no data', 14, 24);
        return;
      }}

      const pad = 40;
      const uniq = [...new Set(ys.filter(y => y !== null && y !== undefined))];
      const colors = ['#667eea','#764ba2','#f093fb','#f5576c','#4facfe','#00f2fe'];
      const colorMap = {{}};
      uniq.forEach((u, i) => colorMap[u] = colors[i % colors.length]);

      g.fillStyle = 'rgba(255,255,255,0.08)';
      g.font = '11px ui-sans-serif';
      g.fillText({json.dumps(title)}, pad, 18);

      const step = (w - pad * 2) / Math.max(1, ys.length - 1);
      const maxY = Math.max(...uniq, 1);
      for (let i = 0; i < ys.length; i++) {{
        const y = ys[i];
        if (y === null || y === undefined) continue;
        const x = pad + i * step;
        const barH = ((y / maxY) * (h - pad - 30));
        const yPos = h - pad - barH;
        g.fillStyle = colorMap[y] || '#888';
        g.fillRect(x, yPos, Math.max(2, step - 2), barH);
      }}
    }}
"""


def render_speech_analysis_extractor_html(npz_path: str, output_path: str) -> str:
    """Offline HTML render (vanilla canvas, no CDN)."""
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    ctx = render_speech_analysis_extractor(npz_data, meta)

    meta_info = ctx.get("meta", {})
    summary = ctx.get("summary", {})
    asr = ctx.get("asr_metrics", {})
    diar = ctx.get("diarization_metrics", {})
    pitch = ctx.get("pitch_metrics", {})

    # Build charts scripts
    charts_script = ""
    charts_html = ""

    # Language distribution bar
    if asr.get("lang_distribution"):
        labels = list(asr["lang_distribution"].keys())
        values = list(asr["lang_distribution"].values())
        charts_html += """
    <div class="card" style="margin-top:16px;">
      <div class="meta">Language distribution</div>
      <canvas id="chartLangDist"></canvas>
    </div>"""
        charts_script += _draw_bar_chart("chartLangDist", labels, values, "Language distribution")

    # Language timeline
    if asr.get("lang_id_by_segment"):
        charts_html += """
    <div class="card" style="margin-top:16px;">
      <div class="meta">Language ID by segment</div>
      <canvas id="chartLangTimeline"></canvas>
    </div>"""
        charts_script += _draw_timeline("chartLangTimeline", asr["lang_id_by_segment"], "Language ID by segment")

    # Speaker distribution
    if diar.get("speaker_ids"):
        spk_counts: Dict[int, int] = {}
        for sid in diar["speaker_ids"]:
            spk_counts[sid] = spk_counts.get(sid, 0) + 1
        labels = [f"Spk {k}" for k in sorted(spk_counts.keys())]
        values = [spk_counts[k] for k in sorted(spk_counts.keys())]
        charts_html += """
    <div class="card" style="margin-top:16px;">
      <div class="meta">Speaker distribution</div>
      <canvas id="chartSpeaker"></canvas>
    </div>"""
        charts_script += _draw_bar_chart("chartSpeaker", labels, values, "Speaker distribution")

    # Pitch distribution
    if pitch.get("enabled") and pitch.get("distribution"):
        labels = list(pitch["distribution"].keys())
        values = list(pitch["distribution"].values())
        charts_html += """
    <div class="card" style="margin-top:16px;">
      <div class="meta">Pitch distribution by octave</div>
      <canvas id="chartPitch"></canvas>
    </div>"""
        charts_script += _draw_bar_chart("chartPitch", labels, values, "Pitch distribution")

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>speech_analysis_extractor — debug</title>
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
    h2 {{ font-size: 16px; margin: 16px 0 8px 0; }}
    .meta {{ color: var(--muted); font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; margin-top: 12px; }}
    .kpi {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px; }}
    .kpi .k {{ color: var(--muted); font-size: 11px; }}
    .kpi .v {{ font-size: 14px; margin-top: 4px; }}
    canvas {{ width: 100%; height: 220px; background: rgba(0,0,0,0.22); border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid rgba(255,255,255,0.12); padding: 8px; text-align: left; }}
    th {{ background: rgba(255,255,255,0.06); color: var(--muted); }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>speech_analysis_extractor</h1>
      <div class="meta" id="metaLine"></div>
      <div class="grid" id="kpis"></div>
    </div>
    {charts_html}
    <div class="card" style="margin-top:16px;">
      <h2>Key scalars</h2>
      <table id="scalars"></table>
    </div>
  </div>

  <script>
    const ctx = {json.dumps(ctx)};
    const meta = ctx.meta || {{}};
    document.getElementById('metaLine').textContent =
      `status=${{meta.status}} schema=${{meta.schema_version}} empty_reason=${{meta.empty_reason}} duration=${{ctx.summary?.duration_sec ?? '—'}}s`;

    const kpiKeys = ['duration_sec', 'sample_rate', 'asr_segments_count', 'asr_token_total', 'asr_token_density_per_sec', 'speaker_count', 'dominant_speaker_share', 'pitch_f0_mean'];
    const kpisEl = document.getElementById('kpis');
    const feats = ctx.features || {{}};
    for (const k of kpiKeys) {{
      const v = feats[k] ?? ctx.summary?.[k];
      if (v === undefined && v !== 0) continue;
      const d = document.createElement('div');
      d.className = 'kpi';
      const disp = (typeof v === 'number' && isNaN(v)) ? '—' : (v ?? '—');
      d.innerHTML = `<div class="k">${{k}}</div><div class="v">${{disp}}</div>`;
      kpisEl.appendChild(d);
    }}

    {charts_script}

    function runCharts() {{
      if (typeof draw_chartLangDist === 'function') draw_chartLangDist();
      if (typeof draw_chartLangTimeline === 'function') draw_chartLangTimeline();
      if (typeof draw_chartSpeaker === 'function') draw_chartSpeaker();
      if (typeof draw_chartPitch === 'function') draw_chartPitch();
    }}
    runCharts();
    window.addEventListener('resize', runCharts);

    const tbl = document.getElementById('scalars');
    const scalarKeys = Object.keys(ctx.features || {{}}).filter(k => !k.startsWith('_'));
    for (const k of scalarKeys) {{
      const tr = document.createElement('tr');
      const v = ctx.features[k];
      const disp = (typeof v === 'number' && isNaN(v)) ? '—' : (v ?? '—');
      tr.innerHTML = `<td>${{k}}</td><td>${{disp}}</td>`;
      tbl.appendChild(tr);
    }}
  </script>
</body>
</html>
"""

    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html)
    import os
    os.replace(tmp_path, output_path)

    logger.info(f"Saved speech_analysis HTML render to {output_path} (status={meta_info.get('status')})")
    return output_path


__all__ = ["render_speech_analysis_extractor", "render_speech_analysis_extractor_html"]
