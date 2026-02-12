from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunRow:
    run_key: str
    video_path: str
    video_id: str
    run_id: str
    status: str  # success|error|queued|running
    run_rs_path: str
    out_dir: Optional[str] = None
    returncode: Optional[int] = None
    oom: Optional[bool] = None


def _safe_load_json(path: str) -> Optional[dict]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            x = json.load(f)
        return x if isinstance(x, dict) else None
    except Exception:
        return None


def _rel_link(from_dir: str, to_path: str) -> str:
    try:
        return os.path.relpath(to_path, start=from_dir)
    except Exception:
        return to_path


def build_index_html(
    *,
    out_base: str,
    rows: List[RunRow],
    scheduler_peaks: Optional[Dict[str, Any]] = None,
) -> str:
    out_base = os.path.abspath(str(out_base))
    Path(out_base).mkdir(parents=True, exist_ok=True)

    # Summaries
    n_total = len(rows)
    n_ok = sum(1 for r in rows if r.status == "success")
    n_err = sum(1 for r in rows if r.status != "success")

    # HTML
    parts: List[str] = []
    parts.append("<html><head><meta charset='utf-8'/>")
    parts.append("<title>DynamicBatch E2E report</title>")
    parts.append(
        "<style>body{font-family:ui-sans-serif,system-ui,Arial;margin:16px}"
        "table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:6px 8px}"
        "th{background:#f6f6f6;text-align:left}code{background:#f2f2f2;padding:2px 4px;border-radius:4px}"
        ".ok{color:#0a7}.err{color:#b00}</style>"
    )
    parts.append("</head><body>")
    parts.append("<h2>DynamicBatch E2E report</h2>")
    parts.append(f"<p>Total: <b>{n_total}</b> | success: <b class='ok'>{n_ok}</b> | error: <b class='err'>{n_err}</b></p>")

    if scheduler_peaks:
        parts.append("<h3>Scheduler resource peaks (global)</h3>")
        parts.append("<pre style='white-space:pre-wrap'>")
        parts.append(html.escape(json.dumps(scheduler_peaks, ensure_ascii=False, indent=2)))
        parts.append("</pre>")

    parts.append("<h3>Per-video runs</h3>")
    parts.append("<table>")
    parts.append(
        "<tr>"
        "<th>video_id</th><th>run_id</th><th>status</th>"
        "<th>links</th>"
        "</tr>"
    )

    for r in rows:
        st_cls = "ok" if r.status == "success" else "err"
        st = html.escape(str(r.status))
        vid = html.escape(r.video_id)
        rid = html.escape(r.run_id)

        links: List[str] = []
        rep_path = os.path.join(r.run_rs_path, "_reports", "scheduler_runtime_report.json")
        man_path = os.path.join(r.run_rs_path, "manifest.json")
        if os.path.exists(rep_path):
            links.append(f"<a href='{html.escape(_rel_link(out_base, rep_path))}'>runtime_report.json</a>")
        if os.path.exists(man_path):
            links.append(f"<a href='{html.escape(_rel_link(out_base, man_path))}'>manifest.json</a>")
        if r.out_dir:
            q = os.path.join(r.out_dir, "quality.html")
            v = os.path.join(r.out_dir, "validation_report.json")
            if os.path.exists(q):
                links.append(f"<a href='{html.escape(_rel_link(out_base, q))}'>quality.html</a>")
            if os.path.exists(v):
                links.append(f"<a href='{html.escape(_rel_link(out_base, v))}'>validation_report.json</a>")

        parts.append(
            "<tr>"
            f"<td>{vid}</td>"
            f"<td><code>{rid}</code></td>"
            f"<td class='{st_cls}'>{st}</td>"
            f"<td>{' | '.join(links) if links else ''}</td>"
            "</tr>"
        )

    parts.append("</table>")
    parts.append("</body></html>")

    return "\n".join(parts)


def write_index_html(
    *,
    out_base: str,
    rows: List[RunRow],
    scheduler_peaks: Optional[Dict[str, Any]] = None,
    filename: str = "index.html",
) -> str:
    html_text = build_index_html(out_base=os.path.abspath(out_base), rows=rows, scheduler_peaks=scheduler_peaks)
    out_path = os.path.join(os.path.abspath(out_base), str(filename))
    Path(out_path).write_text(html_text, encoding="utf-8")
    return out_path


