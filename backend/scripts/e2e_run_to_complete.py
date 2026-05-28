#!/usr/bin/env python3
"""
E2E: создать run по YouTube URL и дождаться статуса completed (или failed).

Требования:
  - Backend API на --base-url (по умолчанию http://localhost:8001)
  - Fetcher API доступен (Backend вызывает его по TF_BACKEND_FETCHER_API_URL)
  - Fetcher Celery worker запущен (обрабатывает metadata → video → comments → finalize)
  - Celery beat + Backend worker для синхронизации статуса (sync_ingestion_run_status)
  - Для полного E2E (--with-dataprocessor): DataProcessor API + worker, TF_BACKEND_DATAPROCESSOR_API_URL.

Режимы:
  - Без --with-dataprocessor: выход по первому ingestion_status=completed (синк из Fetcher).
  - С --with-dataprocessor: ждём processing (задача process_ingestion_run), затем completed после DataProcessor.

Пример:
  cd backend && source .venv/bin/activate
  export TF_BACKEND_DB_DSN='postgresql+psycopg://trendflow:trendflow@localhost:5433/trendflow'
  python scripts/e2e_run_to_complete.py --source-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  python scripts/e2e_run_to_complete.py --source-url "..." --with-dataprocessor --fetcher-url http://localhost:8000
  # каждые 5 с: RAM, load1, GPU — --resource-snapshot-sec 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import httpx

_E2E_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_E2E_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_E2E_SCRIPTS_DIR))
from e2e_host_resources import host_resource_snapshot_dict, host_resource_snapshot_line


def _out(*args, **kwargs) -> None:
    """Print with flush so progress is visible in non-interactive runners."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def _append_e2e_event(path: Path | None, payload: dict) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def _json_safe_fingerprint(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return [_json_safe_fingerprint(x) for x in obj]
    if isinstance(obj, list):
        return [_json_safe_fingerprint(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe_fingerprint(v) for k, v in obj.items()}
    return obj


def _compact_dp_for_event(dp: dict | None) -> dict | None:
    """Компактное состояние DataProcessor для JSONL (без повторения огромных таблиц)."""
    if not dp:
        return None
    comps = dp.get("components") or {}
    if not isinstance(comps, dict):
        return None
    per_proc: dict[str, Any] = {}
    for proc, pdata in comps.items():
        if not isinstance(pdata, dict):
            continue
        pst = pdata.get("status")
        sub = pdata.get("components") or {}
        running: list[str] = []
        errors: list[dict[str, str]] = []
        if isinstance(sub, dict):
            for kn, kv in sorted(sub.items()):
                if not isinstance(kv, dict):
                    continue
                st = (kv.get("status") or "").strip().lower()
                if st == "running":
                    running.append(str(kn))
                elif st == "error":
                    br = _brief_component_error(kv)
                    errors.append(
                        {
                            "component": str(kn)[:120],
                            "hint": (br[:280] + "…") if len(br) > 283 else br,
                        }
                    )
        pe = str(pdata.get("error") or "").strip()
        per_proc[str(proc)] = {
            "status": pst,
            "running": running[:16],
            "errors": errors[:24],
            "processor_error": (pe[:400] + "…") if len(pe) > 403 else pe or None,
        }
    return {
        "status": dp.get("status"),
        "stage": dp.get("stage"),
        "current_processor": dp.get("current_processor"),
        "current_component": dp.get("current_component"),
        "overall": dp.get("overall"),
        "error": (str(dp.get("error") or "").strip()[:500] or None),
        "error_code": dp.get("error_code"),
        "processors": per_proc,
    }


def fetch_fetcher_progress(client: httpx.Client, fetcher_url: str, run_id: str) -> dict | None:
    """GET /api/v1/runs/{run_id} из Fetcher, вернуть progress или None."""
    url = f"{fetcher_url.rstrip('/')}/api/v1/runs/{run_id}"
    try:
        r = client.get(url, timeout=client.timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        progress = data.get("progress") or {}
        return {
            "stage": progress.get("stage"),
            "completed_stages": progress.get("completed_stages") or [],
            "total_stages": progress.get("total_stages"),
            "status": data.get("status"),
            "error": data.get("error"),
        }
    except Exception:
        return None


def fetch_dataprocessor_progress(
    client: httpx.Client,
    dataprocessor_url: str,
    run_id: str,
    api_key: str | None = None,
) -> dict | None:
    """GET /api/v1/runs/{run_id}/status from DataProcessor, return compact progress or None."""
    url = f"{dataprocessor_url.rstrip('/')}/api/v1/runs/{run_id}/status"
    headers = {"X-API-Key": api_key} if api_key else None
    try:
        r = client.get(
            url,
            headers=headers,
            params={"include_components": True},
            timeout=client.timeout,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        progress = data.get("progress") or {}
        cur_proc = progress.get("current_processor")
        return {
            "status": data.get("status"),
            "stage": data.get("stage") or cur_proc,
            "current_processor": cur_proc,
            "current_component": progress.get("current_component"),
            "overall": progress.get("overall"),
            "components": progress.get("components"),
            "updated_at": data.get("updated_at"),
            "started_at": data.get("started_at"),
            "error": data.get("error"),
            "error_code": data.get("error_code"),
        }
    except Exception:
        return None


def fetch_dataprocessor_health(
    client: httpx.Client,
    dataprocessor_url: str,
) -> dict | None:
    """
    GET /api/v1/health — метрики очереди (без API key).
    Нужен, чтобы в логе было видно «worker занят» (active_runs) и размер streams.
    """
    url = f"{dataprocessor_url.rstrip('/')}/api/v1/health"
    try:
        r = client.get(url, timeout=client.timeout)
        if r.status_code != 200:
            body = (r.text or "").strip().replace("\n", " ")
            if len(body) > 100:
                body = body[:97] + "…"
            return {"_error": f"HTTP {r.status_code}", "_detail": body}
        data = r.json()
        m = data.get("metrics") or {}
        return {
            "active_runs": m.get("active_runs"),
            "queue_length": m.get("queue_length"),
            "max_concurrent_runs": m.get("max_concurrent_runs"),
            "api_status": data.get("status"),
        }
    except Exception as e:
        return {"_error": str(e)[:120]}


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


_E2E_LINE_W = 76
_E2E_KEY_W = 18


def _strip_ansi(s: str) -> str:
    """Убираем ANSI из текста API (цвет логов в ошибках), чтобы таблица не «ломалась»."""
    if not s:
        return s
    return re.sub(r"\x1b\[[0-9;]*m", "", str(s))


def _wrap_text_to_width(text: str, width: int) -> list[str]:
    """Перенос по словам; очень длинные «слова» режутся."""
    t = _strip_ansi(text).replace("\n", " ").strip()
    if not t:
        return [""]
    lines: list[str] = []
    while t:
        if len(t) <= width:
            lines.append(t)
            break
        chunk = t[:width]
        cut = chunk.rfind(" ")
        if cut <= width // 4:
            cut = width
        lines.append(t[:cut].rstrip())
        t = t[cut:].lstrip()
    return lines


def _format_change_separator(prev_seconds: float) -> str:
    """Визуальный разделитель между двумя отличающимися состояниями опроса."""
    frag = f" change after {_fmt_duration(prev_seconds)} "
    dash = "─"
    target = max(20, _E2E_LINE_W - 2)
    if len(frag) >= target:
        return f"  {frag.strip()}"
    pad = target - len(frag)
    left = pad // 2
    right = pad - left
    line = f"{dash * left}{frag}{dash * right}"
    return f"  {line}"


def _e2e_box_horizontal(char: str = "─") -> str:
    n = max(8, _E2E_LINE_W - 4)
    return f"  ╭{char * n}╮"


def _e2e_box_bottom() -> str:
    n = max(8, _E2E_LINE_W - 4)
    return f"  ╰{'─' * n}╯"


def _e2e_box_row(key: str, value: str, *, key_w: int = _E2E_KEY_W) -> list[str]:
    """Одна или несколько строк таблицы key │ value (value с переносом)."""
    val_w = max(12, _E2E_LINE_W - 9 - key_w)
    key_p = _strip_ansi(key)[:key_w].ljust(key_w)
    wrapped = _wrap_text_to_width(str(value), val_w)
    out: list[str] = []
    for i, part in enumerate(wrapped):
        kcol = key_p if i == 0 else " " * key_w
        pad = val_w - len(part)
        out.append(f"  │ {kcol} │ {part}{' ' * max(0, pad)} │")
    return out


def _print_run_banner(
    *,
    run_id: str,
    source_url: str,
    base_url: str,
    fetcher_url: str,
    dataprocessor_url: str,
    with_dataprocessor: bool,
) -> None:
    w = 52
    line = "─" * w
    _out(line)
    _out("  E2E run")
    _out(f"  run_id      {run_id}")
    _out(f"  source      {source_url}")
    _out(f"  backend     {base_url}")
    if fetcher_url:
        _out(f"  fetcher     {fetcher_url}  (progress)")
    if with_dataprocessor and dataprocessor_url:
        _out(f"  dataproc    {dataprocessor_url}  (run /status)")
        _out(f"  dataproc Δ  {dataprocessor_url.rstrip('/')}/api/v1/health  (queue metrics)")
    _out(line)
    _out("")


def _fetcher_done(fetcher: dict | None) -> bool:
    if not fetcher:
        return False
    completed = fetcher.get("completed_stages") or []
    n = len(completed) if isinstance(completed, list) else 0
    total = fetcher.get("total_stages")
    return total is not None and n >= int(total)


def _fetcher_in_progress_line(fetcher: dict | None) -> str | None:
    if not fetcher or _fetcher_done(fetcher):
        return None
    completed = fetcher.get("completed_stages") or []
    n = len(completed) if isinstance(completed, list) else 0
    total = fetcher.get("total_stages")
    cur = fetcher.get("stage") or ""
    if total is not None:
        extra = f" · {cur}" if cur else ""
        return f"Fetcher {n}/{int(total)}{extra}"
    return f"Fetcher steps={n}"


def _processor_order() -> Tuple[str, ...]:
    return ("segmenter", "audio", "visual", "text")


# Коды для компактной строки пайплайна (~ running, + done, . wait, ! err, - skip)
_DP_PROC_SYM: dict[str, str] = {
    "running": "~",
    "waiting": ".",
    "queued": ".",
    "pending": ".",
    "success": "+",
    "skipped": "-",
    "empty": "-",
    "error": "!",
    "recovering": "%",
}


def _dp_pipeline_compact(dp: dict | None) -> str:
    """Короткая сводка по процессорам, чтобы queued не выглядел как 'пустой' статус."""
    if not dp:
        return ""
    comps = dp.get("components") or {}
    if not isinstance(comps, dict) or not comps:
        return ""
    order = list(_processor_order()) + [k for k in sorted(comps.keys()) if k not in _processor_order()]
    parts: list[str] = []
    for name in order:
        if name not in comps:
            continue
        pdata = comps[name]
        if not isinstance(pdata, dict):
            continue
        raw = (pdata.get("status") or "waiting").strip().lower()
        sym = _DP_PROC_SYM.get(raw, "?")
        parts.append(f"{name[:3]}{sym}")
    return " ".join(parts) if parts else ""


_TERMINAL_COMP_STATUS: frozenset[str] = frozenset(
    {"success", "empty", "skipped", "ok", "error"}
)


def _parse_iso_dt(raw: Any) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        s = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _component_duration_label(kv: dict[str, Any]) -> str:
    """Wall time for finished components; live elapsed for running (from started_at)."""
    ms = kv.get("duration_ms")
    if isinstance(ms, (int, float)) and float(ms) >= 0:
        sec = float(ms) / 1000.0
        return f"{sec:.2f}s" if sec < 120 else _fmt_duration(sec)
    started = _parse_iso_dt(kv.get("started_at"))
    finished = _parse_iso_dt(kv.get("finished_at"))
    if started and finished:
        sec = max(0.0, (finished - started).total_seconds())
        return f"{sec:.2f}s" if sec < 120 else _fmt_duration(sec)
    st = (kv.get("status") or "").strip().lower()
    if st == "running" and started:
        sec = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
        return f"…{_fmt_duration(sec)}"
    return "—"


def _component_duration_sort_seconds(kv: dict[str, Any]) -> float:
    """Seconds for sorting subcomponent rows (larger = slower). Unknown → 0."""
    ms = kv.get("duration_ms")
    if isinstance(ms, (int, float)) and float(ms) >= 0:
        return float(ms) / 1000.0
    started = _parse_iso_dt(kv.get("started_at"))
    finished = _parse_iso_dt(kv.get("finished_at"))
    if started and finished:
        return max(0.0, (finished - started).total_seconds())
    st = (kv.get("status") or "").strip().lower()
    if st == "running" and started:
        return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
    return 0.0


def _normalize_sub_status(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s == "ok":
        return "success"
    return s or "—"


_GENERIC_SUB_ERROR_CODES = frozenset({"", "non_zero_exit", "exception", "component_failed"})


def _short_exc_label(name: str, *, max_len: int = 32) -> str:
    n = (name or "").strip()
    if n.endswith("Error"):
        n = n[:-5]
    if len(n) > max_len:
        return n[: max_len - 1] + "…"
    return n


def _brief_component_error(kv: dict) -> str:
    """Краткая причина для строки subcomponents (класс исключения без суффикса Error или error_code)."""
    if not isinstance(kv, dict):
        return ""
    text = str(kv.get("error") or "")
    code = str(kv.get("error_code") or "").strip()

    if "EmbeddingServiceUnavailableError" in text or "EmbeddingServiceUnavailable" in text:
        return "EmbeddingServiceUnavailable"
    if "embedding service unavailable" in text.lower():
        return "EmbeddingServiceUnavailable"

    for line in reversed(text.strip().split("\n")[-24:]):
        line = line.strip()
        m = re.match(r"^([\w.]+Error)\s*:", line)
        if m:
            return _short_exc_label(m.group(1).split(".")[-1])

    first = re.sub(r"\s+", " ", text).strip()
    if first:
        # Хвост subprocess часто уводит в успешный [INFO] (render saved) при реальной ошибке раньше.
        m_exit = re.match(r"^(exit=\d+):\s*(.+)$", first)
        if m_exit and re.search(
            r"\[INFO\].*(?:HTML render|saved to|Cosine metrics|render saved)",
            m_exit.group(2),
            re.I,
        ):
            return f"{m_exit.group(1)} (non_zero_exit; см. полный лог DP — в API попал INFO-хвост)"
        return (first[:200] + "…") if len(first) > 203 else first

    if code and code not in _GENERIC_SUB_ERROR_CODES:
        return _short_exc_label(code, max_len=28)
    return ""


def _sub_status_display(kv: dict) -> str:
    st = _normalize_sub_status(str(kv.get("status") or ""))
    if st != "error":
        return st
    br = _brief_component_error(kv)
    return f"error ({br})" if br else st


def _processor_row_display(pdata: dict) -> str:
    """Строка статуса процессора, если в API нет подкомпонентов."""
    st = _normalize_sub_status(str(pdata.get("status") or ""))
    if st != "error":
        return st
    br = _brief_component_error(
        {"error": pdata.get("error"), "error_code": pdata.get("error_code")}
    )
    return f"error ({br})" if br else st


def _dp_subcomponent_fingerprint(dp: dict | None) -> tuple[Any, ...]:
    """
    Сводка подкомпонент для fingerprint (без wall-clock в подписи).
    Учитывает running/recovering/success/error и «пустой» running (ожидание manifest).
    Плюс current_processor / current_component из API.
    """
    api_tail = (None, None)
    if not dp:
        return ((), api_tail)
    api_tail = (dp.get("current_processor"), dp.get("current_component"))
    comps = dp.get("components") or {}
    if not isinstance(comps, dict):
        return ((), api_tail)
    order = list(_processor_order()) + [k for k in sorted(comps.keys()) if k not in _processor_order()]
    blocks: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    interesting = frozenset({"running", "recovering", "success", "error"})
    for proc in order:
        if proc not in comps:
            continue
        pdata = comps[proc]
        if not isinstance(pdata, dict):
            continue
        proc_st = (pdata.get("status") or "").strip().lower()
        if proc_st not in interesting:
            continue
        sub = pdata.get("components") or {}
        if not isinstance(sub, dict):
            sub = {}
        if not sub and proc_st in ("running", "recovering"):
            blocks.append((proc, tuple()))
            continue
        if not sub and proc_st == "error":
            blocks.append((proc, (("(processor)", _processor_row_display(pdata)),)))
            continue
        if not sub and proc_st in ("success", "skipped", "empty"):
            blocks.append((proc, (("(processor)", _processor_row_display(pdata)),)))
            continue
        if not sub:
            continue
        rows: list[tuple[str, str]] = []
        for kn in sorted(sub.keys()):
            kv = sub[kn]
            if not isinstance(kv, dict):
                continue
            rows.append((str(kn), _sub_status_display(kv)))
        blocks.append((proc, tuple(rows)))
    return (tuple(blocks), api_tail)


_PROC_SHOW_SUB: frozenset[str] = frozenset({"running", "recovering", "success", "error"})


def _processor_time_bracket(pdata: dict) -> str:
    ms = pdata.get("duration_ms")
    if isinstance(ms, (int, float)) and float(ms) >= 0:
        return f"{float(ms) / 1000.0:.1f}s"
    d = _component_duration_label(pdata)
    return d if d != "—" else ""


def _format_dp_component_tables(dp: dict | None) -> list[str]:
    """Таблицы подкомпонентов; при пустом sub — блок с ошибкой процессора или пояснение про manifest."""
    if not dp:
        return []
    comps = dp.get("components") or {}
    if not isinstance(comps, dict):
        return []
    order = list(_processor_order()) + [k for k in sorted(comps.keys()) if k not in _processor_order()]
    out: list[str] = []
    w_name, w_time = 38, 12

    for proc in order:
        if proc not in comps:
            continue
        pdata = comps[proc]
        if not isinstance(pdata, dict):
            continue
        proc_st = (pdata.get("status") or "").strip().lower()
        if proc_st not in _PROC_SHOW_SUB:
            continue
        sub = pdata.get("components") or {}
        if not isinstance(sub, dict):
            sub = {}
        ptime = _processor_time_bracket(pdata)
        paren = f" ({ptime})" if ptime else ""
        rows: list[tuple[str, str, str, int, float]] = []
        if not sub:
            if proc_st in ("error", "success", "skipped", "empty"):
                pst = _normalize_sub_status(str(pdata.get("status") or ""))
                pr = 0 if pst == "running" else 1
                _pd = pdata if isinstance(pdata, dict) else {}
                rows.append(
                    (
                        "(processor)",
                        _processor_row_display(pdata),
                        _component_duration_label(_pd),
                        pr,
                        _component_duration_sort_seconds(_pd),
                    )
                )
        else:
            for kn in sorted(sub.keys()):
                kv = sub[kn]
                if not isinstance(kv, dict):
                    continue
                name = str(kn)
                raw_st = _normalize_sub_status(str(kv.get("status") or ""))
                st_disp = _sub_status_display(kv)
                dur = _component_duration_label(kv)
                pri = 0 if raw_st == "running" else 1
                sec = _component_duration_sort_seconds(kv)
                rows.append((name, st_disp, dur, pri, sec))
        if not rows:
            continue
        rows.sort(key=lambda r: (r[3], -r[4], r[0].lower()))
        w_stat = max(12, min(46, max((len(_strip_ansi(r[1])) for r in rows), default=12)))

        inner = w_name + 2 + w_stat + 2 + w_time
        bar_w = inner + 2
        title = f"{proc} · subcomponents{paren} · processor={proc_st}"
        title_vis = _strip_ansi(title)
        if len(title_vis) > inner:
            title_vis = title_vis[: max(12, inner - 1)] + "…"
        block = [
            f"  ╭{'─' * bar_w}╮",
            f"  │ {title_vis.ljust(inner)} │",
            f"  ├{'─' * bar_w}┤",
            f"  │ {'component':<{w_name}}  {'status':<{w_stat}}  {'time':>{w_time}} │",
            f"  ├{'─' * bar_w}┤",
        ]
        for name, st, dur, _pri, _sec in rows:
            st_clean = _strip_ansi(st)
            block.append(f"  │ {name:<{w_name}}  {st_clean:<{w_stat}}  {dur:>{w_time}} │")
        block.append(f"  ╰{'─' * bar_w}╯")
        out.append("\n".join(block))
    return out


def _dp_updated_age_note(dp: dict | None) -> str | None:
    """Сколько времени прошло с последнего updated_at от DataProcessor (живость состояния)."""
    raw = dp.get("updated_at") if dp else None
    if not raw or not isinstance(raw, str):
        return None
    try:
        s = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
        msg = f"DP updated {_fmt_duration(age)} ago"
        # state_reader отдаёт updated_at из run_state.json; пока worker ждёт subprocess
        # VisualProcessor, поле часто не меняется — детали идут из manifest в components.
        st = str((dp.get("stage") or "")).strip().lower()
        curp = str((dp.get("current_processor") or "")).strip().lower()
        if st == "visual" or curp == "visual":
            msg += " — при VP это нормально (см. лог VisualProcessor/worker)"
        return msg
    except Exception:
        return "DP updated_at ?"


def _dp_activity(dp: dict | None) -> tuple[str, str, str]:
    """
    Returns (status, processor_stage, work_hint).
    work_hint: одна короткая строка для основной линии; детали — в таблицах (_format_dp_component_tables).
    """
    if not dp:
        return "—", "—", "—"
    err = dp.get("error") or dp.get("error_code")
    if err:
        snippet = str(err).strip().replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:177] + "…"
        return (
            (dp.get("status") or "—").strip(),
            (dp.get("stage") or "—").strip(),
            f"error: {snippet}",
        )
    st = (dp.get("status") or "—").strip()
    stage = (dp.get("stage") or "—").strip()
    top = dp.get("current_component")
    comps = dp.get("components") or {}
    if not isinstance(comps, dict):
        comps = {}
    order = list(_processor_order()) + [k for k in sorted(comps.keys()) if k not in _processor_order()]
    bits: list[str] = []
    err_bits: list[str] = []
    for proc in order:
        if proc not in comps:
            continue
        pdata = comps[proc]
        if not isinstance(pdata, dict):
            continue
        pst = (pdata.get("status") or "").strip().lower()
        if pst == "error":
            msg = (pdata.get("error") or "").strip().replace("\n", " ")
            code = (str(pdata.get("error_code") or "")).strip()
            if not msg and not code:
                sub = pdata.get("components") or {}
                if isinstance(sub, dict):
                    for _kn, kv in sorted(sub.items()):
                        if not isinstance(kv, dict):
                            continue
                        if (kv.get("status") or "").strip().lower() != "error":
                            continue
                        hint = _brief_component_error(kv)
                        if hint:
                            msg = hint
                            break
                        msg = (kv.get("error") or "").strip().replace("\n", " ")
                        code = (str(kv.get("error_code") or "")).strip()
                        if msg or code:
                            break
            if len(msg) > 52:
                msg = msg[:49] + "…"
            if msg and code:
                err_bits.append(f"{proc} error: {msg} ({code})")
            elif msg:
                err_bits.append(f"{proc} error: {msg}")
            elif code:
                err_bits.append(f"{proc} error ({code})")
            else:
                err_bits.append(f"{proc} error (no message in API)")
            continue
        if pst != "running":
            continue
        sub = pdata.get("components") or {}
        n_run = n_done = 0
        if isinstance(sub, dict):
            for kn, kv in sorted(sub.items()):
                if not isinstance(kv, dict):
                    continue
                s = (kv.get("status") or "").strip().lower()
                if s == "running":
                    n_run += 1
                elif s in _TERMINAL_COMP_STATUS:
                    n_done += 1
        if isinstance(sub, dict) and sub:
            bits.append(f"{proc}: {n_run} running · {n_done} finished")
        else:
            bits.append(f"{proc}: running")
    merged = err_bits + bits
    if merged:
        work = " · ".join(merged)
    elif top:
        work = str(top)
    else:
        pipe = _dp_pipeline_compact(dp)
        st_l = st.lower()
        if pipe:
            if st_l in ("queued", "pending", "recovering"):
                work = f"{pipe} · (pipeline idle — worker/API not advancing yet)"
            else:
                work = pipe
        elif st_l in ("queued", "pending"):
            work = "(no processor snapshot yet — run registering or enqueue in progress)"
        else:
            work = "…"
    return st, stage, work


def _overall_pct(dp: dict | None) -> str:
    if not dp:
        return "—"
    ov = dp.get("overall")
    if isinstance(ov, (int, float)):
        v = float(ov)
        return f"{v * 100:.0f}%" if v <= 1.0 else f"{v:.0f}%"
    return "—"


def _progress_fingerprint(
    *,
    ingestion: str,
    fetcher: dict | None,
    dp: dict | None,
    backend_stage: str,
    dp_health: dict | None,
) -> Tuple[Any, ...]:
    """Emit when this changes (or on heartbeat timer)."""
    fp_completed_n = 0
    fp_total = None
    fp_cur = None
    fetcher_finished = _fetcher_done(fetcher)
    if fetcher and not fetcher_finished:
        cs = fetcher.get("completed_stages") or []
        fp_completed_n = len(cs) if isinstance(cs, list) else 0
        fp_total = fetcher.get("total_stages")
        fp_cur = fetcher.get("stage") or ""
    dp_status = dp.get("status") if dp else None
    dp_stage = dp.get("stage") if dp else None
    sub_fp = _dp_subcomponent_fingerprint(dp)
    dp_err = (dp.get("error") or dp.get("error_code") or "") if dp else ""
    h_ar = None
    h_api = h_err = ""
    if dp_health:
        if dp_health.get("_error"):
            h_err = str(dp_health.get("_error") or "")
            if dp_health.get("_detail"):
                h_err = f"{h_err}: {dp_health['_detail'][:100]}"
        else:
            h_ar = dp_health.get("active_runs")
            h_api = str(dp_health.get("api_status") or "")
    return (
        ingestion,
        backend_stage or "",
        fetcher_finished,
        fp_completed_n if not fetcher_finished else -1,
        fp_total,
        fp_cur or "",
        dp_status or "",
        dp_stage or "",
        sub_fp,
        dp_err,
        h_ar,
        h_err[:120] if h_err else "",
        h_api,
    )


def _format_dp_health_tail(dp: dict | None, dp_health: dict | None) -> str | None:
    """Только ошибка /health или короткая подсказка по очереди (без max_concurrent и stream_msgs)."""
    if not dp_health:
        return None
    if dp_health.get("_error"):
        detail = dp_health.get("_detail") or ""
        msg = f"/health недоступен: {dp_health['_error']}"
        if detail:
            msg += f" ({detail[:60]}{'…' if len(detail) > 60 else ''})"
        return msg
    st = (dp.get("status") or "").lower() if dp else ""
    if st not in ("queued", "pending", "recovering"):
        return None
    ar = dp_health.get("active_runs")
    mx = dp_health.get("max_concurrent_runs")
    if isinstance(ar, int) and ar >= 1 and isinstance(mx, int) and ar >= mx:
        return "queue — лимит concurrent (ждём слот)"
    if isinstance(ar, int) and ar >= 1:
        return "queue — worker занят другим run; этот run ждёт"
    return "queue — ожидание worker / accept в API"


def _dp_live_digest(dp: dict | None, fetcher: dict | None) -> str:
    """Сводка стадий (Fetcher run status + DP; heartbeat отличимее при долгом fetch_comments)."""
    parts: list[str] = []
    if fetcher:
        if _fetcher_done(fetcher):
            parts.append("fetch_run=DONE")
        else:
            fs = (fetcher.get("stage") or "").strip()
            if fs:
                parts.append(f"fetch_stage={fs}")
            frs = (fetcher.get("status") or "").strip()
            if frs:
                parts.append(f"fetch_api={frs}")
            cs = fetcher.get("completed_stages") or []
            tot = fetcher.get("total_stages")
            if isinstance(cs, list) and tot is not None:
                try:
                    parts.append(f"fetch_steps={len(cs)}/{int(tot)}")
                except (TypeError, ValueError):
                    pass
            if isinstance(cs, list) and cs:
                parts.append(f"fetch_last={cs[-1]}")
            err = fetcher.get("error")
            if err:
                e = str(err).strip().replace("\n", " ")
                parts.append(f"fetch_err={e[:48]}{'…' if len(e) > 48 else ''}")
    if not dp:
        return " · ".join(parts)
    cp = dp.get("current_processor")
    cc = dp.get("current_component")
    if cp:
        parts.append(f"DP.proc={cp}")
    stg = (dp.get("stage") or "").strip()
    if stg:
        parts.append(f"DP.stage={stg}")
    if cc:
        parts.append(f"DP.comp={cc}")
    comps = dp.get("components") or {}
    if isinstance(comps, dict):
        for proc in _processor_order():
            pdata = comps.get(proc)
            if not isinstance(pdata, dict):
                continue
            sub = pdata.get("components") or {}
            if not isinstance(sub, dict):
                continue
            for kn in sorted(sub.keys()):
                kv = sub[kn]
                if not isinstance(kv, dict):
                    continue
                if (kv.get("status") or "").strip().lower() != "running":
                    continue
                started = _parse_iso_dt(kv.get("started_at"))
                if started:
                    sec = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
                    parts.append(f"{kn}={int(sec)}s")
                else:
                    parts.append(f"{kn}=running")
                break
    return " · ".join(parts)


def _format_status_core_text(
    *,
    ingestion_note: str | None,
    fetcher_note: str | None,
    dp: dict | None,
    fetcher: dict | None,
    dp_health: dict | None,
    stage_elapsed: float | None,
    heartbeat: bool,
) -> str:
    """
    Стабильная строка содержимого (без wall clock) — для сравнения heartbeat,
    если визуальный блок обёрнут в рамку.
    """
    meta: list[str] = []
    if ingestion_note:
        meta.append(ingestion_note)
    if fetcher_note:
        meta.append(fetcher_note)
    meta_s = (" · ".join(meta) + " · ") if meta else ""

    st, stage, work = _dp_activity(dp)
    pct = _overall_pct(dp)

    health_tail = _format_dp_health_tail(dp, dp_health)
    upd = _dp_updated_age_note(dp)
    tail: list[str] = []
    if health_tail:
        tail.append(health_tail)
    if upd:
        tail.append(upd)

    digest = _dp_live_digest(dp, fetcher)
    core = f"DP {st} · {stage} · {work} · overall {pct}"
    if digest:
        core += f" · ‖ {digest} ‖"
    if tail:
        core += " · " + " · ".join(tail)
    extra = ""
    if stage_elapsed is not None:
        extra = f" · +{_fmt_duration(stage_elapsed)} in stage"
    if heartbeat:
        extra += " ···"
    return f"{meta_s}{core}{extra}"


def _format_status_line(
    *,
    wall_elapsed: float,
    ingestion_note: str | None,
    fetcher_note: str | None,
    dp: dict | None,
    fetcher: dict | None,
    dp_health: dict | None,
    stage_elapsed: float | None,
    heartbeat: bool,
    include_component_tables: bool = True,
) -> str:
    t = _fmt_duration(wall_elapsed)
    st, stage, work = _dp_activity(dp)
    pct = _overall_pct(dp)

    health_tail = _format_dp_health_tail(dp, dp_health)
    upd = _dp_updated_age_note(dp)
    tail_bits: list[str] = []
    if health_tail:
        tail_bits.append(health_tail)
    if upd:
        tail_bits.append(upd)

    digest = _dp_live_digest(dp, fetcher)
    core_for_summary = f"{st} · {stage} · {work} · overall {pct}"

    extra_hint = ""
    if stage_elapsed is not None:
        extra_hint = f"+{_fmt_duration(stage_elapsed)} in stage"
    if heartbeat:
        extra_hint = f"{extra_hint} ···" if extra_hint else "···"

    lines: list[str] = [_e2e_box_horizontal()]
    elapsed_lab = f"{t}" + ("  · heartbeat" if heartbeat else "")
    lines.extend(_e2e_box_row("Elapsed", elapsed_lab))
    if ingestion_note:
        extend = _e2e_box_row("ingestion", ingestion_note)
        lines.extend(extend)
    if fetcher_note:
        lines.extend(_e2e_box_row("fetcher", fetcher_note))
    lines.extend(_e2e_box_row("dataproc", core_for_summary))
    if digest:
        lines.extend(_e2e_box_row("trace ‖ … ‖", digest))
    if tail_bits:
        lines.extend(_e2e_box_row("liveness", " · ".join(tail_bits)))
    if extra_hint:
        lines.extend(_e2e_box_row("stage Δ", extra_hint))

    lines.append(_e2e_box_bottom())
    box = "\n".join(lines)

    if not include_component_tables:
        return box
    tables = _format_dp_component_tables(dp)
    if tables:
        return box + "\n\n" + "\n".join(tables)
    return box


def main() -> int:
    parser = argparse.ArgumentParser(description="Create run and wait for completed")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8001",
        help="Backend API base URL",
    )
    parser.add_argument(
        "--fetcher-url",
        default="",
        help="Fetcher API URL (e.g. http://localhost:8000). If set, poll Fetcher for detailed progress (stage, completed_stages).",
    )
    parser.add_argument(
        "--dataprocessor-url",
        default=os.environ.get("TF_BACKEND_DATAPROCESSOR_API_URL", ""),
        help="DataProcessor API URL (e.g. http://localhost:8002). If set with --with-dataprocessor, poll DataProcessor /status for processor stage and overall progress.",
    )
    parser.add_argument(
        "--email",
        default="e2e@example.com",
        help="User email (register/login)",
    )
    parser.add_argument(
        "--password",
        default="e2etest123",
        help="User password (min 6 chars)",
    )
    parser.add_argument(
        "--source-url",
        default="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        help="Video URL for ingestion",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        metavar="SEC",
        help="Max seconds to wait for completed/failed (default: 600, or 7200 with --with-dataprocessor)",
    )
    parser.add_argument(
        "--http-read-timeout",
        type=float,
        default=120.0,
        metavar="SEC",
        help=(
            "httpx read timeout for Backend/Fetcher/DataProcessor HTTP calls (default 120). "
            "Raise under heavy CPU load so polls do not fail with ReadTimeout."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Seconds between status polls (default 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Extra progress detail: print Backend /runs snapshot when its fields change (not on every DP/Fetcher tick).",
    )
    parser.add_argument(
        "--debug-backend",
        action="store_true",
        help="Same Backend /runs snapshot as --verbose (for explicit debug).",
    )
    parser.add_argument(
        "--progress-heartbeat",
        type=int,
        default=5,
        metavar="SEC",
        help="If progress fingerprint unchanged, emit a line at most every SEC seconds (default 5).",
    )
    parser.add_argument(
        "--resource-snapshot-sec",
        type=int,
        default=0,
        metavar="N",
        help="Every N seconds log host RAM, load1, GPU via nvidia-smi (0=off).",
    )
    parser.add_argument(
        "--e2e-events-jsonl",
        default="",
        metavar="PATH",
        help=(
            "Append JSONL событий (смена состояния + снимки ресурсов) для полной истории без засорения stdout. "
            "Пусто — не писать."
        ),
    )
    parser.add_argument(
        "--with-dataprocessor",
        action="store_true",
        help="Full E2E: wait for ingestion_status=processing then completed (DataProcessor must be running).",
    )
    parser.add_argument(
        "--processing-grace-seconds",
        type=int,
        default=90,
        help="When --with-dataprocessor: max seconds to wait for 'processing' after first 'completed' (default 90).",
    )
    args = parser.parse_args()
    if args.timeout is None:
        args.timeout = 7200 if args.with_dataprocessor else 600

    events_path = Path(args.e2e_events_jsonl).resolve() if str(args.e2e_events_jsonl or "").strip() else None

    base = args.base_url.rstrip("/")
    start_wall = time.monotonic()
    read_sec = max(5.0, float(args.http_read_timeout))
    http_timeout = httpx.Timeout(read_sec, connect=min(30.0, read_sec))

    with httpx.Client(timeout=http_timeout) as client:
        # 1. Register (409 = already exists)
        r = client.post(
            f"{base}/api/auth/register",
            json={"email": args.email, "password": args.password},
        )
        if r.status_code not in (200, 201, 409):
            _out(f"Register failed: {r.status_code} {r.text}", file=sys.stderr)
            return 1

        # 2. Login
        r = client.post(
            f"{base}/api/auth/login",
            json={"email": args.email, "password": args.password},
        )
        if r.status_code != 200:
            _out(f"Login failed: {r.status_code} {r.text}", file=sys.stderr)
            return 1
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 3. Create run
        r = client.post(
            f"{base}/api/runs",
            headers=headers,
            json={"source_url": args.source_url},
        )
        if r.status_code not in (200, 201):
            _out(f"Create run failed: {r.status_code} {r.text}", file=sys.stderr)
            return 1
        data = r.json()
        run_id = data.get("run_id")
        if not run_id:
            _out("Response missing run_id:", data, file=sys.stderr)
            return 1
        _print_run_banner(
            run_id=run_id,
            source_url=args.source_url,
            base_url=base,
            fetcher_url=args.fetcher_url or "",
            dataprocessor_url=args.dataprocessor_url or "",
            with_dataprocessor=bool(args.with_dataprocessor),
        )
        created = data.get("created_at")
        ing0 = data.get("ingestion_status", "?")
        _out(f"  status at create: {ing0}" + (f"  (created_at {created})" if created else ""))
        _out("  Polling…")
        _out("")

        # 4. Poll until completed or failed (or, with --with-dataprocessor: processing → completed)
        deadline = time.monotonic() + args.timeout
        pending_hint_shown = False
        dp_queued_hint_shown = False
        seen_processing = False
        first_completed_at: float | None = None  # when we first saw completed (for full E2E grace)
        prev_fingerprint: Tuple[Any, ...] | None = None
        state_started_at = time.monotonic()
        last_emit_at = 0.0
        last_ingestion_emitted = str(data.get("ingestion_status") or "")
        fetcher_done_announced = False
        # Fingerprint меняется из‑за DP/Fetcher health, а GET /runs часто тот же — не дублировать JSON.
        last_backend_snap_key: str | None = None
        last_hb_core: str | None = None
        last_resource_emit = 0.0
        while time.monotonic() < deadline:
            time.sleep(args.poll_interval)
            wall_elapsed = time.monotonic() - start_wall
            try:
                r = client.get(f"{base}/api/runs/{run_id}", headers=headers)
            except httpx.TimeoutException as e:
                _out(
                    f"  [{_fmt_duration(wall_elapsed)}] Backend poll HTTP timeout ({type(e).__name__}); "
                    f"retrying (read_timeout={read_sec}s).",
                    file=sys.stderr,
                )
                continue
            if r.status_code != 200:
                _out(f"  [{_fmt_duration(wall_elapsed)}] GET run failed: {r.status_code}", file=sys.stderr)
                continue
            data = r.json()
            status = data.get("ingestion_status") or ""
            stage = data.get("fetcher_stage") or ""

            fp = None
            dp = None
            dp_health = None
            dp_api_key = os.environ.get("TF_BACKEND_DATAPROCESSOR_API_KEY") or os.environ.get("DATAPROCESSOR_API_KEY")
            if args.fetcher_url:
                fp = fetch_fetcher_progress(client, args.fetcher_url, run_id)
            if args.with_dataprocessor and args.dataprocessor_url:
                dp = fetch_dataprocessor_progress(
                    client,
                    args.dataprocessor_url,
                    run_id,
                    api_key=dp_api_key,
                )
                dp_health = fetch_dataprocessor_health(client, args.dataprocessor_url)

            fingerprint = _progress_fingerprint(
                ingestion=status,
                fetcher=fp,
                dp=dp,
                backend_stage=stage,
                dp_health=dp_health,
            )
            now = time.monotonic()
            if args.resource_snapshot_sec > 0 and (now - last_resource_emit) >= args.resource_snapshot_sec:
                res_line = host_resource_snapshot_line()
                _out(f"  [host {_fmt_duration(wall_elapsed)}] {res_line}")
                _append_e2e_event(
                    events_path,
                    {
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "wall_s": round(wall_elapsed, 3),
                        "type": "resource_tick",
                        "line": res_line,
                        "host": host_resource_snapshot_dict(),
                    },
                )
                last_resource_emit = now
            if fingerprint != prev_fingerprint:
                if prev_fingerprint is not None:
                    prev_dur = now - state_started_at
                    _out("")
                    _out(_format_change_separator(prev_dur))
                    _out("")
                ingestion_note = None
                if status != last_ingestion_emitted:
                    ingestion_note = f"ingestion → {status}"
                    last_ingestion_emitted = status
                fetcher_note = None
                if fp and args.fetcher_url:
                    if _fetcher_done(fp):
                        if not fetcher_done_announced:
                            fetcher_note = "Fetcher done"
                            fetcher_done_announced = True
                    else:
                        fetcher_note = _fetcher_in_progress_line(fp)
                _out(
                    _format_status_line(
                        wall_elapsed=wall_elapsed,
                        ingestion_note=ingestion_note,
                        fetcher_note=fetcher_note,
                        dp=dp,
                        fetcher=fp,
                        dp_health=dp_health,
                        stage_elapsed=None,
                        heartbeat=False,
                        include_component_tables=True,
                    )
                )
                _append_e2e_event(
                    events_path,
                    {
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "wall_s": round(wall_elapsed, 3),
                        "type": "state_change",
                        "ingestion": status,
                        "backend_fetcher_stage": stage,
                        "fetcher": fp,
                        "dataprocessor": _compact_dp_for_event(dp),
                        "dp_health": dp_health,
                        "progress_fingerprint": _json_safe_fingerprint(fingerprint),
                        "host": host_resource_snapshot_dict(),
                    },
                )
                if args.debug_backend or args.verbose:
                    snap = {
                        k: data.get(k)
                        for k in (
                            "run_id",
                            "ingestion_status",
                            "fetcher_stage",
                            "fetcher_error_code",
                            "fetcher_error_message",
                            "updated_at",
                        )
                    }
                    snap_key = json.dumps(snap, default=str, ensure_ascii=False, sort_keys=True)
                    if snap_key != last_backend_snap_key:
                        _out(_e2e_box_horizontal())
                        for ln in _e2e_box_row(
                            "backend API",
                            json.dumps(snap, default=str, ensure_ascii=False),
                        ):
                            _out(ln)
                        _out(_e2e_box_bottom())
                        last_backend_snap_key = snap_key
                prev_fingerprint = fingerprint
                state_started_at = now
                last_emit_at = now
                last_hb_core = None
            elif args.progress_heartbeat > 0 and (now - last_emit_at) >= args.progress_heartbeat:
                # Пока Fetcher не дошёл до конца, на heartbeat показываем его стадию — иначе долгий
                # download_video выглядит как «зависший» пустой DP.
                hb_fetcher = None
                if fp and args.fetcher_url and not _fetcher_done(fp):
                    hb_fetcher = _fetcher_in_progress_line(fp)
                hb_block = _format_status_line(
                    wall_elapsed=wall_elapsed,
                    ingestion_note=None,
                    fetcher_note=hb_fetcher,
                    dp=dp,
                    fetcher=fp,
                    dp_health=dp_health,
                    stage_elapsed=now - state_started_at,
                    heartbeat=True,
                    include_component_tables=False,
                )
                hb_core = _format_status_core_text(
                    ingestion_note=None,
                    fetcher_note=hb_fetcher,
                    dp=dp,
                    fetcher=fp,
                    dp_health=dp_health,
                    stage_elapsed=now - state_started_at,
                    heartbeat=True,
                )
                if hb_core != last_hb_core:
                    _out(hb_block)
                    last_hb_core = hb_core
                last_emit_at = now

            if data.get("fetcher_error_code") or data.get("fetcher_error_message"):
                _out(
                    f"    fetcher_error_code: {data.get('fetcher_error_code')}  "
                    f"fetcher_error_message: {data.get('fetcher_error_message', '')[:120]}"
                )

            # Подсказка: долго queued у DataProcessor — часто норма (очередь / долгий POST / скачивание в кеш)
            if (
                not dp_queued_hint_shown
                and args.with_dataprocessor
                and wall_elapsed >= 120
                and dp
                and str(dp.get("status") or "").lower() == "queued"
            ):
                dp_queued_hint_shown = True
                _out(
                    "    [подсказка] DataProcessor в статусе queued дольше 2 мин: это не обязательно ошибка. "
                    "Смотрите в строке progress «queue active=… / stream_msgs=…» (источник GET …/api/v1/health): "
                    "если active≥1 при одном worker — вы почти наверняка ждёте другой run. "
                    "Ещё причины: долгий POST /api/v1/process (кеширование video_url), worker не слушает тот же REDIS_URL, что API. "
                    "Если /health даёт ошибку — почините DataProcessor health (он нужен для метрик очереди).",
                    file=sys.stderr,
                )

            # Подсказка один раз: только если реально «пустой» старт (0/7 и стадия всё ещё pending)
            if not pending_hint_shown and wall_elapsed >= 60 and status == "pending":
                completed = (fp.get("completed_stages") or []) if fp else []
                stg = (fp.get("stage") or "").strip().lower() if fp else ""
                idle_fetcher = not fp or (len(completed) == 0 and stg in ("", "pending"))
                if idle_fetcher:
                    pending_hint_shown = True
                    _out(
                        "    [подсказка] Run застрял на pending: Fetcher worker, скорее всего, не получает задачи. "
                        "Убедитесь, что Fetcher API и Fetcher worker используют один и тот же Redis "
                        "(FETCHER_REDIS_URL / CELERY_BROKER_URL). Если API в Docker — из контейнера подключайтесь к Redis на хосте (host.docker.internal:6379 или IP хоста).",
                        file=sys.stderr,
                    )

            if status == "processing":
                seen_processing = True
            if status == "completed":
                if first_completed_at is None:
                    first_completed_at = time.monotonic()
                if not args.with_dataprocessor:
                    _out("\nDone: ingestion_status = completed")
                    return 0
                # Full E2E: exit only if we've seen processing and then completed
                if seen_processing:
                    _out("\nDone: ingestion_status = completed (after DataProcessor)")
                    return 0
                # Still waiting for processing; check grace period
                if first_completed_at is not None and (time.monotonic() - first_completed_at) >= args.processing_grace_seconds:
                    _out(
                        "\nDataProcessor did not start within grace period: ingestion_status stayed completed (from Fetcher sync) "
                        "without transitioning to processing. Is trigger-processing being called? Is Backend Celery worker running? "
                        "Is TF_BACKEND_DATAPROCESSOR_API_URL set and DataProcessor API + worker running?",
                        file=sys.stderr,
                    )
                    return 1
            if status == "failed":
                _out("\nRun failed.")
                _out(f"  fetcher_error_code: {data.get('fetcher_error_code')}")
                _out(f"  fetcher_error_message: {data.get('fetcher_error_message')}")
                return 1

        _out(f"\nTimeout ({args.timeout}s). Last status: {data.get('ingestion_status')}", file=sys.stderr)
        if args.with_dataprocessor and not seen_processing:
            _out(
                "  (full E2E: never saw ingestion_status=processing; DataProcessor may not have started.)",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
