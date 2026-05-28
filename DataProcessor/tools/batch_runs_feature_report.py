#!/usr/bin/env python3
#
# Eдиная точка: прогон render для всех компонентов run + сводная таблица фичей из NPZ meta.
# Несколько run-каталогов: агрегат для пилотных пачек (например по 15 видео) и сравнение.
#
# Запуск (из корня пакета DataProcessor):
#   python tools/batch_runs_feature_report.py --run-dir /path/.../platform/vid/run --regenerate-renders
#   python tools/batch_runs_feature_report.py --runs-file /tmp/paths.txt --output-csv /tmp/pilot.csv
#   python tools/batch_runs_feature_report.py --run-glob '.../youtube/*/*' --max-runs 3
#   python tools/batch_runs_feature_report.py ... --output-by-component-dir /path/batch_by_component
#   text_processor: кроме строки text_processor добавляются строки text_processor/<экстрактор> с tp_* (как отдельные «компоненты»).
#

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import re
import sys
import time
import traceback
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("batch_runs_feature_report")

# Каталоги под run, которые не являются компонентами (state-files, служебное).
_SKIP_RUN_SUBDIRS = frozenset(
    {
        "state",
    }
)

# Порядок колонок в узком CSV / в таблице по компоненту (остальные ключи — по алфавиту после них).
_BASE_COL_ORDER = [
    "platform_id",
    "video_id",
    "run_id",
    "component",
    "component_type",
    "manifest_status",
    "manifest_empty_reason",
    "duration_ms",
    "device_used",
    "npz_error",
    "render_error",
]

# В узком отчёте (output-by-component-dir: index.html + csv/…) не выводим служебные пути и часть meta_*.
_NARROW_EXCLUDE_COLS: frozenset[str] = frozenset(
    {
        "npz",
        "run_path",
        "meta_config_hash",
        "meta_created_at",
        "meta_run_id",
    }
)


def _dp_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_path() -> None:
    r = str(_dp_root())
    if r not in sys.path:
        sys.path.insert(0, r)


_ensure_path()
from qa.component_feature_qa import flatten_meta as _flatten_meta  # noqa: E402


def _textprocessor_src_in_path() -> bool:
    """TextProcessor: импорты `from core.*` (как в run_cli / renderer)."""
    tp_src = _dp_root() / "TextProcessor" / "src"
    s = str(tp_src)
    if not tp_src.is_dir():
        return False
    if s not in sys.path:
        sys.path.insert(0, s)
    return True


def _text_processor_subcomponent_rows(
    base: Dict[str, Any],
    npz_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Одна строка на sub-extractor: component = text_processor/<имя> и только его tp_* поля
    (как отдельные «компоненты» в melt / batch_by_component).
    """
    if not _textprocessor_src_in_path():
        return []
    try:
        from core.text_feature_grouping import (  # type: ignore[import-not-found]
            build_feature_dict_from_npz_data,
            group_text_features_by_extractor,
        )
    except ImportError as e:
        logger.debug("text_processor sub-rows: %s", e)
        return []
    features = build_feature_dict_from_npz_data(npz_data)
    if not features:
        return []
    groups = group_text_features_by_extractor(features)
    common = (
        "platform_id",
        "video_id",
        "run_id",
        "run_path",
        "component_type",
        "manifest_status",
        "manifest_empty_reason",
        "duration_ms",
        "device_used",
        "npz",
    )
    share_meta = (
        "meta_status",
        "meta_producer_version",
        "meta_schema_version",
    )
    out: List[Dict[str, Any]] = []
    for ex_name, feats in sorted(groups.items()):
        if not feats:
            continue
        sub: Dict[str, Any] = {}
        for k in common:
            if k in base:
                sub[k] = base[k]
        if "render_error" in base:
            sub["render_error"] = base["render_error"]
        for k in share_meta:
            if k in base:
                sub[k] = base[k]
        sub["component"] = f"text_processor/{ex_name}"
        sub.update(feats)
        out.append(sub)
    return out


def _parse_run_ids(data: Optional[Dict[str, Any]], run_dir: Path) -> Tuple[str, str, str]:
    run: Dict[str, Any] = {}
    if isinstance(data, dict):
        r0 = data.get("run")
        if isinstance(r0, dict):
            run = r0
    rid = str(run.get("run_id") or "")
    vid = str(run.get("video_id") or "")
    pid = str(run.get("platform_id") or "")
    if not (rid and vid and pid):
        try:
            p = run_dir.resolve()
            parts = p.parts
            if len(parts) >= 3:
                rid = rid or str(parts[-1])
                vid = vid or str(parts[-2])
                pid = pid or str(parts[-3])
        except Exception:
            pass
    return rid, vid, pid


def _iter_run_dirs(
    run_dirs: List[str],
    runs_file: Optional[str],
    run_glob: Optional[str],
) -> List[Path]:
    paths: List[Path] = []
    for s in run_dirs:
        p = Path(s).resolve()
        if p.is_dir():
            paths.append(p)
    if runs_file:
        with open(runs_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = Path(line).resolve()
                if p.is_dir():
                    paths.append(p)
    if run_glob:
        for s in glob(run_glob):
            p = Path(s).resolve()
            if p.is_dir() and p not in paths:
                paths.append(p)
    # Deduplicate, stable order
    seen: Set[str] = set()
    uniq: List[Path] = []
    for p in paths:
        k = str(p)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return sorted(uniq, key=str)


def _load_manifest(manifest_path: Path) -> Optional[Dict[str, Any]]:
    if not manifest_path.is_file():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return None


def _warmup_renderer_imports() -> None:
    """Один раз загрузить numpy/VisualProcessor, чтобы задержка не маскировалась под первый run."""
    _ensure_path()
    t0 = time.perf_counter()
    # Импортируем тот же набор, что и в _collect_rows_for_run (кэш модулей).
    from VisualProcessor.utils import renderer  # noqa: F401

    logger.info("Прогрев импорта (numpy + VisualProcessor.renderer): %.1fs", time.perf_counter() - t0)


def _collect_rows_for_run(
    run_dir: Path,
    manifest_data: Optional[Dict[str, Any]],
    do_render: bool,
    *,
    component_log_every: int = 1,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    _ensure_path()
    from VisualProcessor.utils.renderer import (
        _path_is_nonempty_file,
        find_component_npz,
        load_npz,
        extract_meta,
        render_all_components,
    )

    render_note: Optional[str] = None
    if do_render:
        t0 = time.perf_counter()
        logger.info(
            "  render_all_components: start (долго: пересчёт JSON/HTML по всем NPZ в run) -> %s",
            run_dir,
        )
        try:
            render_all_components(str(run_dir))
            logger.info("  render_all_components: done за %.1fs", time.perf_counter() - t0)
        except Exception as e:
            render_note = f"render_all_failed: {e}"
            logger.warning("  render_all_components: ошибка за %.1fs: %s", time.perf_counter() - t0, e)

    rows: List[Dict[str, Any]] = []
    run_id, video_id, platform_id = _parse_run_ids(manifest_data, run_dir)

    manifest_components: Dict[str, Dict] = {}
    if manifest_data and isinstance(manifest_data.get("components"), list):
        for c in manifest_data["components"]:
            if isinstance(c, dict) and c.get("name"):
                manifest_components[str(c["name"])] = c

    scanned = {p.name for p in run_dir.iterdir() if p.is_dir() and not p.name.startswith("_")}

    comp_names = sorted(manifest_components.keys() | scanned)
    to_walk = [
        n
        for n in comp_names
        if not n.startswith("_")
        and n not in _SKIP_RUN_SUBDIRS
        and (run_dir / n).is_dir()
    ]
    n_comp = len(to_walk)
    logger.info("  компонентов к обходу: %d (каталоги под run)", n_comp)

    for comp_i, comp_name in enumerate(to_walk, start=1):
        comp_type = "core" if comp_name.startswith("core_") else "module"
        cdir = run_dir / comp_name
        if component_log_every > 0 and (
            comp_i == 1 or comp_i == n_comp or comp_i % component_log_every == 0
        ):
            logger.info("  [%d/%d] %s", comp_i, n_comp, comp_name)

        npz = find_component_npz(cdir, comp_name, comp_type)
        mc = manifest_components.get(comp_name) or {}
        base: Dict[str, Any] = {
            "platform_id": platform_id,
            "video_id": video_id,
            "run_id": run_id,
            "run_path": str(run_dir),
            "component": comp_name,
            "component_type": comp_type,
            "manifest_status": str(mc.get("status") or ""),
            "manifest_empty_reason": str(mc.get("empty_reason") or ""),
            "duration_ms": mc.get("duration_ms"),
            "device_used": str(mc.get("device_used") or ""),
            "npz": str(npz) if npz else "",
        }
        if render_note:
            base["render_error"] = render_note

        d: Optional[Dict[str, Any]] = None
        if npz and _path_is_nonempty_file(npz):
            logger.debug("    load_npz: %s", npz)
            t_npz = time.perf_counter()
            try:
                d = load_npz(str(npz))
                meta = extract_meta(d)
                base.update(_flatten_meta(meta, prefix="meta_"))
            except Exception as e:
                base["npz_error"] = str(e)
                d = None
            dt = time.perf_counter() - t_npz
            logger.debug("    load_npz готово за %.3fs", dt)
            if dt > 3.0:
                hint = ""
                s = str(npz)
                if "core_depth_midas" in s or "depth.npz" in s:
                    hint = " (часто норма: большие тензоры глубины + медленный диск)"
                logger.warning("    медленный NPZ %.1fs: %s%s", dt, npz, hint)
        else:
            base["npz_error"] = "no_npz"
            logger.debug("    нет npz: %s / %s", comp_name, cdir)

        rows.append(base)
        if comp_name == "text_processor" and d is not None and "npz_error" not in base:
            rows.extend(_text_processor_subcomponent_rows(base, d))

    return rows, render_note


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("# no rows\n")
        return
    keys: Set[str] = set()
    for r in rows:
        keys |= set(r.keys())
    fieldnames = sorted(keys)
    with open(path, "w", encoding="utf-8", newline="") as wf:
        w = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _safe_component_filename(component: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", (component or "unknown").strip()) or "unknown"
    return s[:180]


def _fieldnames_narrow_for_rows(rows: List[Dict[str, Any]]) -> List[str]:
    present: Set[str] = set()
    for r in rows:
        present |= set(r.keys())
    present -= _NARROW_EXCLUDE_COLS
    out: List[str] = []
    for k in _BASE_COL_ORDER:
        if k in present:
            out.append(k)
    out.extend(sorted(k for k in present if k not in _BASE_COL_ORDER))
    return out


def _write_csv_narrow(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as wf:
        w = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _export_by_component_dir(out_dir: Path, all_rows: List[Dict[str, Any]]) -> int:
    """
    Каталог: csv/<component>.csv (только релевантные колонки) + index.html (мини-таблицы).
    Returns: число компонентов.
    """
    out_dir = out_dir.resolve()
    csv_sub = out_dir / "csv"
    csv_sub.mkdir(parents=True, exist_ok=True)

    by_comp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        by_comp[str(r.get("component") or "unknown")].append(r)

    sections_html: List[str] = []
    toc_li: List[str] = []
    n_comp = 0

    for comp in sorted(by_comp.keys()):
        rows = by_comp[comp]
        fieldnames = _fieldnames_narrow_for_rows(rows)
        safe = _safe_component_filename(comp)
        anchor = re.sub(r"[^\w\-.]+", "-", safe).strip("-")[:160] or "c"
        _write_csv_narrow(str(csv_sub / f"{safe}.csv"), rows, fieldnames)
        n_comp += 1

        esc_cols = [html.escape(c) for c in fieldnames]
        thead = "<tr>" + "".join(f"<th>{h}</th>" for h in esc_cols) + "</tr>"
        trs: List[str] = []
        for row in rows:
            cells = []
            for k in fieldnames:
                v = row.get(k)
                if v is None:
                    v = ""
                cells.append(f"<td>{html.escape(str(v))}</td>")
            trs.append("<tr>" + "".join(cells) + "</tr>")
        table = f'<div class="wrap"><table><thead>{thead}</thead><tbody>{"".join(trs)}</tbody></table></div>'
        meta = (
            f'<p class="meta">{len(rows)} строк · {len(fieldnames)} колонок · '
            f'<a href="csv/{html.escape(safe)}.csv">csv/{html.escape(safe)}.csv</a></p>'
        )
        sections_html.append(
            f'<section id="{html.escape(anchor)}"><h2>{html.escape(comp)}</h2>{meta}{table}</section>'
        )
        toc_li.append(
            f'<li><a href="#{html.escape(anchor)}">{html.escape(comp)}</a> ({len(rows)})</li>'
        )

    nav = "<nav class=\"toc\"><strong>По компонентам</strong><ul>" + "".join(toc_li) + "</ul></nav>"
    body = (
        f"<h1>Отчёт по компонентам</h1>"
        f'<p class="meta">Всего строк: {len(all_rows)} · компонентов: {n_comp}</p>'
        f"{nav}"
        + "\n".join(sections_html)
    )

    index_html = _html_report_shell(
        title="batch_runs by component",
        body=body,
    )
    index_path = out_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    return n_comp


def _html_report_shell(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0f1419;
      --fg: #e7e9ea;
      --muted: #8b98a5;
      --border: #38444d;
    }}
    body {{
      font-family: system-ui, "Segoe UI", Roboto, sans-serif;
      font-size: 13px;
      background: var(--bg);
      color: var(--fg);
      margin: 0;
      padding: 16px 20px 48px;
      line-height: 1.4;
    }}
    h1 {{ font-size: 1.25rem; margin: 0 0 8px; }}
    h2 {{ font-size: 1.05rem; margin: 28px 0 8px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
    .meta {{ color: var(--muted); font-size: 12px; margin: 0 0 12px; }}
    .toc {{
      position: sticky; top: 0; z-index: 5;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 14px;
      margin-bottom: 20px;
      max-height: 40vh;
      overflow: auto;
    }}
    .toc ul {{ margin: 8px 0 0; padding-left: 18px; columns: 2; column-gap: 24px; }}
    @media (max-width: 800px) {{ .toc ul {{ columns: 1; }} }}
    .toc a {{ color: #8ec7ff; }}
    section {{ margin-bottom: 8px; }}
    .wrap {{
      border: 1px solid var(--border);
      border-radius: 8px;
      max-height: 360px;
      overflow: auto;
    }}
    table {{ border-collapse: collapse; min-width: 100%; font-family: "JetBrains Mono", "Consolas", monospace; font-size: 11px; }}
    th {{
      position: sticky; top: 0; z-index: 2;
      background: #1a2332;
      text-align: left; padding: 6px 8px;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }}
    td {{ padding: 5px 8px; border-bottom: 1px solid #2f3943; vertical-align: top; max-width: 28rem; word-break: break-all; }}
    tr:nth-child(even) td {{ background: rgba(255,255,255,0.02); }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def _print_markdown_table(rows: List[Dict[str, Any]], max_rows: int) -> None:
    cols = [
        "platform_id",
        "video_id",
        "run_id",
        "component",
        "manifest_status",
        "duration_ms",
    ]
    if not rows:
        print("_(нет строк)_")
        return
    display = rows[: max(1, max_rows)]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    print(header)
    print(sep)
    for r in display:
        line = []
        for c in cols:
            v = r.get(c, "")
            if v is None:
                v = ""
            s = str(v).replace("|", "/")
            line.append(s[:120])
        print("| " + " | ".join(line) + " |")
    if len(rows) > len(display):
        print(f"\n_…и ещё {len(rows) - len(display)} строк (см. CSV)._")


def _run_summary_by_component(rows: List[Dict[str, Any]]) -> None:
    from collections import defaultdict

    durs: Dict[str, List[int]] = defaultdict(list)
    st: Dict[str, int] = defaultdict(int)
    for r in rows:
        comp = str(r.get("component") or "")
        dm = r.get("duration_ms")
        if isinstance(dm, (int, float)):
            durs[comp].append(int(dm))
        s = r.get("manifest_status") or ""
        st[f"{comp}:{s}"] += 1
    if not durs and not st:
        return
    print("\n### Кратко по компонентам (duration_ms)\n")
    for comp in sorted(durs.keys()):
        xs = durs[comp]
        if not xs:
            continue
        print(
            f"- **{comp}**: n={len(xs)}, min={min(xs)}ms, max={max(xs)}ms, "
            f"avg={sum(xs) // len(xs)}ms"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Рендеры + сводка meta по всем компонентам для одного или нескольких run.",
    )
    ap.add_argument(
        "--run-dir",
        action="append",
        default=[],
        metavar="PATH",
        help="Каталог run (содержит manifest.json), можно повторить",
    )
    ap.add_argument(
        "--runs-file",
        metavar="FILE",
        help="Файл: по одному пути run на строку",
    )
    ap.add_argument(
        "--run-glob",
        metavar="GLOB",
        help="Shell-glob каталогов run (например .../result_store/youtube/*/*/*)",
    )
    ap.add_argument(
        "--regenerate-renders",
        action="store_true",
        help="Вызвать render_all_components для каждого run (JSON + HTML где есть)",
    )
    ap.add_argument(
        "--no-summary-table",
        action="store_true",
        help="Не печатать мини-таблицу в stdout",
    )
    ap.add_argument(
        "--max-print-rows",
        type=int,
        default=50,
        help="Максимум строк в markdown-таблице (по умолчанию 50)",
    )
    ap.add_argument(
        "--output-csv",
        metavar="FILE",
        help="Сохранить широкую таблицу (все плоские meta_* колонки)",
    )
    ap.add_argument(
        "--output-jsonl",
        metavar="FILE",
        help="JSON Lines (по одной записи на компонент×run)",
    )
    ap.add_argument(
        "--output-by-component-dir",
        metavar="DIR",
        help="Каталог: для каждого component узкий csv/columns + index.html с мини-таблицами (вместо одной «простыни» на сотни колонок).",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Подробные логи (уровень DEBUG: каждый load_npz и т.д.)",
    )
    ap.add_argument(
        "--log-level",
        default="",
        help="DEBUG|INFO|WARNING (по умолчанию: INFO, с -v — DEBUG)",
    )
    ap.add_argument(
        "--component-log-every",
        type=int,
        default=5,
        metavar="N",
        help="Каждые N компонентов писать строку прогресса (0 = только начало/конец run; по умолчанию 5)",
    )
    ap.add_argument(
        "--max-runs",
        type=int,
        default=0,
        metavar="N",
        help="Обработать не больше N run подряд (0 = все). Список сортируется по пути — берутся первые N. Удобно для теста.",
    )
    args = ap.parse_args()

    if args.log_level:
        level = getattr(logging, args.log_level.upper(), logging.INFO)
    else:
        level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    run_list = _iter_run_dirs(args.run_dir, args.runs_file, args.run_glob)
    if not run_list:
        print("Нет каталогов run. Укажите --run-dir, --runs-file и/или --run-glob.", file=sys.stderr)
        return 2

    total_discovered = len(run_list)
    max_runs = max(0, int(args.max_runs or 0))
    if max_runs > 0:
        if max_runs < total_discovered:
            run_list = run_list[:max_runs]
            logger.info(
                "Ограничение --max-runs=%d: из %d обнаруженных run обрабатываем первые %d (сортировка по path).",
                max_runs,
                total_discovered,
                len(run_list),
            )
        else:
            logger.info(
                "--max-runs=%d: в наличии %d run — срез не применяется.",
                max_runs,
                total_discovered,
            )
    n_runs = len(run_list)
    logger.info("Старт: run-каталогов=%d, regenerate_renders=%s", n_runs, bool(args.regenerate_renders))
    if args.run_glob:
        logger.info("glob: %s", args.run_glob)

    _warmup_renderer_imports()

    all_rows: List[Dict[str, Any]] = []
    total_t0 = time.perf_counter()
    for i, rd in enumerate(run_list, start=1):
        mf = rd / "manifest.json"
        man = _load_manifest(mf)
        if man is None:
            logger.warning("Пропуск (нет manifest): %s", rd)
            continue
        run_id, vid, plat = _parse_run_ids(man, rd)
        logger.info("Run %d/%d: %s / %s / %s", i, n_runs, plat, vid, run_id)
        logger.info("  path: %s", rd)
        t_run = time.perf_counter()
        try:
            rows, _ = _collect_rows_for_run(
                rd,
                man,
                do_render=bool(args.regenerate_renders),
                component_log_every=max(0, int(args.component_log_every)),
            )
            all_rows.extend(rows)
            logger.info(
                "  готово за %.1fs, строк=%d (всего накоплено %d)",
                time.perf_counter() - t_run,
                len(rows),
                len(all_rows),
            )
        except Exception as e:
            print(f"Ошибка {rd}: {e}\n{traceback.format_exc()}", file=sys.stderr)
            return 1

    logger.info("Все run обработаны за %.1fs, итог строк: %d", time.perf_counter() - total_t0, len(all_rows))

    if not args.no_summary_table:
        head = f"**{n_runs}** run"
        if max_runs > 0 and total_discovered > n_runs:
            head += f" (из {total_discovered} по glob/списку, срез --max-runs={max_runs})"
        print(f"\n## Отчёт: {head}, **{len(all_rows)}** строк (компонент×run)\n")
        _print_markdown_table(all_rows, max_rows=int(args.max_print_rows))
        _run_summary_by_component(all_rows)

    if args.output_csv:
        t_w = time.perf_counter()
        logger.info("Запись CSV: %s (%d строк)…", args.output_csv, len(all_rows))
        _write_csv(args.output_csv, all_rows)
        logger.info("CSV записан за %.1fs: %s", time.perf_counter() - t_w, args.output_csv)
        print(f"CSV: {args.output_csv}")
    if args.output_jsonl:
        t_w = time.perf_counter()
        logger.info("Запись JSONL: %s…", args.output_jsonl)
        with open(args.output_jsonl, "w", encoding="utf-8") as jf:
            for r in all_rows:
                jf.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("JSONL за %.1fs: %s", time.perf_counter() - t_w, args.output_jsonl)
        print(f"JSONL: {args.output_jsonl}")

    if args.output_by_component_dir:
        t_w = time.perf_counter()
        out_comp = Path(args.output_by_component_dir).expanduser()
        n_c = _export_by_component_dir(out_comp, all_rows)
        logger.info(
            "По компонентам: %s — %d компонентов, за %.1fs (index.html + csv/…)",
            out_comp,
            n_c,
            time.perf_counter() - t_w,
        )
        print(f"По компонентам: {out_comp / 'index.html'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
