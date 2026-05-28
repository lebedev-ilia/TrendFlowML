#!/usr/bin/env python3
"""
Единый прогон еженедельного QA (playbook §9): quality audit → опц. drift →
опц. golden compare → опц. валидаторы TextProcessor → melt HTML → реестр → shortlist.

Пример:
  DataProcessor/.data_venv/bin/python DataProcessor/tools/feature_qa_pipeline.py \\
    --batch-csv storage/result_store/batch_features_report_20runs.csv \\
    --batch-label 2026W17_20runs \\
    --baseline-csv storage/result_store/batch_features_report_week0.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tools_dir() -> Path:
    return Path(__file__).resolve().parent


def _result_store_root(repo: Path) -> Path:
    return repo / "storage" / "result_store"


def _run(cmd: List[str], *, cwd: Optional[Path] = None) -> int:
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return int(p.returncode)


def _sanitize_label(s: str) -> str:
    t = re.sub(r"[^\w\-.]+", "_", (s or "").strip())
    return (t[:80] if t else "run")


def _pick_text_features_npz(batch_csv: Path) -> Tuple[Optional[Path], str]:
    """
    Берёт первый подходящий text_features.npz из колонки npz (строка text_processor).
    Возвращает (path или None, заметка для summary).
    """
    prefer: List[Tuple[int, Path]] = []
    fallback: List[Tuple[int, Path]] = []
    with open(batch_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            comp = (row.get("component") or "").strip()
            npz_raw = (row.get("npz") or "").strip()
            if not npz_raw or "text_features.npz" not in npz_raw:
                continue
            if "text_processor" not in comp:
                continue
            p = Path(npz_raw).expanduser()
            if not p.is_file():
                continue
            if comp == "text_processor":
                prefer.append((i, p.resolve()))
            else:
                fallback.append((i, p.resolve()))
    if prefer:
        p = prefer[0][1]
        return p, f"row_order={prefer[0][0]} component=text_processor"
    if fallback:
        p = fallback[0][1]
        return p, f"row_order={fallback[0][0]} component_prefix=text_processor"
    return None, "no_text_features_npz_in_batch"


def _export_shortlist(
    quality_csv: Path,
    out_path: Path,
    *,
    max_rows: int,
    health_below: float,
    nan_min: float,
    oor_min: float,
) -> int:
    """Строки из feature_quality_report: кандидаты на ручной отбор / отключение."""
    rows_out: List[Dict[str, str]] = []
    with open(quality_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []
        for row in r:
            try:
                h = float(row.get("health_score") or 999)
            except ValueError:
                h = 999.0
            try:
                nan_r = float(row.get("nan_rate") or 0)
            except ValueError:
                nan_r = 0.0
            try:
                oor = float(row.get("out_of_range_rate") or 0)
            except ValueError:
                oor = 0.0
            sev = (row.get("severity") or "").strip().lower()
            const = (row.get("constant_like") or "").strip().lower() in ("1", "true", "yes")
            if (
                h < health_below
                or nan_r >= nan_min
                or oor >= oor_min
                or sev in ("high", "medium")
                or (const and h < 95)
            ):
                row = dict(row)
                row["shortlist_reason"] = _shortlist_reason(
                    h, nan_r, oor, sev, const, health_below, nan_min, oor_min
                )
                rows_out.append(row)
    rows_out.sort(
        key=lambda x: (float(x.get("health_score") or 999), -float(x.get("nan_rate") or 0))
    )
    if max_rows > 0 and len(rows_out) > max_rows:
        rows_out = rows_out[:max_rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = (
        list(rows_out[0].keys())
        if rows_out
        else (fieldnames + (["shortlist_reason"] if "shortlist_reason" not in fieldnames else []))
    )
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)
    print(f"Shortlist: {out_path} ({len(rows_out)} rows)", flush=True)
    return len(rows_out)


def _shortlist_reason(
    h: float,
    nan_r: float,
    oor: float,
    sev: str,
    const: bool,
    health_thr: float,
    nan_min: float,
    oor_min: float,
) -> str:
    bits = []
    if h < health_thr:
        bits.append(f"health<{health_thr:g}")
    if nan_r >= nan_min:
        bits.append("nan")
    if oor >= oor_min:
        bits.append("oor")
    if sev in ("high", "medium"):
        bits.append(f"sev={sev}")
    if const:
        bits.append("constant")
    return ";".join(bits) if bits else "other"


def main() -> int:
    repo = _repo_root()
    tools = _tools_dir()
    py = sys.executable
    default_rs = _result_store_root(repo)
    default_qa = default_rs / "view_csv_feature_qa.json"

    ap = argparse.ArgumentParser(description="Weekly feature QA pipeline (audit, drift, HTML, registry, shortlist)")
    ap.add_argument("--batch-csv", type=Path, required=True, help="Wide batch_features_report*.csv")
    ap.add_argument(
        "--batch-label",
        required=True,
        help="Метка прогона (префикс имён отчётов и feature_incident_registry)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Каталог артефактов (по умолчанию: storage/result_store/qa_runs/<batch-label>)",
    )
    ap.add_argument("--qa-config", type=Path, default=default_qa, help="view_csv_feature_qa.json")
    ap.add_argument(
        "--baseline-csv",
        type=Path,
        default=None,
        help="Эталон для feature_batch_drift (если не задан — шаг дрейфа пропускается)",
    )
    ap.add_argument(
        "--golden-compare-csv",
        type=Path,
        default=None,
        help="Эталонный wide CSV второго прогона (те же run): golden_batch_compare.py, A=этот файл, B=--batch-csv",
    )
    ap.add_argument("--golden-abs-eps", type=float, default=1e-5)
    ap.add_argument("--golden-rel-eps", type=float, default=1e-4)
    ap.add_argument("--skip-golden-compare", action="store_true", help="Не вызывать golden_batch_compare")
    ap.add_argument(
        "--text-validators-npz",
        type=Path,
        default=None,
        help="text_features.npz для run_text_extractor_validators.py (явный путь)",
    )
    ap.add_argument(
        "--run-text-validators-from-batch",
        action="store_true",
        help="Взять первый text_features.npz из колонки npz (component text_processor*)",
    )
    ap.add_argument("--skip-text-validators", action="store_true", help="Не звать run_text_extractor_validators")
    ap.add_argument("--skip-melt-html", action="store_true", help="Не вызывать view_csv.py")
    ap.add_argument("--skip-registry", action="store_true", help="Не обновлять feature_incidents.json")
    ap.add_argument("--skip-shortlist", action="store_true")
    ap.add_argument("--shortlist-health-below", type=float, default=80.0)
    ap.add_argument("--shortlist-nan-min", type=float, default=0.05)
    ap.add_argument("--shortlist-oor-min", type=float, default=0.02)
    ap.add_argument("--shortlist-max-rows", type=int, default=500, help="0 = без ограничения")
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Записать сводку путей и кодов выхода (по умолчанию: <out-dir>/feature_qa_pipeline_<label>.summary.json)",
    )
    args = ap.parse_args()

    batch_csv = args.batch_csv.expanduser().resolve()
    if not batch_csv.is_file():
        print(f"batch CSV not found: {batch_csv}", file=sys.stderr)
        return 1

    slug = _sanitize_label(args.batch_label)
    if args.out_dir is None:
        out_dir = (default_rs / "qa_runs" / slug).resolve()
    else:
        out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    html_dir = out_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    incidents_dir = default_rs / "incidents"
    incidents_dir.mkdir(parents=True, exist_ok=True)
    registry_path = incidents_dir / "feature_incidents.json"

    paths: Dict[str, str] = {}
    codes: Dict[str, int] = {}

    fq_json = out_dir / f"feature_quality_report_{slug}.json"
    fq_csv = out_dir / f"feature_quality_report_{slug}.csv"
    fq_md = out_dir / f"feature_quality_report_{slug}.md"
    codes["quality_audit"] = _run(
        [
            py,
            str(tools / "feature_quality_audit.py"),
            "--csv",
            str(batch_csv),
            "--qa-config",
            str(args.qa_config.expanduser().resolve()),
            "--out-json",
            str(fq_json),
            "--out-csv",
            str(fq_csv),
            "--out-md",
            str(fq_md),
        ]
    )
    paths["feature_quality_json"] = str(fq_json)
    paths["feature_quality_csv"] = str(fq_csv)
    paths["feature_quality_md"] = str(fq_md)

    tv_npz: Optional[Path] = None
    tv_note = ""
    if args.text_validators_npz:
        tv_npz = args.text_validators_npz.expanduser().resolve()
        tv_note = "explicit_arg"
    elif args.run_text_validators_from_batch:
        tv_npz, tv_note = _pick_text_features_npz(batch_csv)

    if (
        not args.skip_text_validators
        and tv_npz
        and tv_npz.is_file()
    ):
        tv_json = out_dir / f"text_extractor_validators_report_{slug}.json"
        tv_md = out_dir / f"text_extractor_validators_report_{slug}.md"
        codes["text_validators"] = _run(
            [
                py,
                str(tools / "run_text_extractor_validators.py"),
                str(tv_npz),
                "--out-json",
                str(tv_json),
                "--out-md",
                str(tv_md),
            ]
        )
        paths["text_validators_npz"] = str(tv_npz)
        paths["text_validators_pick_note"] = tv_note
        paths["text_validators_json"] = str(tv_json)
        paths["text_validators_md"] = str(tv_md)
    elif not args.skip_text_validators and (args.text_validators_npz or args.run_text_validators_from_batch):
        print(
            f"skip text validators: npz missing or not found ({tv_npz!r}; {tv_note})",
            file=sys.stderr,
        )
        codes["text_validators"] = 1

    drift_csv: Optional[Path] = None
    if args.baseline_csv:
        baseline = args.baseline_csv.expanduser().resolve()
        if not baseline.is_file():
            print(f"baseline CSV missing, skip drift: {baseline}", file=sys.stderr)
        else:
            drift_json = out_dir / f"feature_batch_drift_{slug}.json"
            drift_csv = out_dir / f"feature_batch_drift_{slug}.csv"
            drift_md = out_dir / f"feature_batch_drift_{slug}.md"
            codes["drift"] = _run(
                [
                    py,
                    str(tools / "feature_batch_drift.py"),
                    "--csv-a",
                    str(baseline),
                    "--csv-b",
                    str(batch_csv),
                    "--out-json",
                    str(drift_json),
                    "--out-csv",
                    str(drift_csv),
                    "--out-md",
                    str(drift_md),
                ]
            )
            paths["drift_json"] = str(drift_json)
            paths["drift_csv"] = str(drift_csv)
            paths["drift_md"] = str(drift_md)

    if not args.skip_golden_compare and args.golden_compare_csv:
        gref = args.golden_compare_csv.expanduser().resolve()
        if not gref.is_file():
            print(f"golden reference CSV missing, skip: {gref}", file=sys.stderr)
            codes["golden_compare"] = 1
        else:
            g_csv = out_dir / f"golden_mismatches_{slug}.csv"
            g_md = out_dir / f"golden_mismatches_{slug}.md"
            codes["golden_compare"] = _run(
                [
                    py,
                    str(tools / "golden_batch_compare.py"),
                    "--csv-a",
                    str(gref),
                    "--csv-b",
                    str(batch_csv),
                    "--out-csv",
                    str(g_csv),
                    "--out-md",
                    str(g_md),
                    "--abs-eps",
                    str(args.golden_abs_eps),
                    "--rel-eps",
                    str(args.golden_rel_eps),
                ]
            )
            paths["golden_reference_csv"] = str(gref)
            paths["golden_mismatches_csv"] = str(g_csv)
            paths["golden_mismatches_md"] = str(g_md)

    if not args.skip_melt_html:
        melt_html = html_dir / f"{batch_csv.stem}.melt.interesting.qa.pipeline.view.html"
        view_py = repo / "storage" / "result_store" / "view_csv.py"
        if not view_py.is_file():
            print(f"view_csv.py not found: {view_py}", file=sys.stderr)
            codes["melt_html"] = 1
        else:
            codes["melt_html"] = _run(
                [
                    py,
                    str(view_py),
                    "--csv",
                    str(batch_csv),
                    "--melt",
                    "--melt-interesting",
                    "--melt-qa",
                    "--out",
                    str(melt_html),
                    "--no-open",
                ],
                cwd=str(repo),
            )
        paths["melt_html"] = str(melt_html)

    if not args.skip_registry:
        reg_cmd = [
            py,
            str(tools / "feature_incident_registry.py"),
            "--registry",
            str(registry_path),
            "--batch-label",
            args.batch_label,
            "--quality-csv",
            str(fq_csv),
        ]
        if drift_csv and drift_csv.is_file():
            reg_cmd.extend(["--drift-csv", str(drift_csv)])
        codes["registry"] = _run(reg_cmd)
        paths["registry"] = str(registry_path)

    if not args.skip_shortlist and fq_csv.is_file():
        sl = out_dir / f"feature_shortlist_{slug}.csv"
        mx = 0 if args.shortlist_max_rows <= 0 else args.shortlist_max_rows
        _export_shortlist(
            fq_csv,
            sl,
            max_rows=mx,
            health_below=args.shortlist_health_below,
            nan_min=args.shortlist_nan_min,
            oor_min=args.shortlist_oor_min,
        )
        paths["shortlist_csv"] = str(sl)

    summary_path = args.summary_json
    if summary_path is None:
        summary_path = out_dir / f"feature_qa_pipeline_{slug}.summary.json"
    else:
        summary_path = summary_path.expanduser().resolve()

    summary: Dict[str, Any] = {
        "schema_version": "feature_qa_pipeline_v1_2",
        "batch_label": args.batch_label,
        "batch_csv": str(batch_csv),
        "artifacts": paths,
        "exit_codes": codes,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}", flush=True)

    worst = max(codes.values()) if codes else 0
    return worst if worst > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
