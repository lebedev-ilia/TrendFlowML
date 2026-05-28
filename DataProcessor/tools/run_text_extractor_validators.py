#!/usr/bin/env python3
"""
Запуск всех per-extractor валидаторов TextProcessor на одном text_features.npz
или на всём result_store (--results-base), с агрегированным JSON/Markdown отчётом.

Чек-лист Autopilot: п.3 «локальные валидаторы + агрегированный лог».
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _discover_validator_scripts() -> List[Path]:
    root = _repo_root() / "DataProcessor" / "TextProcessor" / "src" / "extractors"
    out: List[Path] = []
    for p in sorted(root.rglob("validate*_text_npz.py")):
        if p.is_file():
            out.append(p)
    return out


def _run_one(
    py: str,
    script: Path,
    npz: Optional[str],
    results_base: Optional[str],
    platform_id: str,
    timings: bool,
) -> Dict[str, Any]:
    cmd: List[str] = [py, str(script)]
    if results_base:
        cmd.append("--results-base")
        cmd.append(results_base)
        cmd.extend(["--platform-id", platform_id])
    elif npz:
        cmd.append(npz)
    else:
        raise ValueError("npz or results_base")
    cmd.extend(["--struct", "--ranges"])
    if timings:
        cmd.append("--timings")
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )
    return {
        "script": str(script.relative_to(_repo_root())),
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-8000:] if proc.stdout else "",
        "stderr": proc.stderr[-4000:] if proc.stderr else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Агрегированный прогон validate_*_text_npz.py (TextProcessor)"
    )
    ap.add_argument("npz_path", nargs="?", help="Путь к text_features.npz")
    ap.add_argument(
        "--results-base",
        help="Корень result_store: проверяются все .../text_processor/text_features.npz",
    )
    ap.add_argument("--platform-id", default="youtube")
    ap.add_argument(
        "--timings",
        action="store_true",
        help="Передавать --timings (скрипты без флага выдадут ошибку argparse — см. отчёт)",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        help="Путь для машинного отчёта (по умолчанию рядом с npz или в cwd)",
    )
    ap.add_argument("--out-md", type=Path, help="Краткий Markdown-отчёт")
    args = ap.parse_args()

    if bool(args.npz_path) == bool(args.results_base):
        ap.error("нужен ровно один из: npz_path или --results-base")
        return 2

    py = sys.executable
    scripts = _discover_validator_scripts()
    if not scripts:
        print("не найдены validate*_text_npz.py", file=sys.stderr)
        return 1

    rows: List[Dict[str, Any]] = []
    worst = 0
    for sc in scripts:
        try:
            row = _run_one(
                py, sc, args.npz_path, args.results_base, args.platform_id, args.timings
            )
        except Exception as e:
            row = {"script": str(sc), "exit_code": 99, "error": str(e)}
        rows.append(row)
        ec = int(row.get("exit_code") or 0)
        if ec > worst:
            worst = ec

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": {"npz_path": args.npz_path, "results_base": args.results_base},
        "validators_run": len(rows),
        "max_exit_code": worst,
        "results": rows,
    }

    out_json = args.out_json
    if out_json is None:
        if args.npz_path:
            out_json = Path(args.npz_path).resolve().parent / "text_extractor_validators_report.json"
        else:
            out_json = Path.cwd() / "text_extractor_validators_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")

    if args.out_md:
        lines = [
            "# TextProcessor validators aggregate",
            "",
            f"- UTC: `{summary['generated_at_utc']}`",
            f"- Validators: {len(rows)}",
            f"- Max exit code: {worst}",
            "",
        ]
        bad = [r for r in rows if int(r.get("exit_code") or 0) != 0]
        if not bad:
            lines.append("All validators exited 0.")
        else:
            lines.append("## Non-zero exit")
            for r in bad:
                lines.append(f"- `{r.get('script')}` → **{r.get('exit_code')}**")
                err = (r.get("stderr") or "").strip()
                if err:
                    lines.append(f"  ```\n{err[:500]}\n  ```")
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {args.out_md}")

    return 0 if worst <= 1 else min(worst, 255)


if __name__ == "__main__":
    raise SystemExit(main())
