#!/usr/bin/env python3
"""
Попарное сравнение двух wide batch CSV для одних и тех же (platform_id, video_id, run_id, component).

Playbook §5: повторный прогон / регрессия чисел с допусками.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from feature_quality_audit import EMPTY_LIKE, ID_COLS  # noqa: E402


def _row_key(r: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (
        (r.get("platform_id") or "").strip(),
        (r.get("video_id") or "").strip(),
        (r.get("run_id") or "").strip(),
        (r.get("component") or "").strip(),
    )


def _parse_num(raw: str) -> Optional[float]:
    t = (raw or "").strip().replace(" ", "").replace(",", ".")
    if not t or t.lower() in EMPTY_LIKE:
        return None
    if t.lower() in ("nan", "inf", "-inf", "+inf"):
        try:
            return float(t)
        except ValueError:
            return None
    try:
        return float(t)
    except ValueError:
        return None


def _close(a: float, b: float, abs_eps: float, rel_eps: float) -> bool:
    diff = abs(a - b)
    if diff <= abs_eps:
        return True
    scale = max(abs(a), abs(b), 1.0)
    return diff <= rel_eps * scale


def _load_rows(path: Path) -> Tuple[Dict[Tuple[str, str, str, str], Dict[str, str]], int]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"empty csv: {path}")
        by: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
        dups = 0
        for row in reader:
            k = _row_key(row)
            if not k[0] or not k[1] or not k[2] or not k[3]:
                continue
            if k in by:
                dups += 1
            by[k] = dict(row)
        return by, dups


def main() -> int:
    ap = argparse.ArgumentParser(description="Golden: compare two batch CSVs row-wise")
    ap.add_argument("--csv-a", required=True, help="Прогон A (эталон)")
    ap.add_argument("--csv-b", required=True, help="Прогон B")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--abs-eps", type=float, default=1e-5)
    ap.add_argument("--rel-eps", type=float, default=1e-4)
    args = ap.parse_args()

    pa, pb = Path(args.csv_a).resolve(), Path(args.csv_b).resolve()
    map_a, da = _load_rows(pa)
    map_b, db = _load_rows(pb)
    keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    if not keys:
        raise SystemExit("no overlapping (platform_id, video_id, run_id, component) rows")

    feat_cols = [h for h in map_a[next(iter(map_a))].keys() if h not in ID_COLS]
    mismatches: List[Dict[str, Any]] = []
    cells_ok = 0
    cells_cmp = 0
    for k in keys:
        ra, rb = map_a[k], map_b[k]
        for col in feat_cols:
            sa, sb = str(ra.get(col, "") or ""), str(rb.get(col, "") or "")
            ta, tb = sa.strip(), sb.strip()
            if not ta and not tb:
                continue
            if not ta or not tb:
                mismatches.append(
                    {
                        "platform_id": k[0],
                        "video_id": k[1],
                        "run_id": k[2],
                        "component": k[3],
                        "feature": col,
                        "value_a": sa[:200],
                        "value_b": sb[:200],
                        "reason": "empty_mismatch",
                    }
                )
                cells_cmp += 1
                continue
            fa, fb = _parse_num(sa), _parse_num(sb)
            if fa is not None and fb is not None:
                cells_cmp += 1
                if math.isnan(fa) and math.isnan(fb):
                    cells_ok += 1
                    continue
                if math.isnan(fa) or math.isnan(fb) or math.isinf(fa) or math.isinf(fb):
                    if fa == fb or (math.isnan(fa) and math.isnan(fb)):
                        cells_ok += 1
                    else:
                        mismatches.append(
                            {
                                "platform_id": k[0],
                                "video_id": k[1],
                                "run_id": k[2],
                                "component": k[3],
                                "feature": col,
                                "value_a": sa[:200],
                                "value_b": sb[:200],
                                "reason": "nonfinite_mismatch",
                            }
                        )
                    continue
                if _close(fa, fb, args.abs_eps, args.rel_eps):
                    cells_ok += 1
                else:
                    mismatches.append(
                        {
                            "platform_id": k[0],
                            "video_id": k[1],
                            "run_id": k[2],
                            "component": k[3],
                            "feature": col,
                            "value_a": sa[:200],
                            "value_b": sb[:200],
                            "reason": "numeric_mismatch",
                        }
                    )
                continue
            if ta == tb:
                cells_cmp += 1
                cells_ok += 1
            else:
                if ta.lower() == tb.lower():
                    cells_cmp += 1
                    cells_ok += 1
                else:
                    mismatches.append(
                        {
                            "platform_id": k[0],
                            "video_id": k[1],
                            "run_id": k[2],
                            "component": k[3],
                            "feature": col,
                            "value_a": sa[:200],
                            "value_b": sb[:200],
                            "reason": "string_mismatch",
                        }
                    )
                    cells_cmp += 1

    out_csv = Path(args.out_csv).resolve()
    out_md = Path(args.out_md).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "platform_id",
        "video_id",
        "run_id",
        "component",
        "feature",
        "reason",
        "value_a",
        "value_b",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for m in mismatches:
            w.writerow({c: m.get(c, "") for c in cols})

    rate = (1.0 - len(mismatches) / max(1, cells_cmp)) if cells_cmp else 1.0
    lines = [
        "# Golden batch compare",
        "",
        f"- A: `{pa}`",
        f"- B: `{pb}`",
        f"- Overlapping rows: **{len(keys)}** (dup keys skipped: A={da}, B={db})",
        f"- Cells compared: **{cells_cmp}**, matches: **{cells_ok}**, mismatches: **{len(mismatches)}**",
        f"- Match rate (of compared cells): **{rate:.4f}**",
        "",
    ]
    if mismatches:
        lines.append("## First 40 mismatches")
        lines.append("")
        lines.append("| video_id | component | feature | reason |")
        lines.append("|---|---|---|---|")
        for m in mismatches[:40]:
            lines.append(
                f"| `{m['video_id']}` | `{m['component']}` | `{m['feature']}` | {m['reason']} |"
            )
    else:
        lines.append("No mismatches under tolerance.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OK csv: {out_csv}")
    print(f"OK md:  {out_md}")
    print(f"rows={len(keys)} mismatches={len(mismatches)} compared_cells={cells_cmp}")
    return 0 if not mismatches else 2


if __name__ == "__main__":
    raise SystemExit(main())
