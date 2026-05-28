#!/usr/bin/env python3
"""
Манифест колонок wide batch CSV для подготовки обучения (playbook §6).

Один проход по строкам: для каждой колонки считает nonempty_rate, numeric_rate
(доля значений, которые парсятся в finite float). Помечает id/meta/подозрительные
на leakage имена — эвристика для ручной проверки перед time-split и таргетом.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from feature_quality_audit import EMPTY_LIKE, ID_COLS  # noqa: E402

_LEAK_RE = re.compile(
    r"(target|label|popularity|views?_?|watch_?time|retention|"
    r"future_|post_?hoc|leak|after_?publish|"
    r"day_?7|day_?14|day_?21|horizon|"
    r"subscriber_?gain|like_?count|comment_?count)",
    re.I,
)


def _parse_floatish(raw: str) -> Optional[float]:
    t = (raw or "").strip().replace(" ", "").replace(",", ".")
    if not t or t.lower() in EMPTY_LIKE:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _role(col: str) -> str:
    if col in ID_COLS:
        return "id"
    if col.startswith("meta_"):
        return "meta"
    return "feature"


def _leak_hint(col: str) -> str:
    if _LEAK_RE.search(col):
        return "name_matches_leakage_heuristic"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Manifest of wide batch CSV columns for ML prep")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, default=None, help="Плоская таблица по колонкам")
    ap.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Ограничить число строк (0 = все)",
    )
    args = ap.parse_args()

    path = args.csv.expanduser().resolve()
    if not path.is_file():
        print(f"not found: {path}", file=sys.stderr)
        return 1

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        nempty = {h: 0 for h in fields}
        nnum = {h: 0 for h in fields}
        nrows = 0
        for row in reader:
            nrows += 1
            for h in fields:
                s = str(row.get(h, "") or "").strip()
                if s and s.lower() not in EMPTY_LIKE:
                    nempty[h] += 1
                    v = _parse_floatish(s)
                    if v is not None and math.isfinite(v):
                        nnum[h] += 1
            if args.max_rows > 0 and nrows >= args.max_rows:
                break

    if nrows == 0:
        print("no rows", file=sys.stderr)
        return 1

    columns: List[Dict[str, Any]] = []
    for h in fields:
        ne = nempty[h]
        nn = nnum[h]
        leak = _leak_hint(h)
        columns.append(
            {
                "name": h,
                "role": _role(h),
                "nonempty_rate": round(ne / nrows, 6),
                "numeric_rate": round(nn / nrows, 6),
                "leakage_hint": leak or None,
            }
        )

    feat_like = [c for c in columns if c["role"] == "feature" and c["numeric_rate"] >= 0.01]
    leak_suspects = [c for c in columns if c["leakage_hint"]]

    report = {
        "schema_version": "wide_batch_feature_manifest_v1",
        "source_csv": str(path),
        "rows_scanned": nrows,
        "column_count": len(fields),
        "feature_like_numeric_count": len(feat_like),
        "leakage_name_suspects_count": len(leak_suspects),
        "columns": columns,
        "notes": [
            "leakage_hint — только эвристика по имени; таргет и post-hoc поля нужно явно исключать по контракту.",
            "Для time-based split используйте дату публикации из внешней таблицы, не из этого CSV.",
        ],
    }

    out_j = args.out_json.expanduser().resolve()
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK json: {out_j}  rows={nrows} cols={len(fields)}")

    if args.out_csv:
        out_c = args.out_csv.expanduser().resolve()
        out_c.parent.mkdir(parents=True, exist_ok=True)
        cols = ["name", "role", "nonempty_rate", "numeric_rate", "leakage_hint"]
        with open(out_c, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for c in columns:
                row = dict(c)
                if row.get("leakage_hint") is None:
                    row["leakage_hint"] = ""
                w.writerow(row)
        print(f"OK csv: {out_c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
