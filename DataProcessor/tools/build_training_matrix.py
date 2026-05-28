#!/usr/bin/env python3
"""
Build ML-ready matrix from wide batch CSV + external target table.

Pipeline:
1) Collapse wide batch rows to one row per (platform_id, video_id, run_id).
2) Join target rows by key (default video_id).
3) Keep numeric feature columns only (float parse).
4) Optional denylist from manifest leakage hints + user regex.
5) Optional time-based split by publish datetime column.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from feature_quality_audit import EMPTY_LIKE, ID_COLS  # noqa: E402


def _parse_floatish(raw: str) -> Optional[float]:
    t = (raw or "").strip().replace(" ", "").replace(",", ".")
    if not t or t.lower() in EMPTY_LIKE:
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return v


def _safe_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    return _parse_floatish(str(raw))


def _parse_dt(raw: str) -> Optional[datetime]:
    s = (raw or "").strip()
    if not s:
        return None
    # Common cases: 2026-04-18T12:53:42Z / 2026-04-18 / with timezone offset
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    for fmt in ("%Y-%m-%d",):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _row_group_key(r: Dict[str, str], cols: Sequence[str]) -> Tuple[str, ...]:
    return tuple((r.get(c) or "").strip() for c in cols)


def _merge_first_nonempty(rows: List[Dict[str, str]], header: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for h in header:
        out[h] = ""
    for r in rows:
        for h in header:
            cur = out[h]
            if cur:
                continue
            v = str(r.get(h, "") or "").strip()
            if v and v.lower() not in EMPTY_LIKE:
                out[h] = v
    return out


def _load_manifest_denylist(path: Optional[Path]) -> List[str]:
    if not path or not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    cols = data.get("columns", [])
    out: List[str] = []
    if isinstance(cols, list):
        for c in cols:
            if not isinstance(c, dict):
                continue
            name = c.get("name")
            leak = c.get("leakage_hint")
            if isinstance(name, str) and leak:
                out.append(name)
    return sorted(set(out))


def _compile_patterns(items: Sequence[str]) -> List[re.Pattern[str]]:
    out: List[re.Pattern[str]] = []
    for x in items:
        try:
            out.append(re.compile(x))
        except re.error:
            continue
    return out


def _match_any(name: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(p.search(name) for p in patterns)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build ML matrix from wide batch CSV and target table")
    ap.add_argument("--batch-csv", type=Path, required=True)
    ap.add_argument("--target-csv", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True, help="Joined full dataset")
    ap.add_argument("--out-metadata-json", type=Path, required=True)
    ap.add_argument("--out-train-csv", type=Path, default=None)
    ap.add_argument("--out-val-csv", type=Path, default=None)
    ap.add_argument("--out-test-csv", type=Path, default=None)
    ap.add_argument(
        "--group-key-cols",
        default="platform_id,video_id,run_id",
        help="Columns used to collapse batch rows before join",
    )
    ap.add_argument("--target-key-col", default="video_id")
    ap.add_argument("--target-value-col", required=True, help="Target y column in target CSV")
    ap.add_argument("--target-time-col", default="", help="Datetime column for time split")
    ap.add_argument("--split-ratios", default="0.7,0.15,0.15", help="train,val,test ratios")
    ap.add_argument("--manifest-json", type=Path, default=None, help="wide_batch_feature_manifest*.json")
    ap.add_argument(
        "--denylist-regex",
        action="append",
        default=[],
        help="Regex columns to exclude from X (repeatable)",
    )
    ap.add_argument(
        "--include-meta-features",
        action="store_true",
        help="Keep meta_* numeric columns in X (default off)",
    )
    ap.add_argument("--min-numeric-rate", type=float, default=0.7, help="Min finite numeric share to keep feature")
    args = ap.parse_args()

    batch_csv = args.batch_csv.expanduser().resolve()
    target_csv = args.target_csv.expanduser().resolve()
    if not batch_csv.is_file() or not target_csv.is_file():
        raise SystemExit("batch-csv or target-csv not found")

    group_cols = [x.strip() for x in args.group_key_cols.split(",") if x.strip()]
    if not group_cols:
        raise SystemExit("group-key-cols is empty")

    # 1) Collapse component rows into one row per run/video key.
    with open(batch_csv, "r", encoding="utf-8", newline="") as f:
        rr = csv.DictReader(f)
        header = list(rr.fieldnames or [])
        rows = [dict(x) for x in rr]
    if not header or not rows:
        raise SystemExit("batch CSV is empty")
    missing_group = [c for c in group_cols if c not in header]
    if missing_group:
        raise SystemExit(f"group columns missing in batch CSV: {missing_group}")

    by_key: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_key[_row_group_key(r, group_cols)].append(r)
    collapsed: List[Dict[str, str]] = []
    for key, grp_rows in sorted(by_key.items(), key=lambda x: x[0]):
        agg = _merge_first_nonempty(grp_rows, header)
        for i, col in enumerate(group_cols):
            agg[col] = key[i]
        collapsed.append(agg)

    # 2) Target join.
    with open(target_csv, "r", encoding="utf-8", newline="") as f:
        tr = csv.DictReader(f)
        th = list(tr.fieldnames or [])
        trows = [dict(x) for x in tr]
    if args.target_key_col not in th or args.target_value_col not in th:
        raise SystemExit("target key/value column not found in target CSV")
    tmap: Dict[str, Dict[str, str]] = {}
    for r in trows:
        k = (r.get(args.target_key_col) or "").strip()
        if k:
            tmap[k] = r

    joined: List[Dict[str, Any]] = []
    unmatched = 0
    for r in collapsed:
        key = (r.get(args.target_key_col) or "").strip()
        t = tmap.get(key)
        if not t:
            unmatched += 1
            continue
        y = _safe_float(t.get(args.target_value_col))
        if y is None:
            continue
        out = dict(r)
        out["target"] = y
        if args.target_time_col:
            out["_target_time_raw"] = str(t.get(args.target_time_col, "") or "")
        joined.append(out)

    if not joined:
        raise SystemExit("No joined rows with numeric target")

    # 3) Column selection for X.
    deny_cols: set[str] = set(_load_manifest_denylist(args.manifest_json))
    deny_patterns = _compile_patterns(args.denylist_regex)

    numeric_candidates: List[str] = []
    for col in header:
        if col in ID_COLS or col in ("component", "component_type"):
            continue
        if not args.include_meta_features and col.startswith("meta_"):
            continue
        if col in deny_cols:
            continue
        if _match_any(col, deny_patterns):
            continue
        numeric_candidates.append(col)

    keep_x: List[str] = []
    for col in numeric_candidates:
        finite = 0
        for r in joined:
            if _safe_float(r.get(col)) is not None:
                finite += 1
        rate = finite / max(1, len(joined))
        if rate >= args.min_numeric_rate:
            keep_x.append(col)

    # Build matrix rows.
    ds_rows: List[Dict[str, Any]] = []
    for r in joined:
        row: Dict[str, Any] = {}
        for k in group_cols:
            row[k] = r.get(k, "")
        if "video_id" not in row:
            row["video_id"] = r.get("video_id", "")
        row["target"] = r["target"]
        if args.target_time_col:
            row["_target_time_raw"] = r.get("_target_time_raw", "")
        for c in keep_x:
            row[c] = _safe_float(r.get(c))
        ds_rows.append(row)

    # 4) Optional time split.
    train_rows = val_rows = test_rows = None
    split_info: Dict[str, Any] = {"mode": "none"}
    if args.target_time_col:
        parsed: List[Tuple[datetime, Dict[str, Any]]] = []
        for r in ds_rows:
            dt = _parse_dt(str(r.get("_target_time_raw", "")))
            if dt is not None:
                parsed.append((dt, r))
        if parsed:
            parsed.sort(key=lambda x: x[0])
            parts = [x.strip() for x in args.split_ratios.split(",")]
            if len(parts) != 3:
                raise SystemExit("split-ratios must be train,val,test")
            trn, val, tst = (float(parts[0]), float(parts[1]), float(parts[2]))
            s = trn + val + tst
            if s <= 0:
                raise SystemExit("invalid split-ratios sum")
            trn, val, tst = trn / s, val / s, tst / s
            n = len(parsed)
            i1 = int(round(n * trn))
            i2 = int(round(n * (trn + val)))
            train_rows = [x[1] for x in parsed[:i1]]
            val_rows = [x[1] for x in parsed[i1:i2]]
            test_rows = [x[1] for x in parsed[i2:]]
            split_info = {
                "mode": "time",
                "ratios": {"train": trn, "val": val, "test": tst},
                "counts": {
                    "all_with_time": n,
                    "train": len(train_rows),
                    "val": len(val_rows),
                    "test": len(test_rows),
                },
            }
        else:
            split_info = {"mode": "time_requested_but_no_parseable_datetimes"}

    # 5) Write outputs.
    out_csv = args.out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_fields = list(group_cols)
    if "video_id" not in out_fields:
        out_fields.append("video_id")
    out_fields.extend(["target"])
    if args.target_time_col:
        out_fields.append("_target_time_raw")
    out_fields.extend(keep_x)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(ds_rows)

    def _write_part(path: Optional[Path], part_rows: Optional[List[Dict[str, Any]]]) -> None:
        if not path or part_rows is None:
            return
        p = path.expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as ff:
            ww = csv.DictWriter(ff, fieldnames=out_fields, extrasaction="ignore")
            ww.writeheader()
            ww.writerows(part_rows)

    _write_part(args.out_train_csv, train_rows)
    _write_part(args.out_val_csv, val_rows)
    _write_part(args.out_test_csv, test_rows)

    meta = {
        "schema_version": "training_matrix_v1",
        "batch_csv": str(batch_csv),
        "target_csv": str(target_csv),
        "rows_batch_raw": len(rows),
        "rows_collapsed": len(collapsed),
        "rows_joined": len(joined),
        "rows_unmatched_target": unmatched,
        "feature_count_kept": len(keep_x),
        "group_key_cols": group_cols,
        "target_key_col": args.target_key_col,
        "target_value_col": args.target_value_col,
        "target_time_col": args.target_time_col or None,
        "deny_from_manifest_count": len(deny_cols),
        "deny_regex_count": len(deny_patterns),
        "min_numeric_rate": args.min_numeric_rate,
        "split": split_info,
        "outputs": {
            "dataset_csv": str(out_csv),
            "train_csv": str(args.out_train_csv.expanduser().resolve()) if args.out_train_csv else None,
            "val_csv": str(args.out_val_csv.expanduser().resolve()) if args.out_val_csv else None,
            "test_csv": str(args.out_test_csv.expanduser().resolve()) if args.out_test_csv else None,
        },
        "dropped_by_manifest_examples": sorted(list(deny_cols))[:50],
        "kept_feature_examples": keep_x[:50],
    }
    out_meta = args.out_metadata_json.expanduser().resolve()
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK dataset: {out_csv} rows={len(ds_rows)} features={len(keep_x)}")
    print(f"OK metadata: {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

