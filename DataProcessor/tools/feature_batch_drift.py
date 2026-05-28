#!/usr/bin/env python3
"""
Дрейф фич между двумя wide batch CSV (как у batch_runs_feature_report).

Считает по каждой паре (component, feature):
  - coverage / nan_rate в батче A и B
  - |Δ p50|, |Δ mean| (и относительный сдвиг mean при |mean_a| > eps)
  - двухвыборочный KS-статистик (0..1), только по конечным числам

Использование: еженедельное сравнение «текущий батч» vs «эталонный» (playbook §4, drift).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
from feature_quality_audit import ID_COLS, _is_empty, _parse_floatish, _percentile  # noqa: E402


@dataclass
class DriftRow:
    component: str
    feature: str
    n_a: int
    n_b: int
    coverage_a: float
    coverage_b: float
    nan_rate_a: float
    nan_rate_b: float
    nan_rate_delta: float
    mean_a: Optional[float]
    mean_b: Optional[float]
    mean_abs_diff: Optional[float]
    mean_rel_diff: Optional[float]
    p50_a: Optional[float]
    p50_b: Optional[float]
    p50_abs_diff: Optional[float]
    ks_statistic: Optional[float]
    n_finite_a: int
    n_finite_b: int
    drift_score: float
    severity: str


def _finite_numeric(vals: List[str]) -> List[float]:
    out: List[float] = []
    for s in vals:
        if _is_empty(s):
            continue
        x = _parse_floatish(s)
        if x is None:
            continue
        if math.isfinite(x):
            out.append(float(x))
    return out


def _ks_two_sample(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    aa = np.sort(np.asarray(a, dtype=np.float64))
    bb = np.sort(np.asarray(b, dtype=np.float64))
    n, m = int(aa.size), int(bb.size)
    merged = np.unique(np.concatenate([aa, bb]))
    best = 0.0
    for x in merged:
        fa = float(np.searchsorted(aa, x, side="right") / n)
        fb = float(np.searchsorted(bb, x, side="right") / m)
        d = abs(fa - fb)
        if d > best:
            best = d
    return float(best)


def _iter_component_features(
    rows: List[Dict[str, str]],
    header: Iterable[str],
) -> Iterable[Tuple[str, str, List[str]]]:
    by_comp: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        comp = (r.get("component") or "").strip() or "unknown"
        by_comp.setdefault(comp, []).append(r)
    feats = [h for h in header if h not in ID_COLS]
    for comp, comp_rows in sorted(by_comp.items(), key=lambda x: x[0]):
        for feat in feats:
            vals = [str(rr.get(feat, "") or "") for rr in comp_rows]
            yield comp, feat, vals


def _stats_for_vals(vals: List[str]) -> Tuple[int, float, float, Optional[float], Optional[float]]:
    n = len(vals)
    nonempty = [v for v in vals if not _is_empty(v)]
    cov = float(len(nonempty) / n) if n else 0.0
    parsed = [_parse_floatish(v) for v in nonempty]
    num = [x for x in parsed if x is not None]
    nan_c = sum(1 for x in num if isinstance(x, float) and math.isnan(x))
    denom = max(1, len(nonempty))
    nan_r = float(nan_c / denom)
    finite = [float(x) for x in num if isinstance(x, float) and math.isfinite(x)]
    if not finite:
        return n, cov, nan_r, None, None
    mu = float(sum(finite) / len(finite))
    p50 = _percentile(finite, 50.0)
    return n, cov, nan_r, mu, float(p50)


def _drift_severity(
    *,
    ks: Optional[float],
    nan_delta: float,
    mean_rel: Optional[float],
    p50_diff: Optional[float],
    scale: Optional[float],
) -> Tuple[float, str]:
    """Эвристический drift_score 0..100 (больше = сильнее дрейф)."""
    score = 0.0
    if ks is not None:
        score += 100.0 * min(1.0, ks / 0.25)
    score += min(40.0, abs(nan_delta) * 120.0)
    if mean_rel is not None:
        score += min(35.0, abs(mean_rel) * 35.0)
    if p50_diff is not None and scale is not None and scale > 1e-9:
        score += min(25.0, abs(p50_diff) / scale * 25.0)
    score = max(0.0, min(100.0, score))
    if score >= 55:
        sev = "high"
    elif score >= 30:
        sev = "medium"
    else:
        sev = "low"
    return score, sev


def _build_drift(
    rows_a: List[Dict[str, str]],
    rows_b: List[Dict[str, str]],
    header: List[str],
    min_finite_per_side: int,
) -> List[DriftRow]:
    map_a: Dict[Tuple[str, str], List[str]] = {}
    for comp, feat, vals in _iter_component_features(rows_a, header):
        map_a[(comp, feat)] = vals
    map_b: Dict[Tuple[str, str], List[str]] = {}
    for comp, feat, vals in _iter_component_features(rows_b, header):
        map_b[(comp, feat)] = vals

    keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    out: List[DriftRow] = []
    for comp, feat in keys:
        va, vb = map_a[(comp, feat)], map_b[(comp, feat)]
        n_a, cov_a, nan_a, mean_a, p50_a = _stats_for_vals(va)
        n_b, cov_b, nan_b, mean_b, p50_b = _stats_for_vals(vb)
        fa, fb = _finite_numeric(va), _finite_numeric(vb)
        ks = _ks_two_sample(fa, fb) if len(fa) >= min_finite_per_side and len(fb) >= min_finite_per_side else None
        nan_delta = nan_b - nan_a
        mean_abs = (
            abs(mean_a - mean_b)
            if mean_a is not None and mean_b is not None
            else None
        )
        mean_rel = None
        if mean_a is not None and mean_b is not None and abs(mean_a) > 1e-6:
            mean_rel = (mean_b - mean_a) / mean_a
        p50_diff = (
            abs(p50_a - p50_b)
            if p50_a is not None and p50_b is not None
            else None
        )
        scale = None
        if p50_a is not None and p50_b is not None:
            scale = max(abs(p50_a), abs(p50_b), 1e-9)
        drift_score, severity = _drift_severity(
            ks=ks,
            nan_delta=nan_delta,
            mean_rel=mean_rel,
            p50_diff=(abs(p50_a - p50_b) if p50_a is not None and p50_b is not None else None),
            scale=scale,
        )
        out.append(
            DriftRow(
                component=comp,
                feature=feat,
                n_a=n_a,
                n_b=n_b,
                coverage_a=cov_a,
                coverage_b=cov_b,
                nan_rate_a=nan_a,
                nan_rate_b=nan_b,
                nan_rate_delta=nan_delta,
                mean_a=mean_a,
                mean_b=mean_b,
                mean_abs_diff=mean_abs,
                mean_rel_diff=mean_rel,
                p50_a=p50_a,
                p50_b=p50_b,
                p50_abs_diff=p50_diff,
                ks_statistic=ks,
                n_finite_a=len(fa),
                n_finite_b=len(fb),
                drift_score=drift_score,
                severity=severity,
            )
        )
    return out


def _as_dict(r: DriftRow) -> Dict[str, Any]:
    return {
        "component": r.component,
        "feature": r.feature,
        "n_a": r.n_a,
        "n_b": r.n_b,
        "coverage_a": round(r.coverage_a, 6),
        "coverage_b": round(r.coverage_b, 6),
        "nan_rate_a": round(r.nan_rate_a, 6),
        "nan_rate_b": round(r.nan_rate_b, 6),
        "nan_rate_delta": round(r.nan_rate_delta, 6),
        "mean_a": r.mean_a,
        "mean_b": r.mean_b,
        "mean_abs_diff": r.mean_abs_diff,
        "mean_rel_diff": r.mean_rel_diff,
        "p50_a": r.p50_a,
        "p50_b": r.p50_b,
        "p50_abs_diff": r.p50_abs_diff,
        "ks_statistic": r.ks_statistic,
        "n_finite_a": r.n_finite_a,
        "n_finite_b": r.n_finite_b,
        "drift_score": round(r.drift_score, 3),
        "severity": r.severity,
    }


def _write_md(path: Path, rows: List[DriftRow], csv_a: Path, csv_b: Path) -> None:
    top = sorted(rows, key=lambda x: x.drift_score, reverse=True)[:50]
    lines = [
        "# Feature batch drift",
        "",
        f"- CSV A (reference / baseline): `{csv_a}`",
        f"- CSV B (current): `{csv_b}`",
        f"- Pairs compared: **{len(rows)}**",
        "",
        "## Top drift (by heuristic score)",
        "",
        "| component | feature | drift_score | severity | KS | Δnan_rate | |Δmean| | n_finite A/B |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in top:
        ks_s = f"{r.ks_statistic:.3f}" if r.ks_statistic is not None else "—"
        dm = f"{r.mean_abs_diff:.4g}" if r.mean_abs_diff is not None else "—"
        lines.append(
            f"| `{r.component}` | `{r.feature}` | {r.drift_score:.1f} | {r.severity} | "
            f"{ks_s} | {r.nan_rate_delta:+.3f} | {dm} | {r.n_finite_a}/{r.n_finite_b} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Drift between two batch_features_report CSVs")
    ap.add_argument("--csv-a", required=True, help="Baseline / older batch CSV")
    ap.add_argument("--csv-b", required=True, help="Current batch CSV")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument(
        "--min-finite-per-side",
        type=int,
        default=5,
        help="Minimum finite numeric samples per batch to compute KS",
    )
    args = ap.parse_args()

    pa = Path(args.csv_a).expanduser().resolve()
    pb = Path(args.csv_b).expanduser().resolve()
    for p in (pa, pb):
        if not p.is_file():
            raise SystemExit(f"CSV not found: {p}")

    with open(pa, "r", encoding="utf-8", newline="") as f:
        ra = csv.DictReader(f)
        if not ra.fieldnames:
            raise SystemExit("csv-a empty")
        ha = list(ra.fieldnames)
        rows_a = [dict(x) for x in ra]
    with open(pb, "r", encoding="utf-8", newline="") as f:
        rb = csv.DictReader(f)
        if not rb.fieldnames:
            raise SystemExit("csv-b empty")
        hb = list(rb.fieldnames)
        rows_b = [dict(x) for x in rb]

    if ha != hb:
        # Сравниваем пересечение колонок (типично одна версия отчёта с лишними полями)
        common = [h for h in ha if h in set(hb)]
        if "component" not in common:
            raise SystemExit("csv-a and csv-b headers differ; no safe common columns")
        ha = hb = common
        for row in rows_a:
            for k in list(row.keys()):
                if k not in common:
                    del row[k]
        for row in rows_b:
            for k in list(row.keys()):
                if k not in common:
                    del row[k]

    drift_rows = _build_drift(rows_a, rows_b, ha, args.min_finite_per_side)
    report = {
        "schema_version": "feature_batch_drift_v1",
        "csv_a": str(pa),
        "csv_b": str(pb),
        "rows_a": len(rows_a),
        "rows_b": len(rows_b),
        "pairs_compared": len(drift_rows),
        "drift": [_as_dict(x) for x in drift_rows],
    }

    out_json = Path(args.out_json).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_md = Path(args.out_md).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = list(_as_dict(drift_rows[0]).keys()) if drift_rows else []
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in drift_rows:
            w.writerow(_as_dict(r))

    _write_md(out_md, drift_rows, pa, pb)
    print(f"OK json: {out_json}")
    print(f"OK csv:  {out_csv}")
    print(f"OK md:   {out_md}")
    print(f"Compared pairs: {len(drift_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
