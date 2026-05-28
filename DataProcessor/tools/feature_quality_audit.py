#!/usr/bin/env python3
"""
Feature Quality Audit for wide batch feature CSV.

Builds per-(component, feature) health metrics and writes:
- JSON report
- CSV flat table
- Markdown summary with top issues
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


ID_COLS = {
    "platform_id",
    "video_id",
    "run_id",
    "run_path",
    "component",
    "component_type",
    "manifest_status",
    "manifest_empty_reason",
    "duration_ms",
    "device_used",
    "npz_error",
    "render_error",
    "npz",
}

EMPTY_LIKE = {"", "none", "null", "na", "n/a", "-", "—"}


@dataclass
class FeatureStats:
    component: str
    feature: str
    rows: int
    nonempty_count: int
    coverage: float
    numeric_count: int
    nan_rate: float
    inf_rate: float
    n_unique_nonempty: int
    mean: Optional[float]
    std: Optional[float]
    p01: Optional[float]
    p50: Optional[float]
    p99: Optional[float]
    out_of_range_rate: float
    constant_like: bool
    health_score: float
    severity: str


def _parse_floatish(raw: str) -> Optional[float]:
    t = (raw or "").strip().replace(" ", "").replace(",", ".")
    if not t or t.lower() in EMPTY_LIKE:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _is_empty(raw: str) -> bool:
    t = (raw or "").strip()
    return (not t) or (t.lower() in EMPTY_LIKE)


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    if np is not None:
        return float(np.percentile(np.asarray(xs, dtype=float), q))
    ys = sorted(xs)
    idx = int(round((len(ys) - 1) * q / 100.0))
    idx = max(0, min(len(ys) - 1, idx))
    return float(ys[idx])


def _calc_health(
    *,
    coverage: float,
    nan_rate: float,
    out_of_range_rate: float,
    constant_like: bool,
) -> float:
    score = 100.0
    score -= 40.0 * nan_rate
    score -= 30.0 * out_of_range_rate
    score -= 10.0 * max(0.0, 1.0 - coverage)
    if constant_like:
        score -= 20.0
    return max(0.0, min(100.0, score))


def _severity(score: float) -> str:
    if score < 40.0:
        return "high"
    if score < 70.0:
        return "medium"
    return "low"


def _load_qa(path: Path) -> Any:
    # Local import to avoid hard dependency when script only reads stats.
    import sys

    dp = Path(__file__).resolve().parents[1]  # DataProcessor
    if str(dp) not in sys.path:
        sys.path.insert(0, str(dp))
    from qa.component_feature_qa import load_qa_config  # type: ignore

    return load_qa_config(path)


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


def _build_stats(rows: List[Dict[str, str]], header: List[str], qa_cfg: Any) -> List[FeatureStats]:
    out: List[FeatureStats] = []
    for comp, feat, vals in _iter_component_features(rows, header):
        n = len(vals)
        if n <= 0:
            continue
        nonempty = [v for v in vals if not _is_empty(v)]
        nonempty_count = len(nonempty)
        coverage = float(nonempty_count / n) if n else 0.0

        parsed = [_parse_floatish(v) for v in nonempty]
        num_vals = [x for x in parsed if x is not None]
        numeric_count = len(num_vals)
        finite_vals = [x for x in num_vals if math.isfinite(x)]

        nan_count = sum(1 for x in num_vals if isinstance(x, float) and math.isnan(x))
        inf_count = sum(1 for x in num_vals if isinstance(x, float) and math.isinf(x))
        denom = max(1, nonempty_count)
        nan_rate = float(nan_count / denom)
        inf_rate = float(inf_count / denom)

        uniq = len(set(nonempty))
        mean = std = p01 = p50 = p99 = None
        constant_like = False
        if finite_vals:
            mean = float(sum(finite_vals) / len(finite_vals))
            if np is not None and len(finite_vals) > 1:
                std = float(np.std(np.asarray(finite_vals, dtype=float), ddof=0))
            elif len(finite_vals) > 1:
                mu = mean
                std = float((sum((x - mu) ** 2 for x in finite_vals) / len(finite_vals)) ** 0.5)
            else:
                std = 0.0
            p01 = _percentile(finite_vals, 1.0)
            p50 = _percentile(finite_vals, 50.0)
            p99 = _percentile(finite_vals, 99.0)
            constant_like = bool(coverage >= 0.8 and std is not None and std <= 1e-12)

        warn_count = 0
        if qa_cfg is not None:
            for v in vals:
                w = qa_cfg.warning_for(comp, feat, str(v))
                if w:
                    warn_count += 1
        out_rate = float(warn_count / max(1, n))

        score = _calc_health(
            coverage=coverage,
            nan_rate=nan_rate,
            out_of_range_rate=out_rate,
            constant_like=constant_like,
        )
        out.append(
            FeatureStats(
                component=comp,
                feature=feat,
                rows=n,
                nonempty_count=nonempty_count,
                coverage=coverage,
                numeric_count=numeric_count,
                nan_rate=nan_rate,
                inf_rate=inf_rate,
                n_unique_nonempty=uniq,
                mean=mean,
                std=std,
                p01=p01,
                p50=p50,
                p99=p99,
                out_of_range_rate=out_rate,
                constant_like=constant_like,
                health_score=score,
                severity=_severity(score),
            )
        )
    return out


def _as_dict(x: FeatureStats) -> Dict[str, Any]:
    return {
        "component": x.component,
        "feature": x.feature,
        "rows": x.rows,
        "nonempty_count": x.nonempty_count,
        "coverage": round(x.coverage, 6),
        "numeric_count": x.numeric_count,
        "nan_rate": round(x.nan_rate, 6),
        "inf_rate": round(x.inf_rate, 6),
        "n_unique_nonempty": x.n_unique_nonempty,
        "mean": x.mean,
        "std": x.std,
        "p01": x.p01,
        "p50": x.p50,
        "p99": x.p99,
        "out_of_range_rate": round(x.out_of_range_rate, 6),
        "constant_like": x.constant_like,
        "health_score": round(x.health_score, 3),
        "severity": x.severity,
    }


def _write_csv(path: Path, rows: List[FeatureStats]) -> None:
    cols = list(_as_dict(rows[0]).keys()) if rows else [
        "component",
        "feature",
        "rows",
        "nonempty_count",
        "coverage",
        "numeric_count",
        "nan_rate",
        "inf_rate",
        "n_unique_nonempty",
        "mean",
        "std",
        "p01",
        "p50",
        "p99",
        "out_of_range_rate",
        "constant_like",
        "health_score",
        "severity",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(_as_dict(r))


def _write_md(path: Path, rows: List[FeatureStats], source_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_score = sorted(rows, key=lambda x: x.health_score)[:40]
    by_nan = sorted(rows, key=lambda x: x.nan_rate, reverse=True)[:30]
    by_oor = sorted(rows, key=lambda x: x.out_of_range_rate, reverse=True)[:30]

    lines: List[str] = []
    lines.append("# Feature Quality Report")
    lines.append("")
    lines.append(f"- Source CSV: `{source_csv}`")
    lines.append(f"- Feature rows analyzed: **{len(rows)}**")
    lines.append("")

    lines.append("## Lowest Health Score (Top 40)")
    lines.append("")
    lines.append("| component | feature | score | coverage | nan_rate | out_of_range_rate | severity |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for r in by_score:
        lines.append(
            f"| `{r.component}` | `{r.feature}` | {r.health_score:.2f} | {r.coverage:.3f} | "
            f"{r.nan_rate:.3f} | {r.out_of_range_rate:.3f} | {r.severity} |"
        )
    lines.append("")

    lines.append("## Highest NaN Rate (Top 30)")
    lines.append("")
    lines.append("| component | feature | nan_rate | coverage |")
    lines.append("|---|---|---:|---:|")
    for r in by_nan:
        lines.append(f"| `{r.component}` | `{r.feature}` | {r.nan_rate:.3f} | {r.coverage:.3f} |")
    lines.append("")

    lines.append("## Highest Out-of-Range Rate (Top 30)")
    lines.append("")
    lines.append("| component | feature | out_of_range_rate | coverage |")
    lines.append("|---|---|---:|---:|")
    for r in by_oor:
        lines.append(f"| `{r.component}` | `{r.feature}` | {r.out_of_range_rate:.3f} | {r.coverage:.3f} |")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto feature quality audit for batch_features_report CSV")
    ap.add_argument("--csv", required=True, help="Path to wide batch CSV")
    ap.add_argument(
        "--qa-config",
        default="",
        help="Path to view_csv_feature_qa.json (default: storage/result_store/view_csv_feature_qa.json)",
    )
    ap.add_argument("--out-json", required=True, help="Output JSON report")
    ap.add_argument("--out-csv", required=True, help="Output flat CSV report")
    ap.add_argument("--out-md", required=True, help="Output markdown summary")
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    if args.qa_config:
        qa_path = Path(args.qa_config).expanduser().resolve()
    else:
        qa_path = csv_path.parent / "view_csv_feature_qa.json"
    qa_cfg = _load_qa(qa_path) if qa_path.is_file() else None

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise SystemExit("CSV is empty")
        header = list(r.fieldnames)
        rows = [dict(x) for x in r]

    stats = _build_stats(rows, header, qa_cfg)
    report = {
        "schema_version": "feature_quality_report_v1",
        "source_csv": str(csv_path),
        "qa_config_path": str(qa_path) if qa_path.is_file() else None,
        "total_rows": len(rows),
        "total_features_analyzed": len(stats),
        "features": [_as_dict(x) for x in stats],
    }

    out_json = Path(args.out_json).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_md = Path(args.out_md).expanduser().resolve()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_csv, stats)
    _write_md(out_md, stats, csv_path)

    print(f"OK json: {out_json}")
    print(f"OK csv:  {out_csv}")
    print(f"OK md:   {out_md}")
    print(f"Analyzed features: {len(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

