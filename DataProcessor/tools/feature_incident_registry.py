#!/usr/bin/env python3
"""
Слияние авто-отчётов (feature_quality_audit, feature_batch_drift) в реестр инцидентов.

Playbook §8: component, feature, type, affected batches, reason, status.
Новые записи: status=open, reason пустой (заполняется вручную).
Уже существующие id: обновляются last_seen_batch и metrics_snapshot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _incident_id(component: str, feature: str, typ: str) -> str:
    h = hashlib.sha256(f"{component}\0{feature}\0{typ}".encode("utf-8")).hexdigest()
    return h[:16]


def _load_registry(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": "feature_incidents_v1",
            "updated_at": _now_iso(),
            "incidents": [],
        }
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"schema_version": "feature_incidents_v1", "updated_at": _now_iso(), "incidents": []}
    if "incidents" not in data or not isinstance(data["incidents"], list):
        data["incidents"] = []
    return data


def _parse_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _merge_incident(
    incidents: List[Dict[str, Any]],
    *,
    component: str,
    feature: str,
    typ: str,
    batch_label: str,
    metrics: Dict[str, Any],
) -> None:
    iid = _incident_id(component, feature, typ)
    for inc in incidents:
        if inc.get("id") == iid:
            inc["last_seen_batch"] = batch_label
            inc["updated_at"] = _now_iso()
            inc["metrics_snapshot"] = metrics
            batches = inc.setdefault("seen_in_batches", [])
            if batch_label not in batches:
                batches.append(batch_label)
            return
    incidents.append(
        {
            "id": iid,
            "component": component,
            "feature": feature,
            "type": typ,
            "status": "open",
            "reason": "",
            "first_seen_batch": batch_label,
            "last_seen_batch": batch_label,
            "seen_in_batches": [batch_label],
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "metrics_snapshot": metrics,
        }
    )


def _ingest_quality_csv(
    path: Path,
    *,
    batch_label: str,
    incidents: List[Dict[str, Any]],
    nan_thr: float,
    oor_thr: float,
    health_thr: float,
    include_constant: bool,
) -> int:
    added = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            comp = (row.get("component") or "").strip()
            feat = (row.get("feature") or "").strip()
            if not comp or not feat:
                continue
            nan_r = _parse_float(row.get("nan_rate", "")) or 0.0
            oor_r = _parse_float(row.get("out_of_range_rate", "")) or 0.0
            health = _parse_float(row.get("health_score", ""))
            cov = _parse_float(row.get("coverage", "")) or 0.0
            const = (row.get("constant_like") or "").strip().lower() in ("1", "true", "yes")
            sev = (row.get("severity") or "").strip().lower()

            if nan_r >= nan_thr:
                _merge_incident(
                    incidents,
                    component=comp,
                    feature=feat,
                    typ="nan_spike",
                    batch_label=batch_label,
                    metrics={"nan_rate": nan_r, "coverage": cov, "source": str(path)},
                )
                added += 1
            if oor_r >= oor_thr:
                _merge_incident(
                    incidents,
                    component=comp,
                    feature=feat,
                    typ="out_of_range",
                    batch_label=batch_label,
                    metrics={"out_of_range_rate": oor_r, "source": str(path)},
                )
                added += 1
            if include_constant and const and cov >= 0.85:
                _merge_incident(
                    incidents,
                    component=comp,
                    feature=feat,
                    typ="constant",
                    batch_label=batch_label,
                    metrics={"coverage": cov, "std": row.get("std", ""), "source": str(path)},
                )
                added += 1
            if health is not None and health < health_thr:
                _merge_incident(
                    incidents,
                    component=comp,
                    feature=feat,
                    typ="health_low",
                    batch_label=batch_label,
                    metrics={"health_score": health, "severity": sev, "source": str(path)},
                )
                added += 1
    return added


_SEV_ORDER = {"low": 0, "medium": 1, "high": 2}


def _severity_at_least(sev: str, minimum: str) -> bool:
    return _SEV_ORDER.get(sev.lower(), 0) >= _SEV_ORDER.get(minimum.lower(), 0)


def _ingest_drift_csv(
    path: Path,
    *,
    batch_label: str,
    incidents: List[Dict[str, Any]],
    drift_score_min: float,
    drift_min_severity: str,
) -> int:
    added = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            comp = (row.get("component") or "").strip()
            feat = (row.get("feature") or "").strip()
            if not comp or not feat:
                continue
            sev = (row.get("severity") or "").strip().lower()
            ds = _parse_float(row.get("drift_score", "")) or 0.0
            if not _severity_at_least(sev, drift_min_severity):
                continue
            if ds < drift_score_min:
                continue
            _merge_incident(
                incidents,
                component=comp,
                feature=feat,
                typ="drift",
                batch_label=batch_label,
                metrics={
                    "drift_score": ds,
                    "severity": sev,
                    "ks": row.get("ks_statistic", ""),
                    "nan_rate_delta": row.get("nan_rate_delta", ""),
                    "source": str(path),
                },
            )
            added += 1
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge QA/drift CSV into feature_incidents.json")
    ap.add_argument(
        "--registry",
        default="",
        help="JSON реестр (default: <repo>/storage/result_store/feature_incidents.json)",
    )
    ap.add_argument(
        "--batch-label",
        required=True,
        help="Метка прогона (например 2026W17_20runs)",
    )
    ap.add_argument("--quality-csv", type=Path, default=None, help="feature_quality_report*.csv")
    ap.add_argument("--drift-csv", type=Path, default=None, help="feature_batch_drift*.csv")
    ap.add_argument("--nan-rate-min", type=float, default=0.2, help="Порог nan_spike")
    ap.add_argument("--oor-min", type=float, default=0.05, help="Порог out_of_range")
    ap.add_argument("--health-max", type=float, default=40.0, help="health_low если score ниже")
    ap.add_argument(
        "--drift-score-min",
        type=float,
        default=50.0,
        help="Мин. drift_score (после фильтра по severity)",
    )
    ap.add_argument(
        "--drift-min-severity",
        choices=("low", "medium", "high"),
        default="high",
        help="Минимальный severity из отчёта дрейфа (по умолчанию только high)",
    )
    ap.add_argument(
        "--include-constant",
        action="store_true",
        help="Добавлять инциденты constant (много шума на meta_*; по умолчанию выкл.)",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    reg_path = Path(args.registry).expanduser().resolve() if args.registry else repo / "storage" / "result_store" / "feature_incidents.json"

    data = _load_registry(reg_path)
    incidents: List[Dict[str, Any]] = data["incidents"]
    n_before = len(incidents)
    n_ops = 0
    if args.quality_csv and args.quality_csv.is_file():
        n_ops += _ingest_quality_csv(
            args.quality_csv,
            batch_label=args.batch_label,
            incidents=incidents,
            nan_thr=args.nan_rate_min,
            oor_thr=args.oor_min,
            health_thr=args.health_max,
            include_constant=args.include_constant,
        )
    if args.drift_csv and args.drift_csv.is_file():
        n_ops += _ingest_drift_csv(
            args.drift_csv,
            batch_label=args.batch_label,
            incidents=incidents,
            drift_score_min=args.drift_score_min,
            drift_min_severity=args.drift_min_severity,
        )

    data["updated_at"] = _now_iso()
    data["incidents"] = incidents
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Registry: {reg_path}")
    print(f"Incidents before: {n_before}, merge operations: {n_ops}, unique incidents now: {len(incidents)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
