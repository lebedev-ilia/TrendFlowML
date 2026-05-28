from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _coerce_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _coerce_str(x: Any) -> str:
    return str(x) if x is not None else ""


def _summarize_components(components: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_status: Dict[str, int] = {}
    durations: List[int] = []
    per_component: List[Dict[str, Any]] = []

    for c in components:
        name = _coerce_str(c.get("name")).strip() or "unknown"
        kind = _coerce_str(c.get("kind")).strip() or "other"
        status = _coerce_str(c.get("status")).strip() or "unknown"
        by_status[status] = by_status.get(status, 0) + 1

        d = _coerce_int(c.get("duration_ms"))
        if isinstance(d, int) and d >= 0:
            durations.append(d)

        artifacts = c.get("artifacts") if isinstance(c.get("artifacts"), list) else []
        per_component.append(
            {
                "name": name,
                "kind": kind,
                "status": status,
                "duration_ms": d,
                "device_used": c.get("device_used"),
                "schema_version": c.get("schema_version"),
                "producer_version": c.get("producer_version"),
                "empty_reason": c.get("empty_reason"),
                "error_code": c.get("error_code"),
                "artifact_count": len(artifacts),
            }
        )

    per_component.sort(key=lambda x: (str(x.get("kind") or ""), str(x.get("name") or "")))
    total_duration_ms = sum(durations) if durations else 0
    max_duration_ms = max(durations) if durations else 0

    return {
        "counts": {
            "total": len(components),
            "by_status": dict(sorted(by_status.items(), key=lambda kv: kv[0])),
        },
        "timing": {
            "total_duration_ms_sum": total_duration_ms,
            "max_component_duration_ms": max_duration_ms,
        },
        "components": per_component,
    }


def summarize_manifest(manifest_path: str) -> Dict[str, Any]:
    m = _load_json(manifest_path)
    run = m.get("run") if isinstance(m.get("run"), dict) else {}
    comps_raw = m.get("components") if isinstance(m.get("components"), list) else []
    comps: List[Dict[str, Any]] = [c for c in comps_raw if isinstance(c, dict)]

    summary = _summarize_components(comps)

    return {
        "schema_version": "run_manifest_summary_v1",
        "created_at": _utc_iso_now(),
        "manifest_path": os.path.abspath(manifest_path),
        "run": {
            "platform_id": run.get("platform_id"),
            "video_id": run.get("video_id"),
            "run_id": run.get("run_id"),
            "config_hash": run.get("config_hash"),
            "sampling_policy_version": run.get("sampling_policy_version"),
            "dataprocessor_version": run.get("dataprocessor_version"),
            "status": run.get("status"),
            "created_at": run.get("created_at"),
            "updated_at": run.get("updated_at"),
        },
        "summary": summary,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Summarize manifest.json into a compact _reports JSON.")
    p.add_argument("--manifest", required=True, help="Path to manifest.json")
    p.add_argument(
        "--out",
        default="",
        help="Output JSON path. Default: <run_dir>/_reports/run_manifest_summary.json",
    )
    args = p.parse_args(argv)

    manifest_path = os.path.abspath(args.manifest)
    run_dir = os.path.dirname(manifest_path)
    out_path = os.path.abspath(args.out) if args.out else os.path.join(run_dir, "_reports", "run_manifest_summary.json")

    payload = summarize_manifest(manifest_path)
    _atomic_write_json(out_path, payload)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

