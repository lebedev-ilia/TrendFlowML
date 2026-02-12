#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional


def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def _guess_runtime(metrics: Dict[str, Any]) -> str:
    if "vram_triton_peak_mb" in metrics or "vram_triton_delta_run_mb" in metrics:
        return "triton"
    return "inprocess"


def _default_device_profile() -> Dict[str, Any]:
    # MVP: best-effort probe using nvidia-smi; keep minimal stable keys.
    profile: Dict[str, Any] = {"os": "linux"}
    try:
        import subprocess
        import shutil

        if shutil.which("nvidia-smi") is None:
            return profile
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,nounits,noheader",
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if p.returncode != 0:
            return profile
        line = (p.stdout or "").strip().splitlines()[0] if (p.stdout or "").strip() else ""
        if not line:
            return profile
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 1:
            profile["gpu_name"] = parts[0]
        if len(parts) >= 2:
            try:
                profile["vram_mb"] = int(float(parts[1]))
            except Exception:
                pass
        if len(parts) >= 3:
            profile["driver"] = parts[2]
    except Exception:
        return profile
    return profile


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="import_resource_costs_to_db",
        description="Import seed resource_costs/*.json into Postgres benchmark registry (benchmark_costs_v1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--db-dsn", required=True, help="Postgres DSN, e.g. postgresql://user:pass@localhost:5432/db")
    ap.add_argument("--db-table", default="benchmark_costs_v1")
    ap.add_argument(
        "--resource-costs-dir",
        required=True,
        help="Directory with DataProcessor/docs/models_docs/resource_costs/*.json",
    )
    ap.add_argument("--owner", default="dataprocessor", choices=["dataprocessor", "fetcher", "models"])
    ap.add_argument("--stage", default="baseline", choices=["baseline", "v1", "v2"])
    ap.add_argument("--producer-version", default="seed-import:v1", help="Producer version label for imported rows")
    ap.add_argument("--git-commit", default="unknown", help="Commit hash for imported rows")
    ap.add_argument("--git-dirty", action="store_true")
    ap.add_argument("--schema-version", default="benchmark_costs_v1")
    ap.add_argument("--device-profile-json", default=None, help="Optional JSON string for device_profile override")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    try:
        import psycopg2  # type: ignore
        import psycopg2.extras  # type: ignore
    except Exception as e:
        raise RuntimeError("This importer requires psycopg2 (or psycopg2-binary).") from e

    device_profile = _default_device_profile()
    if args.device_profile_json:
        device_profile = json.loads(args.device_profile_json)
        if not isinstance(device_profile, dict):
            raise RuntimeError("--device-profile-json must decode to a JSON object")

    rows_inserted = 0
    files = [os.path.join(args.resource_costs_dir, n) for n in sorted(os.listdir(args.resource_costs_dir)) if n.endswith(".json")]

    now = _utc_iso_now()

    if args.dry_run:
        print(f"[dry-run] will import {len(files)} files into {args.db_table}")
        return 0

    with psycopg2.connect(args.db_dsn) as conn:
        with conn.cursor() as cur:
            for path in files:
                data = _load_json(path)
                costs = data.get("costs") or []
                if not isinstance(costs, list):
                    continue
                for c in costs:
                    if not isinstance(c, dict):
                        continue
                    component_id = str(c.get("component") or "").strip()
                    unit = str(c.get("unit") or "").strip()
                    # Hard guard: never insert empty identity keys.
                    if not component_id or not unit:
                        continue
                    model_branch = c.get("model_branch")
                    model_branch_s = None if model_branch is None else str(model_branch)
                    metrics = c.get("metrics") if isinstance(c.get("metrics"), dict) else {}
                    runtime = _guess_runtime(metrics)

                    # Close previous active version for same key.
                    close_sql = f"""
                    UPDATE {args.db_table}
                    SET valid_to = NOW()
                    WHERE valid_to IS NULL
                      AND component_id = %s
                      AND component_part = 'whole'
                      AND owner = %s
                      AND stage IS NOT DISTINCT FROM %s
                      AND unit = %s
                      AND runtime = %s
                      AND model_signature IS NULL
                      AND model_branch IS NOT DISTINCT FROM %s
                    """
                    cur.execute(
                        close_sql,
                        (
                            component_id,
                            str(args.owner),
                            str(args.stage),
                            unit,
                            runtime,
                            model_branch_s,
                        ),
                    )

                    ins_sql = f"""
                    INSERT INTO {args.db_table} (
                      id, component_id, component_part, owner, stage, unit, runtime,
                      model_signature, model_branch,
                      input_bucket, knobs, device_profile,
                      producer_version, git_commit, git_dirty, schema_version,
                      metrics, artifact_uri, created_at, valid_from, valid_to
                    ) VALUES (
                      %s, %s, 'whole', %s, %s, %s, %s,
                      NULL, %s,
                      %s::jsonb, %s::jsonb, %s::jsonb,
                      %s, %s, %s, %s,
                      %s::jsonb, %s, NOW(), NOW(), NULL
                    )
                    """
                    cur.execute(
                        ins_sql,
                        (
                            str(uuid.uuid4()),
                            component_id,
                            str(args.owner),
                            str(args.stage),
                            unit,
                            runtime,
                            model_branch_s,
                            json.dumps({}),  # input_bucket (unknown in seed)
                            json.dumps({}),  # knobs (unknown in seed)
                            json.dumps(device_profile),
                            str(args.producer_version),
                            str(args.git_commit),
                            bool(args.git_dirty),
                            str(args.schema_version),
                            json.dumps(metrics),
                            f"file://{os.path.abspath(path)}",
                        ),
                    )
                    rows_inserted += 1

    print(f"Imported rows: {rows_inserted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


