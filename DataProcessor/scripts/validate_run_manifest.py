from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GateIssue:
    level: str  # "error" | "warning"
    message: str
    component: Optional[str] = None
    artifact: Optional[str] = None


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _coerce_str(x: Any) -> str:
    return str(x) if x is not None else ""


def _component_terminal_ok(status: str) -> bool:
    s = (status or "").strip().lower()
    return s in ("ok", "empty")


def _component_needs_artifacts(status: str) -> bool:
    s = (status or "").strip().lower()
    return s == "ok"


def _resolve_artifact_path(manifest_path: str, artifact_path: str) -> str:
    # Prefer run-local relative artifacts written by manifest normalizer:
    # <run_dir>/<component>/<file>
    if not artifact_path:
        return ""
    if os.path.isabs(artifact_path):
        return artifact_path
    run_dir = os.path.dirname(os.path.abspath(manifest_path))
    return os.path.join(run_dir, artifact_path)


def _validate_npz(artifact_path: str, *, require_known_schema: bool) -> Tuple[bool, List[GateIssue], Dict[str, Any]]:
    # VisualProcessor is not a package; add VisualProcessor/ to sys.path and import utils.artifact_validator.
    dp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    vp_root = os.path.join(dp_root, "VisualProcessor")
    if vp_root not in sys.path:
        sys.path.insert(0, vp_root)

    try:
        from utils.artifact_validator import validate_npz as _validate  # type: ignore
    except ModuleNotFoundError as e:
        # Most commonly: numpy is not installed in the current interpreter.
        msg = (
            f"cannot validate npz in current python env: missing dependency ({e}). "
            f"Tip: run with VisualProcessor venv python (DataProcessor/VisualProcessor/.vp_venv/bin/python) "
            f"or install numpy."
        )
        return False, [GateIssue(level="error", message=msg, artifact=artifact_path)], {}

    ok, issues, meta = _validate(
        artifact_path,
        validate_schema=True,
        require_known_schema=require_known_schema,
    )
    gate_issues: List[GateIssue] = [
        GateIssue(level=i.level, message=i.message, artifact=artifact_path) for i in issues
    ]
    return ok, gate_issues, meta


def validate_manifest(
    manifest_path: str,
    *,
    require_known_schema: bool,
    strict_empty_reason: bool,
) -> Tuple[bool, List[GateIssue]]:
    m = _load_json(manifest_path)
    comps = m.get("components") or []
    if not isinstance(comps, list):
        return False, [GateIssue("error", "manifest.components is not a list")]

    issues: List[GateIssue] = []
    all_ok = True

    for c in comps:
        if not isinstance(c, dict):
            issues.append(GateIssue("error", "component entry is not an object"))
            all_ok = False
            continue

        name = _coerce_str(c.get("name")).strip()
        status = _coerce_str(c.get("status")).strip()
        empty_reason = c.get("empty_reason")
        artifacts = c.get("artifacts") or []

        if not name:
            issues.append(GateIssue("error", "component.name is missing/empty"))
            all_ok = False
            continue

        if not _component_terminal_ok(status):
            issues.append(GateIssue("error", f"component status is not terminal ok: status={status!r}", component=name))
            all_ok = False
            continue

        if status.lower() == "empty":
            if strict_empty_reason and not _coerce_str(empty_reason).strip():
                issues.append(GateIssue("error", "empty component missing empty_reason", component=name))
                all_ok = False
            # empty is allowed to have no artifacts
            continue

        # status == ok
        if _component_needs_artifacts(status) and (not isinstance(artifacts, list) or len(artifacts) == 0):
            issues.append(GateIssue("error", "ok component has no artifacts", component=name))
            all_ok = False
            continue

        for a in artifacts:
            if not isinstance(a, dict):
                issues.append(GateIssue("error", "artifact entry is not an object", component=name))
                all_ok = False
                continue
            ap = _coerce_str(a.get("path")).strip()
            atype = _coerce_str(a.get("type")).strip().lower()
            if not ap:
                issues.append(GateIssue("error", "artifact.path missing/empty", component=name))
                all_ok = False
                continue

            resolved = _resolve_artifact_path(manifest_path, ap)
            if not os.path.exists(resolved):
                issues.append(GateIssue("error", f"artifact missing on disk: {resolved}", component=name, artifact=resolved))
                all_ok = False
                continue

            if atype == "npz" or resolved.lower().endswith(".npz"):
                ok_npz, npz_issues, _meta = _validate_npz(resolved, require_known_schema=require_known_schema)
                for ni in npz_issues:
                    ni.component = name
                issues.extend(npz_issues)
                if not ok_npz:
                    all_ok = False
            else:
                # For non-NPZ, just existence check for now (render/logs etc.)
                continue

    return all_ok, issues


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Quality gate: validate a run manifest and its artifacts.")
    p.add_argument("--manifest", required=True, help="Path to manifest.json inside a run directory")
    p.add_argument(
        "--require-known-schema",
        action="store_true",
        help="Fail if schema_version is unknown to schema registry (strict mode).",
    )
    p.add_argument(
        "--strict-empty-reason",
        action="store_true",
        help="Fail if a component status=empty has no empty_reason.",
    )
    args = p.parse_args(argv)

    ok, issues = validate_manifest(
        args.manifest,
        require_known_schema=bool(args.require_known_schema),
        strict_empty_reason=bool(args.strict_empty_reason),
    )

    for i in issues:
        prefix = i.level.upper()
        where = ""
        if i.component:
            where += f" component={i.component}"
        if i.artifact:
            where += f" artifact={i.artifact}"
        print(f"[{prefix}]{where} {i.message}")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

