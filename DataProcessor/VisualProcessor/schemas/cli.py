from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np

from .npz_validator import validate_npz_against_schema
from .registry import load_all_schema_docs


def _iter_npz_paths(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith(".npz"):
                    out.append(os.path.join(p, name))
        else:
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate NPZ artifacts against VisualProcessor schemas.")
    ap.add_argument("paths", nargs="+", help="NPZ file paths or directories containing .npz files")
    ap.add_argument(
        "--schemas-dir",
        default=os.path.join(os.path.dirname(__file__)),
        help="Directory containing schema JSON files (default: this package dir)",
    )
    ap.add_argument("--require-known-schema", action="store_true", help="Fail if meta.schema_version has no schema doc")
    args = ap.parse_args()

    reg, issues = load_all_schema_docs(str(args.schemas_dir))
    if issues:
        for i in issues:
            print(f"[schema-registry] warning: {i}", file=sys.stderr)

    paths = _iter_npz_paths([str(p) for p in args.paths])
    any_errors = False

    for path in paths:
        if not os.path.exists(path):
            print(f"[error] missing file: {path}")
            any_errors = True
            continue
        try:
            npz = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"[error] failed to load npz: {path}: {e}")
            any_errors = True
            continue

        try:
            meta = None
            if "meta" in npz.files:
                try:
                    meta_arr = npz["meta"]
                    meta = meta_arr.item() if meta_arr.shape == () else meta_arr.flat[0].item()
                except Exception:
                    meta = None

            schema_version = str((meta or {}).get("schema_version") or "")
            if not schema_version:
                print(f"[error] {path}: missing meta.schema_version")
                any_errors = True
                continue

            sd = reg.get(schema_version)
            if sd is None:
                msg = f"{path}: unknown schema_version={schema_version}"
                if bool(args.require_known_schema):
                    print(f"[error] {msg}")
                    any_errors = True
                else:
                    print(f"[warn] {msg}")
                continue

            sch_issues = validate_npz_against_schema(npz, schema_doc=sd.doc)
            errs = [x for x in sch_issues if x.level == "error"]
            if errs:
                any_errors = True
                print(f"[error] {path}: schema validation failed ({len(errs)} errors)")
                for e in errs[:50]:
                    print(f"  - {e.message}")
            else:
                print(f"[ok] {path}: {schema_version}")
        finally:
            try:
                npz.close()
            except Exception:
                pass

    return 1 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())


