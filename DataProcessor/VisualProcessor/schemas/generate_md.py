from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional


def _fmt_shape(shape: Any) -> str:
    if shape is None:
        return "—"
    if isinstance(shape, list) and len(shape) == 0:
        return "()"
    if isinstance(shape, list):
        return "(" + ", ".join(str(x) for x in shape) + ")"
    return str(shape)


def _fmt_dtype(dtype: Any) -> str:
    if isinstance(dtype, list):
        return " | ".join(str(x) for x in dtype)
    return str(dtype)


def render_schema_md(doc: Dict[str, Any]) -> str:
    schema_version = str(doc.get("schema_version") or "")
    producer = str(doc.get("producer") or "")
    allow_extra = bool(doc.get("allow_extra_keys", True))
    meta = dict(doc.get("meta") or {})
    fields = dict(doc.get("fields") or {})

    lines: List[str] = []
    lines.append(f"# Schema: `{schema_version}`")
    lines.append("")
    lines.append(f"- **producer**: `{producer}`")
    lines.append(f"- **artifact_kind**: `{doc.get('artifact_kind')}`")
    lines.append(f"- **allow_extra_keys**: `{allow_extra}`")
    lines.append(f"- **schema_system_version**: `{doc.get('schema_system_version')}`")
    lines.append("")

    req_meta = meta.get("required_keys") or []
    opt_meta = meta.get("optional_keys") or []
    lines.append("## Meta")
    lines.append("")
    lines.append("### Required meta keys")
    lines.append("")
    for k in req_meta:
        lines.append(f"- `{k}`")
    if not req_meta:
        lines.append("- (none)")
    lines.append("")
    lines.append("### Optional meta keys")
    lines.append("")
    for k in opt_meta:
        lines.append(f"- `{k}`")
    if not opt_meta:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Fields")
    lines.append("")
    lines.append("| key | required | tier | dtype | shape | description |")
    lines.append("|---|---:|---|---|---|---|")
    for key in sorted(fields.keys()):
        spec = fields.get(key) or {}
        required = bool(spec.get("required"))
        tier = str(spec.get("tier") or "")
        dtype = _fmt_dtype(spec.get("dtype", "any"))
        shape = _fmt_shape(spec.get("shape", None))
        desc = str(spec.get("description") or "")
        lines.append(f"| `{key}` | `{required}` | `{tier}` | `{dtype}` | `{shape}` | {desc} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate SCHEMA.md from a VisualProcessor schema JSON.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input schema JSON path")
    ap.add_argument("--out", dest="out_path", required=False, help="Output markdown path (default: stdout)")
    args = ap.parse_args()

    with open(str(args.in_path), "r", encoding="utf-8") as f:
        doc = json.load(f)

    md = render_schema_md(doc)
    if args.out_path:
        out_path = str(args.out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        return 0

    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


