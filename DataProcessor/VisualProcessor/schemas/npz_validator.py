from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class SchemaIssue:
    level: str  # "error" | "warning"
    message: str


_DIM_RE = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9_]*)\s*(?P<op>[+-])?\s*(?P<delta>\d+)?$")


def _dtype_matches(arr: np.ndarray, spec: Union[str, List[str]]) -> bool:
    specs = [spec] if isinstance(spec, str) else list(spec)
    dt = arr.dtype

    for s in specs:
        token = str(s).strip().lower()
        if token in ("any", "*"):
            return True
        if token == "object":
            if dt == object:
                return True
            continue
        if token == "str" or token == "string":
            if dt.kind in ("U", "S"):
                return True
            continue
        if token == "bool" or token == "boolean":
            if dt == np.dtype(bool):
                return True
            continue
        # numeric exact match
        try:
            if dt == np.dtype(token):
                return True
        except Exception:
            pass

    return False


def _shape_is_scalar(arr: np.ndarray) -> bool:
    return isinstance(arr, np.ndarray) and arr.shape == ()


def _parse_dim_expr(expr: str) -> Optional[Tuple[str, int]]:
    """
    Returns (var_name, delta) where delta can be negative/positive.
    Examples:
      "N" -> ("N", 0)
      "N-1" -> ("N", -1)
      "K+2" -> ("K", +2)
    """
    m = _DIM_RE.match(str(expr).strip())
    if not m:
        return None
    name = str(m.group("name"))
    op = m.group("op")
    delta_raw = m.group("delta")
    if not op or not delta_raw:
        return name, 0
    delta = int(delta_raw)
    if op == "-":
        delta = -delta
    return name, delta


def _check_shape(
    *,
    key: str,
    arr: np.ndarray,
    shape_spec: Optional[List[Union[int, str]]],
    dims: Dict[str, int],
) -> List[SchemaIssue]:
    issues: List[SchemaIssue] = []
    if shape_spec is None:
        return issues  # no shape validation for this field

    if not isinstance(arr, np.ndarray):
        issues.append(SchemaIssue("error", f"{key}: value is not a numpy array"))
        return issues

    if shape_spec == []:
        if not _shape_is_scalar(arr):
            issues.append(SchemaIssue("error", f"{key}: expected scalar shape (), got {arr.shape}"))
        return issues

    if int(arr.ndim) != int(len(shape_spec)):
        issues.append(SchemaIssue("error", f"{key}: expected ndim={len(shape_spec)}, got ndim={arr.ndim} shape={arr.shape}"))
        return issues

    actual = list(map(int, arr.shape))
    for i, dim_spec in enumerate(shape_spec):
        a = int(actual[i])
        if isinstance(dim_spec, int):
            if a != int(dim_spec):
                issues.append(SchemaIssue("error", f"{key}: dim[{i}] expected {dim_spec}, got {a}"))
            continue
        # symbolic / expression
        expr = str(dim_spec).strip()
        if expr in ("*", "any"):
            continue
        parsed = _parse_dim_expr(expr)
        if parsed is None:
            issues.append(SchemaIssue("error", f"{key}: invalid dim spec '{expr}'"))
            continue
        var, delta = parsed
        if var not in dims:
            dims[var] = a - int(delta)
        expected = int(dims[var] + int(delta))
        if a != expected:
            issues.append(SchemaIssue("error", f"{key}: dim[{i}] expected {var}{'+' if delta>0 else ''}{delta if delta else ''}={expected}, got {a}"))
    return issues


def validate_npz_against_schema(
    npz: np.lib.npyio.NpzFile,
    *,
    schema_doc: Dict[str, Any],
    require_all_required_fields: bool = True,
) -> List[SchemaIssue]:
    """
    Validates an already-loaded NPZ file against a schema document.
    Only validates keys/dtype/shape (no semantic value checks).
    """
    issues: List[SchemaIssue] = []

    if str(schema_doc.get("schema_system_version")) != "vp_schema_v1":
        issues.append(SchemaIssue("error", f"schema_system_version must be vp_schema_v1, got {schema_doc.get('schema_system_version')!r}"))
        return issues

    allow_extra = bool(schema_doc.get("allow_extra_keys", True))
    fields: Dict[str, Any] = dict(schema_doc.get("fields") or {})

    # Required fields list
    required_fields = [k for k, v in fields.items() if bool((v or {}).get("required"))]

    # Key presence
    npz_keys = set(map(str, npz.files))
    if require_all_required_fields:
        for k in required_fields:
            if k not in npz_keys:
                issues.append(SchemaIssue("error", f"missing required npz key: {k}"))

    if not allow_extra:
        known = set(fields.keys())
        prefixes_raw = schema_doc.get("allowed_extra_key_prefixes") or []
        prefixes = [str(p) for p in prefixes_raw if str(p).strip()]
        extra: List[str] = []
        for k in sorted(npz_keys):
            if k in known:
                continue
            if prefixes and any(k.startswith(p) for p in prefixes):
                continue
            extra.append(k)
        if extra:
            issues.append(SchemaIssue("error", f"unexpected npz keys (allow_extra_keys=false): {extra}"))

    # dtype/shape checks (for keys described in schema and present in NPZ)
    dims: Dict[str, int] = {}
    for key, spec in fields.items():
        if key not in npz_keys:
            continue
        if not isinstance(spec, dict):
            issues.append(SchemaIssue("error", f"{key}: invalid field spec (must be object)"))
            continue
        try:
            arr = npz[key]
        except Exception as e:
            issues.append(SchemaIssue("error", f"{key}: failed to read from npz: {e}"))
            continue

        dtype_spec = spec.get("dtype", "any")
        if not _dtype_matches(arr, dtype_spec):
            issues.append(SchemaIssue("error", f"{key}: dtype mismatch: got {arr.dtype}, expected {dtype_spec}"))

        shape_spec_raw = spec.get("shape", None)
        shape_spec: Optional[List[Union[int, str]]]
        if shape_spec_raw is None:
            shape_spec = None
        elif isinstance(shape_spec_raw, list):
            shape_spec = []
            for d in shape_spec_raw:
                if isinstance(d, int):
                    shape_spec.append(int(d))
                else:
                    shape_spec.append(str(d))
        else:
            issues.append(SchemaIssue("error", f"{key}: shape must be list|null, got {type(shape_spec_raw).__name__}"))
            continue

        issues.extend(_check_shape(key=key, arr=arr, shape_spec=shape_spec, dims=dims))

    return issues


