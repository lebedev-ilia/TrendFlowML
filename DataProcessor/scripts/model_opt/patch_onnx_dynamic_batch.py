#!/usr/bin/env python3
"""
Patch an existing ONNX model to support Triton batching by making the leading
batch dimension dynamic for selected model inputs/outputs.

This is a pragmatic fallback when re-exporting with dynamic batch is difficult
in the current torch exporter (e.g. some MiDaS/DPT variants under torch 2.9).

It does NOT change weights/ops; it only edits the ONNX graph IO shapes.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional, Set


def _set_first_dim_dynamic(value_info, *, dim_param: str) -> bool:  # type: ignore[no-untyped-def]
    try:
        tt = value_info.type.tensor_type
        if not tt.HasField("shape"):
            return False
        if len(tt.shape.dim) < 1:
            return False
        d0 = tt.shape.dim[0]
        # Triton batching requirement: first dim must be dynamic.
        # Use dim_param (symbolic name) instead of dim_value=-1 for better ONNX Runtime compatibility.
        d0.ClearField("dim_value")
        d0.dim_param = dim_param
        return True
    except Exception:
        return False


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser("patch onnx IO batch dim to dynamic")
    ap.add_argument("--in", dest="in_path", required=True, help="Input ONNX path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output ONNX path")
    ap.add_argument("--dim-param", default="batch_size", help="Symbolic name for batch dim")
    ap.add_argument("--inputs", default="", help="Comma-separated input names to patch (empty = all graph inputs)")
    ap.add_argument("--outputs", default="", help="Comma-separated output names to patch (empty = all graph outputs)")
    args = ap.parse_args(argv)

    in_path = os.path.abspath(str(args.in_path))
    out_path = os.path.abspath(str(args.out_path))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    try:
        import onnx  # type: ignore
    except Exception as e:
        raise RuntimeError(f"onnx package is required: {e}") from e

    m = onnx.load(in_path)

    only_inputs: Set[str] = set([s.strip() for s in str(args.inputs).split(",") if s.strip()])
    only_outputs: Set[str] = set([s.strip() for s in str(args.outputs).split(",") if s.strip()])

    patched = 0
    for vi in m.graph.input:
        if only_inputs and vi.name not in only_inputs:
            continue
        if _set_first_dim_dynamic(vi, dim_param=str(args.dim_param)):
            patched += 1
    for vi in m.graph.output:
        if only_outputs and vi.name not in only_outputs:
            continue
        if _set_first_dim_dynamic(vi, dim_param=str(args.dim_param)):
            patched += 1

    onnx.save(m, out_path)
    print(f"Patched {patched} IO tensors -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


