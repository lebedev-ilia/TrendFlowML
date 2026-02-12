#!/usr/bin/env python3
"""
Smoke-check for Stage-0 batch API.

Goal:
- Ensure MainProcessor.run_batch([doc]) is equivalent to MainProcessor.run(doc) for key fields.

Usage:
  cd DataProcessor/TextProcessor
  python3 scripts/smoke_batch.py --input-json example_input.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    return {k: d.get(k) for k in keys}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True, help="Path to VideoDocument JSON")
    parser.add_argument("--devices-config-json", default=None, help="JSON string like {'cpu':[...],'gpu':[...]} (optional)")
    args = parser.parse_args()

    # Allow running as a standalone script (so `import src.*` works).
    tp_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(tp_root))

    from src.core.main_processor import MainProcessor, load_document_from_json  # local import

    devices_cfg = None
    if args.devices_config_json:
        devices_cfg = json.loads(args.devices_config_json)

    doc = load_document_from_json(args.input_json)
    p = MainProcessor(devices_config=devices_cfg)

    single = p.run(doc) or {}
    batch = (p.run_batch([doc]) or [{}])[0] or {}

    # Compare a small stable subset (full deep-compare is too noisy at this stage).
    keys = [
        "status",
        "empty_reason",
        "error",
    ]
    s1 = _pick(single, keys)
    s2 = _pick(batch, keys)

    ok = s1 == s2
    print("single:", s1)
    print("batch :", s2)
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())


