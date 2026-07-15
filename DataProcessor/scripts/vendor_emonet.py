#!/usr/bin/env python3
"""
Vendor EmoNet architecture sources into dp_models/emonet/ (expected by create_emonet).

Layout after success:
  DataProcessor/dp_models/emonet/emonet/models/emonet.py

Usage:
  python DataProcessor/scripts/vendor_emonet.py
  python DataProcessor/scripts/vendor_emonet.py --check
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EMONET_REPO = "https://github.com/face-analysis/emonet.git"
TARGET_REL = Path("DataProcessor/dp_models/emonet/emonet/models/emonet.py")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def emonet_py_path(root: Path) -> Path:
    return root / TARGET_REL


def check(root: Path) -> bool:
    p = emonet_py_path(root)
    if p.is_file():
        print(f"OK: {p}")
        return True
    print(f"MISSING: {p}", file=sys.stderr)
    return False


def vendor(root: Path) -> int:
    dest = root / "DataProcessor/dp_models/emonet"
    marker = emonet_py_path(root)
    if marker.is_file():
        print(f"Already vendored: {marker}")
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)

    print(f"Cloning {EMONET_REPO} → {dest} …")
    with tempfile.TemporaryDirectory(prefix="emonet_vendor_") as tmp:
        clone_dir = Path(tmp) / "src"
        subprocess.run(
            ["git", "clone", "--depth", "1", EMONET_REPO, str(clone_dir)],
            check=True,
        )
        # Repo root contains package dir `emonet/` with models/emonet.py
        pkg = clone_dir / "emonet"
        if not (pkg / "models" / "emonet.py").is_file():
            print(f"FAIL: unexpected repo layout under {clone_dir}", file=sys.stderr)
            return 1
        shutil.copytree(clone_dir, dest)

    if not marker.is_file():
        print(f"FAIL: expected {marker} after clone", file=sys.stderr)
        return 1
    print(f"Vendored EmoNet → {marker}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Vendor EmoNet sources for emotion_face")
    ap.add_argument("--check", action="store_true", help="Only verify emonet.py exists")
    args = ap.parse_args()
    root = _repo_root()
    if args.check:
        return 0 if check(root) else 1
    return vendor(root)


if __name__ == "__main__":
    raise SystemExit(main())
