#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

_here = os.path.dirname(__file__)
if _here and _here not in sys.path:
    sys.path.insert(0, _here)

from dynamicbatch.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


