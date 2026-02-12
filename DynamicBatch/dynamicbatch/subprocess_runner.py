from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional


_OOM_PATTERNS = [
    re.compile(r"cuda\\s+out\\s+of\\s+memory", re.IGNORECASE),
    re.compile(r"out\\s+of\\s+memory", re.IGNORECASE),
    re.compile(r"cudart", re.IGNORECASE),
    re.compile(r"oom", re.IGNORECASE),
]


def looks_like_oom(text: str) -> bool:
    t = text or ""
    for p in _OOM_PATTERNS:
        if p.search(t):
            return True
    return False


@dataclass
class RunResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    oom: bool
    cmd: List[str]
    env: Dict[str, str]


def run_subprocess(
    cmd: List[str],
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> RunResult:
    merged_env = dict(os.environ)
    if env:
        merged_env.update({k: str(v) for k, v in env.items()})

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=merged_env,
        check=False,
    )

    out = p.stdout or ""
    err = p.stderr or ""
    oom = (p.returncode != 0) and (looks_like_oom(out) or looks_like_oom(err))
    return RunResult(
        ok=(p.returncode == 0),
        returncode=int(p.returncode),
        stdout=out,
        stderr=err,
        oom=bool(oom),
        cmd=list(cmd),
        env={k: merged_env.get(k, "") for k in (env or {}).keys()},
    )


