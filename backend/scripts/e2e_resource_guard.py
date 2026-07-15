#!/usr/bin/env python3
"""E2E resource guard: аварийная остановка стека при критической нагрузке ОЗУ/GPU/диска.

Запускается параллельно с ``e2e_full_max_run.py`` (см. ``e2e_run_full_green.sh``).
При превышении порогов N раз подряд:
  1) пишет ``backend/.e2e/state/emergency_stop.json``;
  2) SIGTERM наблюдаемому PID (оркестратор);
  3) ``stop_e2e_stack.sh``;
  4) ``e2e_triton_docker.sh stop`` (если есть).

Пороги по умолчанию — 99%% (переопределяются env ``E2E_GUARD_*``).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from e2e_host_resources import evaluate_resource_pressure, host_resource_snapshot_line  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _emergency_shutdown(
    *,
    repo_root: Path,
    reason: str,
    metrics: dict,
    watch_pid: int,
    stop_stack: bool,
    stop_triton: bool,
    log_path: Path | None,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    payload = {
        "ts": ts,
        "reason": reason,
        "metrics": metrics,
        "watch_pid": watch_pid,
        "host_line": host_resource_snapshot_line(),
    }
    state_dir = repo_root / "backend" / ".e2e" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    stop_file = state_dir / "emergency_stop.json"
    stop_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    msg = f"[e2e_resource_guard] EMERGENCY STOP @ {ts}: {reason}\n  {payload['host_line']}\n"
    print(msg, flush=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg)

    if _pid_alive(watch_pid):
        print(f"[e2e_resource_guard] SIGTERM watch_pid={watch_pid}", flush=True)
        try:
            os.kill(watch_pid, signal.SIGTERM)
        except OSError as exc:
            print(f"[e2e_resource_guard] kill failed: {exc}", flush=True)

    scripts = repo_root / "backend" / "scripts"
    if stop_stack and (scripts / "stop_e2e_stack.sh").is_file():
        print("[e2e_resource_guard] running stop_e2e_stack.sh --quiet", flush=True)
        subprocess.run(
            ["bash", str(scripts / "stop_e2e_stack.sh"), "--quiet"],
            cwd=str(repo_root),
            timeout=120,
            check=False,
        )

    triton_sh = scripts / "e2e_triton_docker.sh"
    if stop_triton and triton_sh.is_file():
        print("[e2e_resource_guard] running e2e_triton_docker.sh stop", flush=True)
        subprocess.run(["bash", str(triton_sh), "stop"], cwd=str(repo_root), timeout=60, check=False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watch-pid", type=int, required=True, help="PID оркестратора E2E (родительский shell/python)")
    p.add_argument("--poll-sec", type=float, default=_env_float("E2E_GUARD_POLL_SEC", 2.0))
    p.add_argument("--breach-streak", type=int, default=_env_int("E2E_GUARD_BREACH_STREAK", 3))
    p.add_argument("--ram-used-pct-max", type=float, default=_env_float("E2E_GUARD_RAM_USED_PCT", 99.0))
    p.add_argument("--swap-used-pct-max", type=float, default=_env_float("E2E_GUARD_SWAP_USED_PCT", 99.0))
    p.add_argument("--gpu-mem-pct-max", type=float, default=_env_float("E2E_GUARD_GPU_MEM_PCT", 99.0))
    p.add_argument("--disk-used-pct-max", type=float, default=_env_float("E2E_GUARD_DISK_USED_PCT", 99.0))
    p.add_argument(
        "--disk-path",
        type=Path,
        default=None,
        help="Точка монтирования для проверки диска (default: STORAGE_ROOT или repo/storage)",
    )
    p.add_argument("--no-stop-stack", action="store_true")
    p.add_argument("--no-stop-triton", action="store_true")
    p.add_argument("--log-file", type=Path, default=None)
    args = p.parse_args()

    repo = _repo_root()
    disk_path = args.disk_path
    if disk_path is None:
        storage = os.environ.get("STORAGE_ROOT") or os.environ.get("TREND_FS_ROOT")
        disk_path = Path(storage) if storage else repo / "storage"

    log_file = args.log_file or (repo / "backend" / ".e2e" / "logs" / "resource_guard.log")

    print(
        f"[e2e_resource_guard] watching pid={args.watch_pid} "
        f"RAM<={args.ram_used_pct_max}% swap<={args.swap_used_pct_max}% "
        f"GPU_VRAM<={args.gpu_mem_pct_max}% disk<={args.disk_used_pct_max}% "
        f"poll={args.poll_sec}s streak={args.breach_streak}",
        flush=True,
    )

    streak = 0
    while _pid_alive(args.watch_pid):
        breach, reason, metrics = evaluate_resource_pressure(
            ram_used_pct_max=args.ram_used_pct_max,
            swap_used_pct_max=args.swap_used_pct_max,
            gpu_mem_used_pct_max=args.gpu_mem_pct_max,
            disk_used_pct_max=args.disk_used_pct_max,
            disk_path=disk_path,
        )
        if breach:
            streak += 1
            line = f"[e2e_resource_guard] breach {streak}/{args.breach_streak}: {reason} | {host_resource_snapshot_line()}"
            print(line, flush=True)
            if streak >= args.breach_streak:
                _emergency_shutdown(
                    repo_root=repo,
                    reason=reason,
                    metrics=metrics,
                    watch_pid=args.watch_pid,
                    stop_stack=not args.no_stop_stack,
                    stop_triton=not args.no_stop_triton,
                    log_path=log_file,
                )
                return 2
        else:
            streak = 0
        time.sleep(max(0.5, args.poll_sec))

    print(f"[e2e_resource_guard] watch_pid={args.watch_pid} exited — guard done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
