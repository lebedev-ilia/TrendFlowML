#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System cleanup script for benchmark preparation.

Clears:
- System buffers and caches
- GPU memory and cache
- Drop system caches
- Sync filesystem

Usage:
    python benchmarks/cleanup_system.py [--force]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from typing import Optional


def _check_root() -> bool:
    """Check if running as root (required for drop_caches)."""
    return os.geteuid() == 0


def _print_step(msg: str) -> None:
    """Print a step message."""
    print(f"[cleanup] {msg}")


def _run_cmd(cmd: list, check: bool = False, capture_output: bool = True) -> tuple:
    """Run command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE if capture_output else None, stderr=subprocess.PIPE, text=True, check=check
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        return result.returncode, stdout, stderr
    except Exception as e:
        return 1, "", str(e)


def clear_gpu_memory(force: bool = False) -> bool:
    """Clear GPU memory and cache."""
    _print_step("Clearing GPU memory...")

    if not shutil.which("nvidia-smi"):
        _print_step("  ⚠ nvidia-smi not found, skipping GPU cleanup")
        return False

    # Check GPU processes
    rc, stdout, _ = _run_cmd(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    if rc == 0 and stdout.strip():
        pids = []
        for line in stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 1:
                try:
                    pids.append(int(parts[0].strip()))
                except Exception:
                    pass

        if pids and not force:
            _print_step(f"  ⚠ Found {len(pids)} GPU process(es). Use --force to kill them:")
            for pid in pids:
                _print_step(f"    PID: {pid}")
            _print_step("  Skipping GPU cleanup (use --force to kill processes)")
            return False

        # Kill GPU processes if force
        if pids and force:
            _print_step(f"  Killing {len(pids)} GPU process(es)...")
            for pid in pids:
                try:
                    os.kill(pid, 15)  # SIGTERM
                    time.sleep(0.1)
                except ProcessLookupError:
                    pass
                except Exception as e:
                    _print_step(f"    ⚠ Failed to kill PID {pid}: {e}")

            # Wait a bit for processes to terminate
            time.sleep(1.0)

            # Force kill if still running
            for pid in pids:
                try:
                    os.kill(pid, 0)  # Check if exists
                    os.kill(pid, 9)  # SIGKILL
                except ProcessLookupError:
                    pass
                except Exception:
                    pass

    # Reset GPU (requires root or proper permissions)
    # Try to reset GPU state via nvidia-smi
    _print_step("  Resetting GPU state...")
    rc, _, _ = _run_cmd(["nvidia-smi", "--gpu-reset"], check=False)
    if rc != 0:
        # If reset fails, try to clear memory via simpler method
        _print_step("  Clearing GPU memory via nvidia-smi...")
        # Just query to ensure nvidia-smi works, memory clearing happens via process termination
        _run_cmd(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"], check=False)

    # Check final GPU memory
    rc, stdout, _ = _run_cmd(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"])
    if rc == 0:
        for line in stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    used = int(float(parts[0].strip()))
                    total = int(float(parts[1].strip()))
                    _print_step(f"  ✓ GPU memory: {used}/{total} MB")
                    if used > total * 0.05:  # More than 5% used
                        _print_step("  ⚠ GPU memory still has some usage (may be system reserved)")
                except Exception:
                    pass

    return True


def clear_system_cache(force: bool = False) -> bool:
    """Clear system page cache, dentries, and inodes."""
    _print_step("Clearing system cache...")

    if not _check_root():
        _print_step("  ⚠ Not running as root, skipping cache drop")
        _print_step("  Note: To drop caches, run with sudo:")
        _print_step("    sudo python benchmarks/cleanup_system.py")
        return False

    try:
        # Drop caches: 1 = page cache, 2 = dentries/inodes, 3 = both
        # Use 3 to clear everything
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")

        _print_step("  ✓ Dropped page cache, dentries, and inodes")
        return True
    except Exception as e:
        _print_step(f"  ✗ Failed to drop caches: {e}")
        return False


def sync_filesystem() -> bool:
    """Sync filesystem to ensure all writes are flushed."""
    _print_step("Syncing filesystem...")

    try:
        # Run sync command
        rc, _, _ = _run_cmd(["sync"], check=False)
        if rc == 0:
            _print_step("  ✓ Filesystem synced")
            return True
        else:
            # Fallback: try Python's os.sync if available
            try:
                os.sync()
                _print_step("  ✓ Filesystem synced (via os.sync)")
                return True
            except AttributeError:
                _print_step("  ⚠ sync command failed, but continuing...")
                return False
    except Exception as e:
        _print_step(f"  ⚠ Filesystem sync failed: {e}")
        return False


def clear_python_cache() -> bool:
    """Clear Python bytecode cache."""
    _print_step("Clearing Python cache...")

    try:
        import shutil as sh

        # Find and remove __pycache__ directories
        cache_dirs = []
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        for root, dirs, files in os.walk(repo_root):
            if "__pycache__" in dirs:
                cache_path = os.path.join(root, "__pycache__")
                cache_dirs.append(cache_path)

        if cache_dirs:
            removed = 0
            for cache_dir in cache_dirs:
                try:
                    sh.rmtree(cache_dir)
                    removed += 1
                except Exception:
                    pass

            if removed > 0:
                _print_step(f"  ✓ Removed {removed} __pycache__ directory(ies)")
            else:
                _print_step("  ✓ No Python cache found")
        else:
            _print_step("  ✓ No Python cache found")

        return True
    except Exception as e:
        _print_step(f"  ⚠ Failed to clear Python cache: {e}")
        return False


def check_gpu_processes() -> bool:
    """Check if there are GPU processes running."""
    if not shutil.which("nvidia-smi"):
        return False

    rc, stdout, _ = _run_cmd(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    if rc == 0 and stdout.strip():
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean system for benchmark preparation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force cleanup (kill GPU processes, etc.)",
    )
    parser.add_argument(
        "--skip-python-cache",
        action="store_true",
        help="Skip Python cache cleanup",
    )
    parser.add_argument(
        "--skip-fsync",
        action="store_true",
        help="Skip filesystem sync",
    )
    parser.add_argument(
        "--only-gpu",
        action="store_true",
        help="Only clear GPU memory",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("System Cleanup for Benchmark Preparation")
    print("=" * 70)
    print()

    success_count = 0
    total_count = 0

    # Check GPU processes first
    if check_gpu_processes() and not args.force:
        print("[cleanup] ⚠ WARNING: GPU processes detected!")
        print("[cleanup]   Use --force to kill them, or manually stop them first.")
        print()

    # Clear GPU memory
    if not args.only_gpu:
        if clear_gpu_memory(force=args.force):
            success_count += 1
        total_count += 1
        print()

    # Clear system cache
    if not args.only_gpu:
        if clear_system_cache(force=args.force):
            success_count += 1
        total_count += 1
        print()

    # Sync filesystem
    if not args.only_gpu and not args.skip_fsync:
        if sync_filesystem():
            success_count += 1
        total_count += 1
        print()

    # Clear Python cache
    if not args.only_gpu and not args.skip_python_cache:
        if clear_python_cache():
            success_count += 1
        total_count += 1
        print()

    # Final GPU memory check
    print("=" * 70)
    print("Final Status")
    print("=" * 70)

    if shutil.which("nvidia-smi"):
        _print_step("GPU Memory Status:")
        rc, stdout, _ = _run_cmd(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"])
        if rc == 0:
            for line in stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    idx = parts[0]
                    name = parts[1]
                    used = parts[2]
                    total = parts[3]
                    _print_step(f"  GPU {idx} ({name}): {used} / {total} MB")
        print()

    # Summary
    print("=" * 70)
    if success_count == total_count:
        print(f"✓ Cleanup completed successfully ({success_count}/{total_count} steps)")
    else:
        print(f"⚠ Cleanup completed with warnings ({success_count}/{total_count} steps)")
        if not _check_root() and not args.only_gpu:
            print()
            print("Note: Some steps require root privileges.")
            print("      Run with sudo for full cleanup:")
            print("      sudo python benchmarks/cleanup_system.py")
    print("=" * 70)

    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())

