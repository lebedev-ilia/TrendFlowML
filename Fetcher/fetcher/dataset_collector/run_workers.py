from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List

_SHUTDOWN = False


def _fetcher_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cli_cmd(args: List[str]) -> List[str]:
    return [sys.executable, "-m", "fetcher.dataset_collector.cli", *args]


def _request_shutdown(*_args) -> None:
    global _SHUTDOWN
    _SHUTDOWN = True


def _pass_had_work(result: dict) -> bool:
    """True when the queue pass found and attempted at least one actionable item."""
    return int(result.get("attempted") or 0) > 0


def _idle_sleep(idle_interval_sec: int) -> None:
    """Sleep in small slices so SIGINT/SIGTERM can stop the worker promptly."""
    deadline = time.monotonic() + max(idle_interval_sec, 10)
    while not _SHUTDOWN and time.monotonic() < deadline:
        time.sleep(min(5, deadline - time.monotonic()))


def _run_queue_pass(
    kind: str,
    *,
    config_path: str,
    category: str | None,
) -> dict:
    from fetcher.dataset_collector.config import load_campaign_config
    from fetcher.dataset_collector.cookies import CookieRotator
    from fetcher.dataset_collector.downloads import run_download_queue
    from fetcher.dataset_collector.hf_queues import (
        run_hf_shard_upload_queue,
        run_hf_video_upload_queue,
    )
    from fetcher.dataset_collector.metadata_enrichment import run_metadata_enrich_queue
    from fetcher.dataset_collector.state import DatasetState

    config = load_campaign_config(config_path)
    state = DatasetState(config)
    state.initialize()
    cookie_rotator = CookieRotator.from_config(config)

    if kind == "download":
        queue_path = state.download_dir / "queue.jsonl"
        return run_download_queue(
            state,
            config,
            queue_path,
            cookie_rotator=cookie_rotator,
        )
    if kind == "enrich-metadata":
        return run_metadata_enrich_queue(
            state,
            config,
            category=category,
            cookie_rotator=cookie_rotator,
        )
    if kind == "upload-hf-shards":
        return run_hf_shard_upload_queue(state, config, category=category)
    if kind == "upload-hf-videos":
        return run_hf_video_upload_queue(state, config, category=category)
    raise ValueError(f"unknown queue worker kind: {kind}")


def _queue_worker_daemon(
    *,
    name: str,
    kind: str,
    config_path: str,
    category: str | None,
    idle_interval_sec: int,
    log_dir: Path | str,
    once: bool = False,
) -> None:
    """Long-running queue worker: drain pending work, sleep only when idle."""
    log_dir = Path(log_dir)
    log_path = log_dir / f"{name}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"=== {name} worker; idle_interval={idle_interval_sec}s "
            f"(sleep only when queue has no pending work) ===\n"
        )
        log.flush()

        while not _SHUTDOWN:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log.write(f"\n--- {stamp} ---\n")
            log.flush()

            try:
                result = _run_queue_pass(kind, config_path=config_path, category=category)
            except Exception as exc:
                log.write(f"ERROR {type(exc).__name__}: {exc}\n")
                log.flush()
                if once or _SHUTDOWN:
                    break
                _idle_sleep(idle_interval_sec)
                continue

            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            log.flush()

            if once:
                break

            if _pass_had_work(result):
                # More items may have been enqueued while we worked — recheck immediately.
                time.sleep(1)
            else:
                _idle_sleep(idle_interval_sec)


def _worker_loop(
    *,
    name: str,
    args: List[str],
    interval_sec: int,
    log_dir: Path | str,
    once: bool = False,
) -> None:
    """Legacy subprocess loop for long commands such as discover."""
    log_dir = Path(log_dir)
    log_path = log_dir / f"{name}.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"=== {name} worker; interval={interval_sec}s ===\n")
        while True:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log.write(f"\n--- {stamp} ---\n")
            log.flush()
            proc = subprocess.run(
                _cli_cmd(args),
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(_fetcher_root()),
            )
            log.write(f"exit_code={proc.returncode}\n")
            log.flush()
            if once:
                break
            _idle_sleep(interval_sec)


def _start_inventory_metrics_thread(
    *,
    config_path: str,
    metrics_port: int,
    refresh_sec: int = 30,
) -> None:
    from fetcher.dataset_collector.config import load_campaign_config
    from fetcher.dataset_collector.inventory import refresh_summary
    from fetcher.dataset_collector.metrics import start_metrics_server, update_inventory_gauges
    from fetcher.dataset_collector.state import DatasetState

    config = load_campaign_config(config_path)
    state = DatasetState(config)
    state.initialize()
    start_metrics_server(metrics_port)
    print(f"Prometheus metrics: http://127.0.0.1:{metrics_port}/metrics", flush=True)

    def _loop() -> None:
        while True:
            try:
                summary = refresh_summary(state)
                update_inventory_gauges(summary)
            except Exception as exc:
                print(f"[metrics] inventory refresh failed: {exc}", flush=True)
            time.sleep(max(refresh_sec, 10))

    thread = threading.Thread(target=_loop, name="inventory-metrics", daemon=True)
    thread.start()


QueueWorkerSpec = tuple[str, str, bool]


def run_all_workers(
    *,
    config_path: str,
    category: str | None,
    log_dir: Path,
    interval_sec: int = 120,
    metrics_port: int | None = 9095,
    with_discover: bool = True,
    once: bool = False,
) -> None:
    """Run discover + queue workers in parallel (separate OS processes)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    if metrics_port:
        _start_inventory_metrics_thread(config_path=config_path, metrics_port=metrics_port)

    subprocess_workers: list[tuple[str, list[str], bool]] = []
    queue_workers: list[QueueWorkerSpec] = []

    if with_discover:
        discover_args = ["discover", config_path]
        if category:
            discover_args.extend(["--category", category])
        subprocess_workers.append(("discover", discover_args, True))

    queue_workers.extend(
        [
            ("enrich-metadata", "enrich-metadata", False),
            ("download", "download", False),
            ("upload-hf-shards", "upload-hf-shards", False),
            ("upload-hf-videos", "upload-hf-videos", False),
        ]
    )

    processes: list[tuple[str, subprocess.Popen]] = []

    for name, args, run_once in subprocess_workers:
        log_path = log_dir / f"{name}.log"
        log_file = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "from fetcher.dataset_collector.run_workers import _worker_loop; "
                    f"_worker_loop(name={name!r}, args={args!r}, interval_sec={interval_sec}, "
                    f"log_dir={str(log_dir)!r}, once={once or run_once})"
                ),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(_fetcher_root()),
        )
        processes.append((name, proc))
        print(f"started {name} (pid={proc.pid}) -> {log_path}", flush=True)

    for name, kind, run_once in queue_workers:
        log_path = log_dir / f"{name}.log"
        log_file = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "from fetcher.dataset_collector.run_workers import _queue_worker_daemon; "
                    f"_queue_worker_daemon(name={name!r}, kind={kind!r}, "
                    f"config_path={config_path!r}, category={category!r}, "
                    f"idle_interval_sec={interval_sec}, log_dir={str(log_dir)!r}, "
                    f"once={once or run_once})"
                ),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(_fetcher_root()),
        )
        processes.append((name, proc))
        print(f"started {name} (pid={proc.pid}) -> {log_path}", flush=True)

    try:
        for _, proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\nStopping workers...", flush=True)
        for name, proc in processes:
            proc.terminate()
        for name, proc in processes:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"killed {name}", flush=True)


def main(argv: List[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run all dataset collector workers in parallel.")
    parser.add_argument("config", help="Path to dataset_campaign.json")
    parser.add_argument("--category", default="Sport")
    parser.add_argument(
        "--interval",
        type=int,
        default=120,
        help="Seconds to wait when a queue worker is idle (does not interrupt active downloads).",
    )
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--metrics-port", type=int, default=9095)
    parser.add_argument("--no-discover", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    log_dir = args.log_dir
    if log_dir is None:
        from fetcher.dataset_collector.config import load_campaign_config

        cfg = load_campaign_config(args.config)
        log_dir = Path(cfg.output_dir) / "logs" / "workers"

    run_all_workers(
        config_path=args.config,
        category=args.category,
        log_dir=log_dir,
        interval_sec=args.interval,
        metrics_port=args.metrics_port,
        with_discover=not args.no_discover,
        once=args.once,
    )


if __name__ == "__main__":
    main()
