#!/usr/bin/env python3
"""Разовый, побочно-безопасный (ничего не собирает, только читает локальное state) статус
снапшотов — для hourly_report.py (automation/fetcher/deploy.py::read_snapshot_status). Печатает
JSON в stdout. Запускается на самом поде (SSH), не локально.

Usage: python3 scripts/snapshot_status.py <path-to-campaign-config.json>
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Fetcher/ -> `fetcher` package importable

from fetcher.dataset_collector.config import load_campaign_config
from fetcher.dataset_collector.state import DatasetState
from fetcher.dataset_collector.snapshots import snapshot_poll_report, snapshot_follow_up_indices


def _default(o):
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)


def main() -> None:
    config_path = sys.argv[1]
    config = load_campaign_config(config_path)
    state = DatasetState(config)
    state.initialize()
    indices = snapshot_follow_up_indices(config)
    report = snapshot_poll_report(state, config, now=datetime.now(timezone.utc))
    report["indices_configured"] = indices
    report["schedule_size"] = len(state.load_schedule())
    print(json.dumps(report, default=_default, ensure_ascii=False))


if __name__ == "__main__":
    main()
