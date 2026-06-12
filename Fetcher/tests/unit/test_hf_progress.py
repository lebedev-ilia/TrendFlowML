from __future__ import annotations

import json
from pathlib import Path

import pytest

from fetcher.dataset_collector.checkpoint import DiscoveryCheckpoint
from fetcher.dataset_collector.config import default_campaign_config
from fetcher.dataset_collector.hf_progress import (
    _merge_jsonl_union_key,
    discover_week_allows_run,
    register_discover_daily_session,
)
from fetcher.dataset_collector.state import DatasetState, utcnow


@pytest.mark.unit
def test_merge_jsonl_union_key(tmp_path):
    local = tmp_path / "seen.jsonl"
    remote = tmp_path / "remote.jsonl"
    local.write_text('{"key": "youtube:a", "category": "Sport"}\n', encoding="utf-8")
    remote.write_text(
        '{"key": "youtube:b", "category": "Sport"}\n{"key": "youtube:a", "category": "Old"}\n',
        encoding="utf-8",
    )
    added = _merge_jsonl_union_key(local, remote, key_field="key")
    rows = [json.loads(line) for line in local.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = {row["key"] for row in rows}
    assert keys == {"youtube:a", "youtube:b"}
    assert added == 0
    assert rows[0]["category"] == "Sport"


@pytest.mark.unit
def test_discover_week_session_counter(tmp_path):
    config = default_campaign_config(output_dir=str(tmp_path))
    config.discover_week_days = 7
    state = DatasetState(config)
    state.initialize()
    meta1 = register_discover_daily_session(state, config)
    meta2 = register_discover_daily_session(state, config)
    assert meta1["discover_days_run"] == 1
    assert meta2["discover_days_run"] == 1
    assert discover_week_allows_run(state, config)


@pytest.mark.unit
def test_checkpoint_merge_prefers_newer(tmp_path):
    from fetcher.dataset_collector.hf_progress import _merge_checkpoint

    local = tmp_path / "cp.json"
    remote = tmp_path / "cp_remote.json"
    old = DiscoveryCheckpoint(category="Sport", keyword_index=1, updated_at=utcnow())
    new = DiscoveryCheckpoint(category="Sport", keyword_index=5, updated_at=utcnow())
    local.write_text(json.dumps({"category": "Sport", "keyword_index": 1, "updated_at": old.updated_at.isoformat()}), encoding="utf-8")
    remote.write_text(json.dumps({"category": "Sport", "keyword_index": 5, "updated_at": new.updated_at.isoformat()}), encoding="utf-8")
    _merge_checkpoint(local, remote)
    data = json.loads(local.read_text(encoding="utf-8"))
    assert data["keyword_index"] == 5
