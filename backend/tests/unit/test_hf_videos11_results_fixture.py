"""
Gate для HF videos11 E2E: committed fixture backend/tests/fixtures/hf_videos11_results.json.

CI (GitHub Actions) проверяет контракт fixture без storage/result_store.
Полная §0.1+§0.2 регрессия — e2e_verify_hf_results.py на self-hosted runner (см. hf-e2e-regression.yml).

Обновление fixture после успешного batch:
  ./backend/scripts/ci_sync_hf_results_fixture.sh
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hf_videos11_results.json"
_MANIFEST = Path(__file__).resolve().parents[3] / "example" / "hf_videos11" / "manifest.json"
_RUN_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)


def _load_fixture() -> dict:
    assert _FIXTURE.is_file(), f"missing committed fixture: {_FIXTURE}"
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _manifest_video_ids() -> set[str]:
    if not _MANIFEST.is_file():
        pytest.skip(f"HF manifest not found: {_MANIFEST}")
    data = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    return {str(v["video_id"]) for v in (data.get("videos") or []) if isinstance(v, dict) and v.get("video_id")}


class TestHfVideos11ResultsFixture:
    def test_fixture_has_five_results(self):
        payload = _load_fixture()
        results = payload.get("results")
        assert isinstance(results, list)
        assert len(results) == 5

    def test_video_ids_match_hf_manifest(self):
        expected = _manifest_video_ids()
        payload = _load_fixture()
        got = {str(r["video_id"]) for r in payload["results"]}
        assert got == expected

    def test_all_runs_green_with_core_identity(self):
        payload = _load_fixture()
        for entry in payload["results"]:
            vid = entry["video_id"]
            assert entry.get("e2e_exit") == 0, vid
            assert entry.get("quality_exit") == 0, vid
            assert entry.get("green_exit") == 0, vid
            assert entry.get("core_identity") is True, vid
            run_id = str(entry.get("run_id") or "")
            assert _RUN_ID_RE.match(run_id), f"{vid}: invalid run_id {run_id!r}"

    def test_no_duplicate_video_or_run_ids(self):
        payload = _load_fixture()
        vids: list[str] = []
        runs: list[str] = []
        for entry in payload["results"]:
            vids.append(str(entry["video_id"]))
            runs.append(str(entry["run_id"]))
        assert len(vids) == len(set(vids))
        assert len(runs) == len(set(runs))
