from __future__ import annotations

import json
from pathlib import Path

from fetcher.services.credentials import CredentialsStore


def test_credentials_store_reads_json(tmp_path: Path, monkeypatch):
    cred_dir = tmp_path / "credentials"
    cred_dir.mkdir()
    (cred_dir / "tiktok.json").write_text(
        json.dumps({"client_key": "k", "client_secret": "s", "access_token": "t", "ms_token": "m"}),
        encoding="utf-8",
    )
    monkeypatch.delenv("FETCHER_TIKTOK_CLIENT_KEY", raising=False)
    store = CredentialsStore(credentials_dir=cred_dir)
    creds = store.tiktok()
    assert creds.client_key == "k"
    assert creds.api_configured
    assert creds.sdk_configured


def test_env_overrides_file(tmp_path: Path, monkeypatch):
    cred_dir = tmp_path / "credentials"
    cred_dir.mkdir()
    (cred_dir / "twitch.json").write_text(json.dumps({"client_id": "file_id"}), encoding="utf-8")
    monkeypatch.setenv("FETCHER_TWITCH_CLIENT_ID", "env_id")
    monkeypatch.setenv("FETCHER_TWITCH_ACCESS_TOKEN", "env_token")
    store = CredentialsStore(credentials_dir=cred_dir)
    creds = store.twitch()
    assert creds.client_id == "env_id"
    assert creds.access_token == "env_token"
