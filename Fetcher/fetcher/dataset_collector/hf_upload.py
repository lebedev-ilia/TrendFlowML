from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from fetcher.dataset_collector.schemas import CampaignConfig


class HuggingFaceUploadError(RuntimeError):
    pass


def upload_paths(config: CampaignConfig, paths: Iterable[Path], *, repo_id: str | None = None) -> dict[str, int]:
    target_repo = repo_id or config.hf_repo_id
    if not target_repo:
        raise HuggingFaceUploadError("hf_repo_id is not configured")
    token = os.getenv(config.hf_token_env)
    if not token:
        raise HuggingFaceUploadError(f"{config.hf_token_env} is not set")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise HuggingFaceUploadError("Install huggingface_hub to enable HF uploads") from exc

    api = HfApi(token=token)
    uploaded = 0
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        remote_path = f"{config.hf_path_prefix.strip('/')}/{path.name}".strip("/")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote_path,
            repo_id=target_repo,
            repo_type="dataset",
        )
        uploaded += 1
    return {"uploaded": uploaded}


def maybe_upload_recent_shards(config: CampaignConfig, root: Path, shard_paths: list[Path]) -> dict[str, int]:
    if not config.hf_upload_enabled or not shard_paths:
        return {"uploaded": 0}
    every = max(config.hf_upload_every_shards, 1)
    if len(shard_paths) % every != 0:
        return {"uploaded": 0}
    return upload_paths(config, shard_paths[-every:])
