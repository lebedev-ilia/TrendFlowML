from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable

from fetcher.dataset_collector.schemas import CampaignConfig


class HuggingFaceUploadError(RuntimeError):
    pass


def resolve_hf_token(config: CampaignConfig) -> str:
    token = os.getenv(config.hf_token_env)
    if not token:
        raise HuggingFaceUploadError(f"{config.hf_token_env} is not set")
    return token


def resolve_shards_repo_id(config: CampaignConfig, *, repo_id: str | None = None) -> str:
    target = repo_id or config.hf_shards_repo_id or config.hf_repo_id
    if not target:
        raise HuggingFaceUploadError("hf_shards_repo_id / hf_repo_id is not configured")
    return target


def resolve_videos_repo_id(config: CampaignConfig, *, repo_id: str | None = None) -> str:
    target = repo_id or config.hf_videos_repo_id or config.hf_repo_id
    if not target:
        raise HuggingFaceUploadError("hf_videos_repo_id / hf_repo_id is not configured")
    return target


def remote_shard_path(config: CampaignConfig, shard_relpath: str) -> str:
    prefix = (config.hf_shards_path_prefix or "").strip("/")
    rel = shard_relpath.lstrip("/")
    return f"{prefix}/{rel}" if prefix else rel


def remote_video_path(config: CampaignConfig, *, category: str, video_id: str) -> str:
    prefix = (config.hf_videos_path_prefix or "videos").strip("/")
    return f"{prefix}/{category}/{video_id}.mp4"


def get_hf_api(config: CampaignConfig):
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise HuggingFaceUploadError("Install huggingface_hub to enable HF uploads") from exc
    return HfApi(token=resolve_hf_token(config))


def upload_local_file(
    config: CampaignConfig,
    local_path: Path,
    *,
    repo_id: str,
    path_in_repo: str,
) -> None:
    if not local_path.is_file():
        raise HuggingFaceUploadError(f"Local file not found: {local_path}")
    api = get_hf_api(config)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo.lstrip("/"),
        repo_id=repo_id,
        repo_type="dataset",
    )


def upload_paths(
    config: CampaignConfig,
    paths: Iterable[Path],
    *,
    repo_id: str | None = None,
    path_builder: Callable[[Path], str] | None = None,
) -> dict[str, int]:
    target_repo = resolve_shards_repo_id(config, repo_id=repo_id)
    uploaded = 0
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        if path_builder is not None:
            remote = path_builder(path)
        else:
            prefix = (config.hf_path_prefix or config.hf_shards_path_prefix or "").strip("/")
            remote = f"{prefix}/{path.name}".strip("/") if prefix else path.name
        upload_local_file(config, path, repo_id=target_repo, path_in_repo=remote)
        uploaded += 1
    return {"uploaded": uploaded}


def maybe_upload_recent_shards(config: CampaignConfig, root: Path, shard_paths: list[Path]) -> dict[str, int]:
    """Legacy inline upload during discover; prefer hf_shard_upload_queue + upload-hf-shards."""
    if not config.hf_upload_enabled or not shard_paths:
        return {"uploaded": 0}
    every = max(config.hf_upload_every_shards, 1)
    if len(shard_paths) % every != 0:
        return {"uploaded": 0}
    return upload_paths(config, shard_paths[-every:])
