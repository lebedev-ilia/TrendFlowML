from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class CategoryConfig(BaseModel):
    name: str
    keywords: List[str] = Field(default_factory=list)
    target_count: int = 5500
    collect_count: int = 6000
    platform_weights: Dict[str, float] = Field(default_factory=lambda: {"youtube": 1.0})
    filters: Dict[str, Any] = Field(default_factory=dict)

    @validator("keywords")
    def require_keywords(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("category must include at least one keyword")
        return value


class CampaignConfig(BaseModel):
    name: str
    output_dir: str
    baseline_accepted: int = Field(
        0,
        description="Videos already collected in prior datasets (for progress totals and dedup context).",
    )
    categories: List[CategoryConfig]
    snapshot_schedule_days: List[int] = Field(default_factory=lambda: [0, 7, 14, 21])
    default_filters: Dict[str, Any] = Field(default_factory=dict)
    platform_weights: Dict[str, float] = Field(default_factory=lambda: {"youtube": 1.0})
    time_interval_buckets: List[Dict[str, Any]] = Field(default_factory=list)
    shard_size: int = 100
    comments_per_snapshot: int = 100
    min_videos_per_keyword: int = Field(
        20,
        description="Minimum unique accepted videos required per keyword before moving on.",
    )
    keyword_search_multiplier: int = Field(
        10,
        description="Search budget per keyword = min_videos_per_keyword * multiplier (covers dupes/filters).",
    )
    hf_repo_id: Optional[str] = None
    hf_shards_repo_id: Optional[str] = Field(
        None,
        description="HF dataset repo for metadata shards; defaults to hf_repo_id.",
    )
    hf_videos_repo_id: Optional[str] = Field(
        None,
        description="HF dataset repo for video files; defaults to hf_repo_id.",
    )
    hf_token_env: str = "HF_TOKEN"
    hf_upload_enabled: bool = False
    hf_upload_every_shards: int = 10
    hf_path_prefix: str = ""
    hf_shards_path_prefix: str = Field(
        "shards/metadata",
        description="Path prefix inside HF repo for metadata shard JSON files.",
    )
    hf_videos_path_prefix: str = Field(
        "videos",
        description="Path prefix inside HF repo for downloaded mp4 files.",
    )
    cookie_files_dir: Optional[str] = None
    cookie_file_glob: str = "*.txt"
    youtube_keys_file: Optional[str] = None
    proxies_file: Optional[str] = None
    proxy_default_scheme: str = "http"
    include_local_proxies_for_discovery: bool = False

    @validator("snapshot_schedule_days")
    def schedule_starts_with_zero(cls, value: List[int]) -> List[int]:
        if not value or value[0] != 0:
            raise ValueError("snapshot_schedule_days must start with 0")
        return value


class Snapshot(BaseModel):
    snapshot_index: int
    time_get: str
    collected_at: datetime
    viewCount: Optional[str] = None
    likeCount: Optional[str] = None
    commentCount: Optional[str] = None
    subscriberCount: Optional[int] = None
    videoCount: Optional[int] = None
    viewCount_channel: Optional[int] = None
    comments: List[Dict[str, Any]] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)


class CollectedVideo(BaseModel):
    platform: str
    video_id: str
    url: str
    category: str
    query: str
    metadata: Dict[str, Any]
    snapshot_0: Snapshot
    time_interval: Optional[str] = None
    discovered_at: datetime
    platform_capabilities: Dict[str, bool] = Field(default_factory=dict)

    @property
    def dedup_key(self) -> str:
        return f"{self.platform}:{self.video_id}"


class RejectedRecord(BaseModel):
    platform: str
    video_id: str
    category: str
    query: str
    reason: str
    record: Dict[str, Any] = Field(default_factory=dict)
    rejected_at: datetime


class ScheduleEntry(BaseModel):
    platform: str
    video_id: str
    category: str
    url: str
    baseline_collected_at: datetime
    due_at: Dict[str, datetime]
    completed: Dict[str, datetime] = Field(default_factory=dict)
    status: str = "pending"

    @property
    def dedup_key(self) -> str:
        return f"{self.platform}:{self.video_id}"


class CampaignManifest(BaseModel):
    name: str
    created_at: datetime
    updated_at: datetime
    output_dir: str
    baseline_accepted: int = 0
    legacy_seen_imported: int = 0
    session_started_at: Optional[datetime] = None
    session_counters: Dict[str, int] = Field(default_factory=dict)
    counters: Dict[str, int] = Field(default_factory=dict)
    category_counters: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    shards: List[str] = Field(default_factory=list)
    snapshot_shards: List[str] = Field(default_factory=list)
    rejected_shards: List[str] = Field(default_factory=list)
