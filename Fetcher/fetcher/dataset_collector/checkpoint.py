from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DiscoveryCheckpoint(BaseModel):
    """Resume position for discovery across process restarts."""

    category: str
    bucket_name: Optional[str] = None
    platform: str = "youtube"
    keyword_index: int = 0
    keyword: Optional[str] = None
    updated_at: datetime = Field(default_factory=_utcnow)

    class Config:
        extra = "ignore"
