from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Set

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class KeywordProgressEntry(BaseModel):
    """One finished keyword attempt (append-only log)."""

    category: str
    bucket_name: Optional[str] = None
    platform: str = "youtube"
    keyword_index: int
    keyword: str
    accepted: int
    min_required: int
    scanned: int = 0
    duplicate: int = 0
    rejected: int = 0
    status: str = Field(description="'done' if accepted >= min_required else 'low'")
    completed_at: datetime = Field(default_factory=_utcnow)

    class Config:
        extra = "ignore"

    @property
    def is_done(self) -> bool:
        # Done if we hit the target, OR if the keyword is saturated
        # (yielded results but all were duplicates — no new unique videos).
        saturated = self.scanned > 0 and self.accepted == 0
        return self.accepted >= self.min_required or saturated


def progress_scope_key(
    *,
    category: str,
    bucket_name: Optional[str],
    platform: str,
    keyword_index: int,
) -> tuple[str, Optional[str], str, int]:
    return (category, bucket_name, platform, keyword_index)
