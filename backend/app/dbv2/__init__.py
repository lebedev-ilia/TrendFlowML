"""Database schema v2 (SQLAlchemy 2.0 typed mappings).

This module is intentionally kept separate from the legacy `app/models.py`
to avoid breaking existing API routes while we migrate incrementally.
"""

from .base import Base, SoftDeleteMixin, TimestampMixin
from .enums import (
    AnalysisStatus,
    SourceType,
    SubscriptionStatus,
    VideoType,
    WorkspaceRole,
)
from .models import (
    AnalysisJob,
    Channel,
    Prediction,
    Subscription,
    SubscriptionPlan,
    User,
    UserOAuthAccount,
    UserSecurity,
    Video,
    Workspace,
    WorkspaceMember,
)

__all__ = [
    "Base",
    "TimestampMixin",
    "SoftDeleteMixin",
    # enums
    "WorkspaceRole",
    "SubscriptionStatus",
    "VideoType",
    "SourceType",
    "AnalysisStatus",
    # models
    "User",
    "UserOAuthAccount",
    "UserSecurity",
    "Workspace",
    "WorkspaceMember",
    "SubscriptionPlan",
    "Subscription",
    "Channel",
    "Video",
    "AnalysisJob",
    "Prediction",
]


