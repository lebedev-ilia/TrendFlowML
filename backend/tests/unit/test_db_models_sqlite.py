"""
Unit-тесты core v2 моделей (app.dbv2.models) на уровне метаданных.

Цель: убедиться, что модели описаны консистентно (таблицы, схемы, поля, связи),
без поднятия реальной БД. Это частично закрывает TESTING_PLAN § 3.3.2.
"""

from __future__ import annotations

import pytest

from app.dbv2 import enums
from app.dbv2 import models as m


pytestmark = pytest.mark.unit


class TestCoreModelsMetadata:
    """Проверка структуры core.* моделей через SQLAlchemy metadata."""

    def test_user_table_basic_columns_and_schema(self):
        """User использует схему core и имеет обязательные поля email/email_verified."""
        table = m.User.__table__
        assert table.schema == "core"
        assert table.name == "users"
        assert "email" in table.c
        assert "email_verified" in table.c

    def test_workspace_and_member_relationships_defined(self):
        """Workspace и WorkspaceMember связаны по workspace_id и user_id."""
        ws_table = m.Workspace.__table__
        member_table = m.WorkspaceMember.__table__
        assert ws_table.schema == "core"
        assert member_table.schema == "core"
        # Проверяем наличие ожидаемых FK-колонок.
        assert "owner_user_id" in ws_table.c
        assert "workspace_id" in member_table.c
        assert "user_id" in member_table.c
        # Enum роли хранится в колонке role.
        assert "role" in member_table.c
        assert enums.WorkspaceRole.admin in list(enums.WorkspaceRole)

    def test_video_and_analysis_job_and_prediction_tables(self):
        """Video, AnalysisJob и Prediction присутствуют и связаны по внешним ключам."""
        video_table = m.Video.__table__
        job_table = m.AnalysisJob.__table__
        pred_table = m.Prediction.__table__

        assert video_table.schema == "core"
        assert job_table.schema == "core"
        assert pred_table.schema == "core"

        # Базовые поля идентификации/связи
        assert "channel_id" in video_table.c
        assert "video_id" in job_table.c
        assert "analysis_job_id" in pred_table.c


