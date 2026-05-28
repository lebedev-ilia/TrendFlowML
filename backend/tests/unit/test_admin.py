"""
Unit-тесты для admin: admin_email_set (Settings), WorkspaceRole.admin, логика «админ по email».

См. backend/docs/TESTING_PLAN.md § 3.6.4 (Admin: проверка admin_emails / роли).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.dbv2.enums import WorkspaceRole


pytestmark = pytest.mark.unit


class TestAdminEmailSet:
    """Парсинг TF_BACKEND_ADMIN_EMAILS в множество email."""

    def test_empty_default(self, monkeypatch):
        """По умолчанию admin_emails пустая строка → пустое множество."""
        monkeypatch.delenv("TF_BACKEND_ADMIN_EMAILS", raising=False)
        s = Settings()
        assert s.admin_email_set() == set()

    def test_single_email(self, monkeypatch):
        """Один email возвращается в множестве в нижнем регистре."""
        monkeypatch.setenv("TF_BACKEND_ADMIN_EMAILS", "Admin@Example.COM")
        s = Settings()
        assert s.admin_email_set() == {"admin@example.com"}

    def test_multiple_emails_normalized(self, monkeypatch):
        """Несколько email через запятую, пробелы и регистр нормализуются."""
        monkeypatch.setenv("TF_BACKEND_ADMIN_EMAILS", " a@b.com , B@B.COM  ")
        s = Settings()
        assert s.admin_email_set() == {"a@b.com", "b@b.com"}

    def test_empty_after_split_ignored(self, monkeypatch):
        """Пустые элементы между запятыми не попадают в множество."""
        monkeypatch.setenv("TF_BACKEND_ADMIN_EMAILS", "x@x.com,,,y@y.com")
        s = Settings()
        assert s.admin_email_set() == {"x@x.com", "y@y.com"}


class TestWorkspaceRoleAdmin:
    """Роль admin в workspace (для будущих проверок по роли)."""

    def test_admin_role_value(self):
        """WorkspaceRole.admin имеет значение 'admin'."""
        assert WorkspaceRole.admin == WorkspaceRole("admin")
        assert WorkspaceRole.admin.value == "admin"
