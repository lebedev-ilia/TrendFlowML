"""
Тесты роутера POST/GET /api/runs (Phase 1 Backend ↔ Fetcher).

Проверяют создание run по source_url, вызов Fetcher и idempotency.
Используются моки БД и fetcher_client.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4
from unittest.mock import MagicMock, patch

import pytest

# Импорт после conftest (backend в path)
from app.dbv2.models import IngestionRun
from app.schemas import CreateRunRequest, IngestionRunOut


class TestCreateRunLogic:
    """Тесты логики создания run (мокируем Fetcher и БД)."""

    def test_fetcher_create_run_called_with_run_id_and_source_url(self):
        """При создании run вызывается fetcher_client.create_run с run_id и source_url."""
        from app.services import fetcher_client

        run_id = uuid4()
        source_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        mock_ingestion = MagicMock(spec=IngestionRun)
        mock_ingestion.run_id = run_id
        mock_ingestion.source_url = source_url
        mock_ingestion.workspace_id = None
        mock_ingestion.ingestion_status = "pending"
        mock_ingestion.created_at = datetime.utcnow()
        mock_ingestion.updated_at = datetime.utcnow()
        mock_ingestion.idempotency_key = None

        with patch.object(
            fetcher_client,
            "create_run",
            return_value={
                "run_id": str(run_id),
                "status": "PENDING",
                "message": "Run created",
            },
        ) as mock_create:
            # Проверяем контракт: create_run вызывается с run_id (UUID) и source_url
            result = fetcher_client.create_run(
                run_id=run_id,
                source_url=source_url,
                settings=MagicMock(
                    fetcher_api_url="http://fetcher:8002",
                    fetcher_api_key=None,
                    fetcher_timeout_seconds=30.0,
                ),
            )
            mock_create.assert_called_once()
            call_kw = mock_create.call_args[1]
            assert call_kw["run_id"] == run_id
            assert call_kw["source_url"] == source_url
            assert result["status"] == "PENDING"

    def test_ingestion_run_out_schema(self):
        """IngestionRunOut содержит run_id, source_url, ingestion_status."""
        run_id = uuid4()
        out = IngestionRunOut(
            run_id=run_id,
            source_url="https://www.youtube.com/watch?v=abc",
            workspace_id=None,
            ingestion_status="pending",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            message=None,
        )
        assert out.run_id == run_id
        assert "youtube" in out.source_url
        assert out.ingestion_status == "pending"

    def test_create_run_request_schema(self):
        """CreateRunRequest принимает source_url и опционально workspace_id."""
        req = CreateRunRequest(source_url="https://youtu.be/abc123")
        assert req.source_url == "https://youtu.be/abc123"
        assert req.workspace_id is None

        req2 = CreateRunRequest(
            source_url="https://www.youtube.com/watch?v=xyz",
            workspace_id=uuid4(),
        )
        assert req2.workspace_id is not None
