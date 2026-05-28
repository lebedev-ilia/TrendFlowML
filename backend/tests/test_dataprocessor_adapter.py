"""
Тесты адаптера Backend v2 → DataProcessor (legacy payload).

Проверяют prepare_dataprocessor_payload и resolve_run_paths_v2 без реальной БД (моки).
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4
from unittest.mock import MagicMock, patch

import pytest

from app.services.dataprocessor_adapter import (
    DataProcessorPayload,
    prepare_dataprocessor_payload,
    resolve_run_paths_v2,
)


pytestmark = pytest.mark.integration


class TestPrepareDataprocessorPayload:
    """Тесты преобразования AnalysisJob → DataProcessor payload."""

    def test_payload_has_required_fields_for_dataprocessor(
        self, tmp_path, sample_profile_config
    ):
        """Payload содержит run_id, video_id, platform_id, config_hash, video_path, profile_config."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"x")

        run_id = uuid4()
        video_id = uuid4()
        channel_id = uuid4()
        config_id = uuid4()

        mock_analysis_job = MagicMock()
        mock_analysis_job.id = run_id
        mock_analysis_job.video_id = video_id
        mock_analysis_job.processing_config_id = config_id

        mock_video = MagicMock()
        mock_video.id = video_id
        mock_video.channel_id = channel_id
        mock_video.external_video_id = "ext-123"
        mock_video.storage_path = str(video_path)
        mock_video.checksum = None

        mock_channel = MagicMock()
        mock_channel.platform = "youtube"

        mock_profile = MagicMock()
        mock_profile.config_hash = sample_profile_config.get("config_hash", "hash")
        mock_profile.config_json = sample_profile_config

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            mock_video,
            mock_channel,
            mock_profile,
        ]

        with patch(
            "app.services.dataprocessor_adapter.Settings"
        ) as MockSettings:
            mock_settings = MagicMock()
            mock_settings.resolve_paths.return_value = MagicMock(
                visual_cfg_default=tmp_path / "visual.yaml"
            )
            MockSettings.return_value = mock_settings

            payload = prepare_dataprocessor_payload(mock_db, mock_analysis_job)

        assert isinstance(payload, DataProcessorPayload)
        assert payload.run_id == str(run_id)
        assert payload.video_id == "ext-123"
        assert payload.platform_id == "youtube"
        assert payload.config_hash == sample_profile_config.get("config_hash", "hash")
        assert payload.video_path == video_path
        assert "processors" in payload.profile_config
        assert payload.profile_config.get("visual", {}).get("cfg_path")

    def test_video_id_fallback_to_uuid_when_no_external_id(self, tmp_path):
        """Если external_video_id пустой, video_id = str(video.id)."""
        video_path = tmp_path / "v.mp4"
        video_path.write_bytes(b"x")
        run_id = uuid4()
        video_id = uuid4()
        channel_id = uuid4()
        config_id = uuid4()

        mock_analysis_job = MagicMock()
        mock_analysis_job.id = run_id
        mock_analysis_job.video_id = video_id
        mock_analysis_job.processing_config_id = config_id

        mock_video = MagicMock()
        mock_video.id = video_id
        mock_video.channel_id = channel_id
        mock_video.external_video_id = None  # нет внешнего id
        mock_video.storage_path = str(video_path)
        mock_video.checksum = None

        mock_channel = MagicMock()
        mock_channel.platform = "upload"

        mock_profile = MagicMock()
        mock_profile.config_hash = "h"
        mock_profile.config_json = {"processors": {}, "visual": {"cfg_path": str(tmp_path)}}

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            mock_video,
            mock_channel,
            mock_profile,
        ]

        with patch(
            "app.services.dataprocessor_adapter.Settings"
        ) as MockSettings:
            mock_settings = MagicMock()
            mock_settings.resolve_paths.return_value = MagicMock(
                visual_cfg_default=tmp_path / "v.yaml"
            )
            MockSettings.return_value = mock_settings

            payload = prepare_dataprocessor_payload(mock_db, mock_analysis_job)

        assert payload.video_id == str(video_id)
        assert payload.platform_id == "upload"

    def test_raises_when_video_not_found(self):
        """ValueError если Video не найден."""
        run_id = uuid4()
        mock_analysis_job = MagicMock()
        mock_analysis_job.id = run_id
        mock_analysis_job.video_id = uuid4()
        mock_analysis_job.processing_config_id = uuid4()

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        with pytest.raises(ValueError, match="Video not found"):
            prepare_dataprocessor_payload(mock_db, mock_analysis_job)

    def test_raises_when_channel_not_found(self, tmp_path):
        """ValueError если Channel не найден."""
        video_path = tmp_path / "v.mp4"
        video_path.write_bytes(b"x")
        mock_analysis_job = MagicMock()
        mock_analysis_job.id = uuid4()
        mock_analysis_job.video_id = uuid4()
        mock_analysis_job.processing_config_id = uuid4()

        mock_video = MagicMock()
        mock_video.id = uuid4()
        mock_video.channel_id = uuid4()
        mock_video.external_video_id = "x"
        mock_video.storage_path = str(video_path)
        mock_video.checksum = None

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            mock_video,
            None,  # channel not found
        ]

        with pytest.raises(ValueError, match="Channel not found"):
            prepare_dataprocessor_payload(mock_db, mock_analysis_job)


class TestResolveRunPathsV2:
    """Тесты путей результатов (совпадение с DataProcessor)."""

    def test_paths_structure_matches_dataprocessor(self, tmp_path):
        """Структура каталогов совпадает с тем, куда пишет DataProcessor."""
        result_store_base = tmp_path / "result_store"
        analysis_job_id = uuid4()
        paths = resolve_run_paths_v2(
            platform_id="youtube",
            video_id="dQw4w9WgXcQ",
            analysis_job_id=analysis_job_id,
            result_store_base=result_store_base,
        )
        run_id = str(analysis_job_id)
        assert paths["run_rs_path"] == (
            result_store_base / "youtube" / "dQw4w9WgXcQ" / run_id
        )
        assert paths["manifest_path"] == paths["run_rs_path"] / "manifest.json"
        assert paths["state_events_path"] == (
            result_store_base.parent
            / "state"
            / "youtube"
            / "dQw4w9WgXcQ"
            / run_id
            / "state_events.jsonl"
        )
