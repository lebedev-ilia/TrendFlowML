"""
Unit тесты для Validators
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from api.utils.validators import (
    validate_video_path,
    validate_profile_config,
    validate_run_id,
    validate_platform_id,
    _get_allowed_video_paths
)
from api.utils.errors import InvalidPayloadError


class TestValidateVideoPath:
    """Тесты для функции validate_video_path."""
    
    def test_validate_video_path_success(self):
        """Успешная валидация существующего файла."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_file = Path(tmpdir) / "test_video.mp4"
            video_file.write_bytes(b"test video content")
            
            with patch("api.utils.validators.config") as mock_config:
                mock_config.allowed_video_paths = tmpdir
                
                # Не должно вызывать исключение
                validate_video_path(str(video_file))
    
    def test_validate_video_path_not_found(self):
        """Валидация несуществующего файла."""
        with pytest.raises(InvalidPayloadError) as exc_info:
            validate_video_path("/nonexistent/path/video.mp4")
        
        assert "Video file not found" in str(exc_info.value)
        assert exc_info.value.details["field"] == "video_path"
    
    def test_validate_video_path_not_file(self):
        """Валидация пути, который не является файлом."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(InvalidPayloadError) as exc_info:
                validate_video_path(tmpdir)
            
            assert "Path is not a file" in str(exc_info.value)
    
    def test_validate_video_path_outside_allowed(self):
        """Валидация пути вне разрешённых директорий."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_file = Path(tmpdir) / "test_video.mp4"
            video_file.write_bytes(b"test video content")
            
            with patch("api.utils.validators.config") as mock_config:
                mock_config.allowed_video_paths = "/other/allowed/path"
                
                with pytest.raises(InvalidPayloadError) as exc_info:
                    validate_video_path(str(video_file))
                
                assert "Video path outside allowed directories" in str(exc_info.value)
                assert "allowed_paths" in exc_info.value.details
    
    def test_validate_video_path_file_too_large(self):
        """Валидация файла превышающего размер."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_file = Path(tmpdir) / "test_video.mp4"
            # Создать большой файл (больше 10GB)
            video_file.write_bytes(b"x" * (11 * 1024 * 1024 * 1024))  # 11GB
            
            with patch("api.utils.validators.config") as mock_config:
                mock_config.allowed_video_paths = tmpdir
                with patch.dict(os.environ, {"MAX_VIDEO_SIZE_BYTES": str(10 * 1024 * 1024 * 1024)}):
                    with pytest.raises(InvalidPayloadError) as exc_info:
                        validate_video_path(str(video_file))
                    
                    assert "Video file too large" in str(exc_info.value)
    
    def test_validate_video_path_no_allowed_paths(self):
        """Валидация когда разрешённые пути не настроены."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_file = Path(tmpdir) / "test_video.mp4"
            video_file.write_bytes(b"test video content")
            
            with patch("api.utils.validators.config") as mock_config:
                mock_config.allowed_video_paths = ""
                
                # Не должно вызывать исключение если allowed_paths пуст
                validate_video_path(str(video_file))


class TestValidateProfileConfig:
    """Тесты для функции validate_profile_config."""
    
    def test_validate_profile_config_success(self):
        """Успешная валидация валидной конфигурации."""
        profile_config = {
            "processors": {
                "segmenter": {"enabled": True, "required": True},
                "visual": {"enabled": True}
            }
        }
        
        # Не должно вызывать исключение
        validate_profile_config(profile_config)
    
    def test_validate_profile_config_not_dict(self):
        """Валидация конфигурации не являющейся словарём."""
        with pytest.raises(InvalidPayloadError) as exc_info:
            validate_profile_config("not a dict")
        
        assert "profile_config must be a dictionary" in str(exc_info.value)
    
    def test_validate_profile_config_missing_processors(self):
        """Валидация конфигурации без ключа processors."""
        with pytest.raises(InvalidPayloadError) as exc_info:
            validate_profile_config({"other_key": "value"})
        
        assert "profile_config must contain 'processors'" in str(exc_info.value)
    
    def test_validate_profile_config_processors_not_dict(self):
        """Валидация когда processors не словарь."""
        with pytest.raises(InvalidPayloadError) as exc_info:
            validate_profile_config({"processors": "not a dict"})
        
        assert "profile_config.processors must be a dictionary" in str(exc_info.value)
    
    def test_validate_profile_config_invalid_processor(self):
        """Валидация конфигурации с невалидным процессором."""
        profile_config = {
            "processors": {
                "invalid_processor": {"enabled": True}
            }
        }
        
        with pytest.raises(InvalidPayloadError) as exc_info:
            validate_profile_config(profile_config)
        
        assert "Unknown processor" in str(exc_info.value)
        assert exc_info.value.details["field"] == "profile_config.processors.invalid_processor"
    
    def test_validate_profile_config_processor_not_dict(self):
        """Валидация когда конфигурация процессора не словарь."""
        profile_config = {
            "processors": {
                "segmenter": "not a dict"
            }
        }
        
        with pytest.raises(InvalidPayloadError) as exc_info:
            validate_profile_config(profile_config)
        
        assert "Processor config for segmenter must be a dictionary" in str(exc_info.value)


class TestValidateRunId:
    """Тесты для функции validate_run_id."""
    
    def test_validate_run_id_success(self):
        """Успешная валидация валидного UUID."""
        valid_uuid = "550e8400-e29b-41d4-a716-446655440000"
        
        # Не должно вызывать исключение
        validate_run_id(valid_uuid)
    
    def test_validate_run_id_invalid_format(self):
        """Валидация невалидного формата UUID."""
        with pytest.raises(InvalidPayloadError) as exc_info:
            validate_run_id("invalid-uuid")
        
        assert "Invalid run_id format" in str(exc_info.value)
        assert exc_info.value.details["field"] == "run_id"
    
    def test_validate_run_id_too_short(self):
        """Валидация слишком короткого UUID."""
        with pytest.raises(InvalidPayloadError):
            validate_run_id("550e8400-e29b-41d4-a716")
    
    def test_validate_run_id_uppercase(self):
        """Валидация UUID в верхнем регистре."""
        valid_uuid = "550E8400-E29B-41D4-A716-446655440000"
        
        # Не должно вызывать исключение (lowercase в regex)
        validate_run_id(valid_uuid)


class TestValidatePlatformId:
    """Тесты для функции validate_platform_id."""
    
    def test_validate_platform_id_youtube(self):
        """Валидация platform_id = youtube."""
        # Не должно вызывать исключение
        validate_platform_id("youtube")
    
    def test_validate_platform_id_upload(self):
        """Валидация platform_id = upload."""
        # Не должно вызывать исключение
        validate_platform_id("upload")
    
    def test_validate_platform_id_invalid(self):
        """Валидация невалидного platform_id."""
        with pytest.raises(InvalidPayloadError) as exc_info:
            validate_platform_id("invalid_platform")
        
        assert "Invalid platform_id" in str(exc_info.value)
        assert exc_info.value.details["field"] == "platform_id"
        assert "valid_platforms" in exc_info.value.details


class TestGetAllowedVideoPaths:
    """Тесты для функции _get_allowed_video_paths."""
    
    def test_get_allowed_video_paths_success(self):
        """Получение разрешённых путей."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("api.utils.validators.config") as mock_config:
                mock_config.allowed_video_paths = f"{tmpdir},/other/path"
                
                paths = _get_allowed_video_paths()
                
                assert len(paths) >= 1
                assert tmpdir in paths or str(Path(tmpdir).resolve()) in paths
    
    def test_get_allowed_video_paths_empty(self):
        """Получение пустого списка когда не настроено."""
        with patch("api.utils.validators.config") as mock_config:
            mock_config.allowed_video_paths = ""
            
            paths = _get_allowed_video_paths()
            
            assert paths == []
    
    def test_get_allowed_video_paths_nonexistent(self):
        """Получение путей с несуществующими директориями."""
        with patch("api.utils.validators.config") as mock_config:
            mock_config.allowed_video_paths = "/nonexistent/path1,/nonexistent/path2"
            
            paths = _get_allowed_video_paths()
            
            # Несуществующие пути должны быть отфильтрованы
            assert len(paths) == 0
    
    def test_get_allowed_video_paths_mixed(self):
        """Получение путей с существующими и несуществующими."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("api.utils.validators.config") as mock_config:
                mock_config.allowed_video_paths = f"{tmpdir},/nonexistent/path"
                
                paths = _get_allowed_video_paths()
                
                # Должен быть только существующий путь
                assert len(paths) >= 1

