"""Integration тесты для resume после сбоя Fetcher."""

import pytest


@pytest.mark.integration
@pytest.mark.resume
class TestResumeIntegration:
    """Integration тесты для resume после сбоя."""

    @pytest.mark.slow
    @pytest.mark.database
    def test_resume_after_crash(self, sample_video_url, sample_run_id):
        """Тест resume после сбоя worker'а."""
        # TODO: Реализовать с реальной БД
        # 1. Создать run в статусе FETCHING_METADATA
        # 2. Симулировать сбой worker'а
        # 3. Запустить resume
        # 4. Проверить, что pipeline продолжается с правильной stage
        pass

    @pytest.mark.slow
    @pytest.mark.database
    def test_partial_resume(self, sample_video_url, sample_run_id):
        """Тест resume частично обработанного run'а."""
        # TODO: Реализовать с реальной БД
        # 1. Создать run с частично готовыми артефактами
        # 2. Запустить resume
        # 3. Проверить, что pipeline продолжается с правильной stage
        pass

