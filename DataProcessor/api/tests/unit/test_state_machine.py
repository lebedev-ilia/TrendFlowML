"""
Unit тесты для State Machine Service
"""

import pytest
from api.services.state_machine import (
    can_transition,
    validate_transition,
    get_allowed_transitions,
    is_final_status,
    parse_status,
    ALLOWED_TRANSITIONS
)
from api.schemas.state import RunStatus


class TestCanTransition:
    """Тесты для функции can_transition."""
    
    def test_can_transition_pending_to_queued(self):
        """Разрешенный переход PENDING -> QUEUED."""
        assert can_transition(RunStatus.PENDING, RunStatus.QUEUED) is True
    
    def test_can_transition_pending_to_cancelled(self):
        """Разрешенный переход PENDING -> CANCELLED."""
        assert can_transition(RunStatus.PENDING, RunStatus.CANCELLED) is True
    
    def test_can_transition_queued_to_running(self):
        """Разрешенный переход QUEUED -> RUNNING."""
        assert can_transition(RunStatus.QUEUED, RunStatus.RUNNING) is True
    
    def test_can_transition_running_to_success(self):
        """Разрешенный переход RUNNING -> SUCCESS."""
        assert can_transition(RunStatus.RUNNING, RunStatus.SUCCESS) is True
    
    def test_can_transition_running_to_error(self):
        """Разрешенный переход RUNNING -> ERROR."""
        assert can_transition(RunStatus.RUNNING, RunStatus.ERROR) is True
    
    def test_can_transition_running_to_recovering(self):
        """Разрешенный переход RUNNING -> RECOVERING."""
        assert can_transition(RunStatus.RUNNING, RunStatus.RECOVERING) is True
    
    def test_can_transition_recovering_to_running(self):
        """Разрешенный переход RECOVERING -> RUNNING."""
        assert can_transition(RunStatus.RECOVERING, RunStatus.RUNNING) is True
    
    def test_can_transition_pending_to_running(self):
        """Недопустимый переход PENDING -> RUNNING."""
        assert can_transition(RunStatus.PENDING, RunStatus.RUNNING) is False
    
    def test_can_transition_success_to_running(self):
        """Недопустимый переход SUCCESS -> RUNNING (финальный статус)."""
        assert can_transition(RunStatus.SUCCESS, RunStatus.RUNNING) is False
    
    def test_can_transition_error_to_running(self):
        """Недопустимый переход ERROR -> RUNNING (финальный статус)."""
        assert can_transition(RunStatus.ERROR, RunStatus.RUNNING) is False
    
    def test_can_transition_cancelled_to_running(self):
        """Недопустимый переход CANCELLED -> RUNNING (финальный статус)."""
        assert can_transition(RunStatus.CANCELLED, RunStatus.RUNNING) is False


class TestValidateTransition:
    """Тесты для функции validate_transition."""
    
    def test_validate_transition_valid(self):
        """Валидация разрешенного перехода."""
        # Не должно вызывать исключение
        validate_transition(RunStatus.PENDING, RunStatus.QUEUED, "test-run-id")
    
    def test_validate_transition_invalid(self):
        """Валидация недопустимого перехода."""
        with pytest.raises(ValueError, match="Invalid status transition"):
            validate_transition(RunStatus.PENDING, RunStatus.RUNNING, "test-run-id")
    
    def test_validate_transition_final_status(self):
        """Валидация перехода из финального статуса."""
        with pytest.raises(ValueError, match="Invalid status transition"):
            validate_transition(RunStatus.SUCCESS, RunStatus.RUNNING, "test-run-id")
    
    def test_validate_transition_new_run_pending(self):
        """Валидация начального статуса для нового run'а (PENDING)."""
        # Не должно вызывать исключение
        validate_transition(None, RunStatus.PENDING, "test-run-id")
    
    def test_validate_transition_new_run_queued(self):
        """Валидация начального статуса для нового run'а (QUEUED)."""
        # Не должно вызывать исключение
        validate_transition(None, RunStatus.QUEUED, "test-run-id")
    
    def test_validate_transition_new_run_invalid(self):
        """Валидация недопустимого начального статуса."""
        with pytest.raises(ValueError, match="Invalid initial status"):
            validate_transition(None, RunStatus.RUNNING, "test-run-id")
    
    def test_validate_transition_without_run_id(self):
        """Валидация перехода без run_id."""
        # Не должно вызывать исключение
        validate_transition(RunStatus.PENDING, RunStatus.QUEUED)


class TestGetAllowedTransitions:
    """Тесты для функции get_allowed_transitions."""
    
    def test_get_allowed_transitions_pending(self):
        """Получение разрешенных переходов из PENDING."""
        transitions = get_allowed_transitions(RunStatus.PENDING)
        assert RunStatus.QUEUED in transitions
        assert RunStatus.CANCELLED in transitions
        assert len(transitions) == 2
    
    def test_get_allowed_transitions_queued(self):
        """Получение разрешенных переходов из QUEUED."""
        transitions = get_allowed_transitions(RunStatus.QUEUED)
        assert RunStatus.RUNNING in transitions
        assert RunStatus.CANCELLED in transitions
        assert len(transitions) == 2
    
    def test_get_allowed_transitions_running(self):
        """Получение разрешенных переходов из RUNNING."""
        transitions = get_allowed_transitions(RunStatus.RUNNING)
        assert RunStatus.SUCCESS in transitions
        assert RunStatus.ERROR in transitions
        assert RunStatus.RECOVERING in transitions
        assert RunStatus.CANCELLED in transitions
        assert len(transitions) == 4
    
    def test_get_allowed_transitions_recovering(self):
        """Получение разрешенных переходов из RECOVERING."""
        transitions = get_allowed_transitions(RunStatus.RECOVERING)
        assert RunStatus.RUNNING in transitions
        assert RunStatus.ERROR in transitions
        assert len(transitions) == 2
    
    def test_get_allowed_transitions_success(self):
        """Получение разрешенных переходов из SUCCESS (финальный статус)."""
        transitions = get_allowed_transitions(RunStatus.SUCCESS)
        assert len(transitions) == 0
    
    def test_get_allowed_transitions_error(self):
        """Получение разрешенных переходов из ERROR (финальный статус)."""
        transitions = get_allowed_transitions(RunStatus.ERROR)
        assert len(transitions) == 0
    
    def test_get_allowed_transitions_cancelled(self):
        """Получение разрешенных переходов из CANCELLED (финальный статус)."""
        transitions = get_allowed_transitions(RunStatus.CANCELLED)
        assert len(transitions) == 0


class TestIsFinalStatus:
    """Тесты для функции is_final_status."""
    
    def test_is_final_status_success(self):
        """SUCCESS является финальным статусом."""
        assert is_final_status(RunStatus.SUCCESS) is True
    
    def test_is_final_status_error(self):
        """ERROR является финальным статусом."""
        assert is_final_status(RunStatus.ERROR) is True
    
    def test_is_final_status_cancelled(self):
        """CANCELLED является финальным статусом."""
        assert is_final_status(RunStatus.CANCELLED) is True
    
    def test_is_final_status_pending(self):
        """PENDING не является финальным статусом."""
        assert is_final_status(RunStatus.PENDING) is False
    
    def test_is_final_status_queued(self):
        """QUEUED не является финальным статусом."""
        assert is_final_status(RunStatus.QUEUED) is False
    
    def test_is_final_status_running(self):
        """RUNNING не является финальным статусом."""
        assert is_final_status(RunStatus.RUNNING) is False
    
    def test_is_final_status_recovering(self):
        """RECOVERING не является финальным статусом."""
        assert is_final_status(RunStatus.RECOVERING) is False


class TestParseStatus:
    """Тесты для функции parse_status."""
    
    def test_parse_status_lowercase(self):
        """Парсинг статуса в нижнем регистре."""
        assert parse_status("pending") == RunStatus.PENDING
        assert parse_status("queued") == RunStatus.QUEUED
        assert parse_status("running") == RunStatus.RUNNING
        assert parse_status("success") == RunStatus.SUCCESS
        assert parse_status("error") == RunStatus.ERROR
        assert parse_status("cancelled") == RunStatus.CANCELLED
        assert parse_status("recovering") == RunStatus.RECOVERING
    
    def test_parse_status_uppercase(self):
        """Парсинг статуса в верхнем регистре."""
        assert parse_status("PENDING") == RunStatus.PENDING
        assert parse_status("QUEUED") == RunStatus.QUEUED
        assert parse_status("RUNNING") == RunStatus.RUNNING
    
    def test_parse_status_mixed_case(self):
        """Парсинг статуса в смешанном регистре."""
        assert parse_status("Pending") == RunStatus.PENDING
        assert parse_status("Running") == RunStatus.RUNNING
    
    def test_parse_status_invalid(self):
        """Парсинг невалидного статуса."""
        with pytest.raises(ValueError, match="Unknown status|Invalid status"):
            parse_status("invalid_status")
    
    def test_parse_status_empty(self):
        """Парсинг пустого статуса."""
        with pytest.raises(ValueError):
            parse_status("")
    
    def test_parse_status_none(self):
        """Парсинг None статуса."""
        with pytest.raises(ValueError):
            parse_status(None)
