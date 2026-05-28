"""Unit тесты для state machine Fetcher."""

import pytest
from fetcher.state_machine import (
    RUN_STATUS_PENDING,
    RUN_STATUS_NORMALIZING_SOURCE,
    RUN_STATUS_CHECKING_CACHE,
    RUN_STATUS_FETCHING_METADATA,
    RUN_STATUS_FINALIZING,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    can_transition,
    validate_transition,
    get_allowed_transitions,
    is_final_status,
)


@pytest.mark.unit
class TestStateMachine:
    """Тесты для state machine."""

    def test_can_transition_valid(self):
        """Тест разрешённых переходов."""
        assert can_transition(RUN_STATUS_PENDING, RUN_STATUS_NORMALIZING_SOURCE) is True
        assert can_transition(RUN_STATUS_NORMALIZING_SOURCE, RUN_STATUS_CHECKING_CACHE) is True
        assert can_transition(RUN_STATUS_FINALIZING, RUN_STATUS_COMPLETED) is True

    def test_can_transition_invalid(self):
        """Тест недопустимых переходов."""
        assert can_transition(RUN_STATUS_COMPLETED, RUN_STATUS_PENDING) is False
        assert can_transition(RUN_STATUS_FINALIZING, RUN_STATUS_PENDING) is False

    def test_validate_transition_valid(self):
        """Тест валидации разрешённых переходов."""
        # Не должно вызывать исключение
        validate_transition(RUN_STATUS_PENDING, RUN_STATUS_NORMALIZING_SOURCE)

    def test_validate_transition_invalid(self):
        """Тест валидации недопустимых переходов."""
        with pytest.raises(ValueError):
            validate_transition(RUN_STATUS_COMPLETED, RUN_STATUS_PENDING)

    def test_get_allowed_transitions(self):
        """Тест получения разрешённых переходов."""
        allowed = get_allowed_transitions(RUN_STATUS_PENDING)
        assert RUN_STATUS_NORMALIZING_SOURCE in allowed
        assert RUN_STATUS_FAILED in allowed

    def test_is_final_status(self):
        """Тест проверки финальных статусов."""
        assert is_final_status(RUN_STATUS_COMPLETED) is True
        assert is_final_status(RUN_STATUS_FAILED) is True
        assert is_final_status(RUN_STATUS_PENDING) is False
        assert is_final_status(RUN_STATUS_FETCHING_METADATA) is False

