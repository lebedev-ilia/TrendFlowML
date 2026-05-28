"""
Unit-тесты модуля auth: JWT (create_access_token, decode_token) и пароли (hash, verify).

Без БД и внешних сервисов. См. backend/docs/TESTING_PLAN.md § 3.6.1, SECURITY.md.
"""

from __future__ import annotations

import pytest

from app.auth import (
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)


pytestmark = pytest.mark.unit


class TestPasswordHashing:
    """Хеширование и проверка паролей."""

    def test_hash_password_returns_non_empty_string(self):
        """hash_password возвращает непустую строку."""
        h = hash_password("secret123")
        assert isinstance(h, str)
        assert len(h) > 0

    def test_verify_password_success(self):
        """verify_password(plain, hash(plain)) == True."""
        plain = "myPassword1"
        h = hash_password(plain)
        assert verify_password(plain, h) is True

    def test_verify_password_wrong_fails(self):
        """verify_password с неверным паролем возвращает False."""
        h = hash_password("correct")
        assert verify_password("wrong", h) is False

    def test_hash_deterministic_different_per_call(self):
        """Разные вызовы hash дают разный результат (соль)."""
        a = hash_password("same")
        b = hash_password("same")
        assert a != b
        assert verify_password("same", a) and verify_password("same", b)


class TestJWT:
    """Создание и декодирование JWT."""

    def test_create_access_token_returns_string(self):
        """create_access_token возвращает строку (JWT)."""
        token = create_access_token("user-id-123")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_decode_token_returns_subject(self):
        """decode_token возвращает sub из payload."""
        token = create_access_token("user-uuid-456")
        sub = decode_token(token)
        assert sub == "user-uuid-456"

    def test_decode_token_invalid_returns_none(self):
        """decode_token при невалидном токене возвращает None."""
        assert decode_token("invalid.jwt.here") is None
        assert decode_token("") is None

    def test_decode_token_expired_returns_none(self, monkeypatch):
        """Истёкший токен при декоде даёт исключение — мы ловим и возвращаем None."""
        # Создаём токен с истечением в прошлом через create с expires_minutes=-1
        token = create_access_token("user-1", expires_minutes=-60)
        result = decode_token(token)
        assert result is None
