"""Проверка политики JWT secret при старте (production vs development)."""

from __future__ import annotations

import pytest

from app.config import Settings, is_weak_jwt_secret, validate_security_at_startup


def test_is_weak_jwt_secret():
    assert is_weak_jwt_secret("") is True
    assert is_weak_jwt_secret("change-me") is True
    assert is_weak_jwt_secret("CHANGE-ME") is True
    assert is_weak_jwt_secret("demo-change-me-in-production") is True
    assert is_weak_jwt_secret("openssl-generated-hex-value") is False


def test_validate_security_production_weak_secret_raises(monkeypatch):
    monkeypatch.setenv("TF_BACKEND_DEPLOYMENT_ENV", "production")
    monkeypatch.setenv("TF_BACKEND_JWT_SECRET", "change-me")
    with pytest.raises(RuntimeError, match="TF_BACKEND_JWT_SECRET"):
        validate_security_at_startup(Settings())


def test_validate_security_staging_weak_secret_raises(monkeypatch):
    monkeypatch.setenv("TF_BACKEND_DEPLOYMENT_ENV", "staging")
    monkeypatch.setenv("TF_BACKEND_JWT_SECRET", "demo-change-me-in-production")
    with pytest.raises(RuntimeError, match="TF_BACKEND_JWT_SECRET"):
        validate_security_at_startup(Settings())


def test_validate_security_development_weak_secret_no_raise(monkeypatch, caplog):
    monkeypatch.setenv("TF_BACKEND_DEPLOYMENT_ENV", "development")
    monkeypatch.setenv("TF_BACKEND_JWT_SECRET", "change-me")
    with caplog.at_level("WARNING"):
        validate_security_at_startup(Settings())
    assert any("TF_BACKEND_JWT_SECRET" in r.message for r in caplog.records)


def test_validate_security_production_strong_secret_ok(monkeypatch):
    monkeypatch.setenv("TF_BACKEND_DEPLOYMENT_ENV", "production")
    monkeypatch.setenv("TF_BACKEND_JWT_SECRET", "a" * 64)
    validate_security_at_startup(Settings())


def test_settings_deployment_env_from_env(monkeypatch):
    monkeypatch.setenv("TF_BACKEND_DEPLOYMENT_ENV", "staging")
    assert Settings().deployment_env == "staging"
