from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from .auth import decode_token
from .config import Settings
from .db import SessionLocal
from .dbv2.models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def get_settings() -> Settings:
    return Settings()


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    user_id = decode_token(token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def require_admin_user(
    user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> User:
    """
    Зависимость для эндпоинтов, доступных только глобальным админам.
    Админы задаются через TF_BACKEND_ADMIN_EMAILS (список email через запятую).
    Если email пользователя не в списке — 403 Forbidden.
    """
    admin_emails = settings.admin_email_set()
    if not admin_emails:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access not configured",
        )
    if (user.email or "").strip().lower() not in admin_emails:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user

