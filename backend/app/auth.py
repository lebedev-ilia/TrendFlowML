from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from jose import jwt
from passlib.context import CryptContext

from .config import Settings


settings = Settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_access_token(subject: str, expires_minutes: Optional[int] = None) -> str:
    if expires_minutes is None:
        expires_minutes = settings.jwt_exp_minutes
    exp = datetime.utcnow() + timedelta(minutes=expires_minutes)
    payload = {"sub": subject, "exp": exp}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return str(payload.get("sub"))
    except Exception:
        return None

