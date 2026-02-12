from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..auth import create_access_token, hash_password, verify_password
from ..deps import get_current_user, get_db
from ..models import User
from ..schemas import TokenOut, UserCreate, UserLogin, UserOut


router = APIRouter(prefix="/api/auth", tags=["auth"])

# TODO(auth):
# - OAuth providers (Google/GitHub)
# - email verification
# - reset password
# - 2FA

@router.post("/register", response_model=UserOut)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already exists")
    user = User(email=payload.email, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenOut)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(user.id)
    return TokenOut(access_token=token)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user

