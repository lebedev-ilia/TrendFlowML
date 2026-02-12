from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..auth import hash_password
from ..config import Settings
from ..deps import get_current_user, get_db
from ..models import User
from ..schemas import AdminUserCreate, AdminUserOut, AdminUserUpdate


router = APIRouter(prefix="/api/admin", tags=["admin"])
settings = Settings()


def require_admin(user: User) -> None:
    if user.role == "admin":
        return
    if user.email.lower() in settings.admin_email_set():
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")


@router.get("/users", response_model=list[AdminUserOut])
def list_users(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    require_admin(user)
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [AdminUserOut(id=u.id, email=u.email, role=u.role, created_at=u.created_at) for u in users]


@router.post("/users", response_model=AdminUserOut)
def create_user(payload: AdminUserCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    require_admin(user)
    exists = db.query(User).filter(User.email == payload.email).first()
    if exists:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already exists")
    new_user = User(email=payload.email, password_hash=hash_password(payload.password), role=payload.role)
    db.add(new_user)
    db.flush()
    return AdminUserOut(id=new_user.id, email=new_user.email, role=new_user.role, created_at=new_user.created_at)


@router.put("/users/{user_id}", response_model=AdminUserOut)
def update_user(
    user_id: str,
    payload: AdminUserUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    require_admin(user)
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if payload.email is not None:
        target.email = payload.email
    if payload.password is not None:
        target.password_hash = hash_password(payload.password)
    if payload.role is not None:
        target.role = payload.role
    db.flush()
    return AdminUserOut(id=target.id, email=target.email, role=target.role, created_at=target.created_at)


@router.delete("/users/{user_id}")
def delete_user(user_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    require_admin(user)
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    db.delete(target)
    db.flush()
    return {"status": "ok"}

