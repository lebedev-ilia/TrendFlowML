from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=72)


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(max_length=72)


class UserOut(BaseModel):
    id: str
    email: EmailStr
    role: str
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ProfileCreate(BaseModel):
    name: str
    description: Optional[str] = None
    is_public: bool = False
    config_json: Dict[str, Any]


class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config_json: Optional[Dict[str, Any]] = None


class ProfileOut(BaseModel):
    id: str
    user_id: Optional[str]
    name: str
    description: Optional[str]
    is_public: bool
    config_json: Dict[str, Any]
    config_hash: str
    created_at: datetime
    updated_at: datetime


class UploadInitOut(BaseModel):
    upload_id: str
    video_id: str


class VideoOut(BaseModel):
    id: str
    platform_id: str
    video_id: str
    source_type: str
    title: Optional[str]
    created_at: datetime


class RunCreate(BaseModel):
    video_id: str
    profile_id: Optional[str]


class RunOut(BaseModel):
    id: str
    video_id: str
    profile_id: Optional[str]
    status: str
    stage: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    error_code: Optional[str]
    error_message: Optional[str]


class RunLogOut(BaseModel):
    ts: datetime
    level: str
    message: str


class RunResultOut(BaseModel):
    run_id: str
    manifest: Dict[str, Any]
    artifacts: List[Dict[str, Any]]


class AdminUserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=72)
    role: str = "user"


class AdminUserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(default=None, max_length=72)
    role: Optional[str] = None


class AdminUserOut(BaseModel):
    id: str
    email: EmailStr
    role: str
    created_at: datetime

