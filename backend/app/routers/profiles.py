from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List

import yaml
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..config import Settings
from ..deps import get_current_user, get_db
from ..models import AnalysisProfile, User
from ..schemas import ProfileCreate, ProfileOut, ProfileUpdate


router = APIRouter(prefix="/api", tags=["profiles"])


def _hash_config(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _normalize_config(config: dict) -> dict:
    settings = Settings()
    paths = settings.resolve_paths()
    if "visual" not in config:
        config["visual"] = {"cfg_path": str(paths.visual_cfg_default)}
    if isinstance(config.get("visual"), dict) and not config["visual"].get("cfg_path"):
        config["visual"]["cfg_path"] = str(paths.visual_cfg_default)
    if "processors" not in config:
        config["processors"] = {"audio": {"enabled": False, "required": False}, "text": {"enabled": False, "required": False}}
    return config


@router.get("/profiles", response_model=List[ProfileOut])
def list_public_profiles(db: Session = Depends(get_db)):
    profiles = db.query(AnalysisProfile).filter(AnalysisProfile.is_public.is_(True)).order_by(AnalysisProfile.created_at).all()
    return profiles


@router.get("/my/profiles", response_model=List[ProfileOut])
def list_my_profiles(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    profiles = db.query(AnalysisProfile).filter(AnalysisProfile.user_id == user.id).order_by(AnalysisProfile.created_at.desc()).all()
    return profiles


@router.post("/my/profiles", response_model=ProfileOut)
def create_profile(payload: ProfileCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    config = _normalize_config(dict(payload.config_json))
    config_hash = _hash_config(config)
    profile = AnalysisProfile(
        user_id=user.id,
        name=payload.name,
        description=payload.description,
        is_public=payload.is_public,
        config_json=config,
        config_hash=config_hash,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(profile)
    db.flush()
    return profile


@router.put("/my/profiles/{profile_id}", response_model=ProfileOut)
def update_profile(
    profile_id: str,
    payload: ProfileUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    profile = db.query(AnalysisProfile).filter(AnalysisProfile.id == profile_id, AnalysisProfile.user_id == user.id).first()
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    if payload.name is not None:
        profile.name = payload.name
    if payload.description is not None:
        profile.description = payload.description
    if payload.config_json is not None:
        config = _normalize_config(dict(payload.config_json))
        profile.config_json = config
        profile.config_hash = _hash_config(config)
    profile.updated_at = datetime.utcnow()
    db.flush()
    return profile


@router.delete("/my/profiles/{profile_id}")
def delete_profile(profile_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    profile = db.query(AnalysisProfile).filter(AnalysisProfile.id == profile_id, AnalysisProfile.user_id == user.id).first()
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    db.delete(profile)
    db.flush()
    return {"status": "ok"}


def seed_public_profiles(db: Session) -> None:
    settings = Settings()
    paths = settings.resolve_paths()
    profiles_dir = paths.dataproc_root / "profiles"
    if not profiles_dir.exists():
        return
    existing = {p.name for p in db.query(AnalysisProfile).filter(AnalysisProfile.is_public.is_(True)).all()}
    for yaml_path in profiles_dir.glob("*.yaml"):
        name = yaml_path.stem
        if name in existing:
            continue
        try:
            content = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        config = _normalize_config(content)
        profile = AnalysisProfile(
            user_id=None,
            name=name,
            description=f"Seeded from {yaml_path.name}",
            is_public=True,
            config_json=config,
            config_hash=_hash_config(config),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(profile)
    db.flush()

