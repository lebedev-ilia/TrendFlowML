"""
Утилиты для профилей анализа: config_hash, загрузка публичных профилей из YAML.

См. backend/docs/PROFILES.md, TESTING_PLAN.md § 3.7.3.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml
from sqlalchemy.orm import Session

from ..models import AnalysisProfile


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Детерминированный config_hash: SHA-256 от JSON с сортировкой ключей.
    Совпадает с логикой, описанной в PROFILES.md.
    """
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seed_public_profiles(db: Session, profiles_dir: Path) -> int:
    """
    Читает YAML-файлы из profiles_dir (*.yaml), создаёт публичные профили
    (AnalysisProfile, is_public=True), если профиля с таким именем ещё нет.

    Имя профиля берётся из имени файла (stem). config_hash вычисляется из содержимого.
    Возвращает количество созданных записей.
    """
    if not profiles_dir.exists() or not profiles_dir.is_dir():
        return 0
    created = 0
    for path in sorted(profiles_dir.glob("*.yaml")):
        try:
            raw = path.read_text(encoding="utf-8")
            config = yaml.safe_load(raw)
        except Exception:
            continue
        if not isinstance(config, dict):
            continue
        name = path.stem
        existing = (
            db.query(AnalysisProfile)
            .filter(AnalysisProfile.name == name, AnalysisProfile.is_public.is_(True))
            .first()
        )
        if existing:
            continue
        config_hash = compute_config_hash(config)
        profile = AnalysisProfile(
            name=name,
            description=config.get("description") or f"Public profile: {name}",
            is_public=True,
            user_id=None,
            config_json=config,
            config_hash=config_hash,
        )
        db.add(profile)
        created += 1
    return created
