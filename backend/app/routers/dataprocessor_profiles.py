from __future__ import annotations

import re
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..config import Settings
from ..deps import get_current_user
from ..models import User


router = APIRouter(prefix="/api/dataprocessor", tags=["dataprocessor-profiles"])


class ProfileListOut(BaseModel):
    profiles: List[str]


class ProfileCreateIn(BaseModel):
    # Optional human name; if omitted server will auto-generate.
    name: Optional[str] = None
    # Optional base profile to copy from (defaults to demo.yaml if exists).
    base_profile: Optional[str] = None


class ProfileCreateOut(BaseModel):
    profile: str
    visual_cfg_path: str


class ProfileGetOut(BaseModel):
    profile: str
    profile_yaml: str
    profile_obj: Dict[str, Any]
    visual_cfg_path: str
    visual_cfg_yaml: str
    visual_cfg_obj: Dict[str, Any]


class ProfileSaveIn(BaseModel):
    # UI sends a flat map like {"core_clip.runtime": "triton", ...}
    param_values: Dict[str, Any]


def _sanitize_profile_name(name: str) -> str:
    """
    Accepts only safe filenames (no paths). Returns filename with .yaml suffix.
    """
    name = name.strip()
    if not name:
        raise ValueError("empty")
    # Allow: letters, digits, dash, underscore, dot
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ValueError("bad_chars")
    if "/" in name or "\\" in name:
        raise ValueError("path")
    if not (name.endswith(".yaml") or name.endswith(".yml")):
        name = f"{name}.yaml"
    return name


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _resolve_cfg_path(repo_root: Path, dataproc_root: Path, cfg_path: str) -> Path:
    """
    cfg_path in profiles is typically relative like "configs/visual_....yaml".
    We resolve it relative to repo_root first (matches existing demo.yaml).
    """
    p = Path(cfg_path)
    if p.is_absolute():
        return p
    # Prefer repo_root (demo.yaml uses repo-level configs/)
    candidate = repo_root / p
    if candidate.exists():
        return candidate
    # Fallback: dataproc_root
    candidate2 = dataproc_root / p
    return candidate2


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid YAML: {path.name}: {e}")


def _dump_yaml(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=120,
    )


@router.get("/profiles", response_model=ProfileListOut)
def list_profiles(user: User = Depends(get_current_user)) -> ProfileListOut:
    _ = user
    paths = Settings().resolve_paths()
    profiles_dir = paths.dataproc_root / "profiles"
    if not profiles_dir.exists():
        return ProfileListOut(profiles=[])
    names = sorted([p.name for p in profiles_dir.glob("*.y*ml") if p.is_file()])
    return ProfileListOut(profiles=names)


@router.post("/profiles", response_model=ProfileCreateOut)
def create_profile(payload: ProfileCreateIn, user: User = Depends(get_current_user)) -> ProfileCreateOut:
    _ = user
    paths = Settings().resolve_paths()
    profiles_dir = paths.dataproc_root / "profiles"

    # Choose filename
    base = (payload.name or "config").strip() if payload.name else "config"
    # sanitize base (no extension yet)
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-") or "config"
    for suffix in ["", f"_{int(time.time())}"] + [f"_{i}" for i in range(1, 1000)]:
        candidate = f"{base}{suffix}.yaml"
        if not (profiles_dir / candidate).exists():
            profile_name = candidate
            break
    else:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Could not allocate filename")

    # Determine base profile (optional copy)
    base_profile = payload.base_profile or ("demo.yaml" if (profiles_dir / "demo.yaml").exists() else None)
    profile_obj: Dict[str, Any]
    if base_profile and (profiles_dir / base_profile).exists():
        profile_obj = _load_yaml_file(profiles_dir / base_profile)
    else:
        profile_obj = {
            "processors": {"audio": {"enabled": False, "required": False}, "text": {"enabled": False, "required": False}},
            "visual": {},
        }

    # Visual cfg path (repo-root configs/)
    stem = Path(profile_name).stem
    visual_cfg_rel = f"configs/visual_{stem}.yaml"
    profile_obj["visual"] = profile_obj.get("visual") if isinstance(profile_obj.get("visual"), dict) else {}
    profile_obj["visual"]["cfg_path"] = visual_cfg_rel

    # Create visual cfg file (copy baseline if exists)
    baseline = paths.repo_root / "configs" / "visual_triton_baseline_gpu_local.yaml"
    visual_cfg_abs = _resolve_cfg_path(paths.repo_root, paths.dataproc_root, visual_cfg_rel)
    if baseline.exists():
        visual_cfg_abs.parent.mkdir(parents=True, exist_ok=True)
        if not visual_cfg_abs.exists():
            visual_cfg_abs.write_text(baseline.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        # minimal placeholder
        if not visual_cfg_abs.exists():
            _atomic_write_text(visual_cfg_abs, _dump_yaml({"global": {}, "core_providers": {}, "modules": {}}))

    # Write profile YAML
    profile_path = profiles_dir / profile_name
    _atomic_write_text(profile_path, _dump_yaml(profile_obj))

    return ProfileCreateOut(profile=profile_name, visual_cfg_path=visual_cfg_rel)


@router.get("/profiles/{profile}", response_model=ProfileGetOut)
def get_profile(profile: str, user: User = Depends(get_current_user)) -> ProfileGetOut:
    _ = user
    paths = Settings().resolve_paths()
    profiles_dir = paths.dataproc_root / "profiles"

    try:
        profile_name = _sanitize_profile_name(profile)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid profile name")

    profile_path = profiles_dir / profile_name
    if not profile_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")

    profile_obj = _load_yaml_file(profile_path)
    vis = profile_obj.get("visual") if isinstance(profile_obj.get("visual"), dict) else {}
    cfg_path = str(vis.get("cfg_path") or "").strip()
    if not cfg_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Profile visual.cfg_path is missing")

    cfg_abs = _resolve_cfg_path(paths.repo_root, paths.dataproc_root, cfg_path)
    if not cfg_abs.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Visual cfg not found: {cfg_path}")

    visual_obj = _load_yaml_file(cfg_abs)

    return ProfileGetOut(
        profile=profile_name,
        profile_yaml=profile_path.read_text(encoding="utf-8"),
        profile_obj=profile_obj,
        visual_cfg_path=cfg_path,
        visual_cfg_yaml=cfg_abs.read_text(encoding="utf-8"),
        visual_cfg_obj=visual_obj,
    )


@router.put("/profiles/{profile}")
def save_profile(profile: str, payload: ProfileSaveIn, user: User = Depends(get_current_user)):
    _ = user
    paths = Settings().resolve_paths()
    profiles_dir = paths.dataproc_root / "profiles"

    try:
        profile_name = _sanitize_profile_name(profile)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid profile name")

    profile_path = profiles_dir / profile_name
    if not profile_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")

    profile_obj = _load_yaml_file(profile_path)
    vis = profile_obj.get("visual") if isinstance(profile_obj.get("visual"), dict) else {}
    cfg_path = str(vis.get("cfg_path") or "").strip()
    if not cfg_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Profile visual.cfg_path is missing")

    cfg_abs = _resolve_cfg_path(paths.repo_root, paths.dataproc_root, cfg_path)
    if not cfg_abs.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Visual cfg not found: {cfg_path}")

    visual_obj = _load_yaml_file(cfg_abs)

    # Apply updates: "component.key" => visual_obj[component][key]
    for dotted, value in payload.param_values.items():
        if not isinstance(dotted, str) or "." not in dotted:
            continue
        component_id, param_key = dotted.split(".", 1)
        if not component_id or not param_key:
            continue
        if not isinstance(component_id, str) or not isinstance(param_key, str):
            continue

        comp = visual_obj.get(component_id)
        if not isinstance(comp, dict):
            comp = {}
            visual_obj[component_id] = comp

        # If YAML currently uses dash-key variant, keep it.
        dash_key = param_key.replace("_", "-")
        if dash_key in comp and param_key not in comp:
            comp[dash_key] = value
        else:
            comp[param_key] = value

    _atomic_write_text(cfg_abs, _dump_yaml(visual_obj))
    return {"status": "ok", "profile": profile_name, "visual_cfg_path": cfg_path}


@router.delete("/profiles/{profile}")
def delete_profile_file(profile: str, user: User = Depends(get_current_user)):
    _ = user
    paths = Settings().resolve_paths()
    profiles_dir = paths.dataproc_root / "profiles"

    try:
        profile_name = _sanitize_profile_name(profile)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid profile name")

    profile_path = profiles_dir / profile_name
    if not profile_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")

    # Best-effort: also delete the referenced visual cfg file if it exists
    cfg_path = None
    try:
        profile_obj = _load_yaml_file(profile_path)
        vis = profile_obj.get("visual") if isinstance(profile_obj.get("visual"), dict) else {}
        cfg_path = str(vis.get("cfg_path") or "").strip() or None
    except Exception:
        cfg_path = None

    with suppress(Exception):
        profile_path.unlink()

    if cfg_path:
        cfg_abs = _resolve_cfg_path(paths.repo_root, paths.dataproc_root, cfg_path)
        with suppress(Exception):
            if cfg_abs.exists():
                cfg_abs.unlink()

    return {"status": "ok", "profile": profile_name, "visual_cfg_path": cfg_path}


