from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "TrendFlow Backend"
    debug: bool = False

    db_dsn: str = "postgresql+psycopg://trendflow:trendflow@localhost:5432/trendflow"
    redis_url: str = "redis://localhost:6379/0"

    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_exp_minutes: int = 60 * 24 * 7
    admin_emails: str = ""

    storage_root: Optional[str] = None
    result_store_base: Optional[str] = None
    frames_dir_base: Optional[str] = None
    raw_uploads_dir: Optional[str] = None
    example_videos_dir: Optional[str] = None

    dataproc_root: Optional[str] = None
    visual_cfg_default: Optional[str] = None

    class Config:
        env_prefix = "TF_BACKEND_"
        env_file = ".env"

    def resolve_paths(self) -> "ResolvedPaths":
        repo_root = Path(__file__).resolve().parents[2]
        storage_root = Path(self.storage_root) if self.storage_root else repo_root / "storage"
        result_store_base = (
            Path(self.result_store_base)
            if self.result_store_base
            else storage_root / "result_store"
        )
        frames_dir_base = (
            Path(self.frames_dir_base) if self.frames_dir_base else storage_root / "frames_dir"
        )
        raw_uploads_dir = (
            Path(self.raw_uploads_dir) if self.raw_uploads_dir else storage_root / "raw"
        )
        example_videos_dir = (
            Path(self.example_videos_dir)
            if self.example_videos_dir
            else repo_root / "example" / "example_videos"
        )
        dataproc_root = (
            Path(self.dataproc_root)
            if self.dataproc_root
            else repo_root / "DataProcessor"
        )
        visual_cfg_default = (
            Path(self.visual_cfg_default)
            if self.visual_cfg_default
            else dataproc_root / "configs" / "visual_triton_baseline_gpu_local.yaml"
        )
        return ResolvedPaths(
            repo_root=repo_root,
            storage_root=storage_root,
            result_store_base=result_store_base,
            frames_dir_base=frames_dir_base,
            raw_uploads_dir=raw_uploads_dir,
            example_videos_dir=example_videos_dir,
            dataproc_root=dataproc_root,
            visual_cfg_default=visual_cfg_default,
        )

    def admin_email_set(self) -> set[str]:
        return {e.strip().lower() for e in self.admin_emails.split(",") if e.strip()}


class ResolvedPaths(BaseModel):
    repo_root: Path
    storage_root: Path
    result_store_base: Path
    frames_dir_base: Path
    raw_uploads_dir: Path
    example_videos_dir: Path
    dataproc_root: Path
    visual_cfg_default: Path

