"""
Worker-модули Fetcher.

Здесь живут:
- metadata worker;
- video download worker;
- comments worker;
- artifact builder.
"""

from .metadata import run_metadata_worker
from .video import run_video_download_worker
from .artifacts import run_artifact_builder
from .comments import run_comments_worker

__all__ = [
    "run_metadata_worker",
    "run_video_download_worker",
    "run_comments_worker",
    "run_artifact_builder",
]



