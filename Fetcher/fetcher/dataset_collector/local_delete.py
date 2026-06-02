from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fetcher.dataset_collector.worker_logging import worker_log

# Colab `drive.mount("/content/drive")` and common Desktop sync mount paths.
_DEFAULT_DRIVE_MOUNT_ROOTS = (
    "/content/drive/MyDrive",
    "/content/drive/Shareddrives",
    "/content/drive/Shared drives",
)

_drive_service = None


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def google_drive_mount_roots() -> tuple[str, ...]:
    extra = os.environ.get("DATASET_DRIVE_MOUNT_ROOTS", "").strip()
    roots = list(_DEFAULT_DRIVE_MOUNT_ROOTS)
    if extra:
        roots.extend(part.strip() for part in extra.split(os.pathsep) if part.strip())
    return tuple(roots)


def is_google_drive_path(path: Path) -> bool:
    resolved = path.resolve()
    for root in google_drive_mount_roots():
        try:
            resolved.relative_to(Path(root).resolve())
            return True
        except ValueError:
            continue
    return False


def should_permanent_delete_on_drive(
    path: Path,
    *,
    output_dir: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> bool:
    if enabled is False:
        return False
    if not is_google_drive_path(path):
        return False
    if enabled is True or _env_truthy("DATASET_DRIVE_PERMANENT_DELETE"):
        return True
    if output_dir is not None:
        return is_google_drive_path(Path(output_dir))
    return True


def _escape_drive_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _pick_drive_child(files: list[dict], *, expect_file: bool) -> Optional[dict]:
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    folders = [item for item in files if item.get("mimeType") == "application/vnd.google-apps.folder"]
    non_folders = [item for item in files if item.get("mimeType") != "application/vnd.google-apps.folder"]
    if expect_file and non_folders:
        return non_folders[0]
    if not expect_file and folders:
        return folders[0]
    return files[0]


def _drive_file_id_for_path(service, path: Path, *, my_drive_root: Path) -> Optional[str]:
    try:
        relative = path.resolve().relative_to(my_drive_root.resolve())
    except ValueError:
        return None
    parts = relative.parts
    if not parts:
        return None

    parent_id = "root"
    for index, part in enumerate(parts):
        query = (
            f"'{parent_id}' in parents and "
            f"name = '{_escape_drive_query_value(part)}' and "
            "trashed = false"
        )
        response = (
            service.files()
            .list(
                q=query,
                fields="files(id, name, mimeType)",
                pageSize=20,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files = response.get("files", [])
        chosen = _pick_drive_child(files, expect_file=index == len(parts) - 1)
        if chosen is None:
            return None
        parent_id = chosen["id"]
    return parent_id


def _my_drive_root_for_path(path: Path) -> Optional[Path]:
    resolved = path.resolve()
    for root in google_drive_mount_roots():
        root_path = Path(root).resolve()
        try:
            resolved.relative_to(root_path)
            return root_path
        except ValueError:
            continue
    return None


def _get_drive_service():
    global _drive_service
    if _drive_service is not None:
        return _drive_service

    try:
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "google-api-python-client is required for permanent Google Drive deletes"
        ) from exc

    credentials = None
    try:
        from google.colab import auth as colab_auth

        colab_auth.authenticate_user()
    except ImportError:
        pass

    try:
        import google.auth

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/drive"]
        )
    except Exception as exc:
        raise RuntimeError("Google Drive credentials are not available") from exc

    _drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    return _drive_service


def delete_drive_file_permanent(path: Path) -> bool:
    path = path.resolve()
    if not path.exists():
        return True

    my_drive_root = _my_drive_root_for_path(path)
    if my_drive_root is None:
        path.unlink()
        return True

    service = _get_drive_service()
    file_id = _drive_file_id_for_path(service, path, my_drive_root=my_drive_root)
    if not file_id:
        raise FileNotFoundError(f"Drive file id not found for {path}")

    service.files().delete(fileId=file_id, supportsAllDrives=True).execute()
    if path.exists():
        path.unlink(missing_ok=True)
    return True


def delete_local_file(
    path: Path,
    *,
    output_dir: Optional[str] = None,
    permanent_on_drive: Optional[bool] = None,
    log_channel: str = "local-delete",
) -> bool:
    """Delete a local file. On Google Drive mounts, optionally bypass Trash via Drive API."""
    path = Path(path)
    if not path.exists():
        return True

    use_permanent = should_permanent_delete_on_drive(
        path,
        output_dir=output_dir,
        enabled=permanent_on_drive,
    )
    if not use_permanent:
        path.unlink()
        return True

    try:
        delete_drive_file_permanent(path)
        worker_log(log_channel, f"permanently deleted {path.name} on Google Drive")
        return True
    except Exception as exc:
        worker_log(
            log_channel,
            f"WARN permanent Drive delete failed for {path}: {type(exc).__name__}: {exc}",
        )
        path.unlink(missing_ok=True)
        worker_log(
            log_channel,
            f"WARN fell back to unlink for {path.name}; file may remain in Drive Trash",
        )
        return False
