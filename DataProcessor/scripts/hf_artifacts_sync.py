#!/usr/bin/env python3
"""
Bidirectional artifacts sync with Hugging Face Hub.

This script syncs a predefined set of files/directories between local repo and
an HF dataset/model repository using a JSON manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: huggingface_hub.\n"
        "Install with: pip install huggingface_hub"
    ) from exc


@dataclass(frozen=True)
class SyncEntry:
    local: str
    remote: str
    kind: str
    required: bool


def _to_posix(path: str) -> str:
    return str(PurePosixPath(path))


def _safe_remote(remote: str) -> str:
    normalized = _to_posix(remote).strip("/")
    if not normalized:
        raise ValueError("Remote path cannot be empty")
    if normalized.startswith(".."):
        raise ValueError(f"Unsafe remote path: {remote}")
    return normalized


def _safe_local(repo_root: Path, local: str) -> Path:
    candidate = (repo_root / local).resolve()
    root = repo_root.resolve()
    if root not in [candidate, *candidate.parents]:
        raise ValueError(f"Local path escapes repo root: {local}")
    return candidate


def _load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "entries" not in payload or not isinstance(payload["entries"], list):
        raise ValueError("Manifest must contain an 'entries' list")
    return payload


def _parse_entries(payload: dict) -> list[SyncEntry]:
    parsed: list[SyncEntry] = []
    for raw in payload["entries"]:
        if not isinstance(raw, dict):
            raise ValueError("Every manifest entry must be an object")
        local = raw.get("local")
        remote = raw.get("remote", local)
        kind = raw.get("kind", "auto")
        required = bool(raw.get("required", False))
        if not local or not isinstance(local, str):
            raise ValueError("Entry.local must be a non-empty string")
        if not remote or not isinstance(remote, str):
            raise ValueError(f"Entry.remote must be string for local={local}")
        if kind not in {"auto", "file", "dir"}:
            raise ValueError(f"Entry.kind must be auto/file/dir for {local}")
        parsed.append(
            SyncEntry(
                local=local,
                remote=_safe_remote(remote),
                kind=kind,
                required=required,
            )
        )
    return parsed


def _resolve_kind(entry: SyncEntry, local_path: Path, default_for_missing: str = "dir") -> str:
    if entry.kind in {"file", "dir"}:
        return entry.kind
    if local_path.exists():
        return "dir" if local_path.is_dir() else "file"
    return default_for_missing


def _iter_prefixed(repo_files: Iterable[str], prefix: str) -> list[str]:
    base = prefix.rstrip("/")
    folder_prefix = f"{base}/"
    return [p for p in repo_files if p == base or p.startswith(folder_prefix)]


def cmd_upload(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    payload = _load_manifest(Path(args.manifest))
    entries = _parse_entries(payload)
    repo_id = args.repo_id or payload.get("repo_id")
    repo_type = args.repo_type or payload.get("repo_type", "dataset")
    revision = args.revision or payload.get("revision", "main")
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    if not repo_id:
        raise ValueError("repo_id is missing. Pass --repo-id or set it in manifest.")

    api = HfApi(token=token)

    if args.create_repo:
        if not args.dry_run:
            api.create_repo(repo_id=repo_id, repo_type=repo_type, private=args.private, exist_ok=True)
        print(f"[upload] ensured repo exists: {repo_type}/{repo_id}")

    for entry in entries:
        local_path = _safe_local(repo_root, entry.local)
        kind = _resolve_kind(entry, local_path, default_for_missing="dir")

        if not local_path.exists():
            if entry.required:
                raise FileNotFoundError(f"Required path is missing: {entry.local}")
            print(f"[upload] skip missing optional: {entry.local}")
            continue

        if kind == "dir" and not local_path.is_dir():
            raise ValueError(f"Entry declares dir, but path is file: {entry.local}")
        if kind == "file" and local_path.is_dir():
            raise ValueError(f"Entry declares file, but path is dir: {entry.local}")

        if args.dry_run:
            print(f"[dry-run][upload] {kind}: {entry.local} -> {entry.remote}")
            continue

        if kind == "dir":
            api.upload_folder(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                folder_path=str(local_path),
                path_in_repo=entry.remote,
                commit_message=f"Upload artifacts folder: {entry.local}",
            )
            print(f"[upload] dir done: {entry.local} -> {entry.remote}")
        else:
            api.upload_file(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                path_or_fileobj=str(local_path),
                path_in_repo=entry.remote,
                commit_message=f"Upload artifact file: {entry.local}",
            )
            print(f"[upload] file done: {entry.local} -> {entry.remote}")

    return 0


def cmd_download(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    payload = _load_manifest(Path(args.manifest))
    entries = _parse_entries(payload)
    repo_id = args.repo_id or payload.get("repo_id")
    repo_type = args.repo_type or payload.get("repo_type", "dataset")
    revision = args.revision or payload.get("revision", "main")
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    if not repo_id:
        raise ValueError("repo_id is missing. Pass --repo-id or set it in manifest.")

    api = HfApi(token=token)
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision)

    for entry in entries:
        local_path = _safe_local(repo_root, entry.local)
        remote_prefix = entry.remote
        matched = _iter_prefixed(repo_files, remote_prefix)
        kind = entry.kind
        if kind == "auto":
            kind = "file" if remote_prefix in repo_files else "dir"

        if not matched:
            if entry.required:
                raise FileNotFoundError(f"Required remote path is missing: {remote_prefix}")
            print(f"[download] skip missing optional: {remote_prefix}")
            continue

        if args.dry_run:
            print(f"[dry-run][download] {kind}: {remote_prefix} -> {entry.local} ({len(matched)} files)")
            continue

        if kind == "file":
            local_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                filename=remote_prefix,
                token=token,
                local_dir=str(local_path.parent),
            )
            print(f"[download] file done: {remote_prefix} -> {entry.local}")
            continue

        local_path.mkdir(parents=True, exist_ok=True)
        base = remote_prefix.rstrip("/")
        for remote_file in matched:
            rel = remote_file[len(base) :].lstrip("/")
            dest = local_path / rel if rel else local_path / Path(remote_file).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                filename=remote_file,
                token=token,
                local_dir=str(dest.parent),
            )
        print(f"[download] dir done: {remote_prefix} -> {entry.local} ({len(matched)} files)")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync local artifacts with Hugging Face Hub.")
    parser.add_argument("command", choices=["upload", "download"], help="Sync direction")
    parser.add_argument(
        "--manifest",
        default="configs/hf_artifacts_manifest.json",
        help="Path to JSON manifest with sync entries",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root for resolving local paths",
    )
    parser.add_argument("--repo-id", help="HF repo id (overrides manifest)")
    parser.add_argument("--repo-type", choices=["dataset", "model", "space"], help="HF repo type")
    parser.add_argument("--revision", help="HF branch/revision")
    parser.add_argument("--token", help="HF token (or use HF_TOKEN env)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without transfer")
    parser.add_argument("--create-repo", action="store_true", help="Create HF repo before upload")
    parser.add_argument("--private", action="store_true", help="Create HF repo as private")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "upload":
        return cmd_upload(args)
    return cmd_download(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
