#!/usr/bin/env python3
"""
Preflight validator: semantic bases / galleries / strict requirements.

Goal:
- Fail-fast BEFORE running the pipeline if required offline bases/galleries/models are missing.
- Read VisualProcessor/config.yaml, detect enabled core providers, validate their required inputs.

This does NOT check result_store artifacts (those are produced at runtime).
It validates:
- db package dirs exist
- required files exist
- ids are consistent with gallery_index.json or contiguous conventions
- required model specs in config are present (non-null)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Issue:
    level: str  # "ERROR" | "WARN"
    component: str
    message: str


def _read_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to read config.yaml. Install it or run inside the project venv."
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"config must be a dict: {path}")
    return obj


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"json must be a dict: {path}")
    return obj


def _read_jsonl_ids(path: str, *, id_key: str = "id") -> List[int]:
    ids: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict) or id_key not in obj:
                raise RuntimeError(f"invalid jsonl row at {path}:{ln} (missing '{id_key}')")
            ids.append(int(obj[id_key]))
    if not ids:
        raise RuntimeError(f"empty jsonl: {path}")
    return ids


def _is_contiguous_0_to_n_minus_1(ids: List[int]) -> bool:
    if not ids:
        return False
    s = sorted(ids)
    return s[0] == 0 and s == list(range(0, len(s)))


def _validate_gallery_alignment(
    *,
    component: str,
    pkg_dir: str,
    ids: List[int],
    gallery_index_json: str,
    issues: List[Issue],
) -> None:
    idx_path = os.path.join(pkg_dir, gallery_index_json)
    if os.path.isfile(idx_path):
        try:
            idx = _read_json(idx_path)
        except Exception as e:
            issues.append(Issue("ERROR", component, f"invalid {gallery_index_json}: {e}"))
            return
        # require all ids present
        missing = []
        for i in ids:
            if str(int(i)) not in idx and int(i) not in idx:
                missing.append(int(i))
        if missing:
            issues.append(Issue("ERROR", component, f"{gallery_index_json} missing ids (first 10): {missing[:10]}"))
        return

    # No index file: require contiguous 0..A-1 convention
    if not _is_contiguous_0_to_n_minus_1(ids):
        issues.append(
            Issue(
                "ERROR",
                component,
                f"{gallery_index_json} not found and ids are not contiguous 0..A-1. "
                f"Provide {gallery_index_json} or reassign ids contiguously.",
            )
        )


def _resolve_repo_root_from_cfg(cfg_path: str) -> str:
    # config.yaml lives in <root>/VisualProcessor/config.yaml in this repo layout.
    p = Path(os.path.abspath(cfg_path))
    # allow either VisualProcessor/config.yaml or any other path; best-effort
    for parent in [p.parent] + list(p.parents):
        if (parent / "VisualProcessor").is_dir() and (parent / "dp_models").is_dir():
            return str(parent)
    # fallback: cwd
    return os.getcwd()


def _resolve_default_pkg(repo_root: str, domain: str, version: str = "v1") -> str:
    return os.path.join(repo_root, "dp_models", "bundled_models", "semantics", domain, version)


def main() -> int:
    ap = argparse.ArgumentParser("check_semantic_bases")
    ap.add_argument("--cfg-path", required=True, help="Path to VisualProcessor/config.yaml")
    ap.add_argument("--strict", action="store_true", help="Treat WARN as ERROR")
    args = ap.parse_args()

    cfg_path = os.path.abspath(str(args.cfg_path))
    cfg = _read_yaml(cfg_path)
    repo_root = _resolve_repo_root_from_cfg(cfg_path)

    core_providers = cfg.get("core_providers") or {}
    if not isinstance(core_providers, dict):
        raise RuntimeError("config.core_providers must be a dict")

    issues: List[Issue] = []

    def is_enabled(name: str) -> bool:
        v = core_providers.get(name)
        return bool(v is True or v == "true" or v == "True" or v == 1)

    # core_brand_semantics
    if is_enabled("core_brand_semantics"):
        block = cfg.get("core_brand_semantics") or {}
        if not isinstance(block, dict):
            block = {}
        db_dir = str(block.get("brand_db_dir") or _resolve_default_pkg(repo_root, "brands"))
        pkg = os.path.abspath(os.path.join(repo_root, db_dir) if not os.path.isabs(db_dir) else db_dir)
        if not os.path.isdir(pkg):
            issues.append(Issue("ERROR", "core_brand_semantics", f"brand_db_dir not found: {pkg}"))
        else:
            for req in ("manifest.json", "brands.jsonl", "gallery_embeddings.npy"):
                if not os.path.isfile(os.path.join(pkg, req)):
                    issues.append(Issue("ERROR", "core_brand_semantics", f"missing required file: {pkg}/{req}"))
            try:
                ids = _read_jsonl_ids(os.path.join(pkg, "brands.jsonl"))
                _validate_gallery_alignment(
                    component="core_brand_semantics",
                    pkg_dir=pkg,
                    ids=ids,
                    gallery_index_json="gallery_index.json",
                    issues=issues,
                )
            except Exception as e:
                issues.append(Issue("ERROR", "core_brand_semantics", f"failed to parse brands.jsonl: {e}"))

    # core_place_semantics
    if is_enabled("core_place_semantics"):
        block = cfg.get("core_place_semantics") or {}
        if not isinstance(block, dict):
            block = {}
        db_dir = str(block.get("places_db_dir") or _resolve_default_pkg(repo_root, "places"))
        pkg = os.path.abspath(os.path.join(repo_root, db_dir) if not os.path.isabs(db_dir) else db_dir)
        if not os.path.isdir(pkg):
            issues.append(Issue("ERROR", "core_place_semantics", f"places_db_dir not found: {pkg}"))
        else:
            for req in ("manifest.json", "places.jsonl", "gallery_embeddings.npy"):
                if not os.path.isfile(os.path.join(pkg, req)):
                    issues.append(Issue("ERROR", "core_place_semantics", f"missing required file: {pkg}/{req}"))
            try:
                ids = _read_jsonl_ids(os.path.join(pkg, "places.jsonl"))
                _validate_gallery_alignment(
                    component="core_place_semantics",
                    pkg_dir=pkg,
                    ids=ids,
                    gallery_index_json="gallery_index.json",
                    issues=issues,
                )
            except Exception as e:
                issues.append(Issue("ERROR", "core_place_semantics", f"failed to parse places.jsonl: {e}"))

    # core_car_semantics
    if is_enabled("core_car_semantics"):
        block = cfg.get("core_car_semantics") or {}
        if not isinstance(block, dict):
            block = {}
        db_dir = str(block.get("cars_db_dir") or _resolve_default_pkg(repo_root, "cars"))
        pkg = os.path.abspath(os.path.join(repo_root, db_dir) if not os.path.isabs(db_dir) else db_dir)
        if not os.path.isdir(pkg):
            issues.append(Issue("ERROR", "core_car_semantics", f"cars_db_dir not found: {pkg}"))
        else:
            for req in ("manifest.json", "makes.jsonl", "models.jsonl", "taxonomy.json"):
                if not os.path.isfile(os.path.join(pkg, req)):
                    issues.append(Issue("ERROR", "core_car_semantics", f"missing required file: {pkg}/{req}"))
            # make gallery is required in strict v1
            if not os.path.isfile(os.path.join(pkg, "make_gallery_embeddings.npy")):
                issues.append(Issue("ERROR", "core_car_semantics", f"missing required file: {pkg}/make_gallery_embeddings.npy"))
            try:
                make_ids = _read_jsonl_ids(os.path.join(pkg, "makes.jsonl"))
                _validate_gallery_alignment(
                    component="core_car_semantics",
                    pkg_dir=pkg,
                    ids=make_ids,
                    gallery_index_json="make_gallery_index.json",
                    issues=issues,
                )
            except Exception as e:
                issues.append(Issue("ERROR", "core_car_semantics", f"failed to parse makes.jsonl: {e}"))

    # core_face_identity
    if is_enabled("core_face_identity"):
        block = cfg.get("core_face_identity") or {}
        if not isinstance(block, dict):
            block = {}
        db_dir = str(block.get("celebs_db_dir") or _resolve_default_pkg(repo_root, "celebs"))
        pkg = os.path.abspath(os.path.join(repo_root, db_dir) if not os.path.isabs(db_dir) else db_dir)
        face_spec = block.get("face_embed_model_spec")
        if not face_spec:
            issues.append(Issue("ERROR", "core_face_identity", "face_embed_model_spec is not set (required)"))
        if not os.path.isdir(pkg):
            issues.append(Issue("ERROR", "core_face_identity", f"celebs_db_dir not found: {pkg}"))
        else:
            for req in ("manifest.json", "celebs.jsonl", "gallery_embeddings.npy"):
                if not os.path.isfile(os.path.join(pkg, req)):
                    issues.append(Issue("ERROR", "core_face_identity", f"missing required file: {pkg}/{req}"))
            try:
                ids = _read_jsonl_ids(os.path.join(pkg, "celebs.jsonl"))
                _validate_gallery_alignment(
                    component="core_face_identity",
                    pkg_dir=pkg,
                    ids=ids,
                    gallery_index_json="gallery_index.json",
                    issues=issues,
                )
            except Exception as e:
                issues.append(Issue("ERROR", "core_face_identity", f"failed to parse celebs.jsonl: {e}"))

    # Print report
    if not issues:
        print("[ok] semantic bases preflight: no issues found")
        return 0

    strict = bool(args.strict)
    exit_code = 0
    for it in issues:
        lvl = it.level
        if strict and lvl == "WARN":
            lvl = "ERROR"
        print(f"[{lvl}] {it.component}: {it.message}")
        if lvl == "ERROR":
            exit_code = 2
    if exit_code == 0:
        print("[ok] only warnings")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


