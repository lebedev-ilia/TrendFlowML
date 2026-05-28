from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SchemaDoc:
    path: str
    doc: Dict[str, Any]

    @property
    def schema_version(self) -> str:
        return str(self.doc.get("schema_version") or "")


def _is_schema_file(path: str) -> bool:
    base = os.path.basename(path)
    return base.endswith(".json") and not base.startswith("_")


def load_all_schema_docs(schemas_dir: str) -> Tuple[Dict[str, SchemaDoc], List[str]]:
    """
    Loads all `*.json` schema docs from `schemas_dir` (non-recursive).

    Returns:
      (by_schema_version, issues_as_strings)
    """
    issues: List[str] = []
    by_ver: Dict[str, SchemaDoc] = {}

    if not os.path.isdir(schemas_dir):
        issues.append(f"schemas_dir does not exist or is not a dir: {schemas_dir}")
        return by_ver, issues

    for name in sorted(os.listdir(schemas_dir)):
        path = os.path.join(schemas_dir, name)
        if not os.path.isfile(path) or not _is_schema_file(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            issues.append(f"failed to load schema json {path}: {e}")
            continue

        schema_version = str(doc.get("schema_version") or "").strip()
        if not schema_version:
            issues.append(f"schema json missing schema_version: {path}")
            continue
        if schema_version in by_ver:
            issues.append(f"duplicate schema_version={schema_version}: {by_ver[schema_version].path} and {path}")
            continue
        by_ver[schema_version] = SchemaDoc(path=path, doc=doc)

    return by_ver, issues


_CACHE: Optional[Tuple[str, Dict[str, SchemaDoc]]] = None


def get_schema_registry_cached(schemas_dir: str) -> Dict[str, SchemaDoc]:
    """
    Best-effort cached registry. Designed for runtime validators.
    """
    global _CACHE
    if _CACHE is not None and _CACHE[0] == schemas_dir:
        return _CACHE[1]

    by_ver, _issues = load_all_schema_docs(schemas_dir)
    _CACHE = (schemas_dir, by_ver)
    return by_ver


