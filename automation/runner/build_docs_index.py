#!/usr/bin/env python3
"""Автогенерация DOCS_INDEX.md — обходит все .md в репо и собирает список с первым заголовком.

Дополняет curated MAIN_INDEX.md (его правят агенты руками). Запуск: python build_docs_index.py
Кладёт результат в automation/DOCS_INDEX.md. Агенты смотрят его, чтобы не терять контекст.
"""
from __future__ import annotations
import re
from pathlib import Path

import config

OUT = config.AUTOMATION_DIR / "DOCS_INDEX.md"
SKIP = {".git", "node_modules", ".venv", "state", "__pycache__"}


def first_heading(p: Path) -> str:
    try:
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"#{1,3}\s+(.*)", line.strip())
            if m:
                return m.group(1).strip()[:100]
    except Exception:
        pass
    return ""


def main():
    root = config.REPO_DIR
    docs = []
    for p in sorted(root.rglob("*.md")):
        if any(part in SKIP for part in p.parts):
            continue
        docs.append((p.relative_to(root), first_heading(p)))
    groups: dict[str, list] = {}
    for rel, title in docs:
        top = rel.parts[0] if len(rel.parts) > 1 else "."
        groups.setdefault(top, []).append((rel, title))
    lines = [f"# DOCS_INDEX (автогенерация, {len(docs)} файлов)",
             "", "> Собрано build_docs_index.py. Curated-точка входа — MAIN_INDEX.md.", ""]
    for top in sorted(groups):
        lines.append(f"## {top}")
        for rel, title in groups[top]:
            lines.append(f"- [{rel}]({rel})" + (f" — {title}" if title else ""))
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Записано {OUT} ({len(docs)} файлов)")


if __name__ == "__main__":
    main()
