#!/usr/bin/env python3
"""Wire Obsidian vault markdown files into a connected navigation graph."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

VAULT = Path(__file__).resolve().parent.parent
NAV_MARKER = "## Навигация"
SKIP_DIRS = {".obsidian", ".git", ".pytest_cache", "scripts"}

LINK_RE = re.compile(
    r"\[\[[^\]]+\]\]|"
    r"\[[^\]]*\]\(([^)]+)\)"
)
WIKI_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


def iter_md_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(VAULT.rglob("*.md")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def rel(from_file: Path, to_file: Path) -> str:
    import os

    return os.path.relpath(to_file, from_file.parent).replace("\\", "/")


def normalize_target(raw: str, source: Path) -> Path | None:
    raw = raw.strip()
    if not raw or raw.startswith(("http://", "https://", "mailto:", "#")):
        return None
    if raw.endswith((".py", ".yaml", ".yml", ".json", ".jsonl", ".sh")):
        return None
    if raw.endswith("/"):
        return None

    target = (source.parent / raw).resolve()
    if target.is_dir():
        for name in ("README.md", "index.md", "INDEX.md", "MAIN_INDEX.md"):
            candidate = target / name
            if candidate.exists():
                target = candidate
                break
        else:
            return None

    if not target.suffix:
        md = target.with_suffix(".md")
        if md.exists():
            target = md
        elif (target / "README.md").exists():
            target = target / "README.md"
        else:
            return None

    try:
        target.relative_to(VAULT.resolve())
    except ValueError:
        return None
    if not target.exists() or target.suffix.lower() != ".md":
        return None
    return target


def extract_links(source: Path, text: str) -> set[Path]:
    links: set[Path] = set()
    for match in WIKI_RE.finditer(text):
        target = normalize_target(match.group(1), source)
        if target:
            links.add(target.resolve())
    for match in LINK_RE.finditer(text):
        raw = match.group(1)
        if not raw:
            continue
        target = normalize_target(raw, source)
        if target:
            links.add(target.resolve())
    return links


def analyze(files: list[Path]) -> dict:
    outbound: dict[Path, set[Path]] = {}
    inbound: dict[Path, set[Path]] = defaultdict(set)
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        outs = extract_links(f, text)
        outbound[f.resolve()] = outs
        for t in outs:
            inbound[t].add(f.resolve())

    orphans = []
    for f in files:
        fr = f.resolve()
        if not outbound.get(fr) and not inbound.get(fr):
            orphans.append(f)

    return {
        "total": len(files),
        "with_out": sum(1 for f in files if outbound.get(f.resolve())),
        "with_in": sum(1 for f in files if inbound.get(f.resolve())),
        "both": sum(
            1
            for f in files
            if outbound.get(f.resolve()) and inbound.get(f.resolve())
        ),
        "orphans": orphans,
        "outbound": outbound,
        "inbound": inbound,
    }


def vault_path(path: Path) -> str:
    return path.relative_to(VAULT).as_posix()


def processor_index(file: Path) -> list[tuple[str, Path]]:
    rel_parts = file.relative_to(VAULT).parts
    hubs: list[tuple[str, Path]] = []

    if "AudioProcessor" in rel_parts:
        hubs.append(("AudioProcessor", VAULT / "DataProcessor/AudioProcessor/docs/MAIN_INDEX.md"))
    if "TextProcessor" in rel_parts:
        hubs.append(("TextProcessor", VAULT / "DataProcessor/TextProcessor/docs/MAIN_INDEX.md"))
    if "VisualProcessor" in rel_parts:
        hubs.append(("VisualProcessor", VAULT / "DataProcessor/VisualProcessor/docs/MAIN_INDEX.md"))
    if rel_parts[0] == "DataProcessor" or "DataProcessor" in rel_parts:
        hubs.append(("DataProcessor", VAULT / "DataProcessor/docs/MAIN_INDEX.md"))
    if rel_parts[0] == "backend" or "backend" in rel_parts:
        hubs.append(("Backend", VAULT / "backend/docs/MAIN_INDEX.md"))
    if rel_parts[0] == "Fetcher":
        hubs.append(("Fetcher", VAULT / "Fetcher/docs/INDEX.md"))
    if rel_parts[0] == "DynamicBatch":
        hubs.append(("DynamicBatch", VAULT / "DynamicBatch/README.md"))
    if rel_parts[0] == "Models":
        hubs.append(("Models", VAULT / "Models/docs/MAIN_INDEX.md"))
    if "dp_models" in rel_parts:
        hubs.append(("dp_models", VAULT / "DataProcessor/dp_models/MAIN_INDEX.md"))
    if "embedding_service" in rel_parts:
        hubs.append(("embedding_service", VAULT / "DataProcessor/embedding_service/MAIN_INDEX.md"))

    hubs.append(("Vault", VAULT / "docs/MAIN_INDEX.md"))
    # dedupe preserving order
    seen: set[Path] = set()
    unique: list[tuple[str, Path]] = []
    for label, p in hubs:
        if p.exists() and p.resolve() not in seen and p.resolve() != file.resolve():
            seen.add(p.resolve())
            unique.append((label, p))
    return unique


def sibling_docs(file: Path) -> list[tuple[str, Path]]:
    siblings: list[tuple[str, Path]] = []
    priority = [
        ("README", "README.md"),
        ("FEATURE_DESCRIPTION", "FEATURE_DESCRIPTION.md"),
        ("FEATURES_DESCRIPTION", "FEATURES_DESCRIPTION.md"),
        ("SCHEMA", "SCHEMA.md"),
        ("TESTING_REPORT", "TESTING_REPORT.md"),
        ("FINAL_TEST_REPORT", "FINAL_TEST_REPORT.md"),
    ]
    parent = file.parent
    for label, name in priority:
        candidate = parent / name
        if candidate.exists() and candidate.resolve() != file.resolve():
            siblings.append((label, candidate))

    # root-level duplicate for extractors
    parts = file.relative_to(VAULT).parts
    if "extractors" in parts and file.parent.name == "docs":
        idx = parts.index("extractors")
        extractor_root = VAULT.joinpath(*parts[: idx + 2])
        for label, name in priority:
            candidate = extractor_root / name
            if candidate.exists() and candidate.resolve() != file.resolve():
                siblings.append((f"{label} (root)", candidate))
        readme = extractor_root / "README.md"
        if readme.exists() and readme.resolve() != file.resolve():
            siblings.append(("README (root)", readme))

    # module README for docs/ subfolder
    if file.parent.name == "docs" and file.name != "README.md":
        module_readme = file.parent.parent / "README.md"
        if module_readme.exists():
            siblings.append(("Module README", module_readme))

    seen: set[Path] = set()
    out: list[tuple[str, Path]] = []
    for label, p in siblings:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            out.append((label, p))
    return out


def audit_links(file: Path) -> list[tuple[str, Path]]:
    links: list[tuple[str, Path]] = []
    name = file.name
    rel_parts = file.relative_to(VAULT).parts

    if "AUDIT_V3" in name:
        component = name.split("_AUDIT_V3")[0]
        if "AudioProcessor" in rel_parts:
            audit_readme = VAULT / "DataProcessor/AudioProcessor/docs/audit_v3/README.md"
            if audit_readme.exists():
                links.append(("Audit v3 index", audit_readme))
            extractor_readme = (
                VAULT
                / "DataProcessor/AudioProcessor/src/extractors"
                / component
                / "docs/README.md"
            )
            if extractor_readme.exists():
                links.append(("Extractor README", extractor_readme))
        if "TextProcessor" in rel_parts:
            audit_readme = VAULT / "DataProcessor/TextProcessor/docs/audit_v3/README.md"
            if audit_readme.exists():
                links.append(("Audit v3 index", audit_readme))

    if "audit_v4" in rel_parts:
        audit_hub = VAULT / "DataProcessor/docs/audit_v4/components/audit_4_2/README.md"
        if audit_hub.exists():
            links.append(("Audit v4 hub", audit_hub))

    return links


def build_nav_section(file: Path) -> str | None:
    links: list[tuple[str, Path]] = []

    links.extend(sibling_docs(file))
    links.extend(audit_links(file))

    for label, hub in processor_index(file):
        links.append((label, hub))

    if not links:
        return None

    seen: set[Path] = set()
    parts: list[str] = []
    for label, target in links:
        if target.resolve() in seen or target.resolve() == file.resolve():
            continue
        seen.add(target.resolve())
        parts.append(f"[{label}]({rel(file, target)})")

    if not parts:
        return None

    return f"\n---\n\n{NAV_MARKER}\n\n" + " · ".join(parts) + "\n"


def has_nav(text: str) -> bool:
    return NAV_MARKER in text


def apply_nav(dry_run: bool = False) -> dict:
    files = iter_md_files()
    stats = {"updated": 0, "skipped_has_nav": 0, "skipped_no_nav": 0}
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        if has_nav(text):
            stats["skipped_has_nav"] += 1
            continue
        section = build_nav_section(f)
        if not section:
            stats["skipped_no_nav"] += 1
            continue
        if not dry_run:
            f.write_text(text.rstrip() + section, encoding="utf-8")
        stats["updated"] += 1
    return stats


def print_report(label: str, data: dict) -> None:
    orphan_pct = 100 * len(data["orphans"]) / data["total"] if data["total"] else 0
    print(f"\n=== {label} ===")
    print(f"Total md files: {data['total']}")
    print(f"With outbound links: {data['with_out']}")
    print(f"With inbound links: {data['with_in']}")
    print(f"Bidirectional: {data['both']}")
    print(f"Orphans: {len(data['orphans'])} ({orphan_pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wire TrendFlowML Obsidian doc graph")
    parser.add_argument("--analyze", action="store_true", help="Analyze link graph only")
    parser.add_argument("--apply", action="store_true", help="Add navigation sections")
    parser.add_argument("--dry-run", action="store_true", help="Show apply stats without writing")
    args = parser.parse_args()

    files = iter_md_files()
    before = analyze(files)
    print_report("Before", before)

    if args.apply or args.dry_run:
        stats = apply_nav(dry_run=args.dry_run)
        print("\n=== Apply navigation ===")
        for k, v in stats.items():
            print(f"{k}: {v}")
        if not args.dry_run:
            after = analyze(iter_md_files())
            print_report("After", after)

    if args.analyze and not args.apply and not args.dry_run:
        print("\nTop orphan patterns:")
        patterns: dict[str, int] = defaultdict(int)
        for f in before["orphans"][:100]:
            rel_p = vault_path(f)
            if "TESTING_REPORT" in rel_p:
                patterns["TESTING_REPORT"] += 1
            elif "FEATURE" in rel_p:
                patterns["FEATURE_*"] += 1
            elif "SCHEMA" in rel_p:
                patterns["SCHEMA"] += 1
            elif "README" in rel_p:
                patterns["README"] += 1
            elif "audit" in rel_p.lower():
                patterns["audit"] += 1
            else:
                patterns["other"] += 1
        for pat, cnt in sorted(patterns.items(), key=lambda x: -x[1]):
            print(f"  {pat}: {cnt}")


if __name__ == "__main__":
    main()
