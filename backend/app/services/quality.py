from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class QualityScript:
    component_name: str
    script_path: Path
    flags: set[str]


def _extract_component_name(script_path: Path) -> Optional[str]:
    parts = script_path.parts
    if "modules" in parts:
        idx = parts.index("modules")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if "model_process" in parts:
        idx = parts.index("model_process")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _detect_flags(text: str) -> set[str]:
    flags = set()
    for m in re.finditer(r"--([a-zA-Z0-9_-]+)", text):
        flags.add(m.group(1))
    return flags


def discover_quality_scripts(dataproc_root: Path) -> Dict[str, QualityScript]:
    scripts: Dict[str, QualityScript] = {}
    for path in dataproc_root.rglob("quality_report/demo_*_quality.py"):
        comp = _extract_component_name(path)
        if not comp:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        flags = _detect_flags(text)
        scripts[comp] = QualityScript(component_name=comp, script_path=path, flags=flags)
    return scripts


def find_component_npz(run_rs_path: Path, component_name: str) -> Optional[Path]:
    comp_dir = run_rs_path / component_name
    if not comp_dir.exists():
        return None
    candidates = sorted(comp_dir.glob("*.npz"))
    if not candidates:
        return None
    # Prefer file containing component name if possible.
    for c in candidates:
        if component_name in c.name:
            return c
    return candidates[0]


def build_quality_command(
    script: QualityScript,
    *,
    run_rs_path: Path,
    frames_dir: Optional[Path],
    video_path: Optional[Path],
    out_dir: Path,
) -> Optional[List[str]]:
    flags = script.flags
    cmd = [os.environ.get("PYTHON", "python3"), str(script.script_path)]

    if "out-dir" in flags:
        cmd.extend(["--out-dir", str(out_dir)])
    if "out-html" in flags:
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--out-html", str(out_dir / f"{script.component_name}_quality.html")])
    if "npz-path" in flags:
        npz = find_component_npz(run_rs_path, script.component_name)
        if not npz:
            return None
        cmd.extend(["--npz-path", str(npz)])
    if "rs-path" in flags:
        cmd.extend(["--rs-path", str(run_rs_path)])
    if "frames-dir" in flags and frames_dir is not None:
        cmd.extend(["--frames-dir", str(frames_dir)])
    if "video-path" in flags and video_path is not None:
        cmd.extend(["--video-path", str(video_path)])

    return cmd


def run_quality_reports(
    scripts: Dict[str, QualityScript],
    *,
    run_rs_path: Path,
    frames_dir: Optional[Path],
    video_path: Optional[Path],
    components: List[str],
) -> List[Tuple[str, Path]]:
    generated: List[Tuple[str, Path]] = []
    for comp in components:
        script = scripts.get(comp)
        if not script:
            continue
        out_dir = run_rs_path / comp / "quality_report"
        cmd = build_quality_command(
            script,
            run_rs_path=run_rs_path,
            frames_dir=frames_dir,
            video_path=video_path,
            out_dir=out_dir,
        )
        if not cmd:
            continue
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception:
            continue

        # Register any HTML outputs in the out_dir.
        for html in out_dir.glob("*.html"):
            generated.append((comp, html))

    return generated

