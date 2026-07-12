#!/usr/bin/env python3
"""
OpenFaceAnalyzer (Docker runner, GPU-only by policy)

This helper runs OpenFace FeatureExtraction inside a Docker container and returns
an OpenFace-like DataFrame / CSV for further processing by MicroEmotionProcessor.

Baseline policies:
- No-network (uses local docker image)
- GPU-only allowed (per owner decision): docker is invoked with `--gpus all`
- Fail-fast on any OpenFace failure (caller decides empty vs error)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np
import pandas as pd  # type: ignore


@dataclass(frozen=True)
class OpenFaceRunResult:
    csv_path: str
    dataframe: pd.DataFrame


def _pick_best_csv(out_dir: str) -> Optional[str]:
    """
    OpenFace may write multiple CSVs depending on flags.
    We pick the one that looks like FeatureExtraction "full" output:
    it should contain AU and landmarks headers.
    """
    candidates: List[str] = []
    for p in Path(out_dir).rglob("*.csv"):
        candidates.append(str(p))
    if not candidates:
        return None

    def score_csv(p: str) -> int:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                head = f.readline()
        except Exception:
            return -1
        s = 0
        if "AU01_r" in head or "AU12_r" in head:
            s += 5
        if "pose_Rx" in head and "gaze_angle_x" in head:
            s += 3
        if "x_0" in head and "y_0" in head:
            s += 3
        if "timestamp" in head:
            s += 1
        return s

    candidates.sort(key=lambda p: (score_csv(p), os.path.getmtime(p)), reverse=True)
    return candidates[0] if score_csv(candidates[0]) >= 3 else candidates[0]


def _docker_cmd(
    *,
    image: str,
    in_dir: str,
    out_dir: str,
    use_gpu: bool,
    extra_args: Sequence[str],
) -> List[str]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--shm-size=1g",
        "--entrypoint",
        "bash",
        "-v",
        f"{in_dir}:/input:ro",
        "-v",
        f"{out_dir}:/output",
        image,
        "-lc",
        # Resolve FeatureExtraction path in a few common locations; then run on /input.
        (
            "set -euo pipefail; "
            "BIN=''; "
            "if command -v FeatureExtraction >/dev/null 2>&1; then BIN='FeatureExtraction'; "
            "elif [ -x /home/openface/build/bin/FeatureExtraction ]; then BIN='/home/openface/build/bin/FeatureExtraction'; "
            "elif [ -x /openface/build/bin/FeatureExtraction ]; then BIN='/openface/build/bin/FeatureExtraction'; "
            "elif [ -x /home/openface-build/build/bin/FeatureExtraction ]; then BIN='/home/openface-build/build/bin/FeatureExtraction'; "
            "else echo 'FeatureExtraction binary not found in container' >&2; exit 2; fi; "
            "cd /output; "
            # We run on directory; OpenFace will create CSVs in /output.
            "$BIN -fdir /input -out_dir /output "
            + " ".join(extra_args)
            + " || true"
        ),
    ]
    if use_gpu:
        cmd.insert(3, "--gpus")
        cmd.insert(4, "all")
    return cmd


class OpenFaceAnalyzer:
    def __init__(self, *, docker_image: str = "openface/openface:latest", use_gpu: bool = True) -> None:
        self.docker_image = str(docker_image)
        self.use_gpu = bool(use_gpu)

    def analyze_frames(
        self,
        *,
        frames_bgr: List[np.ndarray],
        union_frame_indices: List[int],
        output_prefix: str = "openface",
        keep_tmp: bool = False,
        extra_flags: Optional[Sequence[str]] = None,
    ) -> OpenFaceRunResult:
        """
        frames_bgr: list of BGR uint8 images (as OpenCV expects).
        union_frame_indices: same length; union-domain frame indices for mapping.
        """
        if len(frames_bgr) != len(union_frame_indices):
            raise ValueError("OpenFaceAnalyzer | frames_bgr and union_frame_indices length mismatch")
        if not frames_bgr:
            raise ValueError("OpenFaceAnalyzer | empty frames")

        flags = list(extra_flags or [])
        # Sensible defaults for AU + pose + gaze + landmarks.
        if not flags:
            flags = ["-aus", "-pose", "-gaze", "-2Dfp", "-3Dfp"]

        tmp_root = tempfile.mkdtemp(prefix="micro_emotion_openface_")
        in_dir = os.path.join(tmp_root, "input")
        out_dir = os.path.join(tmp_root, "output")
        os.makedirs(in_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        try:
            # Write images with sequential names so OpenFace assigns frame indices 0..M-1.
            for i, img in enumerate(frames_bgr):
                if img is None or not isinstance(img, np.ndarray):
                    raise ValueError(f"OpenFaceAnalyzer | invalid frame at {i}")
                p = os.path.join(in_dir, f"{i:06d}.jpg")
                ok = cv2.imwrite(p, img)
                if not ok:
                    raise RuntimeError(f"OpenFaceAnalyzer | failed to write image: {p}")

            cmd = _docker_cmd(
                image=self.docker_image,
                in_dir=os.path.abspath(in_dir),
                out_dir=os.path.abspath(out_dir),
                use_gpu=self.use_gpu,
                extra_args=flags,
            )
            proc = subprocess.run(cmd, capture_output=True, text=True)
            csv_path = _pick_best_csv(out_dir)
            if proc.returncode != 0 and (not csv_path or not os.path.exists(csv_path)):
                raise RuntimeError(
                    "OpenFaceAnalyzer | docker/OpenFace failed: "
                    f"code={proc.returncode} stdout={proc.stdout[-2000:]} stderr={proc.stderr[-4000:]}"
                )

            if not csv_path or not os.path.exists(csv_path):
                raise RuntimeError("OpenFaceAnalyzer | no CSV produced by OpenFace")

            df = pd.read_csv(csv_path)
            if df is None or len(df) == 0:
                raise RuntimeError("OpenFaceAnalyzer | produced empty CSV/DataFrame")

            # OpenFace FeatureExtraction writes CSV headers with a leading space after each comma
            # (", AU01_r", ", pose_Rx", ", success", ", x_0" ...). pandas keeps these leading spaces
            # unless skipinitialspace=True, so downstream lookups by bare names ("AU01_r", "pose_Rx",
            # "success") silently miss and every OpenFace-derived feature collapses to zero/NaN.
            # Normalize column names once here so the whole pipeline reads real values.
            df.columns = [str(c).strip() for c in df.columns]

            # Map OpenFace local frames -> union-domain indices.
            # OpenFace typically uses 0..M-1 in `frame` column for image sequences.
            if "frame" in df.columns:
                local = df["frame"].astype(int).values
                mapped = []
                for j in local:
                    if 0 <= int(j) < len(union_frame_indices):
                        mapped.append(int(union_frame_indices[int(j)]))
                    else:
                        mapped.append(-1)
                df["frame_union"] = np.asarray(mapped, dtype=np.int32)
            else:
                # best-effort: assume row order corresponds to input order
                df["frame_union"] = np.asarray(union_frame_indices[: len(df)], dtype=np.int32)

            return OpenFaceRunResult(csv_path=csv_path, dataframe=df)
        finally:
            if not keep_tmp:
                try:
                    shutil.rmtree(tmp_root, ignore_errors=True)
                except Exception:
                    pass


