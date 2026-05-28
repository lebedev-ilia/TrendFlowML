from __future__ import annotations

import os
import warnings
from typing import Any

from ..errors import ModelManagerError
from ..specs import ModelSpec


def _suppress_pyannote_io_noise() -> None:
    """soundfile/waveform input — torchcodec traceback только засоряет логи."""
    warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
    warnings.filterwarnings("ignore", module="pyannote.audio.utils.reproducibility")
    warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")


class PyannoteProvider:
    """
    In-process pyannote.audio Pipeline provider.
    
    Policy:
    - model must be loaded from a **local directory** (no HF id downloads).
    - offline env must be set by ModelManager.
    """

    def supports(self, spec: ModelSpec) -> bool:
        return (spec.runtime == "inprocess") and ("pyannote" in (spec.engine or "").lower())

    def load(
        self,
        *,
        spec: ModelSpec,
        device: str,
        precision: str,
        models_root: str,
        runtime_params: dict | None = None,
    ) -> Any:
        # Pick first directory artifact as the model folder.
        model_dir_rel = None
        for a in spec.local_artifacts:
            if str(a.kind) == "dir":
                model_dir_rel = str(a.path)
                break
        if not model_dir_rel:
            raise ModelManagerError(
                message="PyannoteProvider requires a local_artifacts entry with kind=dir",
                error_code="weights_missing",
                details={"model_name": spec.model_name},
            )
        model_dir = os.path.join(models_root, model_dir_rel) if not os.path.isabs(model_dir_rel) else model_dir_rel
        model_dir = os.path.abspath(model_dir)
        if not os.path.isdir(model_dir):
            raise ModelManagerError(
                message="Local pyannote Pipeline directory not found",
                error_code="weights_missing",
                details={"model_name": spec.model_name, "model_dir": model_dir},
            )
        cfg_path = os.path.join(model_dir, "config.yaml")
        if not os.path.isfile(cfg_path):
            raise ModelManagerError(
                message="Pyannote bundle incomplete (expected config.yaml in model directory)",
                error_code="weights_missing",
                details={
                    "model_name": spec.model_name,
                    "model_dir": model_dir,
                    "hint": "Populate e.g. audio/pyannote_speaker_diarization from HF "
                    "pyannote/speaker-diarization-community-1 (see AudioProcessor/ex.py snapshot_download).",
                },
            )

        _suppress_pyannote_io_noise()
        try:
            from pyannote.audio import Pipeline  # type: ignore
            import torch  # type: ignore
        except Exception as e:
            raise ModelManagerError(
                message="pyannote.audio is not installed",
                error_code="dependency_missing",
                details={"import_error": str(e)},
            ) from e

        # Строго локальная загрузка (как в AudioProcessor/ex.py): без докачки через HF hub.
        try:
            try:
                pipeline = Pipeline.from_pretrained(model_dir, local_files_only=True)
            except TypeError:
                pipeline = Pipeline.from_pretrained(model_dir)
        except Exception as e:
            raise ModelManagerError(
                message="Failed to load pyannote Pipeline from local directory",
                error_code="model_load_failed",
                details={"model_name": spec.model_name, "model_dir": model_dir, "error": str(e)},
            ) from e

        # Move to device
        try:
            if str(device).lower().startswith("cuda"):
                pipeline.to(torch.device(device if ":" in str(device) else "cuda"))
            else:
                pipeline.to(torch.device("cpu"))
        except Exception as e:
            # If device move fails, keep on CPU
            try:
                pipeline.to(torch.device("cpu"))
            except Exception:
                pass

        # Precision best-effort (pyannote Pipeline wraps torch models internally).
        # Note: pyannote pipelines typically work best with fp32
        if str(precision).lower() == "fp16" and ("cuda" in str(device).lower()):
            try:
                # Try to convert to fp16 if requested
                # Note: This may not work for all pyannote models
                pass  # Skip fp16 conversion for now as pyannote models may not support it well
            except Exception:
                pass

        return pipeline

