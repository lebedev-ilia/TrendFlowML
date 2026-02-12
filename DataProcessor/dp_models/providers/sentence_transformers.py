from __future__ import annotations

import os
import sys
import contextlib
from typing import Any
from io import StringIO

from ..errors import ModelManagerError
from ..specs import ModelSpec


class SentenceTransformerProvider:
    """
    In-process SentenceTransformers provider.

    Policy:
    - model must be loaded from a **local directory** (no HF id downloads).
    - offline env must be set by ModelManager.
    """

    def supports(self, spec: ModelSpec) -> bool:
        return (spec.runtime == "inprocess") and ("sentence-transformers" in (spec.engine or "").lower())

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
                message="SentenceTransformerProvider requires a local_artifacts entry with kind=dir",
                error_code="weights_missing",
                details={"model_name": spec.model_name},
            )
        model_dir = os.path.join(models_root, model_dir_rel) if not os.path.isabs(model_dir_rel) else model_dir_rel
        model_dir = os.path.abspath(model_dir)
        if not os.path.isdir(model_dir):
            raise ModelManagerError(
                message="Local SentenceTransformer directory not found",
                error_code="weights_missing",
                details={"model_name": spec.model_name, "model_dir": model_dir},
            )

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            import logging
        except Exception as e:
            raise ModelManagerError(
                message="sentence_transformers is not installed",
                error_code="dependency_missing",
                details={"import_error": str(e)},
            ) from e

        # Suppress "Loading weights" progress bar output
        # Set logging level for sentence_transformers to WARNING to hide progress bars
        st_logger = logging.getLogger("sentence_transformers")
        original_level = st_logger.level
        st_logger.setLevel(logging.WARNING)
        
        # Also suppress transformers progress bars
        transformers_logger = logging.getLogger("transformers")
        transformers_original_level = transformers_logger.level
        transformers_logger.setLevel(logging.WARNING)
        
        # Suppress tqdm progress bars (used by sentence-transformers for "Loading weights")
        # Set environment variable to disable tqdm
        original_tqdm_disable = os.environ.get("TQDM_DISABLE", None)
        os.environ["TQDM_DISABLE"] = "1"
        
        # Also suppress stdout/stderr to catch any direct print statements
        @contextlib.contextmanager
        def suppress_output():
            """Temporarily suppress stdout and stderr."""
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                try:
                    sys.stdout = devnull
                    sys.stderr = devnull
                    yield
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

        # Force local path loading. If this fails due to missing files, we want a hard error.
        try:
            # Suppress output during model loading
            with suppress_output():
                model = SentenceTransformer(model_dir, device=device)
        except Exception as e:
            raise ModelManagerError(
                message="Failed to load SentenceTransformer from local directory",
                error_code="model_load_failed",
                details={"model_name": spec.model_name, "model_dir": model_dir, "error": str(e)},
            ) from e
        finally:
            # Restore original logging levels
            st_logger.setLevel(original_level)
            transformers_logger.setLevel(transformers_original_level)
            # Restore tqdm setting
            if original_tqdm_disable is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = original_tqdm_disable

        # Precision best-effort (SentenceTransformer wraps torch model internally).
        if str(precision).lower() == "fp16" and ("cuda" in str(device).lower()):
            try:
                model = model.half()
            except Exception:
                pass
        try:
            model.eval()
        except Exception:
            pass
        return model


