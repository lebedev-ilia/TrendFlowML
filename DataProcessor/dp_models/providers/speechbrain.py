from __future__ import annotations

import os
from typing import Any

from ..errors import ModelManagerError
from ..specs import ModelSpec


class SpeechBrainProvider:
    """
    In-process SpeechBrain model provider.
    
    Policy:
    - model must be loaded from a **local directory** (no HF id downloads).
    - offline env must be set by ModelManager.
    - Supports Speech_Emotion_Diarization and other SpeechBrain inference interfaces.
    """
    
    def supports(self, spec: ModelSpec) -> bool:
        return (spec.runtime == "inprocess") and ("speechbrain" in (spec.engine or "").lower())
    
    def load(
        self,
        *,
        spec: ModelSpec,
        device: str,
        precision: str,
        models_root: str,
        runtime_params: dict | None = None,
    ) -> Any:
        rp = runtime_params or spec.runtime_params or {}
        if not isinstance(rp, dict):
            rp = {}

        
        # Get model class name from runtime_params (default: Speech_Emotion_Diarization)
        model_class_name = rp.get("model_class", "Speech_Emotion_Diarization")
        
        # Pick first directory artifact as the model folder
        model_dir_rel = None
        for a in spec.local_artifacts:
            if str(a.kind) == "dir":
                model_dir_rel = str(a.path)
                break
        if not model_dir_rel:
            raise ModelManagerError(
                message="SpeechBrainProvider requires a local_artifacts entry with kind=dir",
                error_code="weights_missing",
                details={"model_name": spec.model_name},
            )

        try:
            import torch  # type: ignore
        except Exception as e:
            raise ModelManagerError(
                message="torch is not installed",
                error_code="dependency_missing",
                details={"import_error": str(e)},
            ) from e
        
        # Set HF environment variables BEFORE importing SpeechBrain/transformers
        # This is critical because transformers may cache the cache path on first import
        from pathlib import Path
        
        # Determine HF cache directory
        hf_hub_cache = None
        
        # First, check if HF_HOME is already set (e.g., by enforce_offline_env from ModelManager)
        # This takes priority as it points to models_root/hf_cache
        current_hf_home = os.environ.get("HF_HOME")
        if current_hf_home and os.path.exists(current_hf_home):
            # Use the already-set HF_HOME (likely from enforce_offline_env)
            hf_hub_cache = os.path.join(current_hf_home, "hub")
            if not os.path.exists(hf_hub_cache):
                hf_hub_cache = None
        else:
            # Fallback to standard cache location if not set
            default_hf_home = os.path.join(Path.home(), ".cache", "huggingface")
            if os.path.exists(default_hf_home):
                os.environ.setdefault("HF_HOME", default_hf_home)
                hf_hub_cache = os.path.join(default_hf_home, "hub")
                if not os.path.exists(hf_hub_cache):
                    hf_hub_cache = None
        
        # Set HF cache environment variables to ensure transformers uses the correct cache
        if hf_hub_cache:
            # Use realpath for symlinks (e.g. bundled_models/hf_cache/hub -> ~/.cache/huggingface/hub)
            hf_hub_cache = os.path.realpath(hf_hub_cache)
            os.environ["HF_HUB_CACHE"] = hf_hub_cache
            os.environ["TRANSFORMERS_CACHE"] = hf_hub_cache
            # Also set HF_HOME if not already set
            if not current_hf_home:
                os.environ["HF_HOME"] = os.path.dirname(hf_hub_cache)
        
        # Configure logging BEFORE importing speechbrain to prevent logging errors during import
        import logging
        import sys
        
        # Disable logging error reporting to prevent tracebacks from closed handlers
        old_raise_exceptions = logging.raiseExceptions
        logging.raiseExceptions = False
        
        # Set environment variables to suppress logs
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["PYTHONWARNINGS"] = "ignore"
        
        # Configure loggers BEFORE import (speechbrain logs during import)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.ERROR)
        transformers_logger.propagate = False
        
        hf_logger = logging.getLogger("huggingface_hub")
        hf_logger.setLevel(logging.ERROR)
        hf_logger.propagate = False
        
        accelerate_logger = logging.getLogger("accelerate")
        accelerate_logger.setLevel(logging.ERROR)
        accelerate_logger.propagate = False
        
        sb_logger = logging.getLogger("speechbrain")
        sb_logger.setLevel(logging.ERROR)
        sb_logger.propagate = False
        
        tqdm_logger = logging.getLogger("tqdm")
        tqdm_logger.setLevel(logging.ERROR)
        tqdm_logger.propagate = False
        
        sb_utils_logger = logging.getLogger("speechbrain.utils")
        sb_utils_logger.setLevel(logging.ERROR)
        sb_utils_logger.propagate = False
        
        sb_checkpoints_logger = logging.getLogger("speechbrain.utils.checkpoints")
        sb_checkpoints_logger.setLevel(logging.ERROR)
        sb_checkpoints_logger.propagate = False
        
        # Try to import SpeechBrain (prefer local version from component if available)
        try:
            # First try local speechbrain from emotion_diarization_extractor
            extractor_dir = Path(__file__).resolve().parent.parent.parent / "AudioProcessor" / "src" / "extractors" / "emotion_diarization_extractor"
            speechbrain_path = extractor_dir / "speechbrain"
            if speechbrain_path.exists() and str(speechbrain_path) not in sys.path:
                sys.path.insert(0, str(speechbrain_path))
            
            from speechbrain.inference.diarization import Speech_Emotion_Diarization  # type: ignore
        except ImportError:
            # Fallback to system-installed speechbrain
            try:
                from speechbrain.inference.diarization import Speech_Emotion_Diarization  # type: ignore
            except Exception as e:
                # Restore logging.raiseExceptions before raising
                logging.raiseExceptions = old_raise_exceptions
                raise ModelManagerError(
                    message="speechbrain is not installed (neither local nor system)",
                    error_code="dependency_missing",
                    details={"import_error": str(e), "hint": "Install speechbrain or ensure local copy exists in emotion_diarization_extractor/speechbrain/"},
                ) from e
        # Note: We keep logging.raiseExceptions = False for model loading below

        # Load model from local directory using from_hparams (strict local-only, no-network).
        try:
            import contextlib
            # logging.raiseExceptions is already False from above

            model_dir_abs = os.path.normpath(os.path.join(models_root, model_dir_rel))
            if (not os.path.isdir(model_dir_abs)) or (not os.listdir(model_dir_abs)):
                raise ModelManagerError(
                    message="SpeechBrain model directory is missing or empty",
                    error_code="weights_missing",
                    details={"model_name": spec.model_name, "model_dir": model_dir_abs},
                )

            # Подавляем stdout/stderr во время загрузки модели для скрытия прогресс-баров
            @contextlib.contextmanager
            def suppress_output():
                """Временно подавляет stdout и stderr."""
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

            # Resolve WavLM local path for offline loading (hyperparams reference microsoft/wavlm-large).
            # SpeechBrain Wav2Vec2 passes cache_dir=save_path to transformers, which can override
            # HF_HUB_CACHE. Use explicit local path to avoid "couldn't find in cached files" in offline.
            # Use realpath so symlinks (e.g. bundled_models/hf_cache/hub -> ~/.cache/huggingface/hub)
            # resolve to the actual path where preprocessor_config.json may exist.
            hparams_overrides = {}
            hf_hub_resolved = os.path.realpath(hf_hub_cache) if hf_hub_cache and os.path.exists(hf_hub_cache) else (hf_hub_cache or "")
            wavlm_cache = os.path.join(hf_hub_resolved, "models--microsoft--wavlm-large", "snapshots")
            if hf_hub_resolved and os.path.isdir(wavlm_cache):
                try:
                    revs = sorted(p for p in os.listdir(wavlm_cache) if os.path.isdir(os.path.join(wavlm_cache, p)))
                    if revs:
                        wavlm_local = os.path.join(wavlm_cache, revs[-1])
                        if os.path.isfile(os.path.join(wavlm_local, "config.json")):
                            hparams_overrides["wav2vec2_hub"] = os.path.realpath(wavlm_local)
                except OSError:
                    pass

            with suppress_output():
                # IMPORTANT: do NOT disable offline env here.
                # If the SpeechBrain pipeline internally references transformer checkpoints (e.g. WavLM),
                # those must already be present in the HuggingFace cache configured by ModelManager.
                model = Speech_Emotion_Diarization.from_hparams(
                    source=model_dir_abs,
                    savedir=model_dir_abs,
                    run_opts={"device": device},
                    overrides=hparams_overrides,
                )
        except Exception as e:
            # SpeechBrain will raise its own errors if files are missing
            # We wrap them in ModelManagerError for consistency
            error_msg = str(e)
            raise ModelManagerError(
                message="Failed to load SpeechBrain model",
                error_code="model_load_failed",
                details={"model_name": spec.model_name, "model_dir": model_dir_abs, "error": error_msg, "error_type": type(e).__name__},
            ) from e
        finally:
            # Restore logging.raiseExceptions after model loading (always, even on exception)
            logging.raiseExceptions = old_raise_exceptions
        
        return model

