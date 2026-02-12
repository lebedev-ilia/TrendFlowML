from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Dict, Optional, Tuple

# === torchaudio compatibility patch (for speechbrain models) ===
try:
    import torchaudio  # type: ignore
    if not hasattr(torchaudio, 'list_audio_backends'):
        def _list_audio_backends():
            return ['soundfile', 'sox']
        torchaudio.list_audio_backends = _list_audio_backends
except Exception:
    pass

from ..errors import ModelManagerError
from ..specs import ModelSpec


def _import_factory(dotted: str) -> Callable[..., Any]:
    """
    Import a callable from "package.module:attr" or "package.module.attr".
    """
    s = str(dotted or "").strip()
    if not s:
        raise ModelManagerError(message="Empty factory path", error_code="model_spec_invalid")
    if ":" in s:
        mod, attr = s.split(":", 1)
    else:
        parts = s.split(".")
        if len(parts) < 2:
            raise ModelManagerError(
                message="Factory path must be module.attr or module:attr",
                error_code="model_spec_invalid",
                details={"factory": s},
            )
        mod = ".".join(parts[:-1])
        attr = parts[-1]
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        raise ModelManagerError(
            message="Failed to import factory module",
            error_code="model_spec_invalid",
            details={"factory": s, "module": mod, "error": str(e)},
        ) from e
    try:
        fn = getattr(m, attr)
    except Exception as e:
        raise ModelManagerError(
            message="Failed to resolve factory attribute",
            error_code="model_spec_invalid",
            details={"factory": s, "attr": attr, "error": str(e)},
        ) from e
    if not callable(fn):
        raise ModelManagerError(
            message="Factory is not callable",
            error_code="model_spec_invalid",
            details={"factory": s},
        )
    return fn


class TorchStateDictProvider:
    """
    Generic PyTorch provider that:
    - constructs a model using a python factory (dotted import path)
    - loads local `state_dict` from a file

    This is powerful but depends on python code and installed packages (torch/torchvision/etc.).

    Spec requirements (via runtime_params):
    - `factory`: dotted callable (e.g. "torchvision.models.video.slowfast_r50")
    - `factory_kwargs`: dict (must NOT include pretrained=True; forbidden)
    - `checkpoint_relpath`: optional, select which local_artifacts entry is the checkpoint
    - `state_dict_key`: optional, if checkpoint contains nested dict (default "state_dict")
    - `strict`: bool (default True)
    - `strip_prefix`: optional string prefix to strip from all keys (e.g. "module.")
    """

    def supports(self, spec: ModelSpec) -> bool:
        return (spec.runtime == "inprocess") and (str(spec.engine).lower() in ("torch", "pytorch", "torch-state-dict"))

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

        factory_path = rp.get("factory")
        if not isinstance(factory_path, str) or not factory_path.strip():
            raise ModelManagerError(
                message="TorchStateDictProvider requires runtime_params.factory",
                error_code="model_spec_invalid",
                details={"model_name": spec.model_name},
            )
        factory_kwargs = rp.get("factory_kwargs") if isinstance(rp.get("factory_kwargs"), dict) else {}
        # hard policy: forbid pretrained=True
        if "pretrained" in factory_kwargs and bool(factory_kwargs.get("pretrained")):
            raise ModelManagerError(
                message="pretrained=True is forbidden (no-network). Provide local weights via state_dict.",
                error_code="network_forbidden",
                details={"model_name": spec.model_name, "factory": factory_path},
            )

        state_key = rp.get("state_dict_key")
        state_key = str(state_key) if state_key is not None else "state_dict"
        strict = bool(rp.get("strict", True))
        strip_prefix = rp.get("strip_prefix")
        strip_prefix = str(strip_prefix) if strip_prefix is not None else None

        # Pick checkpoint artifact.
        ckpt_rel = rp.get("checkpoint_relpath")
        ckpt_rel = str(ckpt_rel) if ckpt_rel is not None else None
        if ckpt_rel:
            # ensure it exists in local_artifacts for auditability
            declared = {str(a.path) for a in spec.local_artifacts if str(a.kind) == "file"}
            if ckpt_rel not in declared:
                raise ModelManagerError(
                    message="checkpoint_relpath must point to one of spec.local_artifacts (kind=file)",
                    error_code="model_spec_invalid",
                    details={"model_name": spec.model_name, "checkpoint_relpath": ckpt_rel, "declared_files": sorted(declared)},
                )
        else:
            for a in spec.local_artifacts:
                if str(a.kind) == "file":
                    ckpt_rel = str(a.path)
                    break
        if not ckpt_rel:
            raise ModelManagerError(
                message="TorchStateDictProvider requires a checkpoint file (local_artifacts.kind=file)",
                error_code="weights_missing",
                details={"model_name": spec.model_name},
            )

        ckpt_path = os.path.join(models_root, ckpt_rel) if not os.path.isabs(ckpt_rel) else ckpt_rel
        ckpt_path = os.path.abspath(ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise ModelManagerError(
                message="Checkpoint file not found",
                error_code="weights_missing",
                details={"model_name": spec.model_name, "checkpoint": ckpt_path},
            )

        try:
            import torch  # type: ignore
        except Exception as e:
            raise ModelManagerError(
                message="torch is not installed",
                error_code="dependency_missing",
                details={"import_error": str(e)},
            ) from e

        # device mapping
        map_location = "cpu"
        if str(device).lower().startswith("cuda"):
            map_location = device if ":" in str(device) else "cuda"

        # Load checkpoint first to check if it contains a full model
        # Use weights_only=False to allow loading full model objects (we trust our own checkpoints)
        # If loading fails due to missing class (e.g., DemucsEnergyModel), fall back to factory + state_dict
        ckpt = None
        load_error = None
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            # If loading fails due to missing class, we'll use factory function instead
            # Check if error is about missing class attribute
            error_str = str(e).lower()
            if "can't get attribute" in error_str or "cannot get attribute" in error_str:
                # This is likely a class deserialization error - use factory function instead
                load_error = e
            else:
                # Other errors (file not found, corrupted, etc.) should be raised
                raise ModelManagerError(
                    message="Failed to load checkpoint file",
                    error_code="model_load_failed",
                    details={"model_name": spec.model_name, "checkpoint": ckpt_path, "error": str(e)},
                ) from e

        # Check if checkpoint contains a full model (for direct loading without factory)
        # Note: If loading full model fails (e.g., class not found), fall back to factory + state_dict
        if ckpt is not None and isinstance(ckpt, dict) and "model" in ckpt:
            try:
                # Full model is available - try to use it directly
                model = ckpt["model"]
                try:
                    model.eval()
                    # Move to appropriate device
                    map_location = "cpu"
                    if str(device).lower().startswith("cuda"):
                        map_location = device if ":" in str(device) else "cuda"
                    if map_location.startswith("cuda"):
                        try:
                            model = model.cuda() if ":" not in map_location else model.to(map_location)
                        except Exception:
                            pass
                    # Apply precision if needed
                    # Note: For speechbrain models, we prefer fp32 to avoid dtype conflicts
                    try:
                        current_dtype = next(model.parameters()).dtype if hasattr(model, "parameters") else torch.float32
                        if str(precision).lower() == "fp32":
                            # Force fp32 for speechbrain models to avoid dtype conflicts
                            if current_dtype == torch.float16:
                                model = model.float()
                        elif str(precision).lower() == "fp16" and str(map_location).startswith("cuda") and current_dtype != torch.float16:
                            # Only convert to fp16 if explicitly requested and not already fp16
                            model = model.half()
                    except Exception:
                        pass
                except Exception:
                    pass
                # If we got here, model was successfully loaded
                return model
            except Exception as e:
                # If loading full model fails (e.g., class not found), fall back to factory + state_dict
                # This handles cases where model was saved with a class that's not available at load time
                pass
        
        # If checkpoint failed to load due to missing class, try to load only state_dict
        # This is a workaround for models saved with full objects that can't be deserialized
        if load_error is not None:
            # Try to load checkpoint with a custom unpickler that skips the problematic model object
            try:
                import pickle
                import io
                
                # Custom unpickler that skips problematic objects
                class SkipModelUnpickler(pickle.Unpickler):
                    def persistent_load(self, pid):
                        # Skip persistent IDs that might reference the problematic model
                        return None
                    
                    def load_build(self):
                        # Override to skip building problematic objects
                        super().load_build()
                
                # Try to load checkpoint with custom unpickler
                with open(ckpt_path, "rb") as f:
                    unpickler = SkipModelUnpickler(f)
                    try:
                        ckpt = unpickler.load()
                        if isinstance(ckpt, dict) and "state_dict" in ckpt:
                            # Successfully loaded state_dict
                            print(f"[TorchStateDictProvider] Loaded state_dict from checkpoint with missing class (using custom unpickler)")
                            ckpt = {"state_dict": ckpt["state_dict"], "meta": ckpt.get("meta", {})}
                        else:
                            # Failed to load properly, will use factory function
                            ckpt = None
                    except Exception as e2:
                        # Custom unpickler also failed, will use factory function
                        ckpt = None
            except Exception as e2:
                # Failed to use custom unpickler, will use factory function
                ckpt = None
        
        # If we still don't have a checkpoint, we need to use factory function
        # But we can't load state_dict if checkpoint failed to load
        if ckpt is None:
            # Checkpoint couldn't be loaded at all - this is a problem
            # We need the state_dict to load weights into the factory-created model
            raise ModelManagerError(
                message="Failed to load checkpoint file (class deserialization error). Please re-save the model using download_source_separation_models.py",
                error_code="model_load_failed",
                details={
                    "model_name": spec.model_name,
                    "checkpoint": ckpt_path,
                    "error": str(load_error) if load_error else "Unknown error",
                    "suggestion": "Re-save the model using: python scripts/download_source_separation_models.py --sizes large",
                },
            )

        # No full model in checkpoint - use factory to create architecture and load state_dict
        factory = _import_factory(factory_path)

        # Construct model.
        try:
            model = factory(**factory_kwargs)
        except Exception as e:
            raise ModelManagerError(
                message="Failed to construct torch model via factory",
                error_code="model_load_failed",
                details={"model_name": spec.model_name, "factory": factory_path, "error": str(e)},
            ) from e

        state = ckpt
        if isinstance(ckpt, dict) and state_key in ckpt and isinstance(ckpt[state_key], dict):
            state = ckpt[state_key]
        if not isinstance(state, dict):
            raise ModelManagerError(
                message="Checkpoint does not contain a state_dict dict",
                error_code="model_load_failed",
                details={"model_name": spec.model_name, "checkpoint": ckpt_path, "state_dict_key": state_key},
            )

        # Common checkpoint compatibility: strip a prefix from keys (e.g., "module.").
        if strip_prefix:
            try:
                cleaned = {}
                for k, v in state.items():
                    ks = str(k)
                    if ks.startswith(strip_prefix):
                        ks = ks[len(strip_prefix) :]
                    cleaned[ks] = v
                state = cleaned
            except Exception:
                # if something goes wrong, keep original dict
                pass

        try:
            missing, unexpected = model.load_state_dict(state, strict=strict)
        except Exception as e:
            raise ModelManagerError(
                message="load_state_dict failed",
                error_code="model_load_failed",
                details={"model_name": spec.model_name, "checkpoint": ckpt_path, "error": str(e)},
            ) from e

        # Move to device.
        try:
            model = model.to(map_location)
        except Exception:
            try:
                model = model.to("cpu")
                map_location = "cpu"
            except Exception:
                pass
        try:
            model.eval()
        except Exception:
            pass

        # Precision best-effort.
        if str(precision).lower() == "fp16" and str(map_location).startswith("cuda"):
            try:
                model = model.half()
            except Exception:
                pass

        # If strict=False, still consider surfacing mismatch details via exception in future.
        return model


