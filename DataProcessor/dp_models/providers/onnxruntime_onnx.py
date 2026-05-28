from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ..errors import ModelManagerError
from ..specs import ModelSpec


class OnnxRuntimeOnnxProvider:
    """
    Minimal ONNXRuntime provider for local ONNX model files (no-network).

    Spec contract:
    - runtime: inprocess
    - engine: onnxruntime_onnx
    - local_artifacts: must include the ONNX model file (kind=file)
    - runtime_params:
        - onnx_model_relpath: (optional) explicit relpath selecting one of local_artifacts
        - providers: (optional) list[str] overriding execution providers
    """

    def supports(self, spec: ModelSpec) -> bool:
        return (str(spec.runtime).lower() == "inprocess") and (str(spec.engine).lower() == "onnxruntime_onnx")

    def load(
        self,
        *,
        spec: ModelSpec,
        device: str,
        precision: str,
        models_root: str,
        runtime_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        rp = runtime_params or spec.runtime_params or {}
        if not isinstance(rp, dict):
            rp = {}

        # Pick ONNX model file relpath.
        onnx_rel = rp.get("onnx_model_relpath")
        onnx_rel = str(onnx_rel) if onnx_rel is not None else None
        if onnx_rel:
            declared = {str(a.path) for a in spec.local_artifacts if str(a.kind) == "file"}
            if onnx_rel not in declared:
                raise ModelManagerError(
                    message="onnx_model_relpath must point to one of spec.local_artifacts (kind=file)",
                    error_code="model_spec_invalid",
                    details={"model_name": spec.model_name, "onnx_model_relpath": onnx_rel, "declared_files": sorted(declared)},
                )
        else:
            for a in spec.local_artifacts:
                if str(a.kind) == "file" and str(a.path).lower().endswith(".onnx"):
                    onnx_rel = str(a.path)
                    break
            if not onnx_rel:
                # fallback: first file artifact
                for a in spec.local_artifacts:
                    if str(a.kind) == "file":
                        onnx_rel = str(a.path)
                        break

        if not onnx_rel:
            raise ModelManagerError(
                message="OnnxRuntimeOnnxProvider requires an ONNX file (local_artifacts.kind=file)",
                error_code="weights_missing",
                details={"model_name": spec.model_name},
            )

        onnx_path = os.path.join(models_root, onnx_rel) if not os.path.isabs(onnx_rel) else onnx_rel
        onnx_path = os.path.abspath(onnx_path)
        if not os.path.isfile(onnx_path):
            raise ModelManagerError(
                message="ONNX model file not found",
                error_code="weights_missing",
                details={"model_name": spec.model_name, "onnx_model": onnx_path},
            )

        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise ModelManagerError(
                message="onnxruntime is not installed",
                error_code="dependency_missing",
                details={"import_error": str(e)},
            ) from e

        available = []
        try:
            available = list(ort.get_available_providers())
        except Exception:
            available = []

        # Providers selection policy:
        # - If explicit providers list is given in runtime_params, use it (filtered to available).
        # - Else prefer CUDA EP if requested and available; otherwise CPU EP.
        requested = rp.get("providers")
        providers = None
        if isinstance(requested, list) and requested:
            providers = [str(x) for x in requested if str(x)]
            if available:
                providers = [p for p in providers if p in available]
        if not providers:
            want_cuda = str(device).lower().startswith("cuda")
            if want_cuda and ("CUDAExecutionProvider" in available):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        try:
            sess = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as e:
            raise ModelManagerError(
                message="Failed to create ONNXRuntime session",
                error_code="model_load_failed",
                details={"model_name": spec.model_name, "onnx_model": onnx_path, "providers": providers, "error": str(e)},
            ) from e

        return {
            "session": sess,
            "onnx_path": onnx_path,
            "providers": providers,
            "device": str(device),
            "precision": str(precision),
        }


