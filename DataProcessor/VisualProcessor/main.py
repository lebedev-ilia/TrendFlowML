import os
import sys

_path = os.path.dirname(__file__)

if _path not in sys.path:
    sys.path.append(_path)

import yaml
import argparse
import logging
import subprocess
import json
import hashlib
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from contextlib import suppress
from typing import List, Optional, Dict, Any, Tuple
import shutil
import numpy as np

from utils.logger import get_logger
from utils.results_store import ResultsStore
from utils.manifest import RunManifest, ManifestComponent
from utils.artifact_validator import validate_npz
from utils.resource_probe import get_cuda_mem_info
from utils.video_context import VideoContext
from utils.batch_utils import process_batch_results
from utils.core_clip_batch import process_core_clip_batch
from utils.core_depth_midas_batch import process_core_depth_midas_batch
from utils.place_semantics_batch import process_place_semantics_batch
from utils.core_optical_flow_batch import process_core_optical_flow_batch
from utils.core_object_detections_batch import process_core_object_detections_batch
from utils.core_face_landmarks_batch import process_core_face_landmarks_batch
from utils.ocr_extractor_batch import process_ocr_extractor_batch
from utils.content_domain_batch import process_content_domain_batch
from utils.franchise_recognition_batch import process_franchise_recognition_batch
from utils.brand_semantics_batch import process_brand_semantics_batch
from utils.car_semantics_batch import process_car_semantics_batch
from utils.face_identity_batch import process_face_identity_batch
from utils.cut_detection_batch import process_cut_detection_batch
from utils.scene_classification_batch import process_scene_classification_batch
from utils.video_pacing_batch import process_video_pacing_batch

logger = get_logger("VisualProcessor")


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _get_component_render_flags(
    component_name: str,
    component_cfg: Dict[str, Any],
    default_enable_render: bool = True,
    default_enable_html_render: bool = True,
) -> Tuple[bool, bool]:
    """
    Получить флаги рендеринга для компонента из конфига.
    
    Args:
        component_name: Имя компонента (например, "core_clip")
        component_cfg: Конфигурация компонента из global_config.yaml
        default_enable_render: Значение по умолчанию для enable_render
        default_enable_html_render: Значение по умолчанию для enable_html_render
    
    Returns:
        Tuple[enable_render, enable_html_render]
    """
    if component_cfg:
        render_cfg = component_cfg.get("render", {})
        enable_render = render_cfg.get("enable_render", default_enable_render) if render_cfg else default_enable_render
        enable_html_render = render_cfg.get("enable_html_render", default_enable_html_render) if render_cfg else default_enable_html_render
        return bool(enable_render), bool(enable_html_render)
    return default_enable_render, default_enable_html_render


def _safe_load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_artifact(component_dir: str, exts=(".npz", ".json")) -> list:
    if not os.path.isdir(component_dir):
        return []
    files = []
    for name in os.listdir(component_dir):
        p = os.path.join(component_dir, name)
        if os.path.isfile(p) and any(name.lower().endswith(e) for e in exts):
            files.append(p)
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def _atomic_write_json(path: str, payload: dict) -> None:
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _safe_load_json_optional(path: str) -> Optional[dict]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            x = json.load(f)
        return x if isinstance(x, dict) else None
    except Exception:
        return None


def _merge_dict(dst: dict, src: dict) -> dict:
    """
    Shallow/deep merge for JSON-like dicts:
    - dict values are merged recursively
    - other values from src overwrite dst
    """
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            dst[k] = _merge_dict(dst.get(k) or {}, v)
        else:
            dst[k] = v
    return dst


def _derive_run_context(cfg: dict) -> dict:
    g = cfg.get("global") or {}
    frames_dir = g.get("frames_dir")
    frames_meta = {}
    if frames_dir:
        meta_path = os.path.join(frames_dir, "metadata.json")
        if os.path.exists(meta_path):
            try:
                frames_meta = _safe_load_json(meta_path)
            except Exception:
                frames_meta = {}

    platform_id = g.get("platform_id") or frames_meta.get("platform_id") or "youtube"
    sampling_policy_version = g.get("sampling_policy_version") or frames_meta.get("sampling_policy_version") or "v1"
    dataprocessor_version = g.get("dataprocessor_version") or frames_meta.get("dataprocessor_version") or "unknown"
    analysis_fps = g.get("analysis_fps") or frames_meta.get("analysis_fps")
    analysis_width = g.get("analysis_width") or frames_meta.get("analysis_width")
    analysis_height = g.get("analysis_height") or frames_meta.get("analysis_height")
    resolved_model_mapping = cfg.get("resolved_model_mapping") if isinstance(cfg.get("resolved_model_mapping"), dict) else None

    # Try to derive video_id if not set (best effort).
    video_id = g.get("video_id") or frames_meta.get("video_id")
    if not video_id:
        vp = frames_meta.get("video_path")
        if isinstance(vp, str) and vp:
            video_id = os.path.splitext(os.path.basename(vp))[0]
    video_id = video_id or "unknown_video"

    run_id = g.get("run_id") or frames_meta.get("run_id") or uuid.uuid4().hex[:12]

    # Hash the full config for reproducibility.
    cfg_dump = yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True)
    config_hash = g.get("config_hash") or _sha256_text(cfg_dump)[:16]

    return {
        "platform_id": platform_id,
        "video_id": video_id,
        "run_id": run_id,
        "config_hash": config_hash,
        "sampling_policy_version": sampling_policy_version,
        "dataprocessor_version": dataprocessor_version,
        "analysis_fps": analysis_fps,
        "analysis_width": analysis_width,
        "analysis_height": analysis_height,
        "resolved_model_mapping": resolved_model_mapping,
        "created_at": _utc_iso_now(),
    }


def _build_subprocess_cmd(root_path, name, target, frames_dir, rs_path, cfg):
    """
    Унифицированная сборка команды для запуска модуля / core‑провайдера.
    process_dir: относительный путь от VisualProcessor (например, 'modules' или 'core/model_process').
    """
    vp_root = os.path.join(root_path, "VisualProcessor")

    if target == "core/model_process":
        # Allow per-core venv overrides (some cores have conflicting deps).
        # 1) Explicit override from config (recommended for special cases)
        cfg_venv = None
        try:
            cfg_venv = (cfg or {}).get("venv_path")
        except Exception:
            cfg_venv = None

        if isinstance(cfg_venv, str) and cfg_venv.strip():
            venv = cfg_venv.strip()
        else:
            # 2) Built-in override for known isolated environments
            if name == "core_face_landmarks":
                venv = os.path.join(vp_root, target, "core_face_landmarks", ".core_face_landmarks_venv")
            else:
                # Default: core providers run in the same VisualProcessor venv.
                # (.model_process_venv is deprecated in this repo.)
                venv = os.path.join(vp_root, ".vp_venv")
    else:
        # Allow per-module venv overrides (some modules have conflicting deps, e.g., pytorchvideo).
        # 1) Explicit override from config (recommended for special cases)
        cfg_venv = None
        try:
            cfg_venv = (cfg or {}).get("venv_path")
        except Exception:
            cfg_venv = None

        if isinstance(cfg_venv, str) and cfg_venv.strip():
            venv = cfg_venv.strip()
        else:
            # 2) Built-in override for known isolated module environments
            if name == "action_recognition":
                venv = os.path.join(vp_root, target, name, ".action_recognition_venv")
            else:
                # Default: modules run in the same VisualProcessor venv.
                venv = os.path.join(vp_root, ".vp_venv")

    python_exec = os.path.join(venv, "bin", "python")

    entry = os.path.join(vp_root, target, name, "main.py")
    if target == "core/model_process" and not os.path.exists(entry):
        # Compat: canonical component names may differ from folder names.
        # Prefer canonical names in metadata/rs_path, but allow legacy folder layout.
        core_folder_alias = {
            "core_object_detections": "object_detections",
            "core_depth_midas": "depth_midas",
        }
        alt_name = core_folder_alias.get(name)
        if alt_name:
            alt_entry = os.path.join(vp_root, target, alt_name, "main.py")
            if os.path.exists(alt_entry):
                entry = alt_entry
        
        # Check if component is in core_identity subdirectory
        if not os.path.exists(entry):
            core_identity_components = [
                "content_domain",
                "brand_semantics",
                "core_brand_semantics",
                "car_semantics",
                "core_car_semantics",
                "franchise_recognition",
                "core_franchise_recognition",
                "face_identity",
                "core_face_identity",
                "place_semantics",
                "core_place_semantics",
            ]
            # Remove 'core_' prefix if present for folder lookup
            folder_name = name.replace("core_", "") if name.startswith("core_") else name
            if name in core_identity_components or folder_name in ["content_domain", "brand_semantics", "car_semantics", "franchise_recognition", "face_identity", "place_semantics"]:
                identity_entry = os.path.join(vp_root, target, "core_identity", folder_name, "main.py")
                if os.path.exists(identity_entry):
                    entry = identity_entry

    if not os.path.exists(entry):
        raise FileNotFoundError(f"Entry not found for {target}/{name}: {entry}")

    kwargs = []
    # Component-specific parameter mappings and exclusions
    component_specific_exclusions = {
        "franchise_recognition": ("franchise_db_dir", "clip_text_model_spec", "render"),
        "detalize_face": (
            "ocr_npz",
            "alignment_window_seconds",
            "motion_weight",
            "face_weight",
            "audio_weight",
            "min_ocr_confidence",
            "retain_raw_ocr_text",
            "use_face_data",
            "render",
        ),
    }
    component_specific_mappings = {
        "franchise_recognition": {
            "max_full_labels": "max-franchises-for-full-search",
        },
    }
    
    exclusions = component_specific_exclusions.get(name, ())
    mappings = component_specific_mappings.get(name, {})
    
    for k, v in cfg.items():
        # Orchestrator-only keys (must NOT be forwarded to component CLI).
        if k in ("venv_path", "sampling"):
            continue
        # Component-specific exclusions
        if k in exclusions:
            continue
        # Nested objects are config-only (e.g., sampling dicts); do not forward to CLI.
        if isinstance(v, dict):
            continue
        if v is None or v == "False" or v is False:
            continue
        # Skip empty strings (they cause issues with optional int arguments)
        if v == "" or v == "''" or v == '""':
            continue
        # Apply component-specific mappings
        cli_key = mappings.get(k, k)
        key = f"--{cli_key.replace('_', '-')}"
        if v is True or v == "True":
            kwargs.append(key)
        else:
            kwargs.extend([key, str(v)])

    if not os.path.exists(python_exec):
        logger.warning(
            f"VisualProcessor | main | venv python not found at {python_exec}; "
            f"falling back to current interpreter: {sys.executable}"
        )
        python_exec = sys.executable

    cmd = [
        python_exec,
        entry,
        *kwargs,
        "--frames-dir",
        frames_dir,
        "--rs-path",
        rs_path,
    ]
    return cmd


def _component_uses_gpu(name: str, cfg: dict) -> bool:
    """
    Conservative heuristic:
    - if config has device in {"cuda","gpu","auto"} -> treat as GPU task (serialize by default)
    - else if component name suggests GPU-heavy core -> GPU task
    """
    try:
        dev = str((cfg or {}).get("device", "")).strip().lower()
    except Exception:
        dev = ""
    # Explicit runtime hint: Triton is assumed GPU-backed in this project.
    try:
        rt = str((cfg or {}).get("runtime", "")).strip().lower()
    except Exception:
        rt = ""
    if rt in ("triton", "triton-gpu", "triton_gpu"):
        return True
    if dev in ("cuda", "gpu", "auto"):
        return True
    # Fallback: GPU-heavy cores are usually GPU-bound unless explicitly cpu
    # Note: core_face_landmarks (MediaPipe) is typically CPU (TFLite/XNNPACK) in our baseline setup.
    if name in (
        "core_clip",
        "core_depth_midas",
        "core_object_detections",
        "core_optical_flow",
        # Semantic heads that call Triton CLIP text/image (GPU-backed in our deployments)
        "content_domain",
        "franchise_recognition",
    ):
        return True
    return False


def _device_used_for_component(name: str, cfg: dict) -> str:
    """
    Best-effort device string for manifest.
    Canonical values (MVP): "cpu" | "cuda" | "auto"
    """
    try:
        dev = str((cfg or {}).get("device", "")).strip().lower()
    except Exception:
        dev = ""
    # Explicit runtime hint: Triton is assumed GPU-backed in this project.
    try:
        rt = str((cfg or {}).get("runtime", "")).strip().lower()
    except Exception:
        rt = ""
    if rt in ("triton", "triton-gpu", "triton_gpu"):
        return "cuda"
    if dev in ("cpu", "cuda", "auto"):
        return dev
    if dev in ("gpu",):
        return "cuda"
    return "cuda" if _component_uses_gpu(name, cfg) else "cpu"

def _resolve_gpu_slots(global_cfg: dict) -> int:
    """
    Resolve max concurrent GPU tasks.
    Supported:
    - int (>=1)
    - "auto": 1 for small GPUs, 2 for >= ~20GB (best-effort)
    """
    raw = (global_cfg or {}).get("gpu_max_concurrent", "auto")
    if isinstance(raw, int):
        return max(1, int(raw))
    try:
        s = str(raw).strip().lower()
    except Exception:
        s = "auto"
    if s not in ("auto", ""):
        try:
            return max(1, int(s))
        except Exception:
            return 1
    mem = get_cuda_mem_info()
    if mem is None or mem.total_bytes <= 0:
        return 1
    # Very conservative: allow 2 concurrent GPU tasks only on big VRAM.
    total_gb = float(mem.total_bytes) / (1024.0 ** 3)
    return 2 if total_gb >= 19.0 else 1


def _run_component_subprocess(
    *,
    kind: str,  # "module"|"core"
    global_cfg: dict,
    name: str,
    cfg: dict,
    run_rs_path: str,
    gpu_sem: threading.Semaphore,
) -> tuple:
    """
    Run component in a subprocess with resource gating.
    Returns tuple: (ok, err, artifacts, status, notes, schema_version, producer_version, duration_ms)
    """
    root_path = global_cfg["root_path"]
    frames_dir = global_cfg["frames_dir"]
    rs_path = global_cfg["rs_path"]

    os.makedirs(rs_path, exist_ok=True)

    target = "modules" if kind == "module" else "core/model_process"
    cmd = _build_subprocess_cmd(root_path=root_path, name=name, target=target, frames_dir=frames_dir, rs_path=rs_path, cfg=cfg)

    needs_gpu = _component_uses_gpu(name, cfg)
    acquired = False
    started_at = _utc_iso_now()
    t0 = time.time()
    try:
        if needs_gpu:
            gpu_sem.acquire()
            acquired = True
            logger.info(f"VisualProcessor | main | {kind} {name} | GPU slot acquired")

        # Ensure repo-root packages (e.g., dp_models, dp_triton) are importable inside component venvs.
        env = os.environ.copy()
        repo_root = str(root_path)
        prev_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = repo_root if not prev_pp else (repo_root + os.pathsep + prev_pp)
        
        # Suppress MediaPipe verbose logs for core_face_landmarks
        if name == "core_face_landmarks":
            env["GLOG_minloglevel"] = "2"  # Suppress INFO, WARNING (keep ERROR, FATAL)
            env["GLOG_stderrthreshold"] = "2"  # Only ERROR and FATAL to stderr
            # Filter stderr to suppress MediaPipe logs while keeping real errors
            import subprocess as sp
            import threading
            import sys
            
            process = sp.Popen(
                cmd,
                env=env,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                text=True,
                bufsize=1,
            )
            
            # MediaPipe log patterns to suppress
            suppress_patterns = [
                "gl_context_egl.cc",
                "gl_context.cc",
                "inference_feedback_manager.cc",
                "Created TensorFlow Lite XNNPACK delegate",
                "All log messages before absl::InitializeLog()",
                "I0000",  # INFO level glog messages
                "W0000",  # WARNING level glog messages
            ]
            
            def filter_stderr():
                try:
                    for line in process.stderr:
                        # Suppress MediaPipe verbose logs
                        if any(pattern in line for pattern in suppress_patterns):
                            continue
                        # Keep real errors and our logs
                        sys.stderr.write(line)
                        sys.stderr.flush()
                except (BrokenPipeError, ValueError):
                    pass  # Process ended, stderr closed
            
            stderr_thread = threading.Thread(target=filter_stderr, daemon=True)
            stderr_thread.start()
            
            # Read stdout normally
            try:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            except (BrokenPipeError, ValueError):
                pass  # Process ended, stdout closed
            
            returncode = process.wait()
            stderr_thread.join(timeout=2.0)
            
            if returncode != 0:
                raise sp.CalledProcessError(returncode, cmd)
        else:
            subprocess.run(cmd, check=True, env=env)
        ok, err = True, None
    except Exception as e:
        logger.error(f"VisualProcessor | main | {kind} {name} | Error: {e}")
        ok, err = False, str(e)
    finally:
        duration_ms = int((time.time() - t0) * 1000)
        finished_at = _utc_iso_now()
        if acquired:
            try:
                gpu_sem.release()
            except Exception:
                pass
            logger.info(f"VisualProcessor | main | {kind} {name} | GPU slot released")

    # Collect artifacts + validate
    comp_dir = os.path.join(run_rs_path, name)
    artifacts = [{"path": p, "type": os.path.splitext(p)[1].lstrip(".")} for p in _find_latest_artifact(comp_dir)]

    status = "ok" if ok else "error"
    notes = None
    schema_version = None
    producer_version = None
    empty_reason = None

    npz_files = [a["path"] for a in artifacts if a["path"].lower().endswith(".npz")]
    if ok and npz_files:
        v_ok, issues, meta = validate_npz(npz_files[0])
        if not v_ok:
            status = "error"
            notes = "artifact validation failed: " + "; ".join(i.message for i in issues[:5])
        if isinstance(meta, dict):
            schema_version = meta.get("schema_version")
            producer_version = meta.get("producer_version")
            # If component produced a valid empty artifact, reflect it in manifest status.
            m_status = meta.get("status")
            if m_status in {"ok", "empty", "error"}:
                status = str(m_status)
            empty_reason = meta.get("empty_reason")
        
        # Generate render context if enabled (best-effort: don't fail if render fails)
        if ok and npz_files:
            try:
                from utils.renderer import render_component
                
                # Get render flags from config
                enable_render, enable_html_render = _get_component_render_flags(
                    name, cfg, default_enable_render=True, default_enable_html_render=True
                )
                
                if enable_render:
                    artifact_path = npz_files[0]
                    try:
                        render = render_component(
                            artifact_path,
                            name,
                            component_type=kind,
                            output_dir=comp_dir,
                            enable_render=enable_render,
                            enable_html_render=enable_html_render,
                        )
                        logger.info(f"VisualProcessor | Render generated for {name} (HTML: {enable_html_render})")
                    except Exception as e:
                        # Best-effort: do not fail run if render fails
                        logger.warning(f"VisualProcessor | Failed to generate render-context for {name}: {e}")
                else:
                    logger.debug(f"Render disabled for {name}, skipping render generation")
            except ImportError as e:
                logger.debug(f"Could not import renderer for {name}: {e}")
            except Exception as e:
                # Best-effort: do not fail run if render fails
                logger.debug(f"Render generation skipped for {name}: {e}")

    return (
        ok,
        err,
        artifacts,
        status,
        notes,
        schema_version,
        producer_version,
        empty_reason,
        started_at,
        finished_at,
        duration_ms,
    )

def run_module(global_cfg, module_name, module_cfg, run_rs_path: str, gpu_sem: threading.Semaphore):
    return _run_component_subprocess(
        kind="module",
        global_cfg=global_cfg,
        name=module_name,
        cfg=module_cfg,
        run_rs_path=run_rs_path,
        gpu_sem=gpu_sem,
    )


def run_core_provider(global_cfg, provider_name, provider_cfg, run_rs_path: str, gpu_sem: threading.Semaphore):
    return _run_component_subprocess(
        kind="core",
        global_cfg=global_cfg,
        name=provider_name,
        cfg=provider_cfg,
        run_rs_path=run_rs_path,
        gpu_sem=gpu_sem,
    )


def _process_single_video(
    config: dict,
    video_ctx: VideoContext,
    req_map: dict,
    enforce_requirements: bool,
    gpu_sem: threading.Semaphore,
) -> Dict[str, Any]:
    """
    Обрабатывает одно видео с изоляцией артефактов.
    
    Args:
        config: Конфигурация VisualProcessor
        video_ctx: VideoContext для видео
        req_map: Requirements map для проверки required компонентов
        enforce_requirements: Включить enforce requirements
        gpu_sem: GPU semaphore для gating
    
    Returns:
        Словарь с результатами обработки
    """
    video_id = video_ctx.video_id
    logger.info(f"VisualProcessor | _process_single_video | processing video: {video_id}")
    
    try:
        # Создаем временную конфигурацию для этого видео
        # с переопределением frames_dir и rs_path
        video_config = config.copy()
        g_config = video_config.get("global", {})
        g_config = g_config.copy()
        g_config["frames_dir"] = video_ctx.frames_dir
        g_config["rs_path"] = video_ctx.rs_path
        g_config["video_id"] = video_ctx.video_id
        g_config["platform_id"] = video_ctx.platform_id or g_config.get("platform_id", "youtube")
        if video_ctx.run_id:
            g_config["run_id"] = video_ctx.run_id
        video_config["global"] = g_config
        
        # Загружаем метаданные для валидации
        metadata = video_ctx.load_metadata()
        
        # Проверяем обязательные поля
        required_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
        missing = [k for k in required_keys if not metadata.get(k)]
        if missing:
            error_msg = f"missing required keys: {missing}"
            logger.error(f"VisualProcessor | _process_single_video | video {video_id} {error_msg}")
            return {
                "video_id": video_id,
                "status": "error",
                "error": error_msg,
            }
        
        # Derive run context для этого видео
        run_ctx = _derive_run_context(video_config)
        run_rs_path = video_ctx.rs_path
        os.makedirs(run_rs_path, exist_ok=True)
        
        # Создаем manifest для этого видео
        manifest_path = os.path.join(run_rs_path, "manifest.json")
        manifest = RunManifest(
            path=manifest_path,
            run_meta={
                **run_ctx,
                "frames_dir": video_ctx.frames_dir,
                "root_path": g_config.get("root_path"),
            },
        )
        manifest.flush()
        
        # Получаем список компонентов для обработки
        current_core = get_current_core_providers(video_config)
        current_modules = get_current_modules(video_config)  # Валидация core зависимостей выполняется внутри
        enabled_set = set(current_core) | set(current_modules)
        
        # Получаем execution order если есть
        exec_order = _execution_order(video_config)
        
        # Валидация exec_order если задан
        if exec_order:
            _validate_exec_order_deps(exec_order, enabled_set)
        
        # Обработка компонентов
        def _run_one_component(name: str) -> None:
            if name in current_core:
                provider_cfg = video_config.get(name, {})
                logger.info(f"VisualProcessor | _process_single_video | {video_id} | core_provider {name} start")
                (
                    ok,
                    err,
                    artifacts,
                    status,
                    notes,
                    schema_version,
                    producer_version,
                    empty_reason,
                    started_at,
                    finished_at,
                    duration_ms,
                ) = run_core_provider(g_config, name, provider_cfg, run_rs_path=run_rs_path, gpu_sem=gpu_sem)
                
                manifest.upsert_component(
                    ManifestComponent(
                        name=name,
                        kind="core",
                        status=status,
                        empty_reason=empty_reason,
                        started_at=started_at,
                        finished_at=finished_at,
                        duration_ms=duration_ms,
                        artifacts=artifacts,
                        error=err,
                        error_code=("component_failed" if status == "error" else None),
                        notes=notes,
                        producer_version=producer_version,
                        schema_version=schema_version,
                        device_used=_device_used_for_component(name, provider_cfg),
                    )
                )
                if status == "error":
                    logger.error(f"VisualProcessor | _process_single_video | {video_id} | core_provider {name} failed")
                    if enforce_requirements and _is_required(req_map, name):
                        logger.error(f"VisualProcessor | _process_single_video | {video_id} | required core_provider failed: {name}")
                        raise RuntimeError(f"Required core_provider {name} failed for video {video_id}")
                elif status == "empty":
                    logger.info(f"VisualProcessor | _process_single_video | {video_id} | core_provider {name} completed with empty status (empty_reason: {empty_reason})")
                return
            
            if name in current_modules:
                module_cfg = video_config.get(name) or {}
                if not isinstance(module_cfg, dict):
                    module_cfg = {}
                
                # Auto-inject triton_http_url from core_clip for modules that need it
                if name == "story_structure" and not module_cfg.get("triton_http_url"):
                    core_clip_cfg = video_config.get("core_clip", {})
                    if isinstance(core_clip_cfg, dict) and core_clip_cfg.get("triton_http_url"):
                        module_cfg = module_cfg.copy()
                        module_cfg["triton_http_url"] = core_clip_cfg["triton_http_url"]
                
                logger.info(f"VisualProcessor | _process_single_video | {video_id} | module {name} start")
                
                # Policy: detalize_face must not run if core_face_landmarks indicates no faces.
                if name == "detalize_face":
                    try:
                        core_path = os.path.join(run_rs_path, "core_face_landmarks", "landmarks.npz")
                        if os.path.isfile(core_path):
                            d = np.load(core_path, allow_pickle=True)
                            meta = d.get("meta")
                            if isinstance(meta, np.ndarray) and meta.dtype == object and meta.shape == ():
                                try:
                                    meta = meta.item()
                                except Exception:
                                    meta = None
                            status0 = meta.get("status") if isinstance(meta, dict) else None
                            empty0 = meta.get("empty_reason") if isinstance(meta, dict) else None
                            has_any_face = meta.get("has_any_face") if isinstance(meta, dict) else None
                            if status0 == "empty" and (empty0 == "no_faces_in_video" or has_any_face is False):
                                ts_now = _utc_iso_now()
                                manifest.upsert_component(
                                    ManifestComponent(
                                        name=name,
                                        kind="module",
                                        status="empty",
                                        empty_reason="no_faces_in_video",
                                        started_at=ts_now,
                                        finished_at=ts_now,
                                        duration_ms=0,
                                        artifacts=[],
                                        error=None,
                                        error_code=None,
                                        notes="skipped_by_orchestrator_no_faces",
                                        producer_version=None,
                                        schema_version=None,
                                        device_used=_device_used_for_component(name, module_cfg),
                                    )
                                )
                                logger.info(f"VisualProcessor | _process_single_video | {video_id} | detalize_face skipped (no_faces_in_video)")
                                return
                    except Exception:
                        pass
                
                (
                    ok,
                    err,
                    artifacts,
                    status,
                    notes,
                    schema_version,
                    producer_version,
                    empty_reason,
                    started_at,
                    finished_at,
                    duration_ms,
                ) = run_module(g_config, name, module_cfg, run_rs_path, gpu_sem)
                manifest.upsert_component(
                    ManifestComponent(
                        name=name,
                        kind="module",
                        status=status,
                        empty_reason=empty_reason,
                        started_at=started_at,
                        finished_at=finished_at,
                        duration_ms=duration_ms,
                        artifacts=artifacts,
                        error=err,
                        error_code=("component_failed" if status == "error" else None),
                        notes=notes,
                        producer_version=producer_version,
                        schema_version=schema_version,
                        device_used=_device_used_for_component(name, module_cfg),
                    )
                )
                if not ok:
                    logger.error(f"VisualProcessor | _process_single_video | {video_id} | module {name} failed")
                    if enforce_requirements and status == "error" and _is_required(req_map, name):
                        logger.error(f"VisualProcessor | _process_single_video | {video_id} | required module failed: {name}")
                        raise RuntimeError(f"Required module {name} failed for video {video_id}")
                return
        
        # Выполняем компоненты в правильном порядке
        if exec_order:
            logger.info(f"VisualProcessor | _process_single_video | {video_id} | executing by DAG order (len={len(exec_order)})")
            for name in exec_order:
                if name in enabled_set:
                    _run_one_component(name)
            # Run any remaining enabled components not covered by exec_order
            remaining = [n for n in sorted(enabled_set) if n not in exec_order]
            if remaining:
                logger.warning(f"VisualProcessor | _process_single_video | {video_id} | exec_order missing enabled components: {remaining}")
                for n in remaining:
                    _run_one_component(n)
        else:
            # Backward-compatible behavior: keep old module scheduling (parallelism).
            if current_core:
                for provider in current_core:
                    _run_one_component(provider)
            
            # Modules can be run in parallel (intra-video), with GPU gating.
            if current_modules:
                has_deps = any((MODULE_DEPS.get(m) or []) for m in current_modules)
                if has_deps:
                    logger.info(f"VisualProcessor | _process_single_video | {video_id} | module deps detected → running modules sequentially")
                    for module in current_modules:
                        _run_one_component(module)
                else:
                    max_parallel_modules = int(g_config.get("max_parallel_modules", 1) or 1)
                    max_workers = max(1, int(max_parallel_modules))
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        fut_by_name = {}
                        for module in current_modules:
                            module_cfg = video_config.get(module)
                            if module_cfg is None:
                                raise ValueError(f"❌ Config entry for module '{module}' not found in YAML")
                            logger.info(f"VisualProcessor | _process_single_video | {video_id} | scheduling module: {module}")
                            fut = ex.submit(run_module, g_config, module, module_cfg, run_rs_path, gpu_sem)
                            fut_by_name[fut] = module
                        
                        for fut in as_completed(list(fut_by_name.keys())):
                            module = fut_by_name[fut]
                            try:
                                (
                                    ok,
                                    err,
                                    artifacts,
                                    status,
                                    notes,
                                    schema_version,
                                    producer_version,
                                    empty_reason,
                                    started_at,
                                    finished_at,
                                    duration_ms,
                                ) = fut.result()
                                manifest.upsert_component(
                                    ManifestComponent(
                                        name=module,
                                        kind="module",
                                        status=status,
                                        empty_reason=empty_reason,
                                        started_at=started_at,
                                        finished_at=finished_at,
                                        duration_ms=duration_ms,
                                        artifacts=artifacts,
                                        error=err,
                                        error_code=("component_failed" if status == "error" else None),
                                        notes=notes,
                                        producer_version=producer_version,
                                        schema_version=schema_version,
                                        device_used=_device_used_for_component(module, video_config.get(module, {})),
                                    )
                                )
                                if not ok:
                                    logger.error(f"VisualProcessor | _process_single_video | {video_id} | module {module} failed")
                                    if enforce_requirements and status == "error" and _is_required(req_map, module):
                                        logger.error(f"VisualProcessor | _process_single_video | {video_id} | required module failed: {module}")
                                        raise RuntimeError(f"Required module {module} failed for video {video_id}")
                            except Exception as e:
                                logger.exception(f"VisualProcessor | _process_single_video | {video_id} | module {module} exception: {e}")
                                if enforce_requirements and _is_required(req_map, module):
                                    raise
        
        # Flush manifest
        manifest.flush()
        
        logger.info(f"VisualProcessor | _process_single_video | {video_id} | completed successfully")
        return {
            "video_id": video_id,
            "status": "ok",
        }
        
    except Exception as e:
        logger.exception(
            f"VisualProcessor | _process_single_video | error processing video {video_id}: {e}"
        )
        return {
            "video_id": video_id,
            "status": "error",
            "error": str(e),
        }


def run_batch(
    config: dict,
    video_contexts: List[VideoContext],
    *,
    max_video_workers: Optional[int] = None,
    enable_video_parallel: bool = False,
    enable_gpu_batching: bool = False,
    enable_cpu_parallel: bool = False,
) -> List[Dict[str, Any]]:
    """
    Батчевая обработка нескольких видео.
    
    Stage 1: Реализация с изоляцией артефактов (последовательная обработка).
    Stage 4: Двухуровневая параллельность и GPU batching.
    
    Args:
        config: Конфигурация VisualProcessor (YAML config dict)
        video_contexts: Список VideoContext для каждого видео
        max_video_workers: Количество параллельных воркеров для видео (None = auto)
        enable_video_parallel: Включить параллельную обработку нескольких видео
        enable_gpu_batching: Включить GPU batching для кадров (Stage 2+)
        enable_cpu_parallel: Включить CPU параллелизм (Stage 4+)
    
    Returns:
        Список словарей с результатами обработки для каждого видео
    """
    if not video_contexts:
        return []
    
    logger.info(
        f"VisualProcessor | run_batch | processing {len(video_contexts)} videos"
    )
    
    # Получаем requirements map и enforce flag
    req_map = _requirements_map(config)
    enforce_requirements = bool(req_map)
    
    # GPU semaphore для gating (Stage 1: один семафор для всех видео, Stage 4: оптимизировать)
    gpu_slots = _resolve_gpu_slots(config.get("global", {}))
    gpu_sem = threading.Semaphore(value=max(1, int(gpu_slots)))
    
    # Stage 1: Последовательная обработка каждого видео с изоляцией артефактов
    # Stage 2-3: GPU batching для core_clip и core_depth_midas (если включен)
    
    # Проверяем, нужно ли обрабатывать core_clip батчем
    current_core = get_current_core_providers(config)
    process_core_clip_batch_mode = (
        enable_gpu_batching and 
        "core_clip" in current_core and
        len(video_contexts) > 1  # Batch имеет смысл только для нескольких видео
    )
    
    # Проверяем, нужно ли обрабатывать core_depth_midas батчем
    process_core_depth_midas_batch_mode = (
        enable_gpu_batching and
        "core_depth_midas" in current_core and
        len(video_contexts) > 1  # Batch имеет смысл только для нескольких видео
    )
    
    process_core_optical_flow_batch_mode = (
        enable_gpu_batching and
        "core_optical_flow" in current_core and
        len(video_contexts) > 1  # Batch имеет смысл только для нескольких видео
    )
    
    process_core_object_detections_batch_mode = (
        enable_gpu_batching and
        "core_object_detections" in current_core and
        len(video_contexts) > 1  # Batch имеет смысл только для нескольких видео
    )
    
    # Проверяем, нужно ли обрабатывать ocr_extractor батчем (CPU parallelism)
    process_ocr_extractor_batch_mode = (
        enable_cpu_parallel and
        "ocr_extractor" in current_core and
        len(video_contexts) > 1  # Batch имеет смысл только для нескольких видео
    )
    
    # Проверяем, нужно ли обрабатывать core_face_landmarks батчем (CPU parallelism)
    process_core_face_landmarks_batch_mode = (
        enable_cpu_parallel and
        "core_face_landmarks" in current_core and
        len(video_contexts) > 1  # Batch имеет смысл только для нескольких видео
    )
    
    # Проверяем, нужно ли обрабатывать content_domain батчем (зависит от core_clip)
    process_content_domain_batch_mode = (
        enable_gpu_batching and
        "content_domain" in current_core and
        len(video_contexts) > 1  # Batch имеет смысл только для нескольких видео
    )
    
    # Проверяем, нужно ли обрабатывать franchise_recognition батчем (зависит от core_clip)
    process_franchise_recognition_batch_mode = (
        enable_gpu_batching and
        "franchise_recognition" in current_core and
        len(video_contexts) > 1  # Batch имеет смысл только для нескольких видео
    )
    
    # Проверяем, нужно ли обрабатывать brand_semantics батчем (зависит от core_object_detections)
    process_brand_semantics_batch_mode = (
        enable_gpu_batching and
        "brand_semantics" in current_core and
        len(video_contexts) > 1  # Batch имеет смысл только для нескольких видео
    )
    
    # Проверяем, нужно ли обрабатывать car_semantics батчем (зависит от core_object_detections)
    process_car_semantics_batch_mode = (
        enable_gpu_batching and
        "car_semantics" in current_core and
        len(video_contexts) > 1  # Batch имеет смысл только для нескольких видео
    )
    
    # Проверяем, нужно ли обрабатывать face_identity батчем (зависит от core_face_landmarks)
    process_face_identity_batch_mode = (
        enable_gpu_batching and
        ("face_identity" in current_core or "core_face_identity" in current_core) and
        len(video_contexts) > 1  # Batch имеет смысл только для нескольких видео
    )
    
    # Проверяем, нужно ли обрабатывать place_semantics батчем (зависит от core_object_detections, core_clip)
    process_place_semantics_batch_mode = (
        enable_gpu_batching and
        ("place_semantics" in current_core or "core_place_semantics" in current_core) and
        len(video_contexts) > 1  # Batch имеет смысл только для нескольких видео
    )
    
    # Проверяем, нужно ли обрабатывать cut_detection батчем (зависит от core_optical_flow)
    # cut_detection - это модуль, поэтому проверяем в modules
    current_modules = get_current_modules(config)
    process_cut_detection_batch_mode = (
        (enable_cpu_parallel or enable_gpu_batching) and  # cut_detection может использовать CPU или GPU (CLIP)
        "cut_detection" in current_modules and
        len(video_contexts) > 1  # Batch имеет смысл только для нескольких видео
    )
    
    process_scene_classification_batch_mode = (
        (enable_cpu_parallel or enable_gpu_batching) and  # scene_classification может использовать CPU или GPU (Triton)
        "scene_classification" in current_modules and
        len(video_contexts) > 1  # Batch имеет смысл только для нескольких видео
    )
    
    process_video_pacing_batch_mode = (
        enable_cpu_parallel and  # video_pacing использует CPU
        "video_pacing" in current_modules and
        len(video_contexts) > 1  # Batch имеет смысл только для нескольких видео
    )
    
    # Обрабатываем core_clip батчем если нужно
    core_clip_results = {}
    if process_core_clip_batch_mode:
        logger.info("VisualProcessor | run_batch | processing core_clip in batch mode")
        try:
            core_clip_config = config.get("core_clip", {})
            runtime = core_clip_config.get("runtime", "inprocess")
            batch_size = core_clip_config.get("batch_size", 16)
            max_frames_per_batch = config.get("global", {}).get("batch_processing", {}).get("max_frames_per_gpu_batch")
            
            clip_results = process_core_clip_batch(
                video_contexts=video_contexts,
                config=core_clip_config,
                max_frames_per_batch=max_frames_per_batch,
                runtime=runtime,
                batch_size=batch_size,
            )
            
            # Сохраняем результаты для обновления manifest позже
            for result in clip_results:
                video_id = result.get("video_id")
                if video_id:
                    core_clip_results[video_id] = result
            
            logger.info(
                f"VisualProcessor | run_batch | core_clip batch completed: "
                f"{len([r for r in clip_results if r.get('status') == 'ok'])}/{len(clip_results)} successful"
            )
        except Exception as e:
            logger.exception(f"VisualProcessor | run_batch | core_clip batch failed: {e}")
            # Продолжаем с обычной обработкой для всех видео
    
    # Обрабатываем content_domain батчем если нужно (зависит от core_clip, поэтому после core_clip)
    content_domain_results = {}
    if process_content_domain_batch_mode:
        logger.info("VisualProcessor | run_batch | processing content_domain in batch mode")
        try:
            content_domain_config = config.get("content_domain", {})
            max_frames_per_batch = config.get("global", {}).get("batch_processing", {}).get("max_frames_per_gpu_batch")
            
            domain_results = process_content_domain_batch(
                video_contexts=video_contexts,
                config=content_domain_config,
                max_frames_per_batch=max_frames_per_batch,
            )
            
            # Сохраняем результаты для обновления manifest позже
            for result in domain_results:
                video_id = result.get("video_id")
                if video_id:
                    content_domain_results[video_id] = result
            
            logger.info(
                f"VisualProcessor | run_batch | content_domain batch completed: "
                f"{len([r for r in domain_results if r.get('status') == 'ok'])}/{len(domain_results)} successful"
            )
        except Exception as e:
            logger.exception(f"VisualProcessor | run_batch | content_domain batch failed: {e}")
            # Продолжаем с обычной обработкой для всех видео
    
    # Обрабатываем franchise_recognition батчем если нужно (зависит от core_clip, поэтому после core_clip)
    franchise_recognition_results = {}
    if process_franchise_recognition_batch_mode:
        logger.info("VisualProcessor | run_batch | processing franchise_recognition in batch mode")
        try:
            franchise_recognition_config = config.get("franchise_recognition", {})
            max_frames_per_batch = config.get("global", {}).get("batch_processing", {}).get("max_frames_per_gpu_batch")
            
            franchise_results = process_franchise_recognition_batch(
                video_contexts=video_contexts,
                config=franchise_recognition_config,
                max_frames_per_batch=max_frames_per_batch,
            )
            
            # Сохраняем результаты для обновления manifest позже
            for result in franchise_results:
                video_id = result.get("video_id")
                if video_id:
                    franchise_recognition_results[video_id] = result
            
            logger.info(
                f"VisualProcessor | run_batch | franchise_recognition batch completed: "
                f"{len([r for r in franchise_results if r.get('status') == 'ok'])}/{len(franchise_results)} successful"
            )
        except Exception as e:
            logger.exception(f"VisualProcessor | run_batch | franchise_recognition batch failed: {e}")
            # Продолжаем с обычной обработкой для всех видео
    
    # Обрабатываем brand_semantics батчем если нужно (зависит от core_object_detections)
    brand_semantics_results = {}
    if process_brand_semantics_batch_mode:
        logger.info("VisualProcessor | run_batch | processing brand_semantics in batch mode")
        try:
            brand_semantics_config = config.get("brand_semantics", {})
            max_frames_per_batch = config.get("global", {}).get("batch_processing", {}).get("max_frames_per_gpu_batch")
            batch_size = brand_semantics_config.get("batch_size", 16)
            
            brand_results = process_brand_semantics_batch(
                video_contexts=video_contexts,
                config=brand_semantics_config,
                max_frames_per_batch=max_frames_per_batch,
                batch_size=batch_size,
            )
            
            # Сохраняем результаты для обновления manifest позже
            for result in brand_results:
                video_id = result.get("video_id")
                if video_id:
                    brand_semantics_results[video_id] = result
            
            logger.info(
                f"VisualProcessor | run_batch | brand_semantics batch completed: "
                f"{len([r for r in brand_results if r.get('status') == 'ok'])}/{len(brand_results)} successful"
            )
        except Exception as e:
            logger.exception(f"VisualProcessor | run_batch | brand_semantics batch failed: {e}")
            # Продолжаем с обычной обработкой для всех видео
    
    # Обрабатываем car_semantics батчем если нужно (зависит от core_object_detections)
    car_semantics_results = {}
    if process_car_semantics_batch_mode:
        logger.info("VisualProcessor | run_batch | processing car_semantics in batch mode")
        try:
            car_semantics_config = config.get("car_semantics", {})
            max_frames_per_batch = config.get("global", {}).get("batch_processing", {}).get("max_frames_per_gpu_batch")
            batch_size = car_semantics_config.get("batch_size", 16)
            
            car_results = process_car_semantics_batch(
                video_contexts=video_contexts,
                config=car_semantics_config,
                max_frames_per_batch=max_frames_per_batch,
                batch_size=batch_size,
            )
            
            # Сохраняем результаты для обновления manifest позже
            for result in car_results:
                video_id = result.get("video_id")
                if video_id:
                    car_semantics_results[video_id] = result
            
            logger.info(
                f"VisualProcessor | run_batch | car_semantics batch completed: "
                f"{len([r for r in car_results if r.get('status') == 'ok'])}/{len(car_results)} successful"
            )
        except Exception as e:
            logger.exception(f"VisualProcessor | run_batch | car_semantics batch failed: {e}")
            # Продолжаем с обычной обработкой для всех видео
    
    # Обрабатываем face_identity батчем если нужно (зависит от core_face_landmarks)
    face_identity_results = {}
    if process_face_identity_batch_mode:
        logger.info("VisualProcessor | run_batch | processing face_identity in batch mode")
        try:
            face_identity_config = config.get("face_identity", {}) or config.get("core_face_identity", {})
            max_frames_per_batch = config.get("global", {}).get("batch_processing", {}).get("max_frames_per_gpu_batch")
            batch_size = face_identity_config.get("batch_size", 16)
            
            face_results = process_face_identity_batch(
                video_contexts=video_contexts,
                config=face_identity_config,
                max_frames_per_batch=max_frames_per_batch,
                batch_size=batch_size,
            )
            
            # Сохраняем результаты для обновления manifest позже
            for result in face_results:
                video_id = result.get("video_id")
                if video_id:
                    face_identity_results[video_id] = result
            
            logger.info(
                f"VisualProcessor | run_batch | face_identity batch completed: "
                f"{len([r for r in face_results if r.get('status') == 'ok'])}/{len(face_results)} successful"
            )
        except Exception as e:
            logger.exception(f"VisualProcessor | run_batch | face_identity batch failed: {e}")
            # Продолжаем с обычной обработкой для всех видео
    
    # Обрабатываем place_semantics батчем если нужно (зависит от core_object_detections, core_clip)
    place_semantics_results = {}
    if process_place_semantics_batch_mode:
        logger.info("VisualProcessor | run_batch | processing place_semantics in batch mode")
        try:
            place_semantics_config = config.get("place_semantics", {}) or config.get("core_place_semantics", {})
            max_frames_per_batch = config.get("global", {}).get("batch_processing", {}).get("max_frames_per_gpu_batch")
            batch_size = place_semantics_config.get("batch_size", 16)
            
            place_results = process_place_semantics_batch(
                video_contexts=video_contexts,
                config=place_semantics_config,
                max_frames_per_batch=max_frames_per_batch,
                batch_size=batch_size,
            )
            
            # Сохраняем результаты для обновления manifest позже
            for result in place_results:
                video_id = result.get("video_id")
                if video_id:
                    place_semantics_results[video_id] = result
            
            logger.info(
                f"VisualProcessor | run_batch | place_semantics batch completed: "
                f"{len([r for r in place_results if r.get('status') == 'ok'])}/{len(place_results)} successful"
            )
        except Exception as e:
            logger.exception(f"VisualProcessor | run_batch | place_semantics batch failed: {e}")
            # Продолжаем с обычной обработкой для всех видео
    
    # Обрабатываем core_depth_midas батчем если нужно
    core_depth_midas_results = {}
    core_depth_midas_config = None
    if process_core_depth_midas_batch_mode:
        logger.info("VisualProcessor | run_batch | processing core_depth_midas in batch mode")
        try:
            core_depth_midas_config = config.get("core_depth_midas", {})
            batch_size = core_depth_midas_config.get("batch_size", 16)
            max_frames_per_batch = config.get("global", {}).get("batch_processing", {}).get("max_frames_per_gpu_batch")
            
            depth_results = process_core_depth_midas_batch(
                video_contexts=video_contexts,
                config=core_depth_midas_config,
                max_frames_per_batch=max_frames_per_batch,
                batch_size=batch_size,
            )
            
            # Сохраняем результаты для обновления manifest позже
            for result in depth_results:
                video_id = result.get("video_id")
                if video_id:
                    core_depth_midas_results[video_id] = result
            
            logger.info(
                f"VisualProcessor | run_batch | core_depth_midas batch completed: "
                f"{len([r for r in depth_results if r.get('status') == 'ok'])}/{len(depth_results)} successful"
            )
        except Exception as e:
            logger.exception(f"VisualProcessor | run_batch | core_depth_midas batch failed: {e}")
            # Продолжаем с обычной обработкой для всех видео
    
    # Обрабатываем core_optical_flow батчем если нужно
    core_optical_flow_results = {}
    if process_core_optical_flow_batch_mode:
        logger.info("VisualProcessor | run_batch | processing core_optical_flow in batch mode")
        try:
            core_optical_flow_config = config.get("core_optical_flow", {})
            batch_size = core_optical_flow_config.get("batch_size", 16)
            max_frames_per_batch = config.get("global", {}).get("batch_processing", {}).get("max_frames_per_gpu_batch")
            
            flow_results = process_core_optical_flow_batch(
                video_contexts=video_contexts,
                config=core_optical_flow_config,
                max_frames_per_batch=max_frames_per_batch,
                batch_size=batch_size,
            )
            
            # Сохраняем результаты для обновления manifest позже
            for result in flow_results:
                video_id = result.get("video_id")
                if video_id:
                    core_optical_flow_results[video_id] = result
            
            logger.info(
                f"VisualProcessor | run_batch | core_optical_flow batch completed: "
                f"{len([r for r in flow_results if r.get('status') == 'ok'])}/{len(flow_results)} successful"
            )
        except Exception as e:
            logger.exception(f"VisualProcessor | run_batch | core_optical_flow batch failed: {e}")
            # Продолжаем с обычной обработкой для всех видео
    
    # Обрабатываем core_object_detections батчем если нужно
    core_object_detections_results = {}
    if process_core_object_detections_batch_mode:
        logger.info("VisualProcessor | run_batch | processing core_object_detections in batch mode")
        try:
            core_object_detections_config = config.get("core_object_detections", {})
            batch_size = core_object_detections_config.get("batch_size", 1)
            max_frames_per_batch = config.get("global", {}).get("batch_processing", {}).get("max_frames_per_gpu_batch")
            
            detections_results = process_core_object_detections_batch(
                video_contexts=video_contexts,
                config=core_object_detections_config,
                max_frames_per_batch=max_frames_per_batch,
                batch_size=batch_size,
            )
            
            # Сохраняем результаты для обновления manifest позже
            for result in detections_results:
                video_id = result.get("video_id")
                if video_id:
                    core_object_detections_results[video_id] = result
            
            logger.info(
                f"VisualProcessor | run_batch | core_object_detections batch completed: "
                f"{len([r for r in detections_results if r.get('status') == 'ok'])}/{len(detections_results)} successful"
            )
        except Exception as e:
            logger.exception(f"VisualProcessor | run_batch | core_object_detections batch failed: {e}")
            # Продолжаем с обычной обработкой для всех видео
    
    # Обрабатываем ocr_extractor батчем если нужно (CPU parallelism)
    ocr_extractor_results = {}
    if process_ocr_extractor_batch_mode:
        logger.info("VisualProcessor | run_batch | processing ocr_extractor in batch mode")
        try:
            ocr_extractor_config = config.get("ocr_extractor", {})
            max_workers = config.get("global", {}).get("batch_processing", {}).get("max_video_workers")
            
            ocr_results = process_ocr_extractor_batch(
                video_contexts=video_contexts,
                config=ocr_extractor_config,
                max_workers=max_workers,
            )
            
            # Сохраняем результаты для обновления manifest позже
            for result in ocr_results:
                video_id = result.get("video_id")
                if video_id:
                    ocr_extractor_results[video_id] = result
            
            logger.info(
                f"VisualProcessor | run_batch | ocr_extractor batch completed: "
                f"{len([r for r in ocr_results if r.get('status') == 'ok'])}/{len(ocr_results)} successful"
            )
        except Exception as e:
            logger.exception(f"VisualProcessor | run_batch | ocr_extractor batch failed: {e}")
            # Продолжаем с обычной обработкой для всех видео
    
    # Обрабатываем core_face_landmarks батчем если нужно (CPU parallelism)
    core_face_landmarks_results = {}
    if process_core_face_landmarks_batch_mode:
        logger.info("VisualProcessor | run_batch | processing core_face_landmarks in batch mode")
        try:
            core_face_landmarks_config = config.get("core_face_landmarks", {})
            num_workers = config.get("global", {}).get("batch_processing", {}).get("max_video_workers", 2)
            max_frames_per_batch = config.get("global", {}).get("batch_processing", {}).get("max_frames_per_gpu_batch")
            
            landmarks_results = process_core_face_landmarks_batch(
                video_contexts=video_contexts,
                config=core_face_landmarks_config,
                max_frames_per_batch=max_frames_per_batch,
                num_workers=num_workers,
            )
            
            # Сохраняем результаты для обновления manifest позже
            for result in landmarks_results:
                video_id = result.get("video_id")
                if video_id:
                    core_face_landmarks_results[video_id] = result
            
            logger.info(
                f"VisualProcessor | run_batch | core_face_landmarks batch completed: "
                f"{len([r for r in landmarks_results if r.get('status') == 'ok'])}/{len(landmarks_results)} successful"
            )
        except Exception as e:
            logger.exception(f"VisualProcessor | run_batch | core_face_landmarks batch failed: {e}")
            # Продолжаем с обычной обработкой для всех видео
    
    # Обрабатываем cut_detection батчем если нужно (зависит от core_optical_flow)
    cut_detection_results = {}
    if process_cut_detection_batch_mode:
        logger.info("VisualProcessor | run_batch | processing cut_detection in batch mode")
        try:
            cut_detection_config = config.get("cut_detection", {})
            max_video_workers = config.get("global", {}).get("batch_processing", {}).get("max_video_workers")
            
            cut_results = process_cut_detection_batch(
                video_contexts=video_contexts,
                config=cut_detection_config,
                max_video_workers=max_video_workers,
            )
            
            # Сохраняем результаты для обновления manifest позже
            for result in cut_results:
                video_id = result.get("video_id")
                if video_id:
                    cut_detection_results[video_id] = result
            
            logger.info(
                f"VisualProcessor | run_batch | cut_detection batch completed: "
                f"{len([r for r in cut_results if r.get('status') == 'ok'])}/{len(cut_results)} successful"
            )
        except Exception as e:
            logger.exception(f"VisualProcessor | run_batch | cut_detection batch failed: {e}")
            # Продолжаем с обычной обработкой для всех видео
    
    # Обрабатываем scene_classification батчем если нужно
    scene_classification_results = {}
    if process_scene_classification_batch_mode:
        logger.info("VisualProcessor | run_batch | processing scene_classification in batch mode")
        try:
            scene_classification_config = config.get("scene_classification", {})
            max_video_workers = config.get("global", {}).get("batch_processing", {}).get("max_video_workers")
            
            scene_results = process_scene_classification_batch(
                video_contexts=video_contexts,
                config=scene_classification_config,
                max_video_workers=max_video_workers,
            )
            
            # Сохраняем результаты для обновления manifest позже
            for result in scene_results:
                video_id = result.get("video_id")
                if video_id:
                    scene_classification_results[video_id] = result
            
            logger.info(
                f"VisualProcessor | run_batch | scene_classification batch completed: "
                f"{len([r for r in scene_results if r.get('status') == 'ok'])}/{len(scene_results)} successful"
            )
        except Exception as e:
            logger.exception(f"VisualProcessor | run_batch | scene_classification batch failed: {e}")
            # Продолжаем с обычной обработкой для всех видео
    
    # Обрабатываем video_pacing батчем если нужно
    video_pacing_results = {}
    if process_video_pacing_batch_mode:
        logger.info("VisualProcessor | run_batch | processing video_pacing in batch mode")
        try:
            video_pacing_config = config.get("video_pacing", {})
            max_video_workers = config.get("global", {}).get("batch_processing", {}).get("max_video_workers")
            
            pacing_results = process_video_pacing_batch(
                video_contexts=video_contexts,
                config=video_pacing_config,
                max_video_workers=max_video_workers,
            )
            
            # Сохраняем результаты для обновления manifest позже
            for result in pacing_results:
                video_id = result.get("video_id")
                if video_id:
                    video_pacing_results[video_id] = result
            
            logger.info(
                f"VisualProcessor | run_batch | video_pacing batch completed: "
                f"{len([r for r in pacing_results if r.get('status') == 'ok'])}/{len(pacing_results)} successful"
            )
        except Exception as e:
            logger.exception(f"VisualProcessor | run_batch | video_pacing batch failed: {e}")
            # Продолжаем с обычной обработкой для всех видео
    
    # Обрабатываем остальные компоненты для каждого видео
    # Если core_clip уже обработан батчем, пропускаем его в _process_single_video
    batch_results = []
    
    for video_ctx in video_contexts:
        # Создаем временную конфигурацию без компонентов, которые уже обработаны батчем
        video_config = config.copy()
        if process_core_clip_batch_mode and video_ctx.video_id in core_clip_results:
            # Временно отключаем core_clip для этого видео
            core_providers = video_config.get("core_providers", {})
            if isinstance(core_providers, dict):
                core_providers = core_providers.copy()
                core_providers["core_clip"] = False
                video_config["core_providers"] = core_providers
        
        if process_core_depth_midas_batch_mode and video_ctx.video_id in core_depth_midas_results:
            # Временно отключаем core_depth_midas для этого видео
            core_providers = video_config.get("core_providers", {})
            if isinstance(core_providers, dict):
                core_providers = core_providers.copy()
                core_providers["core_depth_midas"] = False
                video_config["core_providers"] = core_providers
        
        if process_core_optical_flow_batch_mode and video_ctx.video_id in core_optical_flow_results:
            # Временно отключаем core_optical_flow для этого видео
            core_providers = video_config.get("core_providers", {})
            if isinstance(core_providers, dict):
                core_providers = core_providers.copy()
                core_providers["core_optical_flow"] = False
                video_config["core_providers"] = core_providers
        
        if process_core_object_detections_batch_mode and video_ctx.video_id in core_object_detections_results:
            # Временно отключаем core_object_detections для этого видео
            core_providers = video_config.get("core_providers", {})
            if isinstance(core_providers, dict):
                core_providers = core_providers.copy()
                core_providers["core_object_detections"] = False
                video_config["core_providers"] = core_providers
        
        if process_core_face_landmarks_batch_mode and video_ctx.video_id in core_face_landmarks_results:
            # Временно отключаем core_face_landmarks для этого видео
            core_providers = video_config.get("core_providers", {})
            if isinstance(core_providers, dict):
                core_providers = core_providers.copy()
                core_providers["core_face_landmarks"] = False
                video_config["core_providers"] = core_providers
        
        if process_content_domain_batch_mode and video_ctx.video_id in content_domain_results:
            # Временно отключаем content_domain для этого видео
            core_providers = video_config.get("core_providers", {})
            if isinstance(core_providers, dict):
                core_providers = core_providers.copy()
                core_providers["content_domain"] = False
                video_config["core_providers"] = core_providers
        
        if process_franchise_recognition_batch_mode and video_ctx.video_id in franchise_recognition_results:
            # Временно отключаем franchise_recognition для этого видео
            core_providers = video_config.get("core_providers", {})
            if isinstance(core_providers, dict):
                core_providers = core_providers.copy()
                core_providers["franchise_recognition"] = False
                video_config["core_providers"] = core_providers
        
        if process_place_semantics_batch_mode and video_ctx.video_id in place_semantics_results:
            # Временно отключаем place_semantics для этого видео
            core_providers = video_config.get("core_providers", {})
            if isinstance(core_providers, dict):
                core_providers = core_providers.copy()
                core_providers["place_semantics"] = False
                core_providers["core_place_semantics"] = False  # Legacy name support
                video_config["core_providers"] = core_providers
        
        if process_ocr_extractor_batch_mode and video_ctx.video_id in ocr_extractor_results:
            # Временно отключаем ocr_extractor для этого видео
            core_providers = video_config.get("core_providers", {})
            if isinstance(core_providers, dict):
                core_providers = core_providers.copy()
                core_providers["ocr_extractor"] = False
                video_config["core_providers"] = core_providers
        
        if process_core_depth_midas_batch_mode and video_ctx.video_id in core_depth_midas_results:
            # Временно отключаем core_depth_midas для этого видео
            core_providers = video_config.get("core_providers", {})
            if isinstance(core_providers, dict):
                core_providers = core_providers.copy()
                core_providers["core_depth_midas"] = False
                video_config["core_providers"] = core_providers
        
        if process_core_optical_flow_batch_mode and video_ctx.video_id in core_optical_flow_results:
            # Временно отключаем core_optical_flow для этого видео
            core_providers = video_config.get("core_providers", {})
            if isinstance(core_providers, dict):
                core_providers = core_providers.copy()
                core_providers["core_optical_flow"] = False
                video_config["core_providers"] = core_providers
        
        if process_cut_detection_batch_mode and video_ctx.video_id in cut_detection_results:
            # Временно отключаем cut_detection для этого видео
            modules = video_config.get("modules", {})
            if isinstance(modules, dict):
                modules = modules.copy()
                modules["cut_detection"] = False
                video_config["modules"] = modules
        
        if process_scene_classification_batch_mode and video_ctx.video_id in scene_classification_results:
            # Временно отключаем scene_classification для этого видео
            modules = video_config.get("modules", {})
            if isinstance(modules, dict):
                modules = modules.copy()
                modules["scene_classification"] = False
                video_config["modules"] = modules
        
        if process_video_pacing_batch_mode and video_ctx.video_id in video_pacing_results:
            # Временно отключаем video_pacing для этого видео
            modules = video_config.get("modules", {})
            if isinstance(modules, dict):
                modules = modules.copy()
                modules["video_pacing"] = False
                video_config["modules"] = modules
        
        result = _process_single_video(
            config=video_config,
            video_ctx=video_ctx,
            req_map=req_map,
            enforce_requirements=enforce_requirements,
            gpu_sem=gpu_sem,
        )
        
        # Обновляем manifest для компонентов, обработанных батчем
        run_rs_path = video_ctx.rs_path
        manifest_path = os.path.join(run_rs_path, "manifest.json")
        
        if os.path.exists(manifest_path):
            try:
                manifest = RunManifest(path=manifest_path)
                manifest.load()
                
                # Обновляем manifest для core_clip если он был обработан батчем
                if process_core_clip_batch_mode and video_ctx.video_id in core_clip_results:
                    clip_result = core_clip_results[video_ctx.video_id]
                    
                    if clip_result.get("status") == "ok":
                        # Находим артефакты
                        component_dir = os.path.join(run_rs_path, "core_clip")
                        artifacts = []
                        if os.path.exists(component_dir):
                            npz_files = list(Path(component_dir).glob("*.npz"))
                            if npz_files:
                                artifacts = [{"path": str(f), "type": "npz"} for f in npz_files]
                        
                        manifest.upsert_component(
                            ManifestComponent(
                                name="core_clip",
                                kind="core",
                                status="ok",
                                empty_reason=None,
                                started_at=_utc_iso_now(),
                                finished_at=_utc_iso_now(),
                                duration_ms=0,  # TODO: track actual duration
                                artifacts=artifacts,
                                error=None,
                                error_code=None,
                                notes="processed_in_batch",
                            )
                        )
                        manifest.save()
                
                # Обновляем manifest для core_depth_midas если он был обработан батчем
                if process_core_depth_midas_batch_mode and video_ctx.video_id in core_depth_midas_results:
                    depth_result = core_depth_midas_results[video_ctx.video_id]
                    
                    if depth_result.get("status") == "ok":
                        # Находим артефакты
                        component_dir = os.path.join(run_rs_path, "core_depth_midas")
                        artifacts = []
                        if os.path.exists(component_dir):
                            npz_files = list(Path(component_dir).glob("*.npz"))
                            if npz_files:
                                artifacts = [{"path": str(f), "type": "npz"} for f in npz_files]
                        
                        manifest.upsert_component(
                            ManifestComponent(
                                name="core_depth_midas",
                                kind="core",
                                status="ok",
                                empty_reason=None,
                                started_at=_utc_iso_now(),
                                finished_at=_utc_iso_now(),
                                duration_ms=0,  # TODO: track actual duration
                                artifacts=artifacts,
                                error=None,
                                error_code=None,
                                notes="processed_in_batch",
                                producer_version="2.0",
                                schema_version="core_depth_midas_npz_v1",
                                device_used="cuda",
                            )
                        )
                        manifest.save()
                
                # Обновляем manifest для core_optical_flow если он был обработан батчем
                if process_core_optical_flow_batch_mode and video_ctx.video_id in core_optical_flow_results:
                    flow_result = core_optical_flow_results[video_ctx.video_id]
                    
                    if flow_result.get("status") == "ok":
                        # Находим артефакты
                        component_dir = os.path.join(run_rs_path, "core_optical_flow")
                        artifacts = []
                        if os.path.exists(component_dir):
                            npz_files = list(Path(component_dir).glob("*.npz"))
                            if npz_files:
                                artifacts = [{"path": str(f), "type": "npz"} for f in npz_files]
                        
                        manifest.upsert_component(
                            ManifestComponent(
                                name="core_optical_flow",
                                kind="core",
                                status="ok",
                                empty_reason=None,
                                started_at=_utc_iso_now(),
                                finished_at=_utc_iso_now(),
                                duration_ms=0,  # TODO: track actual duration
                                artifacts=artifacts,
                                error=None,
                                error_code=None,
                                notes="processed_in_batch",
                                producer_version="2.0",
                                schema_version="core_optical_flow_npz_v1",
                                device_used="cuda",
                            )
                        )
                        manifest.save()
                
                # Обновляем manifest для place_semantics если он был обработан батчем
                if process_place_semantics_batch_mode and video_ctx.video_id in place_semantics_results:
                    place_result = place_semantics_results[video_ctx.video_id]
                    
                    if place_result.get("status") == "ok":
                        # Находим артефакты
                        component_dir = os.path.join(run_rs_path, "place_semantics")
                        artifacts = []
                        if os.path.exists(component_dir):
                            npz_files = list(Path(component_dir).glob("*.npz"))
                            if npz_files:
                                artifacts = [{"path": str(f), "type": "npz"} for f in npz_files]
                        
                        manifest.upsert_component(
                            ManifestComponent(
                                name="place_semantics",
                                kind="core",
                                status="ok",
                                empty_reason=None,
                                started_at=_utc_iso_now(),
                                finished_at=_utc_iso_now(),
                                duration_ms=0,  # TODO: track actual duration
                                artifacts=artifacts,
                                error=None,
                                error_code=None,
                                notes="processed_in_batch",
                                producer_version="0.1",
                                schema_version="place_semantics_npz_v1",
                                device_used="cpu",  # Embedding Service runs on server
                            )
                        )
                        manifest.save()
                
                # Обновляем manifest для scene_classification если он был обработан батчем
                if process_scene_classification_batch_mode and video_ctx.video_id in scene_classification_results:
                    scene_result = scene_classification_results[video_ctx.video_id]
                    
                    if scene_result.get("status") == "ok":
                        # Находим артефакты
                        component_dir = os.path.join(run_rs_path, "scene_classification")
                        artifacts = []
                        if os.path.exists(component_dir):
                            npz_files = list(Path(component_dir).glob("*.npz"))
                            if npz_files:
                                artifacts = [{"path": str(f), "type": "npz"} for f in npz_files]
                        
                        scene_classification_config = config.get("scene_classification", {})
                        manifest.upsert_component(
                            ManifestComponent(
                                name="scene_classification",
                                kind="module",
                                status="ok",
                                empty_reason=None,
                                started_at=_utc_iso_now(),
                                finished_at=_utc_iso_now(),
                                duration_ms=0,  # TODO: track actual duration
                                artifacts=artifacts,
                                error=None,
                                error_code=None,
                                notes="processed_in_batch",
                                producer_version="1.0",
                                schema_version="scene_classification_npz_v1",
                                device_used="cuda" if scene_classification_config.get("runtime") == "triton" else "cpu",
                            )
                        )
                        manifest.save()
                
                # Обновляем manifest для cut_detection если он был обработан батчем
                if process_cut_detection_batch_mode and video_ctx.video_id in cut_detection_results:
                    cut_result = cut_detection_results[video_ctx.video_id]
                    
                    if cut_result.get("status") == "ok":
                        # Находим артефакты
                        component_dir = os.path.join(run_rs_path, "cut_detection")
                        artifacts = []
                        if os.path.exists(component_dir):
                            npz_files = list(Path(component_dir).glob("*.npz"))
                            if npz_files:
                                artifacts = [{"path": str(f), "type": "npz"} for f in npz_files]
                        
                        manifest.upsert_component(
                            ManifestComponent(
                                name="cut_detection",
                                kind="module",
                                status="ok",
                                empty_reason=None,
                                started_at=_utc_iso_now(),
                                finished_at=_utc_iso_now(),
                                duration_ms=0,  # TODO: track actual duration
                                artifacts=artifacts,
                                error=None,
                                error_code=None,
                                notes="processed_in_batch",
                                producer_version="2.0",
                                schema_version="cut_detection_npz_v1",
                                device_used="cpu",  # cut_detection is mostly CPU-bound
                            )
                        )
                        manifest.save()
                
                # Обновляем manifest для video_pacing если он был обработан батчем
                if process_video_pacing_batch_mode and video_ctx.video_id in video_pacing_results:
                    pacing_result = video_pacing_results[video_ctx.video_id]
                    
                    if pacing_result.get("status") == "ok":
                        # Находим артефакты
                        component_dir = os.path.join(run_rs_path, "video_pacing")
                        artifacts = []
                        if os.path.exists(component_dir):
                            npz_files = list(Path(component_dir).glob("*.npz"))
                            if npz_files:
                                artifacts = [{"path": str(f), "type": "npz"} for f in npz_files]
                        
                        manifest.upsert_component(
                            ManifestComponent(
                                name="video_pacing",
                                kind="module",
                                status="ok",
                                empty_reason=None,
                                started_at=_utc_iso_now(),
                                finished_at=_utc_iso_now(),
                                duration_ms=0,  # TODO: track actual duration
                                artifacts=artifacts,
                                error=None,
                                error_code=None,
                                notes="processed_in_batch",
                                producer_version="2.0",
                                schema_version="video_pacing_npz_v2",
                                device_used="cpu",  # video_pacing is CPU-bound
                            )
                        )
                        manifest.save()
            except Exception as e:
                logger.warning(f"VisualProcessor | run_batch | failed to update manifest: {e}")
        
        batch_results.append(result)
    
    # Обрабатываем результаты batch
    success_count = sum(1 for r in batch_results if r.get("status") == "ok")
    error_count = sum(1 for r in batch_results if r.get("status") == "error")
    
    logger.info(
        f"VisualProcessor | run_batch | completed: {len(batch_results)} videos processed "
        f"(success: {success_count}, errors: {error_count})"
    )
    
    return batch_results


def load_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _execution_order(cfg: dict) -> list[str]:
    """
    PR-6: optional DAG execution order (top-level list of component names).
    If provided, VisualProcessor executes enabled components sequentially in this order.
    """
    v = cfg.get("execution_order")
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for x in v:
        if isinstance(x, str) and x:
            out.append(x)
    return out

def _requirements_map(cfg: dict) -> dict:
    """
    PR-4: requirements map enables required/optional enforcement.
    Expected: top-level `requirements: {component_name: bool}`
    If missing/empty -> enforcement is disabled (backward compatible).
    """
    req = cfg.get("requirements")
    return req if isinstance(req, dict) else {}

def _resolved_model_mapping(cfg: dict) -> dict:
    rmm = cfg.get("resolved_model_mapping")
    return rmm if isinstance(rmm, dict) else {}

def _apply_resolved_model_mapping(cfg: dict) -> None:
    """
    PR-8: merge per-component resolved model mapping into component config.
    Only scalar keys matter (nested dict/list values are ignored because they won't be forwarded to component CLI).
    """
    rmm = _resolved_model_mapping(cfg)
    if not rmm:
        return
    for comp, m in rmm.items():
        if not isinstance(comp, str) or not comp:
            continue
        if not isinstance(m, dict) or not m:
            continue
        base = cfg.get(comp)
        if not isinstance(base, dict):
            base = {}
            cfg[comp] = base
        for k, v in m.items():
            if isinstance(v, (dict, list)):
                continue
            base[k] = v

def _is_required(req: dict, component_name: str) -> bool:
    # Default: required=true if map is enabled but key is missing.
    try:
        v = req.get(component_name, True)
    except Exception:
        v = True
    return bool(v)


def get_current_core_providers(config):
    """
    Возвращает список активных core‑провайдеров.
    Ожидает структуру:

    core_providers:
        optical_flow: true
        core_clip: false
    
    Выдает ошибку, если требуемые зависимости не включены.
    """
    core_cfg = config.get("core_providers") or {}
    enabled = [name for name, enabled in core_cfg.items() if enabled]
    return order_core_providers_by_deps(enabled)


def get_current_modules(config):
    """
    Возвращает список активных модулей (modules.<name>: true) в корректном порядке зависимостей.
    Валидирует, что все требуемые core providers включены.
    """
    enabled = [name for name, on in (config.get("modules") or {}).items() if on]
    enabled_core = [name for name, on in (config.get("core_providers") or {}).items() if on]
    return order_modules_by_deps(enabled, enabled_core=enabled_core)


# Module dependency graph (module -> required modules)
# This enforces strict "no-fallback": if a module consumes another module's outputs, it must run after it.
MODULE_DEPS = {
    "shot_quality": ["cut_detection"],
    # video_pacing consumes shot boundaries from cut_detection (baseline policy).
    "video_pacing": ["cut_detection"],
    # high_level_semantic consumes scene structure and emotion signals
    "high_level_semantic": ["cut_detection", "emotion_face"],
    # scene_classification should respect hard cut boundaries (precision) -> depends on cut_detection.
    "scene_classification": ["cut_detection"],
    # optical_flow is a consumer of core_optical_flow (RAFT) only
    "optical_flow": [],
}

# Module -> required core providers.
# This is used to validate PR-6 exec_order (DAG) so modules can't run before core artifacts exist.
# NOTE: modules also implement required_dependencies(), but orchestrator cannot safely import all modules here.
MODULE_CORE_DEPS = {
    # Baseline policy: cut_detection must reuse motion from core_optical_flow (no-fallback).
    "cut_detection": ["core_optical_flow"],
    "optical_flow": ["core_optical_flow"],
    "scene_classification": ["core_clip"],
    "video_pacing": ["core_optical_flow", "core_clip"],
    "shot_quality": ["core_clip", "core_depth_midas", "core_object_detections", "core_face_landmarks"],
    "story_structure": ["core_clip", "core_optical_flow", "core_face_landmarks"],
    "uniqueness": ["core_clip"],
    "behavioral": ["core_face_landmarks"],
    "high_level_semantic": ["core_clip"],
    "micro_emotion": ["core_face_landmarks"],
    "detalize_face": ["core_face_landmarks"],
    "emotion_face": ["core_face_landmarks"],
    "action_recognition": ["core_object_detections"],
    # text_scoring is an OCR consumer; OCR may come from ocr_extractor or external/text pipelines.
    # Baseline policy: OCR is OPTIONAL (module may return status=empty if OCR is missing/empty).
    "text_scoring": [],
}


# Core provider dependency graph (core_provider -> required core_providers)
# This enforces strict "no-fallback": semantic heads must not run before their prerequisites exist.
CORE_DEPS = {
    "brand_semantics": ["core_object_detections"],
    "core_brand_semantics": ["core_object_detections"],  # Legacy name support
    "car_semantics": ["core_object_detections"],
    "core_car_semantics": ["core_object_detections"],  # Legacy name support
    "place_semantics": ["core_object_detections", "core_clip"],
    "core_place_semantics": ["core_object_detections", "core_clip"],  # Legacy name support
    "face_identity": ["core_object_detections", "core_face_landmarks"],
    "core_face_identity": ["core_object_detections", "core_face_landmarks"],  # Legacy name support
    "content_domain": ["core_clip"],
    "franchise_recognition": ["core_clip"],
}


def order_core_providers_by_deps(enabled_core):
    """
    Topologically order enabled core providers according to CORE_DEPS.
    Validates that required deps are enabled, otherwise fail-fast.
    """
    enabled_set = set(enabled_core)

    # validate required deps are enabled
    missing = []
    for m in enabled_core:
        for dep in CORE_DEPS.get(m, []):
            if dep not in enabled_set:
                missing.append((m, dep))
    if missing:
        msg = ", ".join([f"{m} requires {dep}" for m, dep in missing])
        raise ValueError(f"❌ Core provider dependency missing (enable required core provider): {msg}")

    # topo sort (stable-ish)
    order = []
    visiting = set()
    visited = set()

    def dfs(m):
        if m in visited:
            return
        if m in visiting:
            raise ValueError(f"❌ Cycle in core provider dependencies at: {m}")
        visiting.add(m)
        for dep in CORE_DEPS.get(m, []):
            if dep in enabled_set:
                dfs(dep)
        visiting.remove(m)
        visited.add(m)
        order.append(m)

    for m in enabled_core:
        dfs(m)

    # dedupe while preserving topo order
    out = []
    seen = set()
    for m in order:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def order_modules_by_deps(enabled_modules, enabled_core=None):
    """
    Возвращает список модулей в корректном порядке зависимостей.
    
    Args:
        enabled_modules: Список включенных модулей
        enabled_core: Список включенных core providers (для валидации core зависимостей)
    """
    enabled_set = set(enabled_modules)
    enabled_core_set = set(enabled_core or [])

    # validate required module deps are enabled
    missing = []
    for m in enabled_modules:
        for dep in MODULE_DEPS.get(m, []):
            if dep not in enabled_set:
                missing.append((m, dep))
    if missing:
        msg = ", ".join([f"{m} requires {dep}" for m, dep in missing])
        raise ValueError(f"❌ Module dependency missing (enable required module): {msg}")

    # validate required core deps are enabled
    missing_core = []
    for m in enabled_modules:
        for dep in MODULE_CORE_DEPS.get(m, []):
            if dep not in enabled_core_set:
                missing_core.append((m, dep))
    if missing_core:
        msg = ", ".join([f"{m} requires {dep}" for m, dep in missing_core])
        raise ValueError(f"❌ Module core dependency missing (enable required core provider): {msg}")

    # topo sort (stable-ish: preserves original ordering where possible)
    order = []
    visiting = set()
    visited = set()

    def dfs(m):
        if m in visited:
            return
        if m in visiting:
            raise ValueError(f"❌ Cycle in module dependencies at: {m}")
        visiting.add(m)
        for dep in MODULE_DEPS.get(m, []):
            if dep in enabled_set:
                dfs(dep)
        visiting.remove(m)
        visited.add(m)
        order.append(m)

    for m in enabled_modules:
        dfs(m)

    # dedupe while preserving topo order
    out = []
    seen = set()
    for m in order:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def _validate_exec_order_deps(exec_order: list[str], enabled_set: set[str]) -> None:
    """
    Validate PR-6 exec_order ordering across BOTH core providers and modules.
    Fail-fast if:
    - a component appears before its dependency in exec_order
    - a dependency is required but not enabled
    """
    pos = {name: i for i, name in enumerate(exec_order)}

    def deps_for(name: str) -> List[str]:
        if name.startswith("core_"):
            return list(CORE_DEPS.get(name, []))
        # module deps: module->module + module->core
        out: List[str] = []
        out.extend(MODULE_DEPS.get(name, []) or [])
        out.extend(MODULE_CORE_DEPS.get(name, []) or [])
        return out

    missing_enabled = []
    wrong_order = []
    for name in exec_order:
        if name not in enabled_set:
            continue
        for dep in deps_for(name):
            if dep not in enabled_set:
                missing_enabled.append((name, dep))
                continue
            if dep not in pos:
                wrong_order.append((name, dep, "dep missing from exec_order"))
                continue
            if pos[dep] > pos[name]:
                wrong_order.append((name, dep, "dep appears after component"))

    if missing_enabled:
        msg = ", ".join([f"{m} requires {dep}" for m, dep in missing_enabled])
        raise ValueError(f"❌ exec_order invalid: missing enabled dependency: {msg}")
    if wrong_order:
        msg = ", ".join([f"{m} requires {dep} ({why})" for m, dep, why in wrong_order])
        raise ValueError(f"❌ exec_order invalid ordering: {msg}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DataProcessor Controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--cfg-path", type=str, required=True, help="Path to YAML config")

    args = parser.parse_args()

    logger.info(f"VisualProcessor | main | Начало обработки")

    config = load_config(args.cfg_path)
    # PR-8: enrich per-component configs from resolved mapping (profile/DB resolved).
    _apply_resolved_model_mapping(config)
    req_map = _requirements_map(config)
    enforce_requirements = bool(req_map)
    exec_order = _execution_order(config)

    g_config = config.get("global") or {}

    # Auto-detect root_path if missing/invalid (portable across machines).
    if not g_config.get("root_path") or not os.path.isdir(g_config.get("root_path")):
        # VisualProcessor/main.py lives at <root>/VisualProcessor/main.py
        g_config["root_path"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Derive run context + switch rs_path into per-run storage.
    run_ctx = _derive_run_context(config)
    base_rs_path = g_config["rs_path"]
    # Orchestrator may pass the already-resolved per-run directory.
    if bool(g_config.get("rs_path_is_run_dir")):
        run_rs_path = os.path.abspath(str(base_rs_path))
    else:
        run_rs_path = os.path.join(
            base_rs_path,
            run_ctx["platform_id"],
            run_ctx["video_id"],
            run_ctx["run_id"],
        )
    g_config["rs_path"] = run_rs_path
    os.makedirs(run_rs_path, exist_ok=True)

    # ---------------- scheduler runtime report (visual) ----------------
    # We write into the shared per-run report file:
    #   <run_rs_path>/_reports/scheduler_runtime_report.json
    # AudioProcessor may have already written it earlier; we MERGE and add per_processor.visual.
    report_path = os.path.join(run_rs_path, "_reports", "scheduler_runtime_report.json")

    # Best-effort resource sampling (RSS of orchestrator + GPU used MB).
    def _maybe_import_psutil():
        try:
            import psutil  # type: ignore
        except Exception:
            return None
        return psutil

    def _rss_mb() -> Optional[float]:
        psutil = _maybe_import_psutil()
        if psutil is None:
            # Fallback: /proc/self/status (VmRSS)
            try:
                with open("/proc/self/status", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            kb = int(line.split()[1])
                            return float(kb) / 1024.0
            except Exception:
                return None
        try:
            p = psutil.Process(os.getpid())
            return float(p.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            return None

    def _gpu_used_mb0() -> Optional[float]:
        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                return None
        except Exception:
            # Fallback: nvidia-smi if available
            try:
                if shutil.which("nvidia-smi") is None:
                    return None
                import subprocess as _sp

                p = _sp.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                    stdout=_sp.PIPE,
                    stderr=_sp.PIPE,
                    text=True,
                    check=False,
                )
                if p.returncode != 0:
                    return None
                line = (p.stdout or "").strip().splitlines()[0] if (p.stdout or "").strip() else ""
                if not line:
                    return None
                return float(line.strip())
            except Exception:
                return None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            used = float(info.used) / (1024.0 * 1024.0)
            with suppress(Exception):
                pynvml.nvmlShutdown()
            return used
        except Exception:
            try:
                import torch  # type: ignore

                return float(torch.cuda.memory_allocated(0)) / (1024.0 * 1024.0)
            except Exception:
                # Final fallback: nvidia-smi
                try:
                    if shutil.which("nvidia-smi") is None:
                        return None
                    import subprocess as _sp

                    p = _sp.run(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                        stdout=_sp.PIPE,
                        stderr=_sp.PIPE,
                        text=True,
                        check=False,
                    )
                    if p.returncode != 0:
                        return None
                    line = (p.stdout or "").strip().splitlines()[0] if (p.stdout or "").strip() else ""
                    if not line:
                        return None
                    return float(line.strip())
                except Exception:
                    return None

    report_lock = threading.Lock()
    # Mutable container so the sampler thread can update peaks without `nonlocal`
    # (this module runs inside `if __name__ == "__main__"` and has no enclosing function scope).
    report_peaks = {"rss_mb": None, "gpu_used_mb": None}  # type: ignore[var-annotated]
    stop_evt = threading.Event()

    def _sampler_loop():
        while not stop_evt.wait(0.2):
            r = _rss_mb()
            g = _gpu_used_mb0()
            with report_lock:
                if r is not None:
                    prev = report_peaks.get("rss_mb")
                    report_peaks["rss_mb"] = r if prev is None else max(float(prev), float(r))
                if g is not None:
                    prev = report_peaks.get("gpu_used_mb")
                    report_peaks["gpu_used_mb"] = g if prev is None else max(float(prev), float(g))

    sampler_th = threading.Thread(target=_sampler_loop, name="visual_runtime_sampler", daemon=True)
    sampler_th.start()

    manifest_path = os.path.join(run_rs_path, "manifest.json")
    manifest = RunManifest(
        path=manifest_path,
        run_meta={
            **run_ctx,
            "frames_dir": g_config.get("frames_dir"),
            "root_path": g_config.get("root_path"),
        },
    )
    # Always create/refresh manifest on start, even if no components are enabled.
    manifest.flush()

    started_at = _utc_iso_now()
    t0_total = time.time()

    # Resource limits for intra-video parallelism
    max_parallel_modules = int(g_config.get("max_parallel_modules", 1) or 1)
    gpu_slots = _resolve_gpu_slots(g_config)
    gpu_sem = threading.Semaphore(value=max(1, int(gpu_slots)))
    logger.info(
        f"VisualProcessor | main | parallelism: max_parallel_modules={max_parallel_modules} gpu_max_concurrent={gpu_slots}"
    )

    current_core = get_current_core_providers(config)
    current_modules = get_current_modules(config)

    if current_core:
        logger.info("VisualProcessor | main | Текущие core_providers:")
        for provider in current_core:
            logger.info(f"            {provider}")

    logger.info("VisualProcessor | main | Текущие модули:")
    for module in current_modules:
        logger.info(f"            {module}")

    enabled_set = set(current_core) | set(current_modules)

    def _run_one_component(name: str) -> None:
        if name in current_core:
            provider_cfg = config.get(name, {})
            logger.info(f"VisualProcessor | main | core_provider {name} start")
            (
                ok,
                err,
                artifacts,
                status,
                notes,
                schema_version,
                producer_version,
                empty_reason,
                started_at,
                finished_at,
                duration_ms,
            ) = run_core_provider(g_config, name, provider_cfg, run_rs_path=run_rs_path, gpu_sem=gpu_sem)

            manifest.upsert_component(
                ManifestComponent(
                    name=name,
                    kind="core",
                    status=status,
                    empty_reason=empty_reason,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=duration_ms,
                    artifacts=artifacts,
                    error=err,
                    error_code=("component_failed" if status == "error" else None),
                    notes=notes,
                    producer_version=producer_version,
                    schema_version=schema_version,
                    device_used=_device_used_for_component(name, provider_cfg),
                )
            )
            if status == "error":
                logger.error(f"VisualProcessor | main | core_provider {name} failed")
                if enforce_requirements and _is_required(req_map, name):
                    logger.error(f"VisualProcessor | main | required core_provider failed: {name}")
                    raise SystemExit(2)
            elif status == "empty":
                logger.info(f"VisualProcessor | main | core_provider {name} completed with empty status (empty_reason: {empty_reason})")
            return

        if name in current_modules:
            # Module config sections are optional; when absent, run with defaults.
            module_cfg = config.get(name) or {}
            if not isinstance(module_cfg, dict):
                module_cfg = {}
            
            # Auto-inject triton_http_url from core_clip for modules that need it
            if name == "story_structure" and not module_cfg.get("triton_http_url"):
                core_clip_cfg = config.get("core_clip", {})
                if isinstance(core_clip_cfg, dict) and core_clip_cfg.get("triton_http_url"):
                    module_cfg = module_cfg.copy()
                    module_cfg["triton_http_url"] = core_clip_cfg["triton_http_url"]
            
            logger.info(f"VisualProcessor | main | module {name} start")

            # Policy: detalize_face must not run if core_face_landmarks indicates no faces.
            if name == "detalize_face":
                try:
                    core_path = os.path.join(run_rs_path, "core_face_landmarks", "landmarks.npz")
                    if os.path.isfile(core_path):
                        d = np.load(core_path, allow_pickle=True)
                        meta = d.get("meta")
                        if isinstance(meta, np.ndarray) and meta.dtype == object and meta.shape == ():
                            try:
                                meta = meta.item()
                            except Exception:
                                meta = None
                        status0 = meta.get("status") if isinstance(meta, dict) else None
                        empty0 = meta.get("empty_reason") if isinstance(meta, dict) else None
                        has_any_face = meta.get("has_any_face") if isinstance(meta, dict) else None
                        if status0 == "empty" and (empty0 == "no_faces_in_video" or has_any_face is False):
                            ts_now = _utc_iso_now()
                            manifest.upsert_component(
                                ManifestComponent(
                                    name=name,
                                    kind="module",
                                    status="empty",
                                    empty_reason="no_faces_in_video",
                                    started_at=ts_now,
                                    finished_at=ts_now,
                                    duration_ms=0,
                                    artifacts=[],
                                    error=None,
                                    error_code=None,
                                    notes="skipped_by_orchestrator_no_faces",
                                    producer_version=None,
                                    schema_version=None,
                                    device_used=_device_used_for_component(name, module_cfg),
                                )
                            )
                            logger.info("VisualProcessor | main | detalize_face skipped (no_faces_in_video)")
                            return
                except Exception:
                    # If we can't check the core artifact, proceed and let the module fail-fast.
                    pass
            (
                ok,
                err,
                artifacts,
                status,
                notes,
                schema_version,
                producer_version,
                empty_reason,
                started_at,
                finished_at,
                duration_ms,
            ) = run_module(g_config, name, module_cfg, run_rs_path, gpu_sem)
            manifest.upsert_component(
                ManifestComponent(
                    name=name,
                    kind="module",
                    status=status,
                    empty_reason=empty_reason,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=duration_ms,
                    artifacts=artifacts,
                    error=err,
                    error_code=("component_failed" if status == "error" else None),
                    notes=notes,
                    producer_version=producer_version,
                    schema_version=schema_version,
                    device_used=_device_used_for_component(name, module_cfg),
                )
            )
            if not ok:
                logger.error(f"VisualProcessor | main | module {name} failed")
                if enforce_requirements and status == "error" and _is_required(req_map, name):
                    logger.error(f"VisualProcessor | main | required module failed: {name}")
                    raise SystemExit(2)
            return

        # Unknown/unenabled => ignore
        logger.debug(f"VisualProcessor | main | skipping component not enabled: {name}")

    if exec_order:
        logger.info(f"VisualProcessor | main | PR-6: executing by DAG order (len={len(exec_order)})")
        # Fail-fast validation: exec_order must respect core/module dependencies (no-fallback).
        _validate_exec_order_deps(exec_order, enabled_set)
        for name in exec_order:
            if name in enabled_set:
                _run_one_component(name)
        # Run any remaining enabled components not covered by exec_order
        remaining = [n for n in sorted(enabled_set) if n not in exec_order]
        if remaining:
            logger.warning(f"VisualProcessor | main | PR-6: exec_order missing enabled components: {remaining}")
            for n in remaining:
                _run_one_component(n)
    else:
        # Backward-compatible behavior: keep old module scheduling (parallelism).
        if current_core:
            for provider in current_core:
                _run_one_component(provider)

        # Modules can be run in parallel (intra-video), with GPU gating.
        if current_modules:
            has_deps = any((MODULE_DEPS.get(m) or []) for m in current_modules)
            if has_deps:
                logger.info("VisualProcessor | main | module deps detected → running modules sequentially")
                for module in current_modules:
                    _run_one_component(module)
            else:
                max_workers = max(1, int(max_parallel_modules))
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    fut_by_name = {}
                    for module in current_modules:
                        module_cfg = config.get(module)
                        if module_cfg is None:
                            raise ValueError(f"❌ Config entry for module '{module}' not found in YAML")
                        logger.info(f"VisualProcessor | main | scheduling module: {module}")
                        fut = ex.submit(run_module, g_config, module, module_cfg, run_rs_path, gpu_sem)
                        fut_by_name[fut] = module

                    for fut in as_completed(list(fut_by_name.keys())):
                        module = fut_by_name[fut]
                        try:
                            (
                                ok,
                                err,
                                artifacts,
                                status,
                                notes,
                                schema_version,
                                producer_version,
                                empty_reason,
                                started_at,
                                finished_at,
                                duration_ms,
                            ) = fut.result()
                        except Exception as e:
                            ok = False
                            err = str(e)
                            artifacts = []
                            status = "error"
                            notes = "scheduler exception"
                            schema_version = None
                            producer_version = None
                            empty_reason = None
                            started_at = _utc_iso_now()
                            finished_at = _utc_iso_now()
                            duration_ms = 0

                        manifest.upsert_component(
                            ManifestComponent(
                                name=module,
                                kind="module",
                                status=status,
                                empty_reason=empty_reason,
                                started_at=started_at,
                                finished_at=finished_at,
                                duration_ms=duration_ms,
                                artifacts=artifacts,
                                error=err,
                                error_code=("exception" if status == "error" and notes == "scheduler exception" else ("component_failed" if status == "error" else None)),
                                notes=notes,
                                producer_version=producer_version,
                                schema_version=schema_version,
                                device_used=_device_used_for_component(module, config.get(module) or {}),
                            )
                        )
                        if not ok:
                            logger.error(f"VisualProcessor | main | module {module} failed")
                            if enforce_requirements and status == "error" and _is_required(req_map, module):
                                logger.error(f"VisualProcessor | main | required module failed: {module}")
                                raise SystemExit(2)

    finished_at = _utc_iso_now()
    duration_ms_total = int((time.time() - t0_total) * 1000)

    # Stop sampler and freeze maxima
    stop_evt.set()
    with suppress(Exception):
        sampler_th.join(timeout=1.0)
    with report_lock:
        max_rss_mb = report_peaks.get("rss_mb")
        max_gpu_used_mb = report_peaks.get("gpu_used_mb")

    # Build per-component summary from manifest (source-of-truth for component statuses)
    comps = []
    try:
        # Prefer in-memory manifest state if available
        m = _safe_load_json_optional(manifest_path) or {}
        comps = m.get("components") or []
        if not isinstance(comps, list):
            comps = []
    except Exception:
        comps = []

    # Scheduler knobs (as applied by VisualProcessor runtime config)
    # NOTE: batch_size is scheduler-owned; we report what we actually received in cfg.
    per_component_batch_size: dict = {}
    try:
        for name in sorted(enabled_set):
            node = config.get(name)
            if isinstance(node, dict) and isinstance(node.get("batch_size"), int):
                per_component_batch_size[name] = int(node.get("batch_size"))
    except Exception:
        per_component_batch_size = {}

    visual_report = {
        "schema_version": "scheduler_runtime_report_v1",
        "created_at": _utc_iso_now(),
        "platform_id": run_ctx.get("platform_id"),
        "video_id": run_ctx.get("video_id"),
        "run_id": run_ctx.get("run_id"),
        "config_hash": run_ctx.get("config_hash"),
        "scheduler_knobs": {
            "visual.max_parallel_modules": int(max_parallel_modules),
            "visual.gpu_max_concurrent": int(gpu_slots),
            "visual.per_component.batch_size": per_component_batch_size,
            "visual.exec_order_len": int(len(exec_order)) if isinstance(exec_order, list) else 0,
        },
        "per_processor": {
            "visual": {
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_ms": int(duration_ms_total),
                "rss_peak_mb": max_rss_mb,
                "gpu_used_peak_mb": max_gpu_used_mb,
                "components": comps,
            }
        },
    }

    # Merge into shared report file (keep audio/text fields if present).
    try:
        base = _safe_load_json_optional(report_path) or {}
        if not base:
            _atomic_write_json(report_path, visual_report)
        else:
            merged = _merge_dict(base, visual_report)
            _atomic_write_json(report_path, merged)
    except Exception:
        # Do not fail VisualProcessor on report writing.
        pass
