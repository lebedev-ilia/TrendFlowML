"""
Renderer для генерации human-friendly render-context JSON из NPZ артефактов VisualProcessor.

Render-context используется для:
- LLM генерации текстовых описаний
- Frontend визуализаций (timeline, графики, распределения)

Каждый компонент имеет свой файл render.py в core/model_process/<component_name>/render.py
или modules/<module_name>/render.py
"""

import os
import json
import logging
import importlib
from typing import Dict, Any, Optional, List
from pathlib import Path
import inspect

import numpy as np

logger = logging.getLogger(__name__)

def safe_log_warning(logger_instance, message, *args, **kwargs):
    """Safely log a warning message, catching I/O errors from closed handlers."""
    try:
        logger_instance.warning(message, *args, **kwargs)
    except Exception:
        pass


def _convert_numpy_to_python(obj: Any) -> Any:
    """Рекурсивно конвертирует numpy массивы и типы в Python типы."""
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            # Object arrays - рекурсивно обрабатываем каждый элемент
            if obj.size == 1:
                return _convert_numpy_to_python(obj.item())
            else:
                return [_convert_numpy_to_python(item) for item in obj]
        else:
            return obj.tolist() if obj.size > 0 else []
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_to_python(item) for item in obj]
    else:
        return obj


def load_npz(npz_path: str) -> Dict[str, Any]:
    """Загрузить NPZ файл и вернуть словарь."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        result = {}
        for key in data.files:
            arr = data[key]
            # Convert numpy arrays to lists for JSON serialization
            result[key] = _convert_numpy_to_python(arr)
        return result
    except Exception as e:
        logger.error(f"Failed to load NPZ {npz_path}: {e}")
        raise


def _path_is_nonempty_file(p: Path) -> bool:
    """
    True только для существующего обычного файла.
    Не бросает при гонках / сетевом FS (ENOENT, ESTALE и т.д.) — в этом случае False.
    """
    try:
        return p.is_file()
    except OSError:
        return False


def find_component_npz(
    component_dir: Path,
    component_name: str,
    component_type: str = "core",
) -> Optional[Path]:
    """
    Найти основной NPZ в каталоге компонента.

    Сначала проверяются канонические имена (embeddings.npz / features.npz), затем любой *.npz
    с детерминированным приоритетом — чтобы не терять артефакты вроде face_identity.npz.
    """
    try:
        if not component_dir.is_dir():
            return None
    except OSError:
        return None

    preferred: List[str]
    if component_type == "core":
        preferred = ["embeddings.npz", f"{component_name}.npz"]
    else:
        preferred = ["features.npz", f"{component_name}_features.npz", f"{component_name}.npz"]

    for name in preferred:
        p = component_dir / name
        if _path_is_nonempty_file(p):
            return p

    try:
        all_npz = sorted(component_dir.glob("*.npz"))
    except OSError:
        return None
    if not all_npz:
        return None
    all_npz = [p for p in all_npz if _path_is_nonempty_file(p)]
    if not all_npz:
        return None
    # Prefer names that mention the component; else first lexicographically
    mention = [p for p in all_npz if component_name.replace("core_", "") in p.name]
    if mention:
        return mention[0]
    return all_npz[0]


def extract_meta(npz_data: Dict[str, Any]) -> Dict[str, Any]:
    """Извлечь meta из NPZ данных."""
    meta = npz_data.get("meta")
    if meta is None:
        return {}
    # Handle numpy array with object dtype
    if isinstance(meta, np.ndarray) and meta.dtype == object:
        try:
            if hasattr(meta, 'size') and meta.size == 1:
                item = meta.item()
                return item if isinstance(item, dict) else {}
            elif hasattr(meta, 'item'):
                item = meta.item()
                return item if isinstance(item, dict) else {}
        except (AttributeError, ValueError):
            pass
    # Handle list (already converted from numpy array)
    if isinstance(meta, list):
        if len(meta) == 1:
            item = meta[0]
            return item if isinstance(item, dict) else {}
        # If list contains dict, return first dict
        for item in meta:
            if isinstance(item, dict):
                return item
    # Handle dict directly
    if isinstance(meta, dict):
        return meta
    return {}


def _load_renderer(component_name: str, component_type: str = "core"):
    """
    Динамически загрузить renderer для компонента.
    
    Args:
        component_name: Имя компонента (например, "core_clip" или "cut_detection")
        component_type: Тип компонента ("core" или "module")
    
    Returns:
        Функция renderer или None, если не найдена
    """
    try:
        from pathlib import Path
        
        # Определяем путь к файлу render.py в зависимости от типа
        # VisualProcessor находится в DataProcessor/VisualProcessor/
        # core компоненты: VisualProcessor/core/model_process/<component_name>/render.py
        # modules: VisualProcessor/modules/<module_name>/render.py
        
        # Находим корень VisualProcessor (utils/renderer.py -> VisualProcessor/)
        renderer_file = Path(__file__).resolve()
        visual_processor_root = renderer_file.parent.parent  # utils -> VisualProcessor
        
        if component_type == "core":
            # Try standard path first
            render_path = visual_processor_root / "core" / "model_process" / component_name / "render.py"
            # If not found, try core_identity subdirectory (for content_domain, brand_semantics, etc.)
            if not render_path.exists():
                # Remove 'core_' prefix if present for folder lookup
                folder_name = component_name.replace("core_", "") if component_name.startswith("core_") else component_name
                core_identity_components = [
                    "content_domain",
                    "brand_semantics",
                    "core_brand_semantics",
                    "car_semantics",
                    "core_car_semantics",
                    "franchise_recognition",
                    "face_identity",
                    "place_semantics",
                    "core_place_semantics",
                ]
                if component_name in core_identity_components or folder_name in core_identity_components:
                    identity_path = visual_processor_root / "core" / "model_process" / "core_identity" / folder_name / "render.py"
                    logger.debug(f"Trying core_identity path: {identity_path}")
                    if identity_path.exists():
                        render_path = identity_path
                        logger.debug(f"Found render file at core_identity path: {render_path}")
        else:  # module
            render_path = visual_processor_root / "modules" / component_name / "render.py"
        
        if not render_path.exists():
            logger.debug(f"Render file not found: {render_path}")
            return None
        
        logger.debug(f"Loading renderer from: {render_path}")
        
        # Используем importlib.util для загрузки из файла
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"{component_name}.render", render_path)
        if spec is None or spec.loader is None:
            logger.warning(f"Could not create spec for {render_path}")
            return None
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Ищем функцию render_<component_name>
        render_func_name = f"render_{component_name}"
        renderer = getattr(module, render_func_name, None)
        
        if renderer is None:
            logger.warning(f"Renderer function {render_func_name} not found in {render_path}")
            return None
        
        return renderer
    except ImportError as e:
        logger.debug(f"Could not import renderer for {component_name}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error loading renderer for {component_name}: {e}")
        return None


def render_component(
    npz_path: str, 
    component_name: str,
    component_type: str = "core",
    output_dir: Optional[str] = None,
    frames_dir: Optional[str] = None,
    enable_render: bool = True,
    enable_html_render: bool = True,
) -> Dict[str, Any]:
    """
    Генерировать render-context JSON для компонента из NPZ артефакта.
    
    Args:
        npz_path: Путь к NPZ файлу
        component_name: Имя компонента (например, "core_clip")
        component_type: Тип компонента ("core" или "module")
        output_dir: Директория для сохранения render-context (если None, не сохраняет)
        enable_render: Включить генерацию render-context JSON (по умолчанию: True)
        enable_html_render: Включить генерацию HTML debug страницы (по умолчанию: True)
    
    Returns:
        Dict с render-context данными (или пустой dict, если enable_render=False)
    """
    # Если рендеринг отключен, возвращаем пустой dict
    if not enable_render:
        logger.debug(f"Render disabled for {component_name}, skipping render generation")
        return {}
    
    # Load NPZ
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    
    # Get renderer dynamically
    renderer = _load_renderer(component_name, component_type)
    if renderer is None:
        logger.warning(f"No renderer for component {component_name}, returning basic render")
        return {
            "component": component_name,
            "summary": {},
            "timeline": [],
            "distributions": {},
        }
    
    # Prepare render environment (paths available to renderers)
    render_dir = None
    assets_dir = None
    if output_dir is not None:
        render_dir = os.path.join(output_dir, "_render")
        assets_dir = os.path.join(render_dir, "assets")
        try:
            os.makedirs(assets_dir, exist_ok=True)
        except Exception:
            assets_dir = None

    render_env = {
        "npz_path": npz_path,
        "component_name": component_name,
        "component_type": component_type,
        "output_dir": output_dir,
        "render_dir": render_dir,
        "assets_dir": assets_dir,
        "frames_dir": frames_dir,
    }

    # Generate render-context (support both legacy signature and new signature with env)
    try:
        sig = inspect.signature(renderer)
        if len(sig.parameters) >= 3:
            render = renderer(npz_data, meta, render_env)
        else:
            render = renderer(npz_data, meta)
    except Exception:
        # Fallback to legacy signature
        render = renderer(npz_data, meta)
    
    # Add meta information
    render["meta"] = {
        "status": meta.get("status", "unknown"),
        "producer_version": meta.get("producer_version", "unknown"),
        "schema_version": meta.get("schema_version", "unknown"),
        "created_at": meta.get("created_at", ""),
    }
    
    # Save to file if output_dir is provided
    if output_dir is not None:
        render_dir = os.path.join(output_dir, "_render")
        os.makedirs(render_dir, exist_ok=True)
        render_path = os.path.join(render_dir, "render_context.json")
        
        # Atomic write
        tmp_path = render_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(_convert_numpy_to_python(render), f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, render_path)
        
        logger.info(f"Render-context saved to {render_path}")
        
        # Try to generate HTML if HTML renderer function exists and enabled
        if enable_html_render:
            try:
                from pathlib import Path
                import importlib.util
                
                # Находим путь к render.py (аналогично _load_renderer)
                renderer_file = Path(__file__).resolve()
                visual_processor_root = renderer_file.parent.parent
                
                if component_type == "core":
                    # Try standard path first
                    render_path = visual_processor_root / "core" / "model_process" / component_name / "render.py"
                    # If not found, try core_identity subdirectory
                    if not render_path.exists():
                        folder_name = component_name.replace("core_", "") if component_name.startswith("core_") else component_name
                        core_identity_components = [
                            "content_domain",
                            "brand_semantics",
                            "core_brand_semantics",
                            "car_semantics",
                            "core_car_semantics",
                            "franchise_recognition",
                            "face_identity",
                            "place_semantics",
                            "core_place_semantics",
                        ]
                        if component_name in core_identity_components or folder_name in core_identity_components:
                            identity_path = visual_processor_root / "core" / "model_process" / "core_identity" / folder_name / "render.py"
                            if identity_path.exists():
                                render_path = identity_path
                else:
                    render_path = visual_processor_root / "modules" / component_name / "render.py"
                
                if render_path.exists():
                    spec = importlib.util.spec_from_file_location(f"{component_name}.render", render_path)
                    if spec is not None and spec.loader is not None:
                        render_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(render_module)
                        
                        html_renderer_name = f"render_{component_name}_html"
                        html_renderer = getattr(render_module, html_renderer_name, None)
                        if html_renderer is not None and callable(html_renderer):
                            html_path = os.path.join(render_dir, "render.html")
                            logger.info(f"Generating HTML render for {component_name} -> {html_path}")
                            # Optional kwargs for richer HTML renders (frames + assets)
                            try:
                                html_sig = inspect.signature(html_renderer)
                                kwargs = {}
                                if "frames_dir" in html_sig.parameters:
                                    kwargs["frames_dir"] = frames_dir
                                if "assets_dir" in html_sig.parameters:
                                    kwargs["assets_dir"] = os.path.join(render_dir, "assets")
                                if kwargs:
                                    os.makedirs(kwargs.get("assets_dir", os.path.join(render_dir, "assets")), exist_ok=True)
                                html_renderer(npz_path, html_path, **kwargs)
                            except TypeError:
                                # Legacy signature
                                html_renderer(npz_path, html_path)
                            logger.info(f"HTML render saved to {html_path}")
                        else:
                            logger.debug(f"HTML renderer function {html_renderer_name} not found or not callable")
            except Exception as e:
                safe_log_warning(logger, f"Could not generate HTML for {component_name}: {e}")
        else:
            logger.debug(f"HTML render disabled for {component_name}, skipping HTML generation")
    
    return render


def render_all_components(run_rs_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Генерировать render-context для всех компонентов в run.
    
    Args:
        run_rs_path: Путь к run result_store директории
    
    Returns:
        Dict[component_name, render_context]
    """
    results = {}
    
    # Find all component directories
    run_path = Path(run_rs_path)
    if not run_path.exists():
        logger.warning(f"Run path does not exist: {run_rs_path}")
        return results
    
    for component_dir in run_path.iterdir():
        if not component_dir.is_dir():
            continue
        
        component_name = component_dir.name
        if component_name.startswith("_") or component_name == "manifest.json":
            continue
        # Служебные каталоги run (не NPZ-компоненты), см. batch_runs_feature_report._SKIP_RUN_SUBDIRS
        if component_name == "state":
            continue

        # Determine component type (core providers vs modules)
        # Core providers are typically: core_clip, core_depth_midas, etc.
        # Modules are: cut_detection, scene_classification, etc.
        component_type = "core" if component_name.startswith("core_") else "module"
        
        npz_file = find_component_npz(component_dir, component_name, component_type)
        if npz_file is None:
            continue
        
        try:
            render = render_component(str(npz_file), component_name, component_type, str(component_dir))
            results[component_name] = render
        except Exception as e:
            logger.error(f"Failed to render {component_name}: {e}")
            continue
    
    return results

