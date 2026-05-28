"""
Renderer для генерации human-friendly render-context JSON из NPZ артефактов AudioProcessor.

Render-context используется для:
- LLM генерации текстовых описаний (см. docs/contracts/LLM_RENDERING.md)
- Frontend визуализаций (timeline, графики, распределения)

Каждый extractor имеет свой файл render.py в src/extractors/<extractor_name>/render.py
"""

import os
import json
import logging
import importlib
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

def safe_log_warning(logger_instance, message, *args, **kwargs):
    """Safely log a warning message, catching I/O errors from closed handlers."""
    try:
        # Temporarily disable logging error reporting to prevent traceback output
        old_raise_exceptions = logging.raiseExceptions
        logging.raiseExceptions = False
        try:
            logger_instance.warning(message, *args, **kwargs)
        finally:
            # Restore original setting
            logging.raiseExceptions = old_raise_exceptions
    except Exception:
        # Catch ALL exceptions silently - handlers may be closed, streams may be closed,
        # or logging infrastructure may be in an invalid state during shutdown
        # This is expected behavior during cleanup/shutdown phases
        pass


def load_npz(npz_path: str) -> Dict[str, Any]:
    """Загрузить NPZ файл и вернуть словарь."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        result = {}
        for key in data.files:
            arr = data[key]
            # Convert numpy arrays to lists for JSON serialization
            if isinstance(arr, np.ndarray):
                if arr.dtype == object:
                    # Object arrays (like meta dict) - keep as is, will be handled separately
                    result[key] = arr.item() if arr.size == 1 else arr.tolist()
                else:
                    result[key] = arr.tolist() if arr.size > 0 else []
            else:
                result[key] = arr
        return result
    except Exception as e:
        logger.error(f"Failed to load NPZ {npz_path}: {e}")
        raise


def extract_meta(npz_data: Dict[str, Any]) -> Dict[str, Any]:
    """Извлечь meta из NPZ данных."""
    meta = npz_data.get("meta")
    if meta is None:
        return {}
    if isinstance(meta, np.ndarray) and meta.dtype == object:
        if meta.size == 1:
            return meta.item() if isinstance(meta.item(), dict) else {}
        # Multiple meta entries (shouldn't happen, but handle gracefully)
        return meta.item() if hasattr(meta, 'item') else {}
    if isinstance(meta, dict):
        return meta
    return {}


def _load_renderer(component_name: str):
    """
    Динамически загрузить renderer для компонента из модуля extractor'а.
    
    Args:
        component_name: Имя компонента (например, "clap_extractor")
    
    Returns:
        Функция renderer или None, если не найдена
    """
    try:
        # Преобразуем имя компонента в имя модуля
        # clap_extractor -> clap_extractor
        module_name = component_name
        
        # Импортируем модуль render (utils/render.py)
        module_path = f"src.extractors.{module_name}.utils.render"
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            # Fallback: render в корне extractor'а (legacy)
            module_path = f"src.extractors.{module_name}.render"
            module = importlib.import_module(module_path)
        
        # Ищем функцию render_<component_name>
        render_func_name = f"render_{module_name}"
        renderer = getattr(module, render_func_name, None)
        
        if renderer is None:
            logger.warning(f"Renderer function {render_func_name} not found in {module_path}")
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
    output_dir: Optional[str] = None,
    enable_render: bool = True,
    enable_html_render: bool = True,
) -> Dict[str, Any]:
    """
    Генерировать render-context JSON для компонента из NPZ артефакта.
    
    Args:
        npz_path: Путь к NPZ файлу
        component_name: Имя компонента (например, "clap_extractor")
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
    renderer = _load_renderer(component_name)
    if renderer is None:
        logger.warning(f"No renderer for component {component_name}, returning basic render")
        return {
            "component": component_name,
            "summary": {},
            "timeline": [],
            "distributions": {},
        }
    
    # Generate render-context
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
            json.dump(render, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, render_path)
        
        logger.info(f"Render-context saved to {render_path}")
        
        # Try to generate HTML if HTML renderer function exists and enabled
        if enable_html_render:
            try:
                try:
                    render_module = importlib.import_module(f"src.extractors.{component_name}.utils.render")
                except ImportError:
                    render_module = importlib.import_module(f"src.extractors.{component_name}.render")
                html_renderer_name = f"render_{component_name}_html"
                logger.debug(f"Looking for HTML renderer: {html_renderer_name} in module {render_module}")
                html_renderer = getattr(render_module, html_renderer_name, None)
                if html_renderer is not None and callable(html_renderer):
                    html_path = os.path.join(render_dir, "render.html")
                    logger.info(f"Generating HTML render for {component_name} -> {html_path}")
                    html_renderer(npz_path, html_path)
                    logger.info(f"HTML render saved to {html_path}")
                else:
                    logger.debug(f"HTML renderer function {html_renderer_name} not found or not callable in {render_module}")
            except ImportError as e:
                # Module not found - skip HTML generation
                logger.debug(f"Could not import render module for {component_name}: {e}")
            except Exception as e:
                # Best-effort: do not fail if HTML generation fails
                # Use warning level to make errors visible
                safe_log_warning(logger, f"Could not generate HTML for {component_name}: {e}")
                try:
                    import traceback
                    logger.debug(f"HTML generation traceback for {component_name}: {traceback.format_exc()}")
                except (ValueError, OSError, AttributeError):
                    # Logging handler may be closed, ignore
                    pass
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
        
        # Look for NPZ file
        npz_file = component_dir / f"{component_name}_features.npz"
        if not npz_file.exists():
            continue
        
        try:
            render = render_component(str(npz_file), component_name, str(component_dir))
            results[component_name] = render
        except Exception as e:
            logger.error(f"Failed to render {component_name}: {e}")
            continue
    
    return results
