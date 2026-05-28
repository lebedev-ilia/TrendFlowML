"""
Renderer для генерации human-friendly render-context JSON из NPZ артефактов TextProcessor.

Render-context используется для:
- LLM генерации текстовых описаний (см. docs/contracts/LLM_RENDERING.md)
- Frontend визуализаций (статистики, распределения, метрики)

Каждый extractor может иметь свой файл render.py в src/extractors/<extractor_name>/render.py
для детализированного рендеринга своих фич.
"""

import os
import json
import logging
import importlib
import math
from typing import Dict, Any, Optional, List
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


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


def _clean_for_json(obj: Any) -> Any:
    """
    Рекурсивно очистить объект от NaN, Infinity значений для JSON сериализации.
    Заменяет NaN и Infinity на None.
    """
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_clean_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return _clean_for_json(obj.item() if obj.size == 1 else obj.tolist())
        else:
            return _clean_for_json(obj.tolist())
    else:
        return obj


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


def _load_extractor_html_renderer(extractor_name: str):
    """
    Динамически загрузить HTML renderer для extractor'а из модуля.
    
    Args:
        extractor_name: Имя extractor'а (например, "lexico_static_features")
    
    Returns:
        Функция HTML renderer или None, если не найдена
    """
    try:
        module_name = extractor_name
        module_path = f"src.extractors.{module_name}.render"
        module = importlib.import_module(module_path)
        
        # Ищем функцию render_<extractor_name>_html
        html_render_func_name = f"render_{module_name}_html"
        html_renderer = getattr(module, html_render_func_name, None)
        
        if html_renderer is None:
            logger.debug(f"HTML renderer function {html_render_func_name} not found in {module_path}")
            return None
        
        return html_renderer
    except ImportError as e:
        logger.debug(f"Could not import HTML renderer for {extractor_name}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error loading HTML renderer for {extractor_name}: {e}")
        return None


def _load_extractor_renderer(extractor_name: str):
    """
    Динамически загрузить renderer для extractor'а из модуля.
    
    Args:
        extractor_name: Имя extractor'а (например, "lexico_static_features")
    
    Returns:
        Функция renderer или None, если не найдена
    """
    try:
        # Преобразуем имя extractor'а в имя модуля
        # lexico_static_features -> lexico_static_features
        module_name = extractor_name
        
        # Импортируем модуль render
        module_path = f"src.extractors.{module_name}.render"
        module = importlib.import_module(module_path)
        
        # Ищем функцию render_<extractor_name> (с заменой _ на _)
        # lexico_static_features -> render_lexico_static_features
        render_func_name = f"render_{module_name}"
        renderer = getattr(module, render_func_name, None)
        
        if renderer is None:
            logger.debug(f"Renderer function {render_func_name} not found in {module_path}")
            return None
        
        return renderer
    except ImportError as e:
        logger.debug(f"Could not import renderer for {extractor_name}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error loading renderer for {extractor_name}: {e}")
        return None


def render_text_processor(npz_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Генерировать render-context JSON для TextProcessor из NPZ артефакта.
    
    Args:
        npz_path: Путь к NPZ файлу (text_features.npz)
        output_dir: Директория для сохранения render-context (если None, не сохраняет)
    
    Returns:
        Dict с render-context данными
    """
    # Load NPZ
    npz_data = load_npz(npz_path)
    meta = extract_meta(npz_data)
    
    # Extract payload
    payload = npz_data.get("payload")
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        payload = payload.item() if payload.size == 1 else {}
    if not isinstance(payload, dict):
        payload = {}
    
    from core.text_feature_grouping import (
        build_feature_dict_from_npz_data,
        group_text_features_by_extractor,
    )

    features = build_feature_dict_from_npz_data(npz_data)
    extractor_features = group_text_features_by_extractor(features)
    
    # Generate per-extractor renders (if renderers exist)
    extractor_renders: Dict[str, Dict[str, Any]] = {}
    for extractor_name, extractor_feats in extractor_features.items():
        if extractor_name == "unclassified":
            continue
        # Log extractor features found
        if extractor_feats:
            logger.debug(f"Found {len(extractor_feats)} features for {extractor_name}")
        renderer = _load_extractor_renderer(extractor_name)
        if renderer is not None:
            try:
                # Pass extractor-specific features and full payload
                extractor_render = renderer(extractor_feats, payload, meta)
                if isinstance(extractor_render, dict):
                    extractor_renders[extractor_name] = extractor_render
            except Exception as e:
                import traceback
                tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                logger.error(
                    f"Failed to render {extractor_name}:\n"
                    f"  Error: {str(e)}\n"
                    f"  Type: {type(e).__name__}\n"
                    f"  Traceback:\n{tb_str}",
                    exc_info=False
                )
                continue
        else:
            logger.debug(f"No JSON renderer found for {extractor_name}")
    
    # Build main render-context
    render = {
        "component": "text_processor",
        "summary": {
            "total_features": len(feature_names),
            "extractors_count": len(extractor_features),
            "status": meta.get("status", "unknown"),
        },
        "features": features,
        "extractors": extractor_renders,
        "meta": {
            "status": meta.get("status", "unknown"),
            "producer_version": meta.get("producer_version", "unknown"),
            "schema_version": meta.get("schema_version", "unknown"),
            "created_at": meta.get("created_at", ""),
        },
    }
    
    # Add payload summary (privacy-safe)
    if payload:
        payload_summary = {}
        # Extract non-sensitive summary fields
        for key in ["extractors_run", "extractors_successful", "extractors_failed", "extractors_empty"]:
            if key in payload:
                payload_summary[key] = payload[key]
        if payload_summary:
            render["payload_summary"] = payload_summary
    
    # Save to file if output_dir is provided
    if output_dir is not None:
        render_dir = os.path.join(output_dir, "_render")
        os.makedirs(render_dir, exist_ok=True)
        render_path = os.path.join(render_dir, "render_context.json")
        
        # Atomic write (clean NaN values before serialization)
        tmp_path = render_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            cleaned_render = _clean_for_json(render)
            json.dump(cleaned_render, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, render_path)
        
        logger.info(f"Render-context saved to {render_path}")
        
        # Generate HTML reports for each extractor that has a renderer
        # Check both extractors that have renders AND extractors that have features
        html_paths = []
        # Collect all extractor names that might have renderers
        extractor_names_to_check = set(extractor_renders.keys())
        extractor_names_to_check.update(extractor_features.keys())
        extractor_names_to_check.discard("unclassified")
        
        for extractor_name in extractor_names_to_check:
            try:
                html_renderer = _load_extractor_html_renderer(extractor_name)
                if html_renderer is not None:
                    # Get extractor features for this extractor (may be empty if extractor didn't run)
                    extractor_feats = extractor_features.get(extractor_name, {})
                    html_path = os.path.join(render_dir, f"{extractor_name}_report.html")
                    
                    # Call HTML renderer (signature may vary, try common patterns)
                    try:
                        # Pattern 1: (npz_path, output_path, extractor_features, payload, meta)
                        html_path_result = html_renderer(npz_path, html_path, extractor_feats, payload, meta)
                        if html_path_result:
                            html_paths.append(html_path_result)
                    except TypeError as e:
                        # Check if it's a "missing required positional arguments" error
                        error_msg = str(e)
                        if "missing" in error_msg and "required positional argument" in error_msg:
                            # Try Pattern 2: (npz_path, output_path) - fallback
                            try:
                                html_path_result = html_renderer(npz_path, html_path)
                                if html_path_result:
                                    html_paths.append(html_path_result)
                            except Exception as e2:
                                logger.warning(f"Failed to generate HTML for {extractor_name} with pattern 2: {e2}")
                        else:
                            # Other TypeError (e.g., wrong argument types) - log and continue
                            logger.warning(f"Failed to generate HTML for {extractor_name} with pattern 1: {e}")
                    except Exception as e:
                        logger.warning(f"Failed to generate HTML for {extractor_name}: {e}")
            except Exception as e:
                logger.warning(f"Failed to generate HTML report for {extractor_name}: {e}")
                continue
        
        # Add HTML paths to render metadata
        if html_paths:
            render["html_reports"] = html_paths
    
    return render


__all__ = ["render_text_processor", "load_npz", "extract_meta"]

