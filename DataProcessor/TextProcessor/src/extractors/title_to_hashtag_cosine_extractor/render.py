"""
Renderer для title_to_hashtag_cosine_extractor: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление косинусного сходства заголовка и хэштегов для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_title_to_hashtag_cosine_extractor(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для title_to_hashtag_cosine_extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_titlehashcos_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "title_to_hashtag_cosine_extractor",
        "summary": {},
        "similarity": {},
        "presence": {},
        "configuration": {},
        "safety": {},
    }
    
    # Clean NaN values for JSON compatibility
    def _clean_value(v: Any) -> Any:
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        return v
    
    # Extract similarity metric (canonical; legacy tp_title_hashtag_cosine for old NPZ)
    similarity = {}
    if "tp_titlehashcos_cosine" in extractor_features:
        similarity["cosine"] = _clean_value(extractor_features["tp_titlehashcos_cosine"])
    elif "tp_title_hashtag_cosine" in extractor_features:
        similarity["cosine"] = _clean_value(extractor_features["tp_title_hashtag_cosine"])
    
    # Extract presence flags
    presence = {}
    if "tp_titlehashcos_title_present" in extractor_features:
        presence["title"] = bool(extractor_features["tp_titlehashcos_title_present"] > 0.5)
    if "tp_titlehashcos_hashtag_present" in extractor_features:
        presence["hashtag"] = bool(extractor_features["tp_titlehashcos_hashtag_present"] > 0.5)
    
    # Extract configuration
    configuration = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_titlehashcos_") and (key.endswith("_enabled") or key.startswith("tp_titlehashcos_require")):
            feature_name = key.replace("tp_titlehashcos_", "")
            configuration[feature_name] = _clean_value(value)
    
    # Extract safety flags
    safety = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_titlehashcos_") and key.endswith("_flag"):
            feature_name = key.replace("tp_titlehashcos_", "")
            safety[feature_name] = bool(value > 0.5) if value is not None else False
    
    render["similarity"] = similarity
    render["presence"] = presence
    render["configuration"] = configuration
    render["safety"] = safety
    
    # Summary
    render["summary"] = {
        "present": bool(extractor_features.get("tp_titlehashcos_present", 0) > 0.5)
        or bool(extractor_features.get("tp_title_hashtag_cosine_present", 0) > 0.5),
        "cosine": _clean_value(similarity.get("cosine")),
        "title_present": bool(presence.get("title", False)),
        "hashtag_present": bool(presence.get("hashtag", False)),
    }
    
    return render


def render_title_to_hashtag_cosine_extractor_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага title_to_hashtag_cosine_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_title_to_hashtag_cosine_extractor(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    similarity = render.get("similarity", {})
    presence = render.get("presence", {})
    configuration = render.get("configuration", {})
    safety = render.get("safety", {})
    
    cosine_value = summary.get("cosine")
    cosine_color = "green" if cosine_value is not None and cosine_value > 0.7 else "orange" if cosine_value is not None and cosine_value > 0.3 else "red" if cosine_value is not None else "gray"
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Косинусное сходство заголовка и хэштегов</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #fafafa; }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 20px; }}
        .feature-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; margin: 15px 0; }}
        .feature-card {{ padding: 10px; background-color: white; border: 1px solid #ddd; border-radius: 4px; }}
        .feature-name {{ font-weight: bold; color: #4CAF50; }}
        .feature-value {{ color: #333; font-size: 1.1em; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .cosine-indicator {{ font-size: 2em; font-weight: bold; text-align: center; padding: 20px; border-radius: 8px; background-color: #f0f0f0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Косинусное сходство заголовка и хэштегов</h1>
        <p><strong>Компонент:</strong> {render.get("component", "title_to_hashtag_cosine_extractor")}</p>
        <p><strong>Статус:</strong> <span style="color: {'green' if summary.get('present') else 'red'};">{{'✓ OK' if summary.get('present') else '✗ Нет данных'}}</span></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Присутствует</div>
                    <div class="feature-value">{{'Да' if summary.get('present') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Косинусное сходство</div>
                    <div class="feature-value" style="color: {cosine_color}; font-size: 1.5em; font-weight: bold;">{summary.get("cosine", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Заголовок присутствует</div>
                    <div class="feature-value">{{'Да' if summary.get('title_present') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Хэштеги присутствуют</div>
                    <div class="feature-value">{{'Да' if summary.get('hashtag_present') else 'Нет'}}</div>
                </div>
            </div>
            <div class="cosine-indicator" style="background-color: {cosine_color}; color: white; margin-top: 20px;">
                <div>Косинусное сходство</div>
                <div style="font-size: 3em;">{summary.get("cosine", "N/A")}</div>
                <div style="font-size: 0.8em; margin-top: 10px;">
                    {{'Высокое сходство' if cosine_value is not None and cosine_value > 0.7 else 'Среднее сходство' if cosine_value is not None and cosine_value > 0.3 else 'Низкое сходство' if cosine_value is not None else 'N/A'}}
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Косинусное сходство</h2>
            <table>
                <tr>
                    <th>Метрика</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Косинусное сходство</td>
                    <td style="color: {cosine_color}; font-weight: bold;">{similarity.get("cosine", "N/A")}</td>
                    <td>Косинусное сходство между эмбеддингами заголовка и хэштегов (диапазон [-1, 1], где 1 - максимальное сходство)</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Присутствие данных</h2>
            <table>
                <tr>
                    <th>Источник</th>
                    <th>Присутствует</th>
                </tr>
                <tr>
                    <td>Заголовок</td>
                    <td>{{'Да' if presence.get('title') else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Хэштеги</td>
                    <td>{{'Да' if presence.get('hashtag') else 'Нет'}}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Конфигурация</h2>
            <table>
                <tr>
                    <th>Параметр</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Включен</td>
                    <td>{{'Да' if configuration.get('enabled', 0) > 0.5 else 'Нет'}}</td>
                    <td>Был ли включен extractor</td>
                </tr>
                <tr>
                    <td>Требуется эмбеддинг заголовка</td>
                    <td>{{'Да' if configuration.get('require_title_embedding_enabled', 0) > 0.5 else 'Нет'}}</td>
                    <td>Было ли обязательно наличие эмбеддинга заголовка</td>
                </tr>
                <tr>
                    <td>Требуется эмбеддинг хэштегов</td>
                    <td>{{'Да' if configuration.get('require_hashtag_embedding_enabled', 0) > 0.5 else 'Нет'}}</td>
                    <td>Было ли обязательно наличие эмбеддинга хэштегов</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Флаги безопасности</h2>
            <table>
                <tr>
                    <th>Флаг</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Небезопасный relpath</td>
                    <td>{{'Да' if safety.get('unsafe_relpath_flag') else 'Нет'}}</td>
                    <td>Был ли обнаружен подозрительный путь к файлу (возможная попытка path traversal)</td>
                </tr>
                <tr>
                    <td>Несоответствие размерности</td>
                    <td>{{'Да' if safety.get('dim_mismatch_flag') else 'Нет'}}</td>
                    <td>Соответствуют ли размерности эмбеддингов заголовка и хэштегов</td>
                </tr>
                <tr>
                    <td>Нулевая норма</td>
                    <td>{{'Да' if safety.get('zero_norm_flag') else 'Нет'}}</td>
                    <td>Имеет ли один из эмбеддингов нулевую норму (не может быть нормализован)</td>
                </tr>
            </table>
        </div>
    </div>
</body>
</html>
"""
    
    # Atomic write
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    os.replace(tmp_path, output_path)
    
    logger.info(f"Title to hashtag cosine extractor HTML render saved to {output_path}")
    return output_path


__all__ = ["render_title_to_hashtag_cosine_extractor", "render_title_to_hashtag_cosine_extractor_html"]

