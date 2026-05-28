"""
Renderer для comments_embedder extractor: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление эмбеддингов комментариев для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def _truthy_scalar(d: Dict[str, Any], key: str, *, default: float = 0.0) -> bool:
    """0/1 flags after _clean_value may be None; treat like falsy for comparisons."""
    v = d.get(key, default)
    if v is None:
        v = default
    try:
        return float(v) > 0.5
    except (TypeError, ValueError):
        return False


def render_comments_embedder(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для comments_embedder extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_commentsemb_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "comments_embedder",
        "summary": {},
        "embedding": {},
        "selection": {},
        "cache": {},
        "performance": {},
        "configuration": {},
    }
    
    # Clean NaN values for JSON compatibility
    def _clean_value(v: Any) -> Any:
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        return v
    
    # Extract embedding features
    embedding = {}
    for key, value in extractor_features.items():
        if key in ["tp_commentsemb_present", "tp_commentsemb_count", "tp_commentsemb_dim"]:
            feature_name = key.replace("tp_commentsemb_", "")
            embedding[feature_name] = _clean_value(value)
    
    # Extract selection features
    selection = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_commentsemb_n_") or key.startswith("tp_commentsemb_total_chars") or key.startswith("tp_commentsemb_truncated"):
            feature_name = key.replace("tp_commentsemb_", "")
            selection[feature_name] = _clean_value(value)
    
    # Extract cache features
    cache = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_commentsemb_cache") or key.startswith("tp_commentsemb_artifact"):
            feature_name = key.replace("tp_commentsemb_", "")
            cache[feature_name] = _clean_value(value)
    
    # Extract performance features
    performance = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_commentsemb_select_ms") or key.startswith("tp_commentsemb_encode_ms"):
            feature_name = key.replace("tp_commentsemb_", "")
            performance[feature_name] = _clean_value(value)
    
    # Extract configuration features
    configuration = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_commentsemb_") and not any(key.startswith(prefix) for prefix in [
            "tp_commentsemb_present", "tp_commentsemb_count", "tp_commentsemb_dim",
            "tp_commentsemb_n_", "tp_commentsemb_total_chars", "tp_commentsemb_truncated",
            "tp_commentsemb_cache", "tp_commentsemb_artifact", "tp_commentsemb_select_ms",
            "tp_commentsemb_encode_ms"
        ]):
            feature_name = key.replace("tp_commentsemb_", "")
            configuration[feature_name] = _clean_value(value)
    
    render["embedding"] = embedding
    render["selection"] = selection
    render["cache"] = cache
    render["performance"] = performance
    render["configuration"] = configuration
    
    # Summary
    render["summary"] = {
        "present": _truthy_scalar(embedding, "present"),
        "count": _clean_value(embedding.get("count")),
        "dimension": _clean_value(embedding.get("dim")),
        "n_input": _clean_value(selection.get("n_input")),
        "n_deduped": _clean_value(selection.get("n_deduped")),
        "n_selected": _clean_value(selection.get("n_selected")),
        "total_chars_used": _clean_value(selection.get("total_chars_used")),
        "truncated": _truthy_scalar(selection, "truncated_by_total_chars_flag"),
        "cache_hit": (
            None if cache.get("cache_hit") is None else _truthy_scalar(cache, "cache_hit")
        ),
        "select_time_ms": _clean_value(performance.get("select_ms")),
        "encode_time_ms": _clean_value(performance.get("encode_ms")),
        "device": "CUDA" if _truthy_scalar(configuration, "device_cuda") else "CPU",
    }
    
    return render


def render_comments_embedder_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага comments_embedder результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_comments_embedder(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    embedding = render.get("embedding", {})
    selection = render.get("selection", {})
    cache = render.get("cache", {})
    performance = render.get("performance", {})
    configuration = render.get("configuration", {})
    
    # Prepare data for visualization
    numeric_features = {
        "Количество комментариев": embedding.get("count"),
        "Размерность": embedding.get("dim"),
        "Входных комментариев": selection.get("n_input"),
        "После дедупликации": selection.get("n_deduped"),
        "Выбрано": selection.get("n_selected"),
        "Символов использовано": selection.get("total_chars_used"),
    }
    # Filter out None/NaN values
    numeric_features = {k: v for k, v in numeric_features.items() 
                      if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Эмбеддинги комментариев</title>
    <meta charset="UTF-8">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #fafafa; }}
        .metric {{ margin: 10px 0; padding: 8px; background-color: white; border-left: 3px solid #4CAF50; }}
        .metric-label {{ font-weight: bold; color: #333; }}
        .metric-value {{ color: #666; margin-left: 10px; }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 20px; }}
        .feature-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; margin: 15px 0; }}
        .feature-card {{ padding: 10px; background-color: white; border: 1px solid #ddd; border-radius: 4px; }}
        .feature-name {{ font-weight: bold; color: #4CAF50; }}
        .feature-value {{ color: #333; font-size: 1.1em; }}
        .plot-container {{ margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Эмбеддинги комментариев</h1>
        <p><strong>Компонент:</strong> {render.get("component", "comments_embedder")}</p>
        <p><strong>Статус:</strong> <span style="color: {'green' if summary.get('present') else 'red'};">{{'✓ OK' if summary.get('present') else '✗ Нет данных'}}</span></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Присутствует</div>
                    <div class="feature-value">{{'Да' if summary.get('present') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Количество комментариев</div>
                    <div class="feature-value">{summary.get("count", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Размерность</div>
                    <div class="feature-value">{summary.get("dimension", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Входных комментариев</div>
                    <div class="feature-value">{summary.get("n_input", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">После дедупликации</div>
                    <div class="feature-value">{summary.get("n_deduped", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Выбрано</div>
                    <div class="feature-value">{summary.get("n_selected", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Символов использовано</div>
                    <div class="feature-value">{summary.get("total_chars_used", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Обрезано по символам</div>
                    <div class="feature-value">{{'Да' if summary.get('truncated') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Кеш попадание</div>
                    <div class="feature-value">{{'Да' if summary.get('cache_hit') else 'Нет' if summary.get('cache_hit') is False else 'N/A'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Время выбора (мс)</div>
                    <div class="feature-value">{summary.get("select_time_ms", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Время кодирования (мс)</div>
                    <div class="feature-value">{summary.get("encode_time_ms", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Устройство</div>
                    <div class="feature-value">{summary.get("device", "N/A")}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Эмбеддинги</h2>
            <div class="plot-container">
                <div id="embedding-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Присутствует</td>
                    <td>{{'Да' if embedding.get('present', 0) > 0.5 else 'Нет'}}</td>
                    <td>Были ли вычислены эмбеддинги комментариев</td>
                </tr>
                <tr>
                    <td>Количество</td>
                    <td>{embedding.get("count", "N/A")}</td>
                    <td>Количество комментариев, для которых вычислены эмбеддинги</td>
                </tr>
                <tr>
                    <td>Размерность</td>
                    <td>{embedding.get("dim", "N/A")}</td>
                    <td>Размерность вектора эмбеддинга для каждого комментария</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Отбор комментариев</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Входных комментариев</td>
                    <td>{selection.get("n_input", "N/A")}</td>
                    <td>Количество комментариев до обработки</td>
                </tr>
                <tr>
                    <td>После дедупликации</td>
                    <td>{selection.get("n_deduped", "N/A")}</td>
                    <td>Количество уникальных комментариев после удаления дубликатов</td>
                </tr>
                <tr>
                    <td>Выбрано</td>
                    <td>{selection.get("n_selected", "N/A")}</td>
                    <td>Количество комментариев после применения политики отбора</td>
                </tr>
                <tr>
                    <td>Символов использовано</td>
                    <td>{selection.get("total_chars_used", "N/A")}</td>
                    <td>Общее количество символов в выбранных комментариях</td>
                </tr>
                <tr>
                    <td>Обрезано по символам</td>
                    <td>{{'Да' if selection.get('truncated_by_total_chars_flag', 0) > 0.5 else 'Нет'}}</td>
                    <td>Было ли применено ограничение по общему количеству символов</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Кеширование</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Кеш включен</td>
                    <td>{{'Да' if cache.get('cache_enabled', 0) > 0.5 else 'Нет'}}</td>
                    <td>Было ли включено кеширование эмбеддингов</td>
                </tr>
                <tr>
                    <td>Кеш попадание</td>
                    <td>{{'Да' if cache.get('cache_hit', 0) > 0.5 else 'Нет' if cache.get('cache_hit') is not None else 'N/A'}}</td>
                    <td>Было ли попадание в кеш при вычислении эмбеддингов</td>
                </tr>
                <tr>
                    <td>Артефакт записан</td>
                    <td>{{'Да' if cache.get('artifact_written', 0) > 0.5 else 'Нет'}}</td>
                    <td>Был ли записан файл с эмбеддингами</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Производительность</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Время выбора (мс)</td>
                    <td>{performance.get("select_ms", "N/A")}</td>
                    <td>Время, затраченное на отбор комментариев, в миллисекундах</td>
                </tr>
                <tr>
                    <td>Время кодирования (мс)</td>
                    <td>{performance.get("encode_ms", "N/A")}</td>
                    <td>Время, затраченное на кодирование комментариев в эмбеддинги, в миллисекундах</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Конфигурация</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>FP16</td>
                    <td>{{'Да' if (configuration.get('fp16') or 0) > 0.5 else 'Нет'}}</td>
                    <td>Использовалась ли половинная точность (FP16) для вычислений</td>
                </tr>
                <tr>
                    <td>Устройство CUDA</td>
                    <td>{{'Да' if (configuration.get('device_cuda') or 0) > 0.5 else 'Нет'}}</td>
                    <td>Использовался ли GPU для вычислений</td>
                </tr>
                <tr>
                    <td>Вычисление включено</td>
                    <td>{{'Да' if (configuration.get('compute_enabled') or 0) > 0.5 else 'Нет'}}</td>
                    <td>Было ли включено вычисление эмбеддингов</td>
                </tr>
                <tr>
                    <td>Запись артефакта включена</td>
                    <td>{{'Да' if (configuration.get('write_artifact_enabled') or 0) > 0.5 else 'Нет'}}</td>
                    <td>Была ли включена запись файла с эмбеддингами</td>
                </tr>
            </table>
        </div>
    </div>
    
    <script>
        // График эмбеддингов
        var numericFeatures = {json.dumps(numeric_features)};
        if (Object.keys(numericFeatures).length > 0) {{
            var trace = {{
                x: Object.keys(numericFeatures),
                y: Object.values(numericFeatures),
                type: 'bar',
                marker: {{ color: '#4CAF50' }}
            }};
            var layout = {{
                title: 'Характеристики эмбеддингов комментариев',
                xaxis: {{ title: 'Признак' }},
                yaxis: {{ title: 'Значение' }}
            }};
            Plotly.newPlot('embedding-plot', [trace], layout);
        }}
    </script>
</body>
</html>
"""
    
    # Atomic write
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    os.replace(tmp_path, output_path)
    
    logger.info(f"Comments embedder HTML render saved to {output_path}")
    return output_path


__all__ = ["render_comments_embedder", "render_comments_embedder_html"]

