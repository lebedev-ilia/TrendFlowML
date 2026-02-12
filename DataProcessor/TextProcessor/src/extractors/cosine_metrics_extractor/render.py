"""
Renderer для cosine_metrics_extractor: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление метрик косинусного сходства для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_cosine_metrics_extractor(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для cosine_metrics_extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_cos_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "cosine_metrics_extractor",
        "summary": {},
        "similarities": {},
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
    
    # Extract similarity metrics
    similarities = {}
    for key, value in extractor_features.items():
        if key in ["tp_cos_title_desc", "tp_cos_title_transcript", "tp_cos_desc_transcript",
                   "tp_cos_transcript_comments_mean", "tp_cos_transcript_comments_median"]:
            feature_name = key.replace("tp_cos_", "")
            similarities[feature_name] = _clean_value(value)
    
    # Extract presence flags
    presence = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_cos_") and key.endswith("_present"):
            feature_name = key.replace("tp_cos_", "").replace("_present", "")
            presence[feature_name] = bool(value > 0.5)
    
    # Extract empty reasons
    empty_reasons = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_cos_empty_"):
            feature_name = key.replace("tp_cos_empty_", "")
            empty_reasons[feature_name] = bool(value > 0.5)
    
    # Extract configuration
    configuration = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_cos_") and (key.endswith("_enabled") or key.startswith("tp_cos_comments_mode")):
            feature_name = key.replace("tp_cos_", "")
            configuration[feature_name] = _clean_value(value)
    
    # Extract safety flags
    safety = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_cos_") and (key.endswith("_flag") or key.endswith("_mismatch_flag")):
            feature_name = key.replace("tp_cos_", "")
            safety[feature_name] = bool(value > 0.5) if value is not None else False
    
    # Extract extra metrics if available
    extra_metrics = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_cos_") and not any(key.startswith(prefix) for prefix in [
            "tp_cos_title_desc", "tp_cos_title_transcript", "tp_cos_desc_transcript",
            "tp_cos_transcript_comments_mean", "tp_cos_transcript_comments_median",
            "tp_cos_title_present", "tp_cos_desc_present", "tp_cos_transcript_present", "tp_cos_comments_present",
            "tp_cos_empty_", "tp_cos_title_desc_enabled", "tp_cos_title_transcript_enabled",
            "tp_cos_desc_transcript_enabled", "tp_cos_transcript_comments_mean_enabled", "tp_cos_transcript_comments_median_enabled",
            "tp_cos_zero_norm_flag", "tp_cos_dim_mismatch_flag", "tp_cos_pair_dim_mismatch_flag",
            "tp_cos_tc_dim_mismatch_flag", "tp_cos_unsafe_relpath_flag"
        ]):
            feature_name = key.replace("tp_cos_", "")
            extra_metrics[feature_name] = _clean_value(value)
    
    render["similarities"] = similarities
    render["presence"] = presence
    render["empty_reasons"] = empty_reasons
    render["configuration"] = configuration
    render["safety"] = safety
    render["extra_metrics"] = extra_metrics
    
    # Summary
    render["summary"] = {
        "title_desc": _clean_value(similarities.get("title_desc")),
        "title_transcript": _clean_value(similarities.get("title_transcript")),
        "desc_transcript": _clean_value(similarities.get("desc_transcript")),
        "transcript_comments_mean": _clean_value(similarities.get("transcript_comments_mean")),
        "transcript_comments_median": _clean_value(similarities.get("transcript_comments_median")),
        "title_present": bool(presence.get("title", False)),
        "desc_present": bool(presence.get("desc", False)),
        "transcript_present": bool(presence.get("transcript", False)),
        "comments_present": bool(presence.get("comments", False)),
    }
    
    return render


def render_cosine_metrics_extractor_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага cosine_metrics_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_cosine_metrics_extractor(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    similarities = render.get("similarities", {})
    presence = render.get("presence", {})
    empty_reasons = render.get("empty_reasons", {})
    configuration = render.get("configuration", {})
    safety = render.get("safety", {})
    extra_metrics = render.get("extra_metrics", {})
    
    # Prepare data for visualization
    numeric_similarities = {
        "Заголовок ↔ Описание": similarities.get("title_desc"),
        "Заголовок ↔ Транскрипт": similarities.get("title_transcript"),
        "Описание ↔ Транскрипт": similarities.get("desc_transcript"),
        "Транскрипт ↔ Комментарии (среднее)": similarities.get("transcript_comments_mean"),
        "Транскрипт ↔ Комментарии (медиана)": similarities.get("transcript_comments_median"),
    }
    # Filter out None/NaN values
    numeric_similarities = {k: v for k, v in numeric_similarities.items() 
                           if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Метрики косинусного сходства</title>
    <meta charset="UTF-8">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
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
        .plot-container {{ margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Метрики косинусного сходства</h1>
        <p><strong>Компонент:</strong> {render.get("component", "cosine_metrics_extractor")}</p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Заголовок ↔ Описание</div>
                    <div class="feature-value">{summary.get("title_desc", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Заголовок ↔ Транскрипт</div>
                    <div class="feature-value">{summary.get("title_transcript", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Описание ↔ Транскрипт</div>
                    <div class="feature-value">{summary.get("desc_transcript", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Транскрипт ↔ Комментарии (среднее)</div>
                    <div class="feature-value">{summary.get("transcript_comments_mean", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Транскрипт ↔ Комментарии (медиана)</div>
                    <div class="feature-value">{summary.get("transcript_comments_median", "N/A")}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Косинусное сходство</h2>
            <div class="plot-container">
                <div id="similarities-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Пара</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Заголовок ↔ Описание</td>
                    <td>{similarities.get("title_desc", "N/A")}</td>
                    <td>Косинусное сходство между эмбеддингами заголовка и описания</td>
                </tr>
                <tr>
                    <td>Заголовок ↔ Транскрипт</td>
                    <td>{similarities.get("title_transcript", "N/A")}</td>
                    <td>Косинусное сходство между эмбеддингом заголовка и агрегированным эмбеддингом транскрипта</td>
                </tr>
                <tr>
                    <td>Описание ↔ Транскрипт</td>
                    <td>{similarities.get("desc_transcript", "N/A")}</td>
                    <td>Косинусное сходство между эмбеддингом описания и агрегированным эмбеддингом транскрипта</td>
                </tr>
                <tr>
                    <td>Транскрипт ↔ Комментарии (среднее)</td>
                    <td>{similarities.get("transcript_comments_mean", "N/A")}</td>
                    <td>Косинусное сходство между агрегированным эмбеддингом транскрипта и средним эмбеддингом комментариев</td>
                </tr>
                <tr>
                    <td>Транскрипт ↔ Комментарии (медиана)</td>
                    <td>{similarities.get("transcript_comments_median", "N/A")}</td>
                    <td>Косинусное сходство между агрегированным эмбеддингом транскрипта и медианным эмбеддингом комментариев</td>
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
                    <td>Описание</td>
                    <td>{{'Да' if presence.get('desc') else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Транскрипт</td>
                    <td>{{'Да' if presence.get('transcript') else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Комментарии</td>
                    <td>{{'Да' if presence.get('comments') else 'Нет'}}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Конфигурация</h2>
            <table>
                <tr>
                    <th>Метрика</th>
                    <th>Включена</th>
                </tr>
                <tr>
                    <td>Заголовок ↔ Описание</td>
                    <td>{{'Да' if configuration.get('title_desc_enabled', 0) > 0.5 else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Заголовок ↔ Транскрипт</td>
                    <td>{{'Да' if configuration.get('title_transcript_enabled', 0) > 0.5 else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Описание ↔ Транскрипт</td>
                    <td>{{'Да' if configuration.get('desc_transcript_enabled', 0) > 0.5 else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Транскрипт ↔ Комментарии (среднее)</td>
                    <td>{{'Да' if configuration.get('transcript_comments_mean_enabled', 0) > 0.5 else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Транскрипт ↔ Комментарии (медиана)</td>
                    <td>{{'Да' if configuration.get('transcript_comments_median_enabled', 0) > 0.5 else 'Нет'}}</td>
                </tr>
            </table>
        </div>
    </div>
    
    <script>
        // График косинусного сходства
        var numericSimilarities = {json.dumps(numeric_similarities)};
        if (Object.keys(numericSimilarities).length > 0) {{
            var trace = {{
                x: Object.keys(numericSimilarities),
                y: Object.values(numericSimilarities),
                type: 'bar',
                marker: {{ color: '#4CAF50' }}
            }};
            var layout = {{
                title: 'Косинусное сходство между парами эмбеддингов',
                xaxis: {{ title: 'Пара' }},
                yaxis: {{ title: 'Косинусное сходство', range: [-1, 1] }}
            }};
            Plotly.newPlot('similarities-plot', [trace], layout);
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
    
    logger.info(f"Cosine metrics extractor HTML render saved to {output_path}")
    return output_path


__all__ = ["render_cosine_metrics_extractor", "render_cosine_metrics_extractor_html"]

