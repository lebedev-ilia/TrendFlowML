"""
Renderer для comments_aggregator extractor: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление агрегированных эмбеддингов комментариев для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_comments_aggregator(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для comments_aggregator extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_commentsagg_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "comments_aggregator",
        "summary": {},
        "aggregation": {},
        "weights": {},
        "artifacts": {},
        "configuration": {},
    }
    
    # Clean NaN values for JSON compatibility
    def _clean_value(v: Any) -> Any:
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        return v
    
    # Extract aggregation features (use canonical prefix, but also check legacy)
    aggregation = {}
    for key, value in extractor_features.items():
        if key in ["tp_commentsagg_present", "tp_commentsagg_count", "tp_commentsagg_dim", 
                   "tp_commentsagg_mean_std", "tp_commentsagg_median_std"]:
            feature_name = key.replace("tp_commentsagg_", "")
            aggregation[feature_name] = _clean_value(value)
        # Also check legacy prefixes
        elif key in ["tp_comments_agg_present", "tp_comments_agg_count", "tp_comments_agg_dim",
                     "tp_comments_agg_mean_std", "tp_comments_agg_median_std"]:
            feature_name = key.replace("tp_comments_agg_", "")
            if feature_name not in aggregation:  # Prefer canonical
                aggregation[feature_name] = _clean_value(value)
        elif key in ["tp_cagg_present", "tp_cagg_count", "tp_cagg_dim",
                     "tp_cagg_mean_std", "tp_cagg_median_std"]:
            feature_name = key.replace("tp_cagg_", "")
            if feature_name not in aggregation:  # Prefer canonical
                aggregation[feature_name] = _clean_value(value)
    
    # Extract weights features
    weights = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_commentsagg_weights_"):
            feature_name = key.replace("tp_commentsagg_weights_", "")
            weights[feature_name] = _clean_value(value)
        elif key.startswith("tp_comments_agg_weights_"):
            feature_name = key.replace("tp_comments_agg_weights_", "")
            if f"weights_{feature_name}" not in weights:  # Prefer canonical
                weights[feature_name] = _clean_value(value)
    
    # Extract artifacts features
    artifacts = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_commentsagg_artifact_"):
            feature_name = key.replace("tp_commentsagg_artifact_", "")
            artifacts[feature_name] = _clean_value(value)
    
    # Extract configuration features
    configuration = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_commentsagg_") and not any(key.startswith(prefix) for prefix in [
            "tp_commentsagg_present", "tp_commentsagg_count", "tp_commentsagg_dim",
            "tp_commentsagg_mean_std", "tp_commentsagg_median_std", "tp_commentsagg_weights_",
            "tp_commentsagg_artifact_", "tp_commentsagg_unsafe_", "tp_commentsagg_dim_mismatch"
        ]):
            feature_name = key.replace("tp_commentsagg_", "")
            configuration[feature_name] = _clean_value(value)
        # Also check legacy
        elif key.startswith("tp_comments_agg_") and key not in [
            "tp_comments_agg_present", "tp_comments_agg_count", "tp_comments_agg_dim",
            "tp_comments_agg_mean_std", "tp_comments_agg_median_std", "tp_comments_agg_weights_"
        ]:
            feature_name = key.replace("tp_comments_agg_", "")
            if feature_name not in configuration:  # Prefer canonical
                configuration[feature_name] = _clean_value(value)
    
    # Extract safety flags
    safety = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_commentsagg_unsafe_") or key.startswith("tp_commentsagg_dim_mismatch"):
            feature_name = key.replace("tp_commentsagg_", "")
            safety[feature_name] = _clean_value(value)
    
    render["aggregation"] = aggregation
    render["weights"] = weights
    render["artifacts"] = artifacts
    render["configuration"] = configuration
    render["safety"] = safety
    
    # Summary
    render["summary"] = {
        "present": bool(aggregation.get("present", 0) > 0.5),
        "count": _clean_value(aggregation.get("count")),
        "dimension": _clean_value(aggregation.get("dim")),
        "mean_std": _clean_value(aggregation.get("mean_std")),
        "median_std": _clean_value(aggregation.get("median_std")),
        "compute_mean": bool(configuration.get("compute_mean_enabled", 0) > 0.5),
        "compute_median": bool(configuration.get("compute_median_enabled", 0) > 0.5),
        "compute_std": bool(configuration.get("compute_std_enabled", 0) > 0.5),
        "weights_applied": bool(weights.get("applied", 0) > 0.5),
        "mean_written": bool(artifacts.get("mean_written", 0) > 0.5),
        "median_written": bool(artifacts.get("median_written", 0) > 0.5),
    }
    
    return render


def render_comments_aggregator_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага comments_aggregator результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_comments_aggregator(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    aggregation = render.get("aggregation", {})
    weights = render.get("weights", {})
    artifacts = render.get("artifacts", {})
    configuration = render.get("configuration", {})
    safety = render.get("safety", {})
    
    # Prepare data for visualization
    numeric_features = {
        "Количество комментариев": aggregation.get("count"),
        "Размерность": aggregation.get("dim"),
        "Среднее std": aggregation.get("mean_std"),
        "Медиана std": aggregation.get("median_std"),
    }
    # Filter out None/NaN values
    numeric_features = {k: v for k, v in numeric_features.items() 
                      if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Агрегация эмбеддингов комментариев</title>
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
        <h1>Агрегация эмбеддингов комментариев</h1>
        <p><strong>Компонент:</strong> {render.get("component", "comments_aggregator")}</p>
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
                    <div class="feature-name">Среднее std</div>
                    <div class="feature-value">{summary.get("mean_std", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Медиана std</div>
                    <div class="feature-value">{summary.get("median_std", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Вычисление среднего</div>
                    <div class="feature-value">{{'Да' if summary.get('compute_mean') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Вычисление медианы</div>
                    <div class="feature-value">{{'Да' if summary.get('compute_median') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Вычисление std</div>
                    <div class="feature-value">{{'Да' if summary.get('compute_std') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Веса применены</div>
                    <div class="feature-value">{{'Да' if summary.get('weights_applied') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Артефакт среднего записан</div>
                    <div class="feature-value">{{'Да' if summary.get('mean_written') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Артефакт медианы записан</div>
                    <div class="feature-value">{{'Да' if summary.get('median_written') else 'Нет'}}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Агрегация</h2>
            <div class="plot-container">
                <div id="aggregation-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Присутствует</td>
                    <td>{{'Да' if aggregation.get('present', 0) > 0.5 else 'Нет'}}</td>
                    <td>Были ли вычислены агрегированные эмбеддинги</td>
                </tr>
                <tr>
                    <td>Количество</td>
                    <td>{aggregation.get("count", "N/A")}</td>
                    <td>Количество комментариев, использованных для агрегации</td>
                </tr>
                <tr>
                    <td>Размерность</td>
                    <td>{aggregation.get("dim", "N/A")}</td>
                    <td>Размерность агрегированного вектора эмбеддинга</td>
                </tr>
                <tr>
                    <td>Среднее std</td>
                    <td>{aggregation.get("mean_std", "N/A")}</td>
                    <td>Среднее стандартное отклонение для взвешенного среднего агрегата</td>
                </tr>
                <tr>
                    <td>Медиана std</td>
                    <td>{aggregation.get("median_std", "N/A")}</td>
                    <td>Среднее стандартное отклонение для медианного агрегата</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Веса</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Веса применены</td>
                    <td>{{'Да' if weights.get('applied', 0) > 0.5 else 'Нет'}}</td>
                    <td>Были ли применены веса при вычислении взвешенного среднего</td>
                </tr>
                <tr>
                    <td>Маска лайков</td>
                    <td>{{'Да' if weights.get('mask_likes', 0) > 0.5 else 'Нет'}}</td>
                    <td>Использовались ли лайки как веса</td>
                </tr>
                <tr>
                    <td>Маска авторитета</td>
                    <td>{{'Да' if weights.get('mask_authority', 0) > 0.5 else 'Нет'}}</td>
                    <td>Использовался ли авторитет автора как вес</td>
                </tr>
                <tr>
                    <td>Маска актуальности</td>
                    <td>{{'Да' if weights.get('mask_recency', 0) > 0.5 else 'Нет'}}</td>
                    <td>Использовалась ли актуальность комментария как вес</td>
                </tr>
                <tr>
                    <td>Выравнивание присутствует</td>
                    <td>{{'Да' if weights.get('align_present', 0) > 0.5 else 'Нет'}}</td>
                    <td>Были ли доступны индексы для выравнивания весов</td>
                </tr>
                <tr>
                    <td>Форма выравнивания OK</td>
                    <td>{{'Да' if weights.get('align_shape_ok', 0) > 0.5 else 'Нет'}}</td>
                    <td>Соответствует ли форма весов форме эмбеддингов</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Артефакты</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Артефакт среднего записан</td>
                    <td>{{'Да' if artifacts.get('mean_written', 0) > 0.5 else 'Нет'}}</td>
                    <td>Был ли записан файл с агрегированным средним эмбеддингом</td>
                </tr>
                <tr>
                    <td>Артефакт медианы записан</td>
                    <td>{{'Да' if artifacts.get('median_written', 0) > 0.5 else 'Нет'}}</td>
                    <td>Был ли записан файл с агрегированным медианным эмбеддингом</td>
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
                    <td>Вычисление среднего включено</td>
                    <td>{{'Да' if configuration.get('compute_mean_enabled', 0) > 0.5 else 'Нет'}}</td>
                    <td>Было ли включено вычисление взвешенного среднего</td>
                </tr>
                <tr>
                    <td>Вычисление медианы включено</td>
                    <td>{{'Да' if configuration.get('compute_median_enabled', 0) > 0.5 else 'Нет'}}</td>
                    <td>Было ли включено вычисление медианы</td>
                </tr>
                <tr>
                    <td>Вычисление std включено</td>
                    <td>{{'Да' if configuration.get('compute_std_enabled', 0) > 0.5 else 'Нет'}}</td>
                    <td>Было ли включено вычисление стандартного отклонения</td>
                </tr>
                <tr>
                    <td>Запись артефактов включена</td>
                    <td>{{'Да' if configuration.get('write_artifacts_enabled', 0) > 0.5 else 'Нет'}}</td>
                    <td>Была ли включена запись файлов с агрегатами</td>
                </tr>
                <tr>
                    <td>Требуются эмбеддинги комментариев</td>
                    <td>{{'Да' if configuration.get('require_comment_embeddings_enabled', 0) > 0.5 else 'Нет'}}</td>
                    <td>Было ли обязательно наличие эмбеддингов комментариев</td>
                </tr>
            </table>
        </div>
    </div>
    
    <script>
        // График агрегации
        var numericFeatures = {json.dumps(numeric_features)};
        if (Object.keys(numericFeatures).length > 0) {{
            var trace = {{
                x: Object.keys(numericFeatures),
                y: Object.values(numericFeatures),
                type: 'bar',
                marker: {{ color: '#4CAF50' }}
            }};
            var layout = {{
                title: 'Характеристики агрегированных эмбеддингов',
                xaxis: {{ title: 'Признак' }},
                yaxis: {{ title: 'Значение' }}
            }};
            Plotly.newPlot('aggregation-plot', [trace], layout);
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
    
    logger.info(f"Comments aggregator HTML render saved to {output_path}")
    return output_path


__all__ = ["render_comments_aggregator", "render_comments_aggregator_html"]

