"""
Renderer для embedding_stats_extractor: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление статистик эмбеддингов для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_embedding_stats_extractor(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для embedding_stats_extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_embstats_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "embedding_stats_extractor",
        "summary": {},
        "variance": {},
        "topic": {},
        "source": {},
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
    
    # Extract variance metrics
    variance = {}
    if "tp_embstats_l2_variance" in extractor_features:
        variance["l2_variance"] = _clean_value(extractor_features["tp_embstats_l2_variance"])
    
    # Extract top variance slots
    top_variances = []
    top_k_slots = int(extractor_features.get("tp_embstats_top_k_slots", 8))
    for i in range(1, top_k_slots + 1):
        key = f"tp_embstats_topvar_{i}"
        if key in extractor_features:
            top_variances.append(_clean_value(extractor_features[key]))
    
    variance["top_variances"] = top_variances
    
    # Extract topic entropy metrics
    topic = {}
    if "tp_embstats_topic_entropy" in extractor_features:
        topic["entropy"] = _clean_value(extractor_features["tp_embstats_topic_entropy"])
    if "tp_embstats_topic_entropy_norm" in extractor_features:
        topic["entropy_norm"] = _clean_value(extractor_features["tp_embstats_topic_entropy_norm"])
    if "tp_embstats_topic_perplexity" in extractor_features:
        topic["perplexity"] = _clean_value(extractor_features["tp_embstats_topic_perplexity"])
    topic["present"] = bool(extractor_features.get("tp_embstats_topic_entropy_present", 0) > 0.5)
    topic["probs_present"] = bool(extractor_features.get("tp_embstats_topic_probs_present", 0) > 0.5)
    topic["probs_invalid"] = bool(extractor_features.get("tp_embstats_topic_probs_invalid_flag", 0) > 0.5)
    
    # Extract source tracking
    source = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_embstats_source_used_"):
            source_name = key.replace("tp_embstats_source_used_", "")
            source[source_name] = bool(value > 0.5)
    
    # Extract configuration
    configuration = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_embstats_") and (key.endswith("_enabled") or key.startswith("tp_embstats_top") or 
                                                key.startswith("tp_embstats_min") or key.startswith("tp_embstats_variance")):
            feature_name = key.replace("tp_embstats_", "")
            configuration[feature_name] = _clean_value(value)
    
    # Extract safety flags
    safety = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_embstats_") and key.endswith("_flag"):
            feature_name = key.replace("tp_embstats_", "")
            safety[feature_name] = bool(value > 0.5) if value is not None else False
    
    render["variance"] = variance
    render["topic"] = topic
    render["source"] = source
    render["configuration"] = configuration
    render["safety"] = safety
    
    # Summary
    render["summary"] = {
        "present": bool(extractor_features.get("tp_embstats_present", 0) > 0.5),
        "l2_variance": _clean_value(variance.get("l2_variance")),
        "topic_entropy": _clean_value(topic.get("entropy")),
        "topic_entropy_norm": _clean_value(topic.get("entropy_norm")),
        "topic_perplexity": _clean_value(topic.get("perplexity")),
        "topic_present": bool(topic.get("present", False)),
        "n_chunks": _clean_value(extractor_features.get("tp_embstats_n_chunks")),
        "dim": _clean_value(extractor_features.get("tp_embstats_dim")),
    }
    
    return render


def render_embedding_stats_extractor_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага embedding_stats_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_embedding_stats_extractor(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    variance = render.get("variance", {})
    topic = render.get("topic", {})
    source = render.get("source", {})
    configuration = render.get("configuration", {})
    safety = render.get("safety", {})
    
    top_variances = variance.get("top_variances", [])
    
    # Prepare data for visualization
    numeric_variances = {f"Топ-{i+1}": v for i, v in enumerate(top_variances) 
                        if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Статистики эмбеддингов</title>
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
        <h1>Статистики эмбеддингов</h1>
        <p><strong>Компонент:</strong> {render.get("component", "embedding_stats_extractor")}</p>
        <p><strong>Статус:</strong> <span style="color: {'green' if summary.get('present') else 'red'};">{{'✓ OK' if summary.get('present') else '✗ Нет данных'}}</span></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Присутствует</div>
                    <div class="feature-value">{{'Да' if summary.get('present') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">L2 дисперсия</div>
                    <div class="feature-value">{summary.get("l2_variance", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Энтропия топиков</div>
                    <div class="feature-value">{summary.get("topic_entropy", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Нормированная энтропия</div>
                    <div class="feature-value">{summary.get("topic_entropy_norm", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Перплексия</div>
                    <div class="feature-value">{summary.get("topic_perplexity", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Количество чанков</div>
                    <div class="feature-value">{summary.get("n_chunks", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Размерность</div>
                    <div class="feature-value">{summary.get("dim", "N/A")}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Дисперсия эмбеддингов</h2>
            <div class="plot-container">
                <div id="variance-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Метрика</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>L2 дисперсия</td>
                    <td>{variance.get("l2_variance", "N/A")}</td>
                    <td>L2-норма дисперсии эмбеддингов по компонентам между чанками</td>
                </tr>
            </table>
            <h3>Топ-K дисперсий компонент</h3>
            <table>
                <tr>
                    <th>Позиция</th>
                    <th>Дисперсия</th>
                </tr>
"""
    
    # Add top variance rows
    for i in range(len(top_variances)):
        var = top_variances[i] if i < len(top_variances) else "N/A"
        html_content += f"""
                <tr>
                    <td>Топ-{i+1}</td>
                    <td>{var}</td>
                </tr>
"""
    
    html_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>Энтропия топиков</h2>
            <table>
                <tr>
                    <th>Метрика</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Энтропия</td>
                    <td>{topic.get("entropy", "N/A")}</td>
                    <td>Энтропия распределения топиков (мера смешения тем)</td>
                </tr>
                <tr>
                    <td>Нормированная энтропия</td>
                    <td>{topic.get("entropy_norm", "N/A")}</td>
                    <td>Нормированная энтропия H/log(K), где K - количество топиков</td>
                </tr>
                <tr>
                    <td>Перплексия</td>
                    <td>{topic.get("perplexity", "N/A")}</td>
                    <td>Перплексия распределения топиков (e^H)</td>
                </tr>
                <tr>
                    <td>Топики присутствуют</td>
                    <td>{{'Да' if topic.get('present') else 'Нет'}}</td>
                    <td>Были ли доступны распределения топиков</td>
                </tr>
                <tr>
                    <td>Вероятности топиков присутствуют</td>
                    <td>{{'Да' if topic.get('probs_present') else 'Нет'}}</td>
                    <td>Были ли доступны вероятности топиков</td>
                </tr>
                <tr>
                    <td>Вероятности топиков невалидны</td>
                    <td>{{'Да' if topic.get('probs_invalid') else 'Нет'}}</td>
                    <td>Были ли вероятности топиков невалидными</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Источник транскрипта</h2>
            <table>
                <tr>
                    <th>Источник</th>
                    <th>Использован</th>
                </tr>
                <tr>
                    <td>whisper</td>
                    <td>{{'Да' if source.get('whisper') else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>youtube_auto</td>
                    <td>{{'Да' if source.get('youtube_auto') else 'Нет'}}</td>
                </tr>
            </table>
        </div>
    </div>
    
    <script>
        // График топ-K дисперсий
        var numericVariances = {json.dumps(numeric_variances)};
        if (Object.keys(numericVariances).length > 0) {{
            var trace = {{
                x: Object.keys(numericVariances),
                y: Object.values(numericVariances),
                type: 'bar',
                marker: {{ color: '#4CAF50' }}
            }};
            var layout = {{
                title: 'Топ-K дисперсий компонент эмбеддингов',
                xaxis: {{ title: 'Позиция' }},
                yaxis: {{ title: 'Дисперсия' }}
            }};
            Plotly.newPlot('variance-plot', [trace], layout);
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
    
    logger.info(f"Embedding stats extractor HTML render saved to {output_path}")
    return output_path


__all__ = ["render_embedding_stats_extractor", "render_embedding_stats_extractor_html"]

