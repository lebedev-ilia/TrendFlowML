"""
Renderer для embedding_pair_topk_extractor: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление топ-K похожих чанков для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_embedding_pair_topk_extractor(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для embedding_pair_topk_extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_embpair_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "embedding_pair_topk_extractor",
        "summary": {},
        "similarities": {},
        "topk": {},
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
    
    # Extract title-desc similarity
    similarities = {}
    if "tp_embpair_title_desc_cosine" in extractor_features:
        similarities["title_desc"] = _clean_value(extractor_features["tp_embpair_title_desc_cosine"])
    
    # Extract top-k scores (slots)
    topk_scores = []
    topk_indices = []
    top_k_slots = int(extractor_features.get("tp_embpair_top_k_slots", 5))
    for i in range(1, top_k_slots + 1):
        score_key = f"tp_embpair_title_transcript_top{i}"
        idx_key = f"tp_embpair_title_transcript_top{i}_idx"
        if score_key in extractor_features:
            topk_scores.append(_clean_value(extractor_features[score_key]))
        if idx_key in extractor_features:
            topk_indices.append(_clean_value(extractor_features[idx_key]))
    
    # Extract summary metrics
    topk_summary = {}
    if "tp_embpair_title_transcript_topk_max" in extractor_features:
        topk_summary["max"] = _clean_value(extractor_features["tp_embpair_title_transcript_topk_max"])
    if "tp_embpair_title_transcript_topk_mean" in extractor_features:
        topk_summary["mean"] = _clean_value(extractor_features["tp_embpair_title_transcript_topk_mean"])
    
    # Extract presence flags
    presence = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_embpair_") and key.endswith("_present"):
            feature_name = key.replace("tp_embpair_", "").replace("_present", "")
            presence[feature_name] = bool(value > 0.5)
    
    # Extract configuration
    configuration = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_embpair_") and (key.endswith("_enabled") or key.startswith("tp_embpair_top_k") or 
                                               key.startswith("tp_embpair_use_faiss") or key.startswith("tp_embpair_min_corpus")):
            feature_name = key.replace("tp_embpair_", "")
            configuration[feature_name] = _clean_value(value)
    
    # Extract safety flags
    safety = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_embpair_") and key.endswith("_flag"):
            feature_name = key.replace("tp_embpair_", "")
            safety[feature_name] = bool(value > 0.5) if value is not None else False
    
    render["similarities"] = similarities
    render["topk"] = {
        "scores": topk_scores,
        "indices": topk_indices,
        "summary": topk_summary,
    }
    render["presence"] = presence
    render["configuration"] = configuration
    render["safety"] = safety
    
    # Summary
    render["summary"] = {
        "present": bool(extractor_features.get("tp_embpair_present", 0) > 0.5),
        "title_desc_cosine": _clean_value(similarities.get("title_desc")),
        "title_desc_present": bool(presence.get("title_desc", False)),
        "title_transcript_topk_present": bool(presence.get("title_transcript_topk", False)),
        "topk_max": _clean_value(topk_summary.get("max")),
        "topk_mean": _clean_value(topk_summary.get("mean")),
        "top_k": int(extractor_features.get("tp_embpair_top_k", 10)),
        "top_k_slots": int(extractor_features.get("tp_embpair_top_k_slots", 5)),
    }
    
    return render


def render_embedding_pair_topk_extractor_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага embedding_pair_topk_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_embedding_pair_topk_extractor(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    similarities = render.get("similarities", {})
    topk = render.get("topk", {})
    presence = render.get("presence", {})
    configuration = render.get("configuration", {})
    safety = render.get("safety", {})
    
    topk_scores = topk.get("scores", [])
    topk_indices = topk.get("indices", [])
    topk_summary = topk.get("summary", {})
    
    # Prepare data for visualization
    numeric_scores = {f"Топ-{i+1}": score for i, score in enumerate(topk_scores) 
                      if score is not None and not (isinstance(score, float) and (math.isnan(score) or math.isinf(score)))}
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Топ-K похожих чанков транскрипта</title>
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
        <h1>Топ-K похожих чанков транскрипта</h1>
        <p><strong>Компонент:</strong> {render.get("component", "embedding_pair_topk_extractor")}</p>
        <p><strong>Статус:</strong> <span style="color: {'green' if summary.get('present') else 'red'};">{{'✓ OK' if summary.get('present') else '✗ Нет данных'}}</span></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Присутствует</div>
                    <div class="feature-value">{{'Да' if summary.get('present') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Заголовок ↔ Описание</div>
                    <div class="feature-value">{summary.get("title_desc_cosine", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Топ-K максимум</div>
                    <div class="feature-value">{summary.get("topk_max", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Топ-K среднее</div>
                    <div class="feature-value">{summary.get("topk_mean", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">K</div>
                    <div class="feature-value">{summary.get("top_k", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Слотов</div>
                    <div class="feature-value">{summary.get("top_k_slots", "N/A")}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Косинусное сходство</h2>
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
            </table>
        </div>
        
        <div class="section">
            <h2>Топ-K похожих чанков</h2>
            <div class="plot-container">
                <div id="topk-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Позиция</th>
                    <th>Сходство</th>
                    <th>Индекс чанка</th>
                </tr>
"""
    
    # Add top-k rows
    for i in range(len(topk_scores)):
        score = topk_scores[i] if i < len(topk_scores) else "N/A"
        idx = int(topk_indices[i]) if i < len(topk_indices) and topk_indices[i] is not None else "N/A"
        html_content += f"""
                <tr>
                    <td>Топ-{i+1}</td>
                    <td>{score}</td>
                    <td>{idx}</td>
                </tr>
"""
    
    html_content += f"""
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
                    <td>Чанки транскрипта</td>
                    <td>{{'Да' if presence.get('transcript_chunks') else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Заголовок ↔ Описание</td>
                    <td>{{'Да' if presence.get('title_desc') else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Заголовок ↔ Транскрипт (топ-K)</td>
                    <td>{{'Да' if presence.get('title_transcript_topk') else 'Нет'}}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Конфигурация</h2>
            <table>
                <tr>
                    <th>Параметр</th>
                    <th>Значение</th>
                </tr>
                <tr>
                    <td>Вычисление заголовок ↔ описание</td>
                    <td>{{'Да' if configuration.get('compute_title_desc_enabled', 0) > 0.5 else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Вычисление заголовок ↔ транскрипт (топ-K)</td>
                    <td>{{'Да' if configuration.get('compute_title_transcript_topk_enabled', 0) > 0.5 else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Экспорт слотов топ-K</td>
                    <td>{{'Да' if configuration.get('export_topk_slots_enabled', 0) > 0.5 else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Экспорт индексов топ-K</td>
                    <td>{{'Да' if configuration.get('export_topk_indices_enabled', 0) > 0.5 else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Экспорт сводки топ-K</td>
                    <td>{{'Да' if configuration.get('export_topk_summary_enabled', 0) > 0.5 else 'Нет'}}</td>
                </tr>
            </table>
        </div>
    </div>
    
    <script>
        // График топ-K сходств
        var numericScores = {json.dumps(numeric_scores)};
        if (Object.keys(numericScores).length > 0) {{
            var trace = {{
                x: Object.keys(numericScores),
                y: Object.values(numericScores),
                type: 'bar',
                marker: {{ color: '#4CAF50' }}
            }};
            var layout = {{
                title: 'Топ-K похожих чанков транскрипта',
                xaxis: {{ title: 'Позиция' }},
                yaxis: {{ title: 'Косинусное сходство', range: [-1, 1] }}
            }};
            Plotly.newPlot('topk-plot', [trace], layout);
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
    
    logger.info(f"Embedding pair top-k extractor HTML render saved to {output_path}")
    return output_path


__all__ = ["render_embedding_pair_topk_extractor", "render_embedding_pair_topk_extractor_html"]

