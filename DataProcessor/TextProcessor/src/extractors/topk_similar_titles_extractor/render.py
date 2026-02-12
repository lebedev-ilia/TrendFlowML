"""
Renderer для topk_similar_titles_extractor: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление топ-K похожих заголовков для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_topk_similar_titles_extractor(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для topk_similar_titles_extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_topktitles_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "topk_similar_titles_extractor",
        "summary": {},
        "scores": {},
        "corpus": {},
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
    
    # Extract scores
    scores = {}
    if "tp_topktitles_top1_score" in extractor_features:
        scores["top1"] = _clean_value(extractor_features["tp_topktitles_top1_score"])
    if "tp_topktitles_topk_mean_score" in extractor_features:
        scores["topk_mean"] = _clean_value(extractor_features["tp_topktitles_topk_mean_score"])
    
    # Extract corpus metadata
    corpus = {}
    if "tp_topktitles_corpus_size" in extractor_features:
        corpus["size"] = _clean_value(extractor_features["tp_topktitles_corpus_size"])
    if "tp_topktitles_dim" in extractor_features:
        corpus["dim"] = _clean_value(extractor_features["tp_topktitles_dim"])
    
    # Try to get corpus metadata from payload
    corpus_meta = {}
    if isinstance(payload, dict):
        topk_similar = payload.get("topk_similar_corpus_titles", {})
        if isinstance(topk_similar, dict):
            corpus_meta = topk_similar.get("corpus", {})
    
    # Extract configuration
    configuration = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_topktitles_") and (key.endswith("_enabled") or key.startswith("tp_topktitles_k") or 
                                                 key.startswith("tp_topktitles_export") or key.startswith("tp_topktitles_backend") or
                                                 key.startswith("tp_topktitles_cache") or key.startswith("tp_topktitles_max") or
                                                 key.startswith("tp_topktitles_require") or key.startswith("tp_topktitles_allow") or
                                                 key.startswith("tp_topktitles_hnsw")):
            feature_name = key.replace("tp_topktitles_", "")
            configuration[feature_name] = _clean_value(value)
    
    # Extract safety flags
    safety = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_topktitles_") and key.endswith("_flag"):
            feature_name = key.replace("tp_topktitles_", "")
            safety[feature_name] = bool(value > 0.5) if value is not None else False
    
    render["scores"] = scores
    render["corpus"] = corpus
    render["corpus_meta"] = corpus_meta
    render["configuration"] = configuration
    render["safety"] = safety
    
    # Summary
    render["summary"] = {
        "present": bool(extractor_features.get("tp_topktitles_present", 0) > 0.5),
        "top1_score": _clean_value(scores.get("top1")),
        "topk_mean_score": _clean_value(scores.get("topk_mean")),
        "k": int(extractor_features.get("tp_topktitles_k", 5)),
        "corpus_size": _clean_value(corpus.get("size")),
        "dim": _clean_value(corpus.get("dim")),
        "backend_faiss": bool(extractor_features.get("tp_topktitles_backend_faiss", 0) > 0.5),
    }
    
    return render


def render_topk_similar_titles_extractor_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага topk_similar_titles_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_topk_similar_titles_extractor(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    scores = render.get("scores", {})
    corpus = render.get("corpus", {})
    corpus_meta = render.get("corpus_meta", {})
    configuration = render.get("configuration", {})
    safety = render.get("safety", {})
    
    # Try to get top-k IDs and scores from payload
    topk_similar = {}
    if isinstance(payload, dict):
        topk_payload = payload.get("topk_similar_corpus_titles", {})
        if isinstance(topk_payload, dict):
            topk_similar = {
                "ids": topk_payload.get("topk_similar_ids", []),
                "scores": topk_payload.get("topk_similar_scores", []),
            }
    
    # Prepare data for visualization
    numeric_scores = {}
    if scores.get("top1") is not None:
        numeric_scores["Топ-1"] = scores.get("top1")
    if scores.get("topk_mean") is not None:
        numeric_scores["Среднее топ-K"] = scores.get("topk_mean")
    
    # Filter out None/NaN values
    numeric_scores = {k: v for k, v in numeric_scores.items() 
                     if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Топ-K похожих заголовков</title>
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
        <h1>Топ-K похожих заголовков</h1>
        <p><strong>Компонент:</strong> {render.get("component", "topk_similar_titles_extractor")}</p>
        <p><strong>Статус:</strong> <span style="color: {'green' if summary.get('present') else 'red'};">{{'✓ OK' if summary.get('present') else '✗ Нет данных'}}</span></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Присутствует</div>
                    <div class="feature-value">{{'Да' if summary.get('present') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Топ-1 сходство</div>
                    <div class="feature-value">{summary.get("top1_score", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Среднее сходство топ-K</div>
                    <div class="feature-value">{summary.get("topk_mean_score", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">K</div>
                    <div class="feature-value">{summary.get("k", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Размер корпуса</div>
                    <div class="feature-value">{summary.get("corpus_size", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Размерность</div>
                    <div class="feature-value">{summary.get("dim", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Backend</div>
                    <div class="feature-value">{{'FAISS' if summary.get('backend_faiss') else 'NumPy'}}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Сходства</h2>
            <div class="plot-container">
                <div id="scores-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Метрика</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Топ-1 сходство</td>
                    <td>{scores.get("top1", "N/A")}</td>
                    <td>Косинусное сходство с наиболее похожим заголовком из корпуса</td>
                </tr>
                <tr>
                    <td>Среднее сходство топ-K</td>
                    <td>{scores.get("topk_mean", "N/A")}</td>
                    <td>Среднее косинусное сходство с топ-K похожими заголовками</td>
                </tr>
            </table>
"""
    
    # Add top-k list if available
    if topk_similar.get("ids") or topk_similar.get("scores"):
        html_content += f"""
            <h3>Топ-K похожих заголовков</h3>
            <table>
                <tr>
                    <th>Позиция</th>
                    <th>ID</th>
                    <th>Сходство</th>
                </tr>
"""
        ids = topk_similar.get("ids", [])
        scores_list = topk_similar.get("scores", [])
        for i in range(max(len(ids), len(scores_list))):
            idx = ids[i] if i < len(ids) else "N/A"
            score = scores_list[i] if i < len(scores_list) else "N/A"
            html_content += f"""
                <tr>
                    <td>Топ-{i+1}</td>
                    <td>{idx}</td>
                    <td>{score}</td>
                </tr>
"""
        html_content += """
            </table>
"""
    
    html_content += f"""
        </div>
        
        <div class="section">
            <h2>Корпус</h2>
            <table>
                <tr>
                    <th>Параметр</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Размер корпуса</td>
                    <td>{corpus.get("size", "N/A")}</td>
                    <td>Количество заголовков в корпусе</td>
                </tr>
                <tr>
                    <td>Размерность</td>
                    <td>{corpus.get("dim", "N/A")}</td>
                    <td>Размерность эмбеддингов в корпусе</td>
                </tr>
                <tr>
                    <td>Spec Name</td>
                    <td>{corpus_meta.get("corpus_spec_name", "N/A") if isinstance(corpus_meta, dict) else "N/A"}</td>
                    <td>Имя спецификации корпуса в dp_models</td>
                </tr>
                <tr>
                    <td>Версия корпуса</td>
                    <td>{corpus_meta.get("corpus_version", "N/A") if isinstance(corpus_meta, dict) else "N/A"}</td>
                    <td>Версия корпуса</td>
                </tr>
                <tr>
                    <td>Weights Digest</td>
                    <td>{corpus_meta.get("corpus_weights_digest", "N/A") if isinstance(corpus_meta, dict) else "N/A"}</td>
                    <td>Хеш весов модели, использованной для создания корпуса</td>
                </tr>
                <tr>
                    <td>Backend</td>
                    <td>{corpus_meta.get("backend", "N/A") if isinstance(corpus_meta, dict) else "N/A"}</td>
                    <td>Backend для поиска (FAISS или NumPy)</td>
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
                    <td>K</td>
                    <td>{configuration.get("k", "N/A")}</td>
                    <td>Количество похожих заголовков для поиска</td>
                </tr>
                <tr>
                    <td>Backend FAISS</td>
                    <td>{{'Да' if configuration.get('backend_faiss', 0) > 0.5 else 'Нет'}}</td>
                    <td>Использовался ли FAISS для поиска</td>
                </tr>
                <tr>
                    <td>FAISS доступен</td>
                    <td>{{'Да' if configuration.get('faiss_available', 0) > 0.5 else 'Нет'}}</td>
                    <td>Была ли доступна библиотека FAISS</td>
                </tr>
                <tr>
                    <td>Требуется эмбеддинг заголовка</td>
                    <td>{{'Да' if configuration.get('require_title_embedding_enabled', 0) > 0.5 else 'Нет'}}</td>
                    <td>Было ли обязательно наличие эмбеддинга заголовка</td>
                </tr>
                <tr>
                    <td>Кеш включен</td>
                    <td>{{'Да' if configuration.get('cache_enabled', 0) > 0.5 else 'Нет'}}</td>
                    <td>Было ли включено кеширование индекса</td>
                </tr>
            </table>
        </div>
    </div>
    
    <script>
        // График сходств
        var numericScores = {json.dumps(numeric_scores)};
        if (Object.keys(numericScores).length > 0) {{
            var trace = {{
                x: Object.keys(numericScores),
                y: Object.values(numericScores),
                type: 'bar',
                marker: {{ color: '#4CAF50' }}
            }};
            var layout = {{
                title: 'Сходства с похожими заголовками',
                xaxis: {{ title: 'Метрика' }},
                yaxis: {{ title: 'Косинусное сходство', range: [-1, 1] }}
            }};
            Plotly.newPlot('scores-plot', [trace], layout);
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
    
    logger.info(f"Top-K similar titles extractor HTML render saved to {output_path}")
    return output_path


__all__ = ["render_topk_similar_titles_extractor", "render_topk_similar_titles_extractor_html"]

