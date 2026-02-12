"""
Renderer для semantics_topics_keyphrases: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление тем, ключевых фраз и стилистических признаков для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_semantics_topics_keyphrases(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для semantics_topics_keyphrases.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_topics_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "semantics_topics_keyphrases",
        "summary": {},
        "topics": {},
        "keyphrases": {},
        "style": {},
        "configuration": {},
        "presence": {},
    }
    
    # Clean NaN values for JSON compatibility
    def _clean_value(v: Any) -> Any:
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        return v
    
    # Extract topic metrics
    topics = {}
    if "tp_topics_topic_top1_id" in extractor_features:
        topics["top1_id"] = _clean_value(extractor_features["tp_topics_topic_top1_id"])
    if "tp_topics_topic_top1_score" in extractor_features:
        topics["top1_score"] = _clean_value(extractor_features["tp_topics_topic_top1_score"])
    if "tp_topics_topic_top1_prob" in extractor_features:
        topics["top1_prob"] = _clean_value(extractor_features["tp_topics_topic_top1_prob"])
    if "tp_topics_entropy_topk" in extractor_features:
        topics["entropy_topk"] = _clean_value(extractor_features["tp_topics_entropy_topk"])
    if "tp_topics_entropy_topk_norm" in extractor_features:
        topics["entropy_topk_norm"] = _clean_value(extractor_features["tp_topics_entropy_topk_norm"])
    if "tp_topics_perplexity_topk" in extractor_features:
        topics["perplexity_topk"] = _clean_value(extractor_features["tp_topics_perplexity_topk"])
    
    # Extract top-K topics (fixed slots)
    topk_topics = []
    for i in range(1, 11):  # Check up to 10 slots
        id_key = f"tp_topics_topic_top{i}_id"
        score_key = f"tp_topics_topic_top{i}_score"
        prob_key = f"tp_topics_topic_top{i}_prob"
        if id_key in extractor_features or score_key in extractor_features or prob_key in extractor_features:
            topic_id = _clean_value(extractor_features.get(id_key))
            score = _clean_value(extractor_features.get(score_key))
            prob = _clean_value(extractor_features.get(prob_key))
            if topic_id is not None or score is not None or prob is not None:
                topk_topics.append({
                    "rank": i,
                    "id": topic_id,
                    "score": score,
                    "prob": prob,
                })
    topics["topk"] = topk_topics
    
    # Extract keyphrase metrics
    keyphrases = {}
    if "tp_topics_keyphrases_count" in extractor_features:
        keyphrases["count"] = _clean_value(extractor_features["tp_topics_keyphrases_count"])
    if "tp_topics_keyphrases_dim" in extractor_features:
        keyphrases["embedding_dim"] = _clean_value(extractor_features["tp_topics_keyphrases_dim"])
    
    # Extract keyphrase score summaries
    if "tp_topics_keyphrase_score_top1" in extractor_features:
        keyphrases["score_top1"] = _clean_value(extractor_features["tp_topics_keyphrase_score_top1"])
    if "tp_topics_keyphrase_score_mean" in extractor_features:
        keyphrases["score_mean"] = _clean_value(extractor_features["tp_topics_keyphrase_score_mean"])
    
    # Extract keyphrase hashed slots
    keyphrase_slots = []
    for i in range(1, 11):  # Check up to 10 slots
        present_key = f"tp_topics_kp_top{i}_present"
        hash_key = f"tp_topics_kp_top{i}_hash01"
        len_key = f"tp_topics_kp_top{i}_len"
        if present_key in extractor_features:
            present = bool(extractor_features[present_key] > 0.5)
            if present:
                keyphrase_slots.append({
                    "rank": i,
                    "present": True,
                    "hash01": _clean_value(extractor_features.get(hash_key)),
                    "len": _clean_value(extractor_features.get(len_key)),
                })
    keyphrases["slots"] = keyphrase_slots
    
    # Extract style flags
    style = {}
    if "tp_topics_style_faq_qmarks" in extractor_features:
        style["faq_qmarks"] = _clean_value(extractor_features["tp_topics_style_faq_qmarks"])
    if "tp_topics_style_instructional_flag" in extractor_features:
        style["instructional"] = bool(extractor_features["tp_topics_style_instructional_flag"] > 0.5)
    if "tp_topics_style_audience_flag" in extractor_features:
        style["audience"] = bool(extractor_features["tp_topics_style_audience_flag"] > 0.5)
    if "tp_topics_style_cta_flag" in extractor_features:
        style["call_to_action"] = bool(extractor_features["tp_topics_style_cta_flag"] > 0.5)
    
    # Extract presence flags
    presence = {}
    if "tp_topics_has_asr" in extractor_features:
        presence["asr"] = bool(extractor_features["tp_topics_has_asr"] > 0.5)
    if "tp_topics_has_title" in extractor_features:
        presence["title"] = bool(extractor_features["tp_topics_has_title"] > 0.5)
    if "tp_topics_has_description" in extractor_features:
        presence["description"] = bool(extractor_features["tp_topics_has_description"] > 0.5)
    
    # Extract configuration
    configuration = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_topics_") and (key.endswith("_enabled") or key.startswith("tp_topics_enable_") or
                                            key.startswith("tp_topics_export_") or key.startswith("tp_topics_allow_") or
                                            key.startswith("tp_topics_top_k_") or key.startswith("tp_topics_temperature")):
            feature_name = key.replace("tp_topics_", "")
            configuration[feature_name] = _clean_value(value)
    
    render["topics"] = topics
    render["keyphrases"] = keyphrases
    render["style"] = style
    render["configuration"] = configuration
    render["presence"] = presence
    
    # Summary
    render["summary"] = {
        "top1_topic_id": _clean_value(topics.get("top1_id")),
        "top1_topic_prob": _clean_value(topics.get("top1_prob")),
        "entropy_topk": _clean_value(topics.get("entropy_topk")),
        "perplexity_topk": _clean_value(topics.get("perplexity_topk")),
        "keyphrases_count": _clean_value(keyphrases.get("count")),
        "keyphrases_dim": _clean_value(keyphrases.get("embedding_dim")),
    }
    
    return render


def render_semantics_topics_keyphrases_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага semantics_topics_keyphrases результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_semantics_topics_keyphrases(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    topics = render.get("topics", {})
    keyphrases = render.get("keyphrases", {})
    style = render.get("style", {})
    configuration = render.get("configuration", {})
    presence = render.get("presence", {})
    
    # Prepare data for visualization
    topk_scores = {}
    topk_probs = {}
    for topic in topics.get("topk", []):
        rank = topic.get("rank")
        score = topic.get("score")
        prob = topic.get("prob")
        if score is not None:
            topk_scores[f"Топ-{rank}"] = score
        if prob is not None:
            topk_probs[f"Топ-{rank}"] = prob
    
    # Filter out None/NaN values
    topk_scores = {k: v for k, v in topk_scores.items() 
                  if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    topk_probs = {k: v for k, v in topk_probs.items() 
                 if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Темы и ключевые фразы</title>
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
        <h1>Темы и ключевые фразы</h1>
        <p><strong>Компонент:</strong> {render.get("component", "semantics_topics_keyphrases")}</p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Топ-1 тема (ID)</div>
                    <div class="feature-value">{summary.get("top1_topic_id", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Топ-1 тема (вероятность)</div>
                    <div class="feature-value">{summary.get("top1_topic_prob", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Энтропия топ-K</div>
                    <div class="feature-value">{summary.get("entropy_topk", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Perplexity топ-K</div>
                    <div class="feature-value">{summary.get("perplexity_topk", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Количество ключевых фраз</div>
                    <div class="feature-value">{summary.get("keyphrases_count", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Размерность эмбеддингов ключевых фраз</div>
                    <div class="feature-value">{summary.get("keyphrases_dim", "N/A")}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Темы</h2>
            <div class="plot-container">
                <div id="topics-scores-plot"></div>
            </div>
            <div class="plot-container">
                <div id="topics-probs-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Ранг</th>
                    <th>ID темы</th>
                    <th>Сходство</th>
                    <th>Вероятность</th>
                </tr>
"""
    
    # Add top-K topics table
    for topic in topics.get("topk", []):
        rank = topic.get("rank")
        topic_id = topic.get("id", "N/A")
        score = topic.get("score", "N/A")
        prob = topic.get("prob", "N/A")
        html_content += f"""
                <tr>
                    <td>Топ-{rank}</td>
                    <td>{topic_id}</td>
                    <td>{score}</td>
                    <td>{prob}</td>
                </tr>
"""
    
    html_content += f"""
            </table>
            <table>
                <tr>
                    <th>Метрика</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Энтропия топ-K</td>
                    <td>{topics.get("entropy_topk", "N/A")}</td>
                    <td>Энтропия Шеннона распределения вероятностей по топ-K темам</td>
                </tr>
                <tr>
                    <td>Нормализованная энтропия топ-K</td>
                    <td>{topics.get("entropy_topk_norm", "N/A")}</td>
                    <td>Энтропия, нормализованная на log(K)</td>
                </tr>
                <tr>
                    <td>Perplexity топ-K</td>
                    <td>{topics.get("perplexity_topk", "N/A")}</td>
                    <td>Perplexity = exp(энтропия), эффективное количество тем</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Ключевые фразы</h2>
            <table>
                <tr>
                    <th>Метрика</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Количество ключевых фраз</td>
                    <td>{keyphrases.get("count", "N/A")}</td>
                    <td>Общее количество извлеченных ключевых фраз</td>
                </tr>
                <tr>
                    <td>Размерность эмбеддингов</td>
                    <td>{keyphrases.get("embedding_dim", "N/A")}</td>
                    <td>Размерность векторов эмбеддингов ключевых фраз</td>
                </tr>
                <tr>
                    <td>Топ-1 оценка</td>
                    <td>{keyphrases.get("score_top1", "N/A")}</td>
                    <td>Оценка наиболее важной ключевой фразы</td>
                </tr>
                <tr>
                    <td>Средняя оценка</td>
                    <td>{keyphrases.get("score_mean", "N/A")}</td>
                    <td>Средняя оценка всех ключевых фраз</td>
                </tr>
            </table>
"""
    
    # Add keyphrase slots if available
    if keyphrases.get("slots"):
        html_content += f"""
            <h3>Ключевые фразы (хешированные слоты)</h3>
            <table>
                <tr>
                    <th>Ранг</th>
                    <th>Хеш (01)</th>
                    <th>Длина</th>
                </tr>
"""
        for slot in keyphrases.get("slots", []):
            rank = slot.get("rank")
            hash01 = slot.get("hash01", "N/A")
            length = slot.get("len", "N/A")
            html_content += f"""
                <tr>
                    <td>Топ-{rank}</td>
                    <td>{hash01}</td>
                    <td>{length}</td>
                </tr>
"""
        html_content += """
            </table>
"""
    
    html_content += f"""
        </div>
        
        <div class="section">
            <h2>Стилистические признаки</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>FAQ-подобные вопросы</td>
                    <td>{style.get("faq_qmarks", "N/A")}</td>
                    <td>Количество предложений, заканчивающихся на "?"</td>
                </tr>
                <tr>
                    <td>Инструктивный язык</td>
                    <td>{{'Да' if style.get('instructional') else 'Нет'}}</td>
                    <td>Присутствие инструктивных ключевых слов</td>
                </tr>
                <tr>
                    <td>Обращение к аудитории</td>
                    <td>{{'Да' if style.get('audience') else 'Нет'}}</td>
                    <td>Присутствие обращений к аудитории</td>
                </tr>
                <tr>
                    <td>Призыв к действию</td>
                    <td>{{'Да' if style.get('call_to_action') else 'Нет'}}</td>
                    <td>Присутствие призывов к действию</td>
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
                    <td>ASR</td>
                    <td>{{'Да' if presence.get('asr') else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Заголовок</td>
                    <td>{{'Да' if presence.get('title') else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Описание</td>
                    <td>{{'Да' if presence.get('description') else 'Нет'}}</td>
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
                    <td>Включено распределение тем</td>
                    <td>{{'Да' if configuration.get('enable_topic_distribution', 0) > 0.5 else 'Нет'}}</td>
                    <td>Было ли включено извлечение тем</td>
                </tr>
                <tr>
                    <td>Включены ключевые фразы</td>
                    <td>{{'Да' if configuration.get('enable_keyphrases', 0) > 0.5 else 'Нет'}}</td>
                    <td>Было ли включено извлечение ключевых фраз</td>
                </tr>
                <tr>
                    <td>Включены эмбеддинги ключевых фраз</td>
                    <td>{{'Да' if configuration.get('enable_keyphrase_embeddings', 0) > 0.5 else 'Нет'}}</td>
                    <td>Были ли вычислены эмбеддинги ключевых фраз</td>
                </tr>
                <tr>
                    <td>Включены стилистические флаги</td>
                    <td>{{'Да' if configuration.get('enable_style_flags', 0) > 0.5 else 'Нет'}}</td>
                    <td>Были ли вычислены стилистические признаки</td>
                </tr>
                <tr>
                    <td>Топ-K тем</td>
                    <td>{configuration.get("top_k_topics", "N/A")}</td>
                    <td>Количество тем для извлечения</td>
                </tr>
                <tr>
                    <td>Слотов топ-K</td>
                    <td>{configuration.get("top_k_slots", "N/A")}</td>
                    <td>Количество слотов для топ-K тем</td>
                </tr>
                <tr>
                    <td>Температура</td>
                    <td>{configuration.get("temperature", "N/A")}</td>
                    <td>Температура для softmax при вычислении вероятностей</td>
                </tr>
            </table>
        </div>
    </div>
    
    <script>
        // График сходств тем
        var topkScores = {json.dumps(topk_scores)};
        if (Object.keys(topkScores).length > 0) {{
            var trace1 = {{
                x: Object.keys(topkScores),
                y: Object.values(topkScores),
                type: 'bar',
                marker: {{ color: '#4CAF50' }},
                name: 'Сходство'
            }};
            var layout1 = {{
                title: 'Сходства тем (топ-K)',
                xaxis: {{ title: 'Ранг' }},
                yaxis: {{ title: 'Сходство' }}
            }};
            Plotly.newPlot('topics-scores-plot', [trace1], layout1);
        }}
        
        // График вероятностей тем
        var topkProbs = {json.dumps(topk_probs)};
        if (Object.keys(topkProbs).length > 0) {{
            var trace2 = {{
                x: Object.keys(topkProbs),
                y: Object.values(topkProbs),
                type: 'bar',
                marker: {{ color: '#2196F3' }},
                name: 'Вероятность'
            }};
            var layout2 = {{
                title: 'Вероятности тем (топ-K)',
                xaxis: {{ title: 'Ранг' }},
                yaxis: {{ title: 'Вероятность', range: [0, 1] }}
            }};
            Plotly.newPlot('topics-probs-plot', [trace2], layout2);
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
    
    logger.info(f"Semantics topics keyphrases HTML render saved to {output_path}")
    return output_path


__all__ = ["render_semantics_topics_keyphrases", "render_semantics_topics_keyphrases_html"]

