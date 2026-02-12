"""
Renderer для qa_embedding_pairs_extractor: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление извлеченных вопросов и их эмбеддингов для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_qa_embedding_pairs_extractor(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для qa_embedding_pairs_extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_qa_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "qa_embedding_pairs_extractor",
        "summary": {},
        "questions": {},
        "sources": {},
        "configuration": {},
        "artifacts": {},
    }
    
    # Clean NaN values for JSON compatibility
    def _clean_value(v: Any) -> Any:
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        return v
    
    # Extract question metrics
    questions = {}
    if "tp_qa_num_questions" in extractor_features:
        questions["count"] = _clean_value(extractor_features["tp_qa_num_questions"])
    if "tp_qa_embedding_dim" in extractor_features:
        questions["embedding_dim"] = _clean_value(extractor_features["tp_qa_embedding_dim"])
    
    # Extract per-source question counts
    sources = {}
    if "tp_qa_q_title" in extractor_features:
        sources["title"] = _clean_value(extractor_features["tp_qa_q_title"])
    if "tp_qa_q_description" in extractor_features:
        sources["description"] = _clean_value(extractor_features["tp_qa_q_description"])
    if "tp_qa_q_transcript" in extractor_features:
        sources["transcript"] = _clean_value(extractor_features["tp_qa_q_transcript"])
    if "tp_qa_q_comments" in extractor_features:
        sources["comments"] = _clean_value(extractor_features["tp_qa_q_comments"])
    
    # Extract configuration
    configuration = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_qa_") and (key.endswith("_enabled") or key.startswith("tp_qa_use_") or 
                                         key.startswith("tp_qa_require_") or key.startswith("tp_qa_max_") or
                                         key.startswith("tp_qa_min_") or key.startswith("tp_qa_transcript_source_policy_") or
                                         key.startswith("tp_qa_allow_") or key.startswith("tp_qa_dedup_") or
                                         key.startswith("tp_qa_write_")):
            feature_name = key.replace("tp_qa_", "")
            configuration[feature_name] = _clean_value(value)
    
    # Extract artifacts status
    artifacts = {}
    if "tp_qa_hashes_written" in extractor_features:
        artifacts["hashes_written"] = bool(extractor_features["tp_qa_hashes_written"] > 0.5)
    if "tp_qa_source_ids_written" in extractor_features:
        artifacts["source_ids_written"] = bool(extractor_features["tp_qa_source_ids_written"] > 0.5)
    
    # Extract extra metrics if available
    extra_metrics = {}
    if "tp_qa_questions_per_min" in extractor_features:
        extra_metrics["questions_per_min"] = _clean_value(extractor_features["tp_qa_questions_per_min"])
    if "tp_qa_questions_per_1k_chars" in extractor_features:
        extra_metrics["questions_per_1k_chars"] = _clean_value(extractor_features["tp_qa_questions_per_1k_chars"])
    if "tp_qa_mean_cosine_to_centroid" in extractor_features:
        extra_metrics["mean_cosine_to_centroid"] = _clean_value(extractor_features["tp_qa_mean_cosine_to_centroid"])
    
    render["questions"] = questions
    render["sources"] = sources
    render["configuration"] = configuration
    render["artifacts"] = artifacts
    render["extra_metrics"] = extra_metrics
    
    # Summary
    render["summary"] = {
        "present": bool(extractor_features.get("tp_qa_present", 0) > 0.5),
        "num_questions": _clean_value(questions.get("count")),
        "embedding_dim": _clean_value(questions.get("embedding_dim")),
        "total_from_sources": sum([v for v in sources.values() if v is not None and isinstance(v, (int, float))]),
    }
    
    return render


def render_qa_embedding_pairs_extractor_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага qa_embedding_pairs_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_qa_embedding_pairs_extractor(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    questions = render.get("questions", {})
    sources = render.get("sources", {})
    configuration = render.get("configuration", {})
    artifacts = render.get("artifacts", {})
    extra_metrics = render.get("extra_metrics", {})
    
    # Prepare data for visualization
    source_counts = {}
    for source_name, count in sources.items():
        if count is not None and not (isinstance(count, float) and (math.isnan(count) or math.isinf(count))):
            source_counts[source_name] = count
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Вопросы и эмбеддинги</title>
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
        <h1>Вопросы и эмбеддинги</h1>
        <p><strong>Компонент:</strong> {render.get("component", "qa_embedding_pairs_extractor")}</p>
        <p><strong>Статус:</strong> <span style="color: {'green' if summary.get('present') else 'red'};">{{'✓ OK' if summary.get('present') else '✗ Нет данных'}}</span></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Присутствует</div>
                    <div class="feature-value">{{'Да' if summary.get('present') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Количество вопросов</div>
                    <div class="feature-value">{summary.get("num_questions", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Размерность эмбеддингов</div>
                    <div class="feature-value">{summary.get("embedding_dim", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Всего из источников</div>
                    <div class="feature-value">{summary.get("total_from_sources", "N/A")}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Вопросы по источникам</h2>
            <div class="plot-container">
                <div id="sources-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Источник</th>
                    <th>Количество вопросов</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Заголовок</td>
                    <td>{sources.get("title", "N/A")}</td>
                    <td>Вопросы, извлеченные из заголовка видео</td>
                </tr>
                <tr>
                    <td>Описание</td>
                    <td>{sources.get("description", "N/A")}</td>
                    <td>Вопросы, извлеченные из описания видео</td>
                </tr>
                <tr>
                    <td>Транскрипт</td>
                    <td>{sources.get("transcript", "N/A")}</td>
                    <td>Вопросы, извлеченные из транскрипта</td>
                </tr>
                <tr>
                    <td>Комментарии</td>
                    <td>{sources.get("comments", "N/A")}</td>
                    <td>Вопросы, извлеченные из комментариев</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Вопросы</h2>
            <table>
                <tr>
                    <th>Метрика</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Количество вопросов</td>
                    <td>{questions.get("count", "N/A")}</td>
                    <td>Общее количество извлеченных вопросов</td>
                </tr>
                <tr>
                    <td>Размерность эмбеддингов</td>
                    <td>{questions.get("embedding_dim", "N/A")}</td>
                    <td>Размерность векторов эмбеддингов вопросов</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Артефакты</h2>
            <table>
                <tr>
                    <th>Артефакт</th>
                    <th>Статус</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Хеши вопросов</td>
                    <td>{{'Записан' if artifacts.get('hashes_written') else 'Не записан'}}</td>
                    <td>Был ли записан артефакт с хешами вопросов</td>
                </tr>
                <tr>
                    <td>ID источников</td>
                    <td>{{'Записан' if artifacts.get('source_ids_written') else 'Не записан'}}</td>
                    <td>Был ли записан артефакт с ID источников вопросов</td>
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
                    <td>Использовать заголовок</td>
                    <td>{{'Да' if configuration.get('use_title', 0) > 0.5 else 'Нет'}}</td>
                    <td>Извлекать ли вопросы из заголовка</td>
                </tr>
                <tr>
                    <td>Использовать описание</td>
                    <td>{{'Да' if configuration.get('use_description', 0) > 0.5 else 'Нет'}}</td>
                    <td>Извлекать ли вопросы из описания</td>
                </tr>
                <tr>
                    <td>Использовать транскрипт</td>
                    <td>{{'Да' if configuration.get('use_transcript', 0) > 0.5 else 'Нет'}}</td>
                    <td>Извлекать ли вопросы из транскрипта</td>
                </tr>
                <tr>
                    <td>Использовать комментарии</td>
                    <td>{{'Да' if configuration.get('use_comments', 0) > 0.5 else 'Нет'}}</td>
                    <td>Извлекать ли вопросы из комментариев</td>
                </tr>
                <tr>
                    <td>Максимум вопросов всего</td>
                    <td>{configuration.get("max_questions_total", "N/A")}</td>
                    <td>Максимальное общее количество вопросов</td>
                </tr>
                <tr>
                    <td>Максимум вопросов на источник</td>
                    <td>{configuration.get("max_questions_per_source", "N/A")}</td>
                    <td>Максимальное количество вопросов из одного источника</td>
                </tr>
            </table>
        </div>
"""
    
    # Add extra metrics if available
    if extra_metrics:
        html_content += f"""
        <div class="section">
            <h2>Дополнительные метрики</h2>
            <table>
                <tr>
                    <th>Метрика</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
"""
        if "questions_per_min" in extra_metrics:
            html_content += f"""
                <tr>
                    <td>Вопросов в минуту</td>
                    <td>{extra_metrics.get("questions_per_min", "N/A")}</td>
                    <td>Плотность вопросов относительно длительности видео</td>
                </tr>
"""
        if "questions_per_1k_chars" in extra_metrics:
            html_content += f"""
                <tr>
                    <td>Вопросов на 1000 символов</td>
                    <td>{extra_metrics.get("questions_per_1k_chars", "N/A")}</td>
                    <td>Плотность вопросов относительно длины текста</td>
                </tr>
"""
        if "mean_cosine_to_centroid" in extra_metrics:
            html_content += f"""
                <tr>
                    <td>Среднее косинусное сходство к центроиду</td>
                    <td>{extra_metrics.get("mean_cosine_to_centroid", "N/A")}</td>
                    <td>Среднее косинусное сходство эмбеддингов вопросов к центроиду</td>
                </tr>
"""
        html_content += """
            </table>
        </div>
"""
    
    html_content += f"""
    </div>
    
    <script>
        // График распределения вопросов по источникам
        var sourceCounts = {json.dumps(source_counts)};
        if (Object.keys(sourceCounts).length > 0) {{
            var trace = {{
                x: Object.keys(sourceCounts),
                y: Object.values(sourceCounts),
                type: 'bar',
                marker: {{ color: '#4CAF50' }}
            }};
            var layout = {{
                title: 'Распределение вопросов по источникам',
                xaxis: {{ title: 'Источник' }},
                yaxis: {{ title: 'Количество вопросов' }}
            }};
            Plotly.newPlot('sources-plot', [trace], layout);
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
    
    logger.info(f"QA embedding pairs extractor HTML render saved to {output_path}")
    return output_path


__all__ = ["render_qa_embedding_pairs_extractor", "render_qa_embedding_pairs_extractor_html"]

