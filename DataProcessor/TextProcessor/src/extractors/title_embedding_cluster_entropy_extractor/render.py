"""
Renderer для title_embedding_cluster_entropy_extractor: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление энтропии распределения эмбеддинга заголовка по кластерам для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_title_embedding_cluster_entropy_extractor(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для title_embedding_cluster_entropy_extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_titleclent_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "title_embedding_cluster_entropy_extractor",
        "summary": {},
        "entropy": {},
        "clusters": {},
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
    
    # Extract entropy metrics
    entropy = {}
    if "tp_titleclent_entropy_raw" in extractor_features:
        entropy["raw"] = _clean_value(extractor_features["tp_titleclent_entropy_raw"])
    if "tp_titleclent_entropy_norm" in extractor_features:
        entropy["normalized"] = _clean_value(extractor_features["tp_titleclent_entropy_norm"])
    if "tp_titleclent_perplexity" in extractor_features:
        entropy["perplexity"] = _clean_value(extractor_features["tp_titleclent_perplexity"])
    
    # Extract cluster metrics
    clusters = {}
    if "tp_titleclent_distinct_clusters_topk" in extractor_features:
        clusters["distinct_topk"] = _clean_value(extractor_features["tp_titleclent_distinct_clusters_topk"])
    if "tp_titleclent_top_k_used" in extractor_features:
        clusters["top_k_used"] = _clean_value(extractor_features["tp_titleclent_top_k_used"])
    if "tp_titleclent_top_k_slots" in extractor_features:
        clusters["top_k_slots"] = _clean_value(extractor_features["tp_titleclent_top_k_slots"])
    if "tp_titleclent_top_k_slots_requested" in extractor_features:
        clusters["top_k_slots_requested"] = _clean_value(extractor_features["tp_titleclent_top_k_slots_requested"])
    if "tp_titleclent_top_k_slots_clamped" in extractor_features:
        v = extractor_features["tp_titleclent_top_k_slots_clamped"]
        clusters["top_k_slots_clamped"] = bool(v > 0.5) if isinstance(v, (float, int, np.floating, np.integer)) else None
    if "tp_titleclent_schema_top_k_slots_max" in extractor_features:
        clusters["schema_top_k_slots_max"] = _clean_value(extractor_features["tp_titleclent_schema_top_k_slots_max"])
    
    # Try to get cluster metadata from payload
    cluster_meta = {}
    if isinstance(payload, dict):
        title_cluster_entropy_meta = payload.get("title_cluster_entropy_meta", {})
        if isinstance(title_cluster_entropy_meta, dict):
            cluster_meta = title_cluster_entropy_meta
    
    # Extract configuration
    configuration = {}
    if "tp_titleclent_temperature" in extractor_features:
        configuration["temperature"] = _clean_value(extractor_features["tp_titleclent_temperature"])
    if "tp_titleclent_backend_faiss" in extractor_features:
        configuration["backend_faiss"] = bool(extractor_features["tp_titleclent_backend_faiss"] > 0.5)
    
    # Extract extra metrics if available
    extra_metrics = {}
    if "tp_titleclent_n_clusters" in extractor_features:
        extra_metrics["n_clusters"] = _clean_value(extractor_features["tp_titleclent_n_clusters"])
    if "tp_titleclent_model_orig_dim" in extractor_features:
        extra_metrics["model_orig_dim"] = _clean_value(extractor_features["tp_titleclent_model_orig_dim"])
    if "tp_titleclent_model_reduced_dim" in extractor_features:
        extra_metrics["model_reduced_dim"] = _clean_value(extractor_features["tp_titleclent_model_reduced_dim"])
    if "tp_titleclent_margin_top2" in extractor_features:
        extra_metrics["margin_top2"] = _clean_value(extractor_features["tp_titleclent_margin_top2"])
    if "tp_titleclent_compute_ms" in extractor_features:
        extra_metrics["compute_ms"] = _clean_value(extractor_features["tp_titleclent_compute_ms"])
    
    # Extract safety flags
    safety = {}
    if "tp_titleclent_dim_mismatch_flag" in extractor_features:
        safety["dim_mismatch_flag"] = bool(extractor_features["tp_titleclent_dim_mismatch_flag"] > 0.5)
    if "tp_titleclent_title_present" in extractor_features:
        safety["title_present"] = bool(extractor_features["tp_titleclent_title_present"] > 0.5)
    
    render["entropy"] = entropy
    render["clusters"] = clusters
    render["cluster_meta"] = cluster_meta
    render["configuration"] = configuration
    render["extra_metrics"] = extra_metrics
    render["safety"] = safety
    
    # Summary
    render["summary"] = {
        "present": bool(extractor_features.get("tp_titleclent_present", 0) > 0.5),
        "entropy_raw": _clean_value(entropy.get("raw")),
        "entropy_normalized": _clean_value(entropy.get("normalized")),
        "perplexity": _clean_value(entropy.get("perplexity")),
        "distinct_clusters": _clean_value(clusters.get("distinct_topk")),
        "top_k_used": _clean_value(clusters.get("top_k_used")),
        "temperature": _clean_value(configuration.get("temperature")),
        "backend_faiss": bool(configuration.get("backend_faiss", False)),
    }
    
    return render


def render_title_embedding_cluster_entropy_extractor_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага title_embedding_cluster_entropy_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_title_embedding_cluster_entropy_extractor(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    entropy = render.get("entropy", {})
    clusters = render.get("clusters", {})
    cluster_meta = render.get("cluster_meta", {})
    configuration = render.get("configuration", {})
    extra_metrics = render.get("extra_metrics", {})
    safety = render.get("safety", {})
    
    # Try to get top-k distribution from payload
    topk_distribution = {}
    if isinstance(payload, dict):
        title_cluster_entropy_meta = payload.get("title_cluster_entropy_meta", {})
        if isinstance(title_cluster_entropy_meta, dict):
            topk_distribution = title_cluster_entropy_meta.get("topk", {})
    
    # Prepare data for visualization
    entropy_data = {}
    if entropy.get("raw") is not None:
        entropy_data["Сырая энтропия"] = entropy.get("raw")
    if entropy.get("normalized") is not None:
        entropy_data["Нормализованная энтропия"] = entropy.get("normalized")
    if entropy.get("perplexity") is not None:
        entropy_data["Perplexity"] = entropy.get("perplexity")
    
    # Filter out None/NaN values
    entropy_data = {k: v for k, v in entropy_data.items() 
                   if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    
    # Prepare top-k distribution data for visualization
    topk_plot_data = {}
    if isinstance(topk_distribution, dict):
        cluster_ids = topk_distribution.get("cluster_ids", [])
        probs = topk_distribution.get("probs", [])
        if cluster_ids and probs:
            topk_plot_data = {f"Кластер {cid}": prob for cid, prob in zip(cluster_ids, probs)}
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Энтропия кластеров эмбеддинга заголовка</title>
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
        <h1>Энтропия кластеров эмбеддинга заголовка</h1>
        <p><strong>Компонент:</strong> {render.get("component", "title_embedding_cluster_entropy_extractor")}</p>
        <p><strong>Статус:</strong> <span style="color: {'green' if summary.get('present') else 'red'};">{{'✓ OK' if summary.get('present') else '✗ Нет данных'}}</span></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Присутствует</div>
                    <div class="feature-value">{{'Да' if summary.get('present') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Сырая энтропия</div>
                    <div class="feature-value">{summary.get("entropy_raw", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Нормализованная энтропия</div>
                    <div class="feature-value">{summary.get("entropy_normalized", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Perplexity</div>
                    <div class="feature-value">{summary.get("perplexity", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Различных кластеров (топ-K)</div>
                    <div class="feature-value">{summary.get("distinct_clusters", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Использовано K</div>
                    <div class="feature-value">{summary.get("top_k_used", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Температура</div>
                    <div class="feature-value">{summary.get("temperature", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Backend</div>
                    <div class="feature-value">{{'FAISS' if summary.get('backend_faiss') else 'NumPy'}}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Энтропия</h2>
            <div class="plot-container">
                <div id="entropy-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Метрика</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Сырая энтропия</td>
                    <td>{entropy.get("raw", "N/A")}</td>
                    <td>Энтропия Шеннона распределения вероятностей по кластерам (биты)</td>
                </tr>
                <tr>
                    <td>Нормализованная энтропия</td>
                    <td>{entropy.get("normalized", "N/A")}</td>
                    <td>Энтропия, нормализованная на log(K), где K - количество кластеров (диапазон [0, 1])</td>
                </tr>
                <tr>
                    <td>Perplexity</td>
                    <td>{entropy.get("perplexity", "N/A")}</td>
                    <td>Perplexity = exp(энтропия), эффективное количество кластеров</td>
                </tr>
            </table>
        </div>
"""
    
    # Add top-k distribution if available
    if topk_distribution and isinstance(topk_distribution, dict):
        cluster_ids = topk_distribution.get("cluster_ids", [])
        probs = topk_distribution.get("probs", [])
        scores = topk_distribution.get("scores", [])
        if cluster_ids:
            html_content += f"""
        <div class="section">
            <h2>Распределение по кластерам (топ-K)</h2>
            <div class="plot-container">
                <div id="topk-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Позиция</th>
                    <th>ID кластера</th>
                    <th>Вероятность</th>
                    <th>Сходство</th>
                </tr>
"""
            for i in range(len(cluster_ids)):
                cid = cluster_ids[i] if i < len(cluster_ids) else "N/A"
                prob = probs[i] if i < len(probs) else "N/A"
                score = scores[i] if i < len(scores) else "N/A"
                html_content += f"""
                <tr>
                    <td>Топ-{i+1}</td>
                    <td>{cid}</td>
                    <td>{prob}</td>
                    <td>{score}</td>
                </tr>
"""
            html_content += """
            </table>
        </div>
"""
    
    html_content += f"""
        <div class="section">
            <h2>Кластеры</h2>
            <table>
                <tr>
                    <th>Параметр</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Различных кластеров (топ-K)</td>
                    <td>{clusters.get("distinct_topk", "N/A")}</td>
                    <td>Количество уникальных кластеров в топ-K</td>
                </tr>
                <tr>
                    <td>Использовано K</td>
                    <td>{clusters.get("top_k_used", "N/A")}</td>
                    <td>Фактическое количество кластеров, использованных для вычисления энтропии</td>
                </tr>
                <tr>
                    <td>Слотов K</td>
                    <td>{clusters.get("top_k_slots", "N/A")}</td>
                    <td>Максимальное количество кластеров для рассмотрения (конфигурация)</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Метаданные кластеров</h2>
            <table>
                <tr>
                    <th>Параметр</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Spec Name</td>
                    <td>{cluster_meta.get("clusters_spec_name", "N/A") if isinstance(cluster_meta, dict) else "N/A"}</td>
                    <td>Имя спецификации кластеров в dp_models</td>
                </tr>
                <tr>
                    <td>Spec Version</td>
                    <td>{cluster_meta.get("clusters_spec_version", "N/A") if isinstance(cluster_meta, dict) else "N/A"}</td>
                    <td>Версия спецификации кластеров</td>
                </tr>
                <tr>
                    <td>Weights Digest</td>
                    <td>{cluster_meta.get("clusters_weights_digest", "N/A") if isinstance(cluster_meta, dict) else "N/A"}</td>
                    <td>Хеш весов модели, использованной для создания кластеров</td>
                </tr>
                <tr>
                    <td>DB Version</td>
                    <td>{cluster_meta.get("cluster_db_version", "N/A") if isinstance(cluster_meta, dict) else "N/A"}</td>
                    <td>Версия базы данных кластеров</td>
                </tr>
                <tr>
                    <td>Backend</td>
                    <td>{cluster_meta.get("backend", "N/A") if isinstance(cluster_meta, dict) else "N/A"}</td>
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
                    <td>Температура</td>
                    <td>{configuration.get("temperature", "N/A")}</td>
                    <td>Температура для softmax при вычислении вероятностей</td>
                </tr>
                <tr>
                    <td>Backend FAISS</td>
                    <td>{{'Да' if configuration.get('backend_faiss') else 'Нет'}}</td>
                    <td>Использовался ли FAISS для поиска</td>
                </tr>
            </table>
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
        if "n_clusters" in extra_metrics:
            html_content += f"""
                <tr>
                    <td>Количество кластеров</td>
                    <td>{extra_metrics.get("n_clusters", "N/A")}</td>
                    <td>Общее количество кластеров в таксономии</td>
                </tr>
"""
        if "model_orig_dim" in extra_metrics:
            html_content += f"""
                <tr>
                    <td>Исходная размерность модели</td>
                    <td>{extra_metrics.get("model_orig_dim", "N/A")}</td>
                    <td>Размерность эмбеддингов до PCA</td>
                </tr>
"""
        if "model_reduced_dim" in extra_metrics:
            html_content += f"""
                <tr>
                    <td>Приведенная размерность модели</td>
                    <td>{extra_metrics.get("model_reduced_dim", "N/A")}</td>
                    <td>Размерность после PCA</td>
                </tr>
"""
        if "margin_top2" in extra_metrics:
            html_content += f"""
                <tr>
                    <td>Margin (топ-1 - топ-2)</td>
                    <td>{extra_metrics.get("margin_top2", "N/A")}</td>
                    <td>Разница между сходством с топ-1 и топ-2 кластерами</td>
                </tr>
"""
        if "compute_ms" in extra_metrics:
            html_content += f"""
                <tr>
                    <td>Время вычисления (мс)</td>
                    <td>{extra_metrics.get("compute_ms", "N/A")}</td>
                    <td>Время вычисления энтропии в миллисекундах</td>
                </tr>
"""
        html_content += """
            </table>
        </div>
"""
    
    html_content += f"""
        <div class="section">
            <h2>Флаги безопасности</h2>
            <table>
                <tr>
                    <th>Флаг</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Заголовок присутствует</td>
                    <td>{{'Да' if safety.get('title_present') else 'Нет'}}</td>
                    <td>Был ли доступен эмбеддинг заголовка</td>
                </tr>
                <tr>
                    <td>Несоответствие размерности</td>
                    <td>{{'Да' if safety.get('dim_mismatch_flag') else 'Нет'}}</td>
                    <td>Соответствует ли размерность эмбеддинга размерности модели</td>
                </tr>
            </table>
        </div>
    </div>
    
    <script>
        // График энтропии
        var entropyData = {json.dumps(entropy_data)};
        if (Object.keys(entropyData).length > 0) {{
            var trace = {{
                x: Object.keys(entropyData),
                y: Object.values(entropyData),
                type: 'bar',
                marker: {{ color: '#4CAF50' }}
            }};
            var layout = {{
                title: 'Метрики энтропии',
                xaxis: {{ title: 'Метрика' }},
                yaxis: {{ title: 'Значение' }}
            }};
            Plotly.newPlot('entropy-plot', [trace], layout);
        }}
"""
    
    # Add top-k distribution plot if available
    if topk_plot_data:
        html_content += f"""
        // График распределения по кластерам
        var topkData = {json.dumps(topk_plot_data)};
        if (Object.keys(topkData).length > 0) {{
            var trace2 = {{
                x: Object.keys(topkData),
                y: Object.values(topkData),
                type: 'bar',
                marker: {{ color: '#2196F3' }}
            }};
            var layout2 = {{
                title: 'Распределение вероятностей по кластерам (топ-K)',
                xaxis: {{ title: 'Кластер' }},
                yaxis: {{ title: 'Вероятность', range: [0, 1] }}
            }};
            Plotly.newPlot('topk-plot', [trace2], layout2);
        }}
"""
    
    html_content += """
    </script>
</body>
</html>
"""
    
    # Atomic write
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    os.replace(tmp_path, output_path)
    
    logger.info(f"Title embedding cluster entropy extractor HTML render saved to {output_path}")
    return output_path


__all__ = ["render_title_embedding_cluster_entropy_extractor", "render_title_embedding_cluster_entropy_extractor_html"]

