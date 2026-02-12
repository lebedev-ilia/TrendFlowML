"""
Renderer для transcript_aggregator extractor: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление агрегированных эмбеддингов транскрипта для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_transcript_aggregator(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для transcript_aggregator extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_tragg_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "transcript_aggregator",
        "summary": {},
        "sources": {},
        "aggregation": {},
        "configuration": {},
    }
    
    # Clean NaN values for JSON compatibility
    def _clean_value(v: Any) -> Any:
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        return v
    
    # Extract source presence features
    sources = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_tragg_present_"):
            source_name = key.replace("tp_tragg_present_", "")
            sources[source_name] = {"present": bool(value > 0.5)}
    
    # Extract chunk counts per source
    for key, value in extractor_features.items():
        if key.startswith("tp_tragg_") and key.endswith("_n_chunks"):
            source_name = key.replace("tp_tragg_", "").replace("_n_chunks", "")
            if source_name not in sources:
                sources[source_name] = {}
            sources[source_name]["n_chunks"] = _clean_value(value)
    
    # Extract std per source
    for key, value in extractor_features.items():
        if key.endswith("_mean_std") or key.endswith("_max_std"):
            source_name = key.replace("tp_tragg_", "").replace("_mean_std", "").replace("_max_std", "")
            agg_type = "mean" if "_mean_std" in key else "max"
            if source_name not in sources:
                sources[source_name] = {}
            if "std" not in sources[source_name]:
                sources[source_name]["std"] = {}
            sources[source_name]["std"][agg_type] = _clean_value(value)
    
    # Extract aggregation configuration
    aggregation = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_tragg_compute_") or key.startswith("tp_tragg_decay_rate") or key.startswith("tp_tragg_write"):
            feature_name = key.replace("tp_tragg_", "")
            aggregation[feature_name] = _clean_value(value)
    
    # Extract general configuration
    configuration = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_tragg_") and not any(key.startswith(prefix) for prefix in [
            "tp_tragg_present", "tp_tragg_compute_", "tp_tragg_decay_rate", "tp_tragg_write",
            "tp_tragg_whisper_", "tp_tragg_youtube_", "tp_tragg_combined_"
        ]):
            feature_name = key.replace("tp_tragg_", "")
            configuration[feature_name] = _clean_value(value)
    
    render["sources"] = sources
    render["aggregation"] = aggregation
    render["configuration"] = configuration
    
    # Summary
    present = bool(extractor_features.get("tp_tragg_present", 0) > 0.5)
    whisper_present = bool(extractor_features.get("tp_tragg_present_whisper", 0) > 0.5)
    youtube_present = bool(extractor_features.get("tp_tragg_present_youtube", 0) > 0.5)
    combined_present = bool(extractor_features.get("tp_tragg_present_combined", 0) > 0.5)
    
    render["summary"] = {
        "present": present,
        "whisper_present": whisper_present,
        "youtube_present": youtube_present,
        "combined_present": combined_present,
        "decay_rate": _clean_value(aggregation.get("decay_rate")),
        "compute_mean": bool(aggregation.get("compute_mean", 0) > 0.5),
        "compute_max": bool(aggregation.get("compute_max", 0) > 0.5),
        "compute_std": bool(aggregation.get("compute_std", 0) > 0.5),
        "compute_combined": bool(aggregation.get("compute_combined", 0) > 0.5),
        "write_artifacts": bool(aggregation.get("write_artifacts", 0) > 0.5),
    }
    
    return render


def render_transcript_aggregator_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага transcript_aggregator результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_transcript_aggregator(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    sources = render.get("sources", {})
    aggregation = render.get("aggregation", {})
    configuration = render.get("configuration", {})
    
    # Prepare data for visualization
    source_counts = {}
    for source_name, source_data in sources.items():
        if isinstance(source_data, dict) and "n_chunks" in source_data:
            source_counts[source_name] = source_data.get("n_chunks")
    
    # Filter out None/NaN values
    source_counts = {k: v for k, v in source_counts.items() 
                    if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    
    # Extract source data for HTML template (avoid nested dict access in f-string)
    whisper_data = sources.get("whisper", {})
    whisper_present_val = "Да" if whisper_data.get("present") else "Нет"
    whisper_n_chunks = whisper_data.get("n_chunks", "N/A")
    whisper_std = whisper_data.get("std", {})
    whisper_mean_std = whisper_std.get("mean", "N/A") if isinstance(whisper_std, dict) else "N/A"
    whisper_max_std = whisper_std.get("max", "N/A") if isinstance(whisper_std, dict) else "N/A"
    
    youtube_data = sources.get("youtube_auto", {})
    youtube_present_val = "Да" if youtube_data.get("present") else "Нет"
    youtube_n_chunks = youtube_data.get("n_chunks", "N/A")
    youtube_std = youtube_data.get("std", {})
    youtube_mean_std = youtube_std.get("mean", "N/A") if isinstance(youtube_std, dict) else "N/A"
    youtube_max_std = youtube_std.get("max", "N/A") if isinstance(youtube_std, dict) else "N/A"
    
    combined_data = sources.get("combined", {})
    combined_present_val = "Да" if combined_data.get("present") else "Нет"
    combined_n_chunks = combined_data.get("n_chunks", "N/A")
    combined_std = combined_data.get("std", {})
    combined_mean_std = combined_std.get("mean", "N/A") if isinstance(combined_std, dict) else "N/A"
    combined_max_std = combined_std.get("max", "N/A") if isinstance(combined_std, dict) else "N/A"
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Агрегация эмбеддингов транскрипта</title>
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
        <h1>Агрегация эмбеддингов транскрипта</h1>
        <p><strong>Компонент:</strong> {render.get("component", "transcript_aggregator")}</p>
        <p><strong>Статус:</strong> <span style="color: {'green' if summary.get('present') else 'red'};">{{'✓ OK' if summary.get('present') else '✗ Нет данных'}}</span></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Присутствует</div>
                    <div class="feature-value">{{'Да' if summary.get('present') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Whisper присутствует</div>
                    <div class="feature-value">{{'Да' if summary.get('whisper_present') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">YouTube Auto присутствует</div>
                    <div class="feature-value">{{'Да' if summary.get('youtube_present') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Комбинированный присутствует</div>
                    <div class="feature-value">{{'Да' if summary.get('combined_present') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Коэффициент затухания</div>
                    <div class="feature-value">{summary.get("decay_rate", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Вычисление среднего</div>
                    <div class="feature-value">{{'Да' if summary.get('compute_mean') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Вычисление максимума</div>
                    <div class="feature-value">{{'Да' if summary.get('compute_max') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Вычисление стандартного отклонения</div>
                    <div class="feature-value">{{'Да' if summary.get('compute_std') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Вычисление комбинированного</div>
                    <div class="feature-value">{{'Да' if summary.get('compute_combined') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Запись артефактов</div>
                    <div class="feature-value">{{'Да' if summary.get('write_artifacts') else 'Нет'}}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Источники транскрипта</h2>
            <div class="plot-container">
                <div id="sources-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Источник</th>
                    <th>Присутствует</th>
                    <th>Количество чанков</th>
                    <th>Среднее std</th>
                    <th>Максимум std</th>
                </tr>
                <tr>
                    <td>whisper</td>
                    <td>{whisper_present_val}</td>
                    <td>{whisper_n_chunks}</td>
                    <td>{whisper_mean_std}</td>
                    <td>{whisper_max_std}</td>
                </tr>
                <tr>
                    <td>youtube_auto</td>
                    <td>{youtube_present_val}</td>
                    <td>{youtube_n_chunks}</td>
                    <td>{youtube_mean_std}</td>
                    <td>{youtube_max_std}</td>
                </tr>
                <tr>
                    <td>combined</td>
                    <td>{combined_present_val}</td>
                    <td>{combined_n_chunks}</td>
                    <td>{combined_mean_std}</td>
                    <td>{combined_max_std}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Агрегация</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Коэффициент затухания</td>
                    <td>{aggregation.get("decay_rate", "N/A")}</td>
                    <td>Коэффициент экспоненциального затухания для взвешенного среднего</td>
                </tr>
                <tr>
                    <td>Вычисление среднего</td>
                    <td>{{'Да' if aggregation.get('compute_mean', 0) > 0.5 else 'Нет'}}</td>
                    <td>Было ли вычислено взвешенное среднее эмбеддингов чанков</td>
                </tr>
                <tr>
                    <td>Вычисление максимума</td>
                    <td>{{'Да' if aggregation.get('compute_max', 0) > 0.5 else 'Нет'}}</td>
                    <td>Был ли применен max pooling к эмбеддингам чанков</td>
                </tr>
                <tr>
                    <td>Вычисление стандартного отклонения</td>
                    <td>{{'Да' if aggregation.get('compute_std', 0) > 0.5 else 'Нет'}}</td>
                    <td>Было ли вычислено стандартное отклонение эмбеддингов</td>
                </tr>
                <tr>
                    <td>Вычисление комбинированного</td>
                    <td>{{'Да' if aggregation.get('compute_combined', 0) > 0.5 else 'Нет'}}</td>
                    <td>Были ли объединены эмбеддинги из разных источников</td>
                </tr>
                <tr>
                    <td>Запись артефактов</td>
                    <td>{{'Да' if aggregation.get('write_artifacts', 0) > 0.5 else 'Нет'}}</td>
                    <td>Были ли записаны файлы с агрегированными эмбеддингами</td>
                </tr>
            </table>
        </div>
    </div>
    
    <script>
        // График количества чанков по источникам
        var sourceCounts = {json.dumps(source_counts)};
        if (Object.keys(sourceCounts).length > 0) {{
            var trace = {{
                x: Object.keys(sourceCounts),
                y: Object.values(sourceCounts),
                type: 'bar',
                marker: {{ color: '#4CAF50' }}
            }};
            var layout = {{
                title: 'Количество чанков по источникам',
                xaxis: {{ title: 'Источник' }},
                yaxis: {{ title: 'Количество чанков' }}
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
    
    logger.info(f"Transcript aggregator HTML render saved to {output_path}")
    return output_path


__all__ = ["render_transcript_aggregator", "render_transcript_aggregator_html"]

