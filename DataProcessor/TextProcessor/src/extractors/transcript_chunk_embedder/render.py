"""
Renderer для transcript_chunk_embedder extractor: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление эмбеддингов чанков транскрипта для визуализации и LLM.
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


def _optional_scalar_flag(d: Dict[str, Any], key: str) -> Any:
    """Like _truthy_scalar but None if key missing or value is None (post-clean)."""
    if key not in d or d.get(key) is None:
        return None
    return _truthy_scalar(d, key)


def render_transcript_chunk_embedder(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для transcript_chunk_embedder extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_tchunk_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "transcript_chunk_embedder",
        "summary": {},
        "embedding": {},
        "sources": {},
        "chunks": {},
        "confidence": {},
        "cache": {},
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
        if key in ["tp_tchunk_present", "tp_tchunk_embedding_dim"]:
            feature_name = key.replace("tp_tchunk_", "")
            embedding[feature_name] = _clean_value(value)
    
    # Extract sources features
    sources = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_tchunk_sources_count") or key.startswith("tp_tchunk_whisper_present") or key.startswith("tp_tchunk_youtube_auto_present"):
            feature_name = key.replace("tp_tchunk_", "")
            sources[feature_name] = _clean_value(value)
    
    # Extract chunks features
    chunks = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_tchunk_whisper_chunks") or key.startswith("tp_tchunk_youtube_chunks"):
            feature_name = key.replace("tp_tchunk_", "")
            chunks[feature_name] = _clean_value(value)
    
    # Extract confidence features
    confidence = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_tchunk_conf_"):
            feature_name = key.replace("tp_tchunk_", "")
            confidence[feature_name] = _clean_value(value)
    
    # Extract cache features
    cache = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_tchunk_cache"):
            feature_name = key.replace("tp_tchunk_", "")
            cache[feature_name] = _clean_value(value)
    
    # Extract configuration features
    configuration = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_tchunk_") and not any(key.startswith(prefix) for prefix in [
            "tp_tchunk_present", "tp_tchunk_embedding_dim",
            "tp_tchunk_sources_count", "tp_tchunk_whisper_present", "tp_tchunk_youtube_auto_present",
            "tp_tchunk_whisper_chunks", "tp_tchunk_youtube_chunks",
            "tp_tchunk_conf_", "tp_tchunk_cache"
        ]):
            feature_name = key.replace("tp_tchunk_", "")
            configuration[feature_name] = _clean_value(value)
    
    render["embedding"] = embedding
    render["sources"] = sources
    render["chunks"] = chunks
    render["confidence"] = confidence
    render["cache"] = cache
    render["configuration"] = configuration
    
    # Summary
    render["summary"] = {
        "present": _truthy_scalar(embedding, "present"),
        "dimension": _clean_value(embedding.get("embedding_dim")),
        "sources_count": _clean_value(sources.get("sources_count", 0)),
        "whisper_present": _truthy_scalar(sources, "whisper_present"),
        "youtube_auto_present": _truthy_scalar(sources, "youtube_auto_present"),
        "whisper_chunks": _clean_value(chunks.get("whisper_chunks", 0)),
        "youtube_chunks": _clean_value(chunks.get("youtube_chunks", 0)),
        "total_chunks": _clean_value((chunks.get("whisper_chunks", 0) or 0) + (chunks.get("youtube_chunks", 0) or 0)),
        "confidence_present": _truthy_scalar(confidence, "conf_present"),
        "confidence_mean": _clean_value(confidence.get("conf_mean")),
        "confidence_min": _clean_value(confidence.get("conf_min")),
        "confidence_max": _clean_value(confidence.get("conf_max")),
        "cache_enabled": _optional_scalar_flag(cache, "cache_enabled"),
    }
    
    return render


def render_transcript_chunk_embedder_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага transcript_chunk_embedder результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_transcript_chunk_embedder(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    embedding = render.get("embedding", {})
    sources = render.get("sources", {})
    chunks = render.get("chunks", {})
    confidence = render.get("confidence", {})
    cache = render.get("cache", {})
    configuration = render.get("configuration", {})
    
    # Prepare data for visualization
    def _clean_value(v: Any) -> Any:
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        return v
    
    # Embedding metrics for visualization
    embedding_metrics = {}
    if embedding.get("embedding_dim") is not None:
        embedding_metrics["Размерность"] = embedding.get("embedding_dim")
    
    # Chunks statistics
    chunks_stats = {}
    if chunks.get("whisper_chunks") is not None and chunks.get("whisper_chunks", 0) > 0:
        chunks_stats["Whisper"] = chunks.get("whisper_chunks")
    if chunks.get("youtube_chunks") is not None and chunks.get("youtube_chunks", 0) > 0:
        chunks_stats["YouTube Auto"] = chunks.get("youtube_chunks")
    
    # Confidence metrics
    conf_metrics = {}
    if confidence.get("conf_mean") is not None:
        conf_metrics["Среднее"] = confidence.get("conf_mean")
    if confidence.get("conf_min") is not None:
        conf_metrics["Минимум"] = confidence.get("conf_min")
    if confidence.get("conf_max") is not None:
        conf_metrics["Максимум"] = confidence.get("conf_max")
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Эмбеддинги чанков транскрипта</title>
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
        .status-badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; font-weight: bold; }}
        .status-ok {{ background-color: #4CAF50; color: white; }}
        .status-no {{ background-color: #f44336; color: white; }}
        .status-info {{ background-color: #2196F3; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Эмбеддинги чанков транскрипта</h1>
        <p><strong>Компонент:</strong> {render.get("component", "transcript_chunk_embedder")}</p>
        <p><strong>Статус:</strong> <span class="status-badge {'status-ok' if summary.get('present') else 'status-no'}">{'✓ Вычислен' if summary.get('present') else '✗ Не вычислен'}</span></p>
        <p><em>Эмбеддинги по чанкам из транскрипта видео. Разбивает транскрипт на перекрывающиеся чанки (по предложениям) и генерирует векторные представления для каждого чанка</em></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Эмбеддинг вычислен</div>
                    <div class="feature-value">{'Да' if summary.get("present") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Размерность</div>
                    <div class="feature-value">{summary.get("dimension", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Количество источников</div>
                    <div class="feature-value">{summary.get("sources_count", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Whisper присутствует</div>
                    <div class="feature-value">{'Да' if summary.get("whisper_present") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">YouTube Auto присутствует</div>
                    <div class="feature-value">{'Да' if summary.get("youtube_auto_present") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Чанков Whisper</div>
                    <div class="feature-value">{summary.get("whisper_chunks", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Чанков YouTube Auto</div>
                    <div class="feature-value">{summary.get("youtube_chunks", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Всего чанков</div>
                    <div class="feature-value">{summary.get("total_chunks", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Метрики confidence</div>
                    <div class="feature-value">{'Да' if summary.get("confidence_present") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Confidence (среднее)</div>
                    <div class="feature-value">{summary.get("confidence_mean", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Кеш включен</div>
                    <div class="feature-value">{'Да' if summary.get("cache_enabled") else 'Нет' if summary.get("cache_enabled") is not None else "N/A"}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Характеристики эмбеддинга</h2>
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
                    <td>Эмбеддинг вычислен</td>
                    <td>{'Да' if embedding.get("present") else 'Нет'}</td>
                    <td>Был ли успешно вычислен эмбеддинг (хотя бы для одного источника)</td>
                </tr>
                <tr>
                    <td>Размерность</td>
                    <td>{embedding.get("embedding_dim", "N/A")}</td>
                    <td>Размерность вектора эмбеддинга для каждого чанка (зависит от модели, например, 384 для MiniLM-L6-v2, 1024 для multilingual-e5-large)</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Источники транскрипта</h2>
            <table>
                <tr>
                    <th>Источник</th>
                    <th>Присутствует</th>
                    <th>Количество чанков</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Whisper (ASR)</td>
                    <td>{'Да' if (sources.get("whisper_present") or 0) > 0.5 else 'Нет'}</td>
                    <td>{chunks.get("whisper_chunks", "N/A")}</td>
                    <td>Транскрипт из ASR модели Whisper (из doc.asr.segments). Содержит метрики confidence</td>
                </tr>
                <tr>
                    <td>YouTube Auto</td>
                    <td>{'Да' if (sources.get("youtube_auto_present") or 0) > 0.5 else 'Нет'}</td>
                    <td>{chunks.get("youtube_chunks", "N/A")}</td>
                    <td>Транскрипт из автоматических субтитров YouTube (из doc.transcripts["youtube_auto"]). Legacy источник</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Статистика чанков</h2>
            <div class="plot-container">
                <div id="chunks-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Количество источников</td>
                    <td>{sources.get("sources_count", "N/A")}</td>
                    <td>Сколько источников транскрипта было обработано (whisper, youtube_auto)</td>
                </tr>
                <tr>
                    <td>Чанков Whisper</td>
                    <td>{chunks.get("whisper_chunks", "N/A")}</td>
                    <td>Количество чанков, созданных из транскрипта Whisper</td>
                </tr>
                <tr>
                    <td>Чанков YouTube Auto</td>
                    <td>{chunks.get("youtube_chunks", "N/A")}</td>
                    <td>Количество чанков, созданных из транскрипта YouTube Auto</td>
                </tr>
                <tr>
                    <td>Всего чанков</td>
                    <td>{summary.get("total_chunks", "N/A")}</td>
                    <td>Общее количество чанков по всем источникам</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Метрики Confidence (Whisper)</h2>
            <div class="plot-container">
                <div id="confidence-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Confidence присутствует</td>
                    <td>{'Да' if (confidence.get("conf_present") or 0) > 0.5 else 'Нет'}</td>
                    <td>Были ли доступны метрики confidence для чанков Whisper (только для ASR источника)</td>
                </tr>
                <tr>
                    <td>Confidence (среднее)</td>
                    <td>{confidence.get("conf_mean", "N/A")}</td>
                    <td>Среднее значение confidence по всем чанкам Whisper (0.0-1.0, где 1.0 = максимальная уверенность)</td>
                </tr>
                <tr>
                    <td>Confidence (минимум)</td>
                    <td>{confidence.get("conf_min", "N/A")}</td>
                    <td>Минимальное значение confidence среди всех чанков Whisper</td>
                </tr>
                <tr>
                    <td>Confidence (максимум)</td>
                    <td>{confidence.get("conf_max", "N/A")}</td>
                    <td>Максимальное значение confidence среди всех чанков Whisper</td>
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
                    <td>{'Да' if (cache.get("cache_enabled") or 0) > 0.5 else 'Нет' if cache.get("cache_enabled") is not None else "N/A"}</td>
                    <td>Было ли включено дисковое кеширование эмбеддингов чанков</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Конфигурация</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Размер батча</div>
                    <div class="feature-value">{configuration.get("batch_size", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Макс. токенов в чанке</div>
                    <div class="feature-value">{configuration.get("max_chunk_tokens_model", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Overlap ratio</div>
                    <div class="feature-value">{configuration.get("overlap_ratio", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Макс. чанков всего</div>
                    <div class="feature-value">{configuration.get("max_chunks_total", "N/A")}</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // График метрик эмбеддинга
        var embeddingMetrics = {json.dumps(embedding_metrics)};
        if (Object.keys(embeddingMetrics).length > 0) {{
            var trace1 = {{
                x: Object.keys(embeddingMetrics),
                y: Object.values(embeddingMetrics),
                type: 'bar',
                marker: {{ color: '#4CAF50' }},
                name: 'Эмбеддинг'
            }};
            var layout1 = {{
                title: 'Характеристики эмбеддинга',
                xaxis: {{ title: 'Метрика' }},
                yaxis: {{ title: 'Значение' }}
            }};
            Plotly.newPlot('embedding-plot', [trace1], layout1);
        }}
        
        // График статистики чанков
        var chunksStats = {json.dumps(chunks_stats)};
        if (Object.keys(chunksStats).length > 0) {{
            var trace2 = {{
                x: Object.keys(chunksStats),
                y: Object.values(chunksStats),
                type: 'bar',
                marker: {{ color: '#2196F3' }},
                name: 'Количество чанков'
            }};
            var layout2 = {{
                title: 'Количество чанков по источникам',
                xaxis: {{ title: 'Источник' }},
                yaxis: {{ title: 'Количество чанков' }}
            }};
            Plotly.newPlot('chunks-plot', [trace2], layout2);
        }}
        
        // График метрик confidence
        var confMetrics = {json.dumps(conf_metrics)};
        if (Object.keys(confMetrics).length > 0) {{
            var trace3 = {{
                x: Object.keys(confMetrics),
                y: Object.values(confMetrics),
                type: 'bar',
                marker: {{ color: '#FF9800' }},
                name: 'Confidence'
            }};
            var layout3 = {{
                title: 'Метрики Confidence (Whisper)',
                xaxis: {{ title: 'Метрика' }},
                yaxis: {{ title: 'Значение', range: [0, 1] }}
            }};
            Plotly.newPlot('confidence-plot', [trace3], layout3);
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
    
    logger.info(f"Transcript chunk embedder HTML render saved to {output_path}")
    return output_path


__all__ = ["render_transcript_chunk_embedder", "render_transcript_chunk_embedder_html"]

