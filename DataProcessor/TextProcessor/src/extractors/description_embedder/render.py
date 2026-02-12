"""
Renderer для description_embedder extractor: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление эмбеддингов описаний для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_description_embedder(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для description_embedder extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_descemb_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "description_embedder",
        "summary": {},
        "embedding": {},
        "chunking": {},
        "cache": {},
        "performance": {},
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
        if key in ["tp_descemb_present", "tp_descemb_dim", "tp_descemb_norm_raw", "tp_descemb_l2_norm"]:
            feature_name = key.replace("tp_descemb_", "")
            embedding[feature_name] = _clean_value(value)
    
    # Extract chunking features
    chunking = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_descemb_n_chunks") or key.startswith("tp_descemb_avg_chunk_tokens") or key.startswith("tp_descemb_pooling"):
            feature_name = key.replace("tp_descemb_", "")
            chunking[feature_name] = _clean_value(value)
    
    # Extract cache features
    cache = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_descemb_cache") or key.startswith("tp_descemb_artifact"):
            feature_name = key.replace("tp_descemb_", "")
            cache[feature_name] = _clean_value(value)
    
    # Extract performance features
    performance = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_descemb_encode_ms") or key.startswith("tp_descemb_chunk_ms") or key.startswith("tp_descemb_pool_ms") or key.startswith("tp_descemb_device") or key.startswith("tp_descemb_fp16"):
            feature_name = key.replace("tp_descemb_", "")
            performance[feature_name] = _clean_value(value)
    
    # Extract configuration features
    configuration = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_descemb_") and not any(key.startswith(prefix) for prefix in [
            "tp_descemb_present", "tp_descemb_dim", "tp_descemb_norm_raw", "tp_descemb_l2_norm",
            "tp_descemb_cache", "tp_descemb_artifact", "tp_descemb_encode_ms", "tp_descemb_chunk_ms", "tp_descemb_pool_ms",
            "tp_descemb_device", "tp_descemb_fp16", "tp_descemb_n_chunks", "tp_descemb_avg_chunk_tokens", "tp_descemb_pooling"
        ]):
            feature_name = key.replace("tp_descemb_", "")
            configuration[feature_name] = _clean_value(value)
    
    render["embedding"] = embedding
    render["chunking"] = chunking
    render["cache"] = cache
    render["performance"] = performance
    render["configuration"] = configuration
    
    # Summary
    render["summary"] = {
        "present": bool(embedding.get("present", 0) > 0.5),
        "description_present": bool(configuration.get("description_present", 0) > 0.5),
        "dimension": _clean_value(embedding.get("dim")),
        "norm_raw": _clean_value(embedding.get("norm_raw")),
        "l2_norm": _clean_value(embedding.get("l2_norm")),
        "n_chunks": _clean_value(chunking.get("n_chunks")),
        "avg_chunk_tokens": _clean_value(chunking.get("avg_chunk_tokens")),
        "pooling_strategy": "length_weighted_mean" if chunking.get("pooling_length_weighted", 0) > 0.5 else "mean",
        "cache_hit": bool(cache.get("cache_hit", 0) > 0.5) if cache.get("cache_hit") is not None else None,
        "encode_time_ms": _clean_value(performance.get("encode_ms")),
        "chunk_time_ms": _clean_value(performance.get("chunk_ms")),
        "pool_time_ms": _clean_value(performance.get("pool_ms")),
        "device": "CUDA" if performance.get("device_cuda", 0) > 0.5 else "CPU",
    }
    
    return render


def render_description_embedder_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага description_embedder результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_description_embedder(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    embedding = render.get("embedding", {})
    chunking = render.get("chunking", {})
    cache = render.get("cache", {})
    performance = render.get("performance", {})
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
    if embedding.get("dim") is not None:
        embedding_metrics["Размерность"] = embedding.get("dim")
    if embedding.get("norm_raw") is not None:
        embedding_metrics["Норма (raw)"] = embedding.get("norm_raw")
    if embedding.get("l2_norm") is not None:
        embedding_metrics["L2-норма"] = embedding.get("l2_norm")
    
    # Performance metrics
    perf_metrics = {}
    if performance.get("encode_ms") is not None:
        perf_metrics["Кодирование (мс)"] = performance.get("encode_ms")
    if performance.get("chunk_ms") is not None:
        perf_metrics["Разбиение на чанки (мс)"] = performance.get("chunk_ms")
    if performance.get("pool_ms") is not None:
        perf_metrics["Pooling (мс)"] = performance.get("pool_ms")
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Эмбеддинги описаний</title>
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
        <h1>Эмбеддинги описаний</h1>
        <p><strong>Компонент:</strong> {render.get("component", "description_embedder")}</p>
        <p><strong>Статус:</strong> <span class="status-badge {'status-ok' if summary.get('present') else 'status-no'}">{'✓ Вычислен' if summary.get('present') else '✗ Не вычислен'}</span></p>
        <p><em>L2-нормализованные эмбеддинги для описаний видео с использованием sentence transformers и chunk-and-aggregate для длинных текстов</em></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Эмбеддинг вычислен</div>
                    <div class="feature-value">{'Да' if summary.get("present") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Описание присутствует</div>
                    <div class="feature-value">{'Да' if summary.get("description_present") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Размерность</div>
                    <div class="feature-value">{summary.get("dimension", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Норма (raw)</div>
                    <div class="feature-value">{summary.get("norm_raw", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">L2-норма</div>
                    <div class="feature-value">{summary.get("l2_norm", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Количество чанков</div>
                    <div class="feature-value">{summary.get("n_chunks", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Среднее токенов в чанке</div>
                    <div class="feature-value">{summary.get("avg_chunk_tokens", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Стратегия pooling</div>
                    <div class="feature-value">{summary.get("pooling_strategy", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Попадание в кеш</div>
                    <div class="feature-value">{'Да' if summary.get("cache_hit") else 'Нет' if summary.get("cache_hit") is not None else "N/A"}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Время кодирования (мс)</div>
                    <div class="feature-value">{summary.get("encode_time_ms", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Время разбиения (мс)</div>
                    <div class="feature-value">{summary.get("chunk_time_ms", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Время pooling (мс)</div>
                    <div class="feature-value">{summary.get("pool_time_ms", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Устройство</div>
                    <div class="feature-value">{summary.get("device", "N/A")}</div>
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
                    <td>Был ли успешно вычислен эмбеддинг (не просто наличие артефакта)</td>
                </tr>
                <tr>
                    <td>Размерность</td>
                    <td>{embedding.get("dim", "N/A")}</td>
                    <td>Размерность вектора эмбеддинга (зависит от модели, например, 384 для MiniLM-L6-v2, 1024 для multilingual-e5-large)</td>
                </tr>
                <tr>
                    <td>Норма (raw)</td>
                    <td>{embedding.get("norm_raw", "N/A")}</td>
                    <td>L2-норма необработанного вектора (до нормализации). Показывает "силу" эмбеддинга</td>
                </tr>
                <tr>
                    <td>L2-норма</td>
                    <td>{embedding.get("l2_norm", "N/A")}</td>
                    <td>L2-норма нормализованного вектора (после нормализации, должна быть ~1.0)</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Разбиение на чанки (Chunking)</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Количество чанков</td>
                    <td>{chunking.get("n_chunks", "N/A")}</td>
                    <td>На сколько частей был разбит текст описания для обработки (для длинных текстов > max_chunk_tokens_model)</td>
                </tr>
                <tr>
                    <td>Среднее токенов в чанке</td>
                    <td>{chunking.get("avg_chunk_tokens", "N/A")}</td>
                    <td>Среднее количество токенов в одном чанке (используется для весов при pooling)</td>
                </tr>
                <tr>
                    <td>Стратегия pooling</td>
                    <td>{summary.get("pooling_strategy", "N/A")}</td>
                    <td>Метод агрегации эмбеддингов чанков: length_weighted_mean (веса по длине) или mean (равномерные веса)</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Производительность</h2>
            <div class="plot-container">
                <div id="performance-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Время кодирования (мс)</td>
                    <td>{performance.get("encode_ms", "N/A")}</td>
                    <td>Время, затраченное на кодирование чанков через модель (в миллисекундах)</td>
                </tr>
                <tr>
                    <td>Время разбиения (мс)</td>
                    <td>{performance.get("chunk_ms", "N/A")}</td>
                    <td>Время, затраченное на разбиение текста на чанки (token-aware chunking)</td>
                </tr>
                <tr>
                    <td>Время pooling (мс)</td>
                    <td>{performance.get("pool_ms", "N/A")}</td>
                    <td>Время, затраченное на агрегацию эмбеддингов чанков в финальный вектор</td>
                </tr>
                <tr>
                    <td>Устройство</td>
                    <td>{'CUDA' if performance.get("device_cuda", 0) > 0.5 else 'CPU'}</td>
                    <td>Устройство, на котором выполнялось кодирование (CPU или CUDA/GPU)</td>
                </tr>
                <tr>
                    <td>FP16 режим</td>
                    <td>{'Да' if performance.get("fp16", 0) > 0.5 else 'Нет'}</td>
                    <td>Использовался ли режим float16 для экономии памяти (только на GPU)</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Кеширование и артефакты</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Кеш включен</td>
                    <td>{'Да' if cache.get("cache_enabled", 0) > 0.5 else 'Нет'}</td>
                    <td>Было ли включено дисковое кеширование эмбеддингов</td>
                </tr>
                <tr>
                    <td>Попадание в кеш</td>
                    <td>{'Да' if cache.get("cache_hit", 0) > 0.5 else 'Нет' if cache.get("cache_hit") is not None else "N/A"}</td>
                    <td>Был ли эмбеддинг загружен из кеша (1) или вычислен заново (0)</td>
                </tr>
                <tr>
                    <td>Запись артефакта включена</td>
                    <td>{'Да' if cache.get("write_artifact_enabled", 0) > 0.5 else 'Нет'}</td>
                    <td>Было ли включено сохранение артефакта в per-run директорию</td>
                </tr>
                <tr>
                    <td>Артефакт записан</td>
                    <td>{'Да' if cache.get("artifact_written", 0) > 0.5 else 'Нет'}</td>
                    <td>Был ли успешно записан .npy файл с эмбеддингом</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Конфигурация</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Вычисление включено</div>
                    <div class="feature-value">{'Да' if configuration.get("compute_enabled", 0) > 0.5 else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Вычисление нормы raw</div>
                    <div class="feature-value">{'Да' if configuration.get("compute_raw_norm", 0) > 0.5 else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Хеш модели (u24)</div>
                    <div class="feature-value">{configuration.get("model_digest_u24", "N/A")}</div>
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
        
        // График производительности
        var perfMetrics = {json.dumps(perf_metrics)};
        if (Object.keys(perfMetrics).length > 0) {{
            var trace2 = {{
                x: Object.keys(perfMetrics),
                y: Object.values(perfMetrics),
                type: 'bar',
                marker: {{ color: '#2196F3' }},
                name: 'Время (мс)'
            }};
            var layout2 = {{
                title: 'Время выполнения операций',
                xaxis: {{ title: 'Операция' }},
                yaxis: {{ title: 'Время (мс)' }}
            }};
            Plotly.newPlot('performance-plot', [trace2], layout2);
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
    
    logger.info(f"Description embedder HTML render saved to {output_path}")
    return output_path


__all__ = ["render_description_embedder", "render_description_embedder_html"]

