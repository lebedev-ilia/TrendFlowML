"""
Renderer для embedding_shift_indicator_extractor: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление индикатора семантического сдвига для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_embedding_shift_indicator_extractor(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для embedding_shift_indicator_extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_embshift_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "embedding_shift_indicator_extractor",
        "summary": {},
        "shift": {},
        "cosines": {},
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
    
    # Extract shift metrics
    shift = {}
    if "tp_embshift_cosine_begin_end" in extractor_features:
        shift["cosine_begin_end"] = _clean_value(extractor_features["tp_embshift_cosine_begin_end"])
    if "tp_embshift_shift_flag" in extractor_features:
        shift_flag = extractor_features["tp_embshift_shift_flag"]
        if shift_flag is not None and shift_flag == shift_flag:  # not None and not NaN
            shift["shift_flag"] = bool(shift_flag > 0.5)
        else:
            shift["shift_flag"] = None
    if "tp_embshift_margin" in extractor_features:
        shift["margin"] = _clean_value(extractor_features["tp_embshift_margin"])
    if "tp_embshift_cosine_threshold" in extractor_features:
        shift["cosine_threshold"] = _clean_value(extractor_features["tp_embshift_cosine_threshold"])
    
    # Extract extra cosines
    cosines = {}
    if "tp_embshift_cosine_first_last" in extractor_features:
        cosines["first_last"] = _clean_value(extractor_features["tp_embshift_cosine_first_last"])
    if "tp_embshift_mean_cosine_last_to_start_window" in extractor_features:
        cosines["mean_last_to_start_window"] = _clean_value(extractor_features["tp_embshift_mean_cosine_last_to_start_window"])
    
    # Extract source tracking
    source = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_embshift_source_used_"):
            source_name = key.replace("tp_embshift_source_used_", "")
            source[source_name] = bool(value > 0.5) if value is not None else False
    if "tp_embshift_used_legacy_key_flag" in extractor_features:
        legacy_flag = extractor_features["tp_embshift_used_legacy_key_flag"]
        source["used_legacy_key"] = bool(legacy_flag > 0.5) if legacy_flag is not None else False
    
    # Extract configuration
    configuration = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_embshift_") and (key.endswith("_enabled") or key.startswith("tp_embshift_n_") or 
                                                key.startswith("tp_embshift_dim") or key.startswith("tp_embshift_require")):
            feature_name = key.replace("tp_embshift_", "")
            configuration[feature_name] = _clean_value(value)
    
    # Extract safety flags
    safety = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_embshift_") and key.endswith("_flag"):
            feature_name = key.replace("tp_embshift_", "")
            safety[feature_name] = bool(value > 0.5) if value is not None else False
    
    render["shift"] = shift
    render["cosines"] = cosines
    render["source"] = source
    render["configuration"] = configuration
    render["safety"] = safety
    
    # Summary
    render["summary"] = {
        "present": bool(extractor_features.get("tp_embshift_present", 0) > 0.5),
        "cosine_begin_end": _clean_value(shift.get("cosine_begin_end")),
        "shift_detected": shift.get("shift_flag"),
        "margin": _clean_value(shift.get("margin")),
        "cosine_threshold": _clean_value(shift.get("cosine_threshold")),
        "n_chunks": _clean_value(configuration.get("n_chunks")),
        "n_window_chunks": _clean_value(configuration.get("n_window_chunks")),
        "dim": _clean_value(configuration.get("dim")),
    }
    
    return render


def render_embedding_shift_indicator_extractor_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага embedding_shift_indicator_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_embedding_shift_indicator_extractor(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    shift = render.get("shift", {})
    cosines = render.get("cosines", {})
    source = render.get("source", {})
    configuration = render.get("configuration", {})
    safety = render.get("safety", {})
    
    # Prepare data for visualization
    cosine_values = {}
    if shift.get("cosine_begin_end") is not None:
        cosine_values["Начало ↔ Конец"] = shift.get("cosine_begin_end")
    if cosines.get("first_last") is not None:
        cosine_values["Первый ↔ Последний"] = cosines.get("first_last")
    if cosines.get("mean_last_to_start_window") is not None:
        cosine_values["Среднее (последние ↔ начало)"] = cosines.get("mean_last_to_start_window")
    
    # Filter out None/NaN values
    cosine_values = {k: v for k, v in cosine_values.items() 
                    if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    
    shift_detected = summary.get("shift_detected")
    shift_status = "Обнаружен" if shift_detected else "Не обнаружен" if shift_detected is False else "N/A"
    shift_color = "red" if shift_detected else "green" if shift_detected is False else "gray"
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Индикатор семантического сдвига</title>
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
        <h1>Индикатор семантического сдвига</h1>
        <p><strong>Компонент:</strong> {render.get("component", "embedding_shift_indicator_extractor")}</p>
        <p><strong>Статус:</strong> <span style="color: {'green' if summary.get('present') else 'red'};">{{'✓ OK' if summary.get('present') else '✗ Нет данных'}}</span></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Присутствует</div>
                    <div class="feature-value">{{'Да' if summary.get('present') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Семантический сдвиг</div>
                    <div class="feature-value" style="color: {shift_color};">{shift_status}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Косинус начало ↔ конец</div>
                    <div class="feature-value">{summary.get("cosine_begin_end", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Порог косинуса</div>
                    <div class="feature-value">{summary.get("cosine_threshold", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Отступ</div>
                    <div class="feature-value">{summary.get("margin", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Количество чанков</div>
                    <div class="feature-value">{summary.get("n_chunks", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Чанков в окне</div>
                    <div class="feature-value">{summary.get("n_window_chunks", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Размерность</div>
                    <div class="feature-value">{summary.get("dim", "N/A")}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Семантический сдвиг</h2>
            <div class="plot-container">
                <div id="cosine-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Метрика</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Косинус начало ↔ конец</td>
                    <td>{shift.get("cosine_begin_end", "N/A")}</td>
                    <td>Косинусное сходство между усредненными эмбеддингами начальных и конечных чанков</td>
                </tr>
                <tr>
                    <td>Порог косинуса</td>
                    <td>{shift.get("cosine_threshold", "N/A")}</td>
                    <td>Пороговое значение косинуса для определения сдвига</td>
                </tr>
                <tr>
                    <td>Отступ</td>
                    <td>{shift.get("margin", "N/A")}</td>
                    <td>Разница между косинусом и порогом (cosine - threshold)</td>
                </tr>
                <tr>
                    <td>Флаг сдвига</td>
                    <td style="color: {shift_color};">{shift_status}</td>
                    <td>Обнаружен ли семантический сдвиг (cosine < threshold)</td>
                </tr>
"""
    
    if cosines.get("first_last") is not None:
        html_content += f"""
                <tr>
                    <td>Косинус первый ↔ последний</td>
                    <td>{cosines.get("first_last", "N/A")}</td>
                    <td>Косинусное сходство между первым и последним чанками</td>
                </tr>
"""
    
    if cosines.get("mean_last_to_start_window") is not None:
        html_content += f"""
                <tr>
                    <td>Среднее косинус (последние ↔ начало)</td>
                    <td>{cosines.get("mean_last_to_start_window", "N/A")}</td>
                    <td>Среднее косинусное сходство между последними чанками и начальным окном</td>
                </tr>
"""
    
    html_content += f"""
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
                <tr>
                    <td>Использован legacy ключ</td>
                    <td>{{'Да' if source.get('used_legacy_key') else 'Нет'}}</td>
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
                    <td>Включен</td>
                    <td>{{'Да' if configuration.get('enabled', 0) > 0.5 else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Требуются чанки транскрипта</td>
                    <td>{{'Да' if configuration.get('require_transcript_chunks_enabled', 0) > 0.5 else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Минимум чанков</td>
                    <td>{configuration.get("require_min_chunks", "N/A")}</td>
                </tr>
                <tr>
                    <td>Вычисление флага сдвига</td>
                    <td>{{'Да' if configuration.get('compute_shift_flag_enabled', 0) > 0.5 else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Вычисление дополнительных косинусов</td>
                    <td>{{'Да' if configuration.get('compute_extra_cosines_enabled', 0) > 0.5 else 'Нет'}}</td>
                </tr>
            </table>
        </div>
    </div>
    
    <script>
        // График косинусных сходств
        var cosineValues = {json.dumps(cosine_values)};
        if (Object.keys(cosineValues).length > 0) {{
            var trace = {{
                x: Object.keys(cosineValues),
                y: Object.values(cosineValues),
                type: 'bar',
                marker: {{ color: '#4CAF50' }}
            }};
            var layout = {{
                title: 'Косинусные сходства для определения сдвига',
                xaxis: {{ title: 'Тип сравнения' }},
                yaxis: {{ title: 'Косинусное сходство', range: [-1, 1] }}
            }};
            Plotly.newPlot('cosine-plot', [trace], layout);
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
    
    logger.info(f"Embedding shift indicator extractor HTML render saved to {output_path}")
    return output_path


__all__ = ["render_embedding_shift_indicator_extractor", "render_embedding_shift_indicator_extractor_html"]

