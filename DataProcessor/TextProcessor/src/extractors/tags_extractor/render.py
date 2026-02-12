"""
Renderer для tags_extractor: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление признаков хэштегов для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_tags_extractor(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для tags_extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_tags_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "tags_extractor",
        "summary": {},
        "title_features": {},
        "description_features": {},
        "hashtag_features": {},
        "top_k_tags": [],
        "statistics": {},
    }
    
    # Clean NaN values for JSON compatibility
    def _clean_value(v: Any) -> Any:
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        return v
    
    # Extract title features (prefix: tp_tags_title_*)
    title_features = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_tags_title_"):
            feature_name = key.replace("tp_tags_title_", "")
            title_features[feature_name] = _clean_value(value)
    
    # Extract description features (prefix: tp_tags_description_*)
    description_features = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_tags_description_"):
            feature_name = key.replace("tp_tags_description_", "")
            description_features[feature_name] = _clean_value(value)
    
    # Extract hashtag features (tp_tags_hashtag_*)
    hashtag_features = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_tags_hashtag_") and not key.startswith("tp_tags_hashtags_"):
            feature_name = key.replace("tp_tags_hashtag_", "")
            hashtag_features[feature_name] = _clean_value(value)
    
    # Extract top-K tags (tp_tags_top{i}_*)
    top_k_slots = int(extractor_features.get("tp_tags_topk_slots", 5))
    top_k_tags = []
    for i in range(1, top_k_slots + 1):
        present = extractor_features.get(f"tp_tags_top{i}_present", 0.0)
        if present and present > 0.5:
            tag_info = {
                "rank": i,
                "present": True,
                "hash01": _clean_value(extractor_features.get(f"tp_tags_top{i}_hash01")),
                "length": _clean_value(extractor_features.get(f"tp_tags_top{i}_len")),
            }
            top_k_tags.append(tag_info)
        else:
            top_k_tags.append({
                "rank": i,
                "present": False,
                "hash01": None,
                "length": None,
            })
    
    # Extract general statistics (tp_tags_* but not title/description/hashtag/top specific)
    stats = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_tags_") and not any(key.startswith(prefix) for prefix in [
            "tp_tags_title_", "tp_tags_description_", "tp_tags_hashtag_", "tp_tags_top"
        ]):
            stats[key.replace("tp_tags_", "")] = _clean_value(value)
    
    render["title_features"] = title_features
    render["description_features"] = description_features
    render["hashtag_features"] = hashtag_features
    render["top_k_tags"] = top_k_tags
    render["statistics"] = stats
    
    # Summary
    render["summary"] = {
        "title_hashtag_count": _clean_value(title_features.get("hashtag_found_count", 0)),
        "description_hashtag_count": _clean_value(description_features.get("hashtag_found_count", 0)),
        "total_hashtag_count": _clean_value(hashtag_features.get("total_found_count", 0)),
        "unique_hashtag_count": _clean_value(hashtag_features.get("unique_count", 0)),
        "avg_hashtag_length": _clean_value(hashtag_features.get("avg_len")),
        "max_hashtag_length": _clean_value(hashtag_features.get("max_len")),
        "top_k_slots": top_k_slots,
        "has_hashtags": bool(hashtag_features.get("total_found_count", 0) > 0),
    }
    
    return render


def render_tags_extractor_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага tags_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости с AudioProcessor API)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_tags_extractor(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    title_features = render.get("title_features", {})
    description_features = render.get("description_features", {})
    hashtag_features = render.get("hashtag_features", {})
    top_k_tags = render.get("top_k_tags", [])
    statistics = render.get("statistics", {})
    
    # Prepare data for visualization
    def _clean_value(v: Any) -> Any:
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        return v
    
    # Hashtag counts for bar chart
    hashtag_counts = {
        "В заголовке": title_features.get("hashtag_found_count", 0),
        "В описании": description_features.get("hashtag_found_count", 0),
        "Всего найдено": hashtag_features.get("total_found_count", 0),
        "Уникальных": hashtag_features.get("unique_count", 0),
    }
    hashtag_counts = {k: v for k, v in hashtag_counts.items() 
                     if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    
    # Top-K tags lengths
    top_k_lengths = {}
    for tag_info in top_k_tags:
        if tag_info.get("present") and tag_info.get("length") is not None:
            rank = tag_info.get("rank", 0)
            top_k_lengths[f"Топ-{rank}"] = tag_info.get("length")
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Извлечение хэштегов</title>
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
        .tag-row {{ background-color: #f9f9f9; }}
        .tag-row.present {{ background-color: #e8f5e9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Извлечение хэштегов</h1>
        <p><strong>Компонент:</strong> {render.get("component", "tags_extractor")}</p>
        <p><strong>Статус:</strong> <span style="color: green;">✓ OK</span></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Хэштегов в заголовке</div>
                    <div class="feature-value">{summary.get("title_hashtag_count", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Хэштегов в описании</div>
                    <div class="feature-value">{summary.get("description_hashtag_count", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Всего найдено</div>
                    <div class="feature-value">{summary.get("total_hashtag_count", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Уникальных</div>
                    <div class="feature-value">{summary.get("unique_hashtag_count", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Средняя длина</div>
                    <div class="feature-value">{summary.get("avg_hashtag_length", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Максимальная длина</div>
                    <div class="feature-value">{summary.get("max_hashtag_length", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Содержит хэштеги</div>
                    <div class="feature-value">{'Да' if summary.get("has_hashtags") else 'Нет'}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Распределение хэштегов</h2>
            <div class="plot-container">
                <div id="hashtag-counts-plot"></div>
            </div>
        </div>
        
        <div class="section">
            <h2>Признаки заголовка</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Присутствует</td>
                    <td>{'Да' if title_features.get("present") else 'Нет'}</td>
                    <td>Наличие заголовка в документе</td>
                </tr>
                <tr>
                    <td>Количество хэштегов</td>
                    <td>{title_features.get("hashtag_found_count", "N/A")}</td>
                    <td>Количество найденных хэштегов в заголовке</td>
                </tr>
                <tr>
                    <td>Плотность хэштегов</td>
                    <td>{title_features.get("hashtag_density_per_char", "N/A")}</td>
                    <td>Плотность хэштегов на символ текста</td>
                </tr>
                <tr>
                    <td>Обрезан</td>
                    <td>{'Да' if title_features.get("truncated_flag") else 'Нет'}</td>
                    <td>Был ли заголовок обрезан из-за превышения max_text_chars</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Признаки описания</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Присутствует</td>
                    <td>{'Да' if description_features.get("present") else 'Нет'}</td>
                    <td>Наличие описания в документе</td>
                </tr>
                <tr>
                    <td>Количество хэштегов</td>
                    <td>{description_features.get("hashtag_found_count", "N/A")}</td>
                    <td>Количество найденных хэштегов в описании</td>
                </tr>
                <tr>
                    <td>Плотность хэштегов</td>
                    <td>{description_features.get("hashtag_density_per_char", "N/A")}</td>
                    <td>Плотность хэштегов на символ текста</td>
                </tr>
                <tr>
                    <td>Обрезан</td>
                    <td>{'Да' if description_features.get("truncated_flag") else 'Нет'}</td>
                    <td>Было ли описание обрезано из-за превышения max_text_chars</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Общие признаки хэштегов</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Всего найдено</td>
                    <td>{hashtag_features.get("total_found_count", "N/A")}</td>
                    <td>Общее количество найденных хэштегов (включая дубликаты)</td>
                </tr>
                <tr>
                    <td>Уникальных</td>
                    <td>{hashtag_features.get("unique_count", "N/A")}</td>
                    <td>Количество уникальных хэштегов (после дедупликации)</td>
                </tr>
                <tr>
                    <td>Средняя длина</td>
                    <td>{hashtag_features.get("avg_len", "N/A")}</td>
                    <td>Средняя длина хэштега в символах</td>
                </tr>
                <tr>
                    <td>Максимальная длина</td>
                    <td>{hashtag_features.get("max_len", "N/A")}</td>
                    <td>Максимальная длина хэштега в символах</td>
                </tr>
                <tr>
                    <td>Обрезаны</td>
                    <td>{'Да' if hashtag_features.get("truncated_flag") else 'Нет'}</td>
                    <td>Были ли хэштеги обрезаны из-за превышения max_tags_total</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Топ-K хэштегов (privacy-safe)</h2>
            <div class="plot-container">
                <div id="top-k-lengths-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Ранг</th>
                    <th>Присутствует</th>
                    <th>Хэш (hash01)</th>
                    <th>Длина</th>
                    <th>Описание</th>
                </tr>
"""
    
    # Add top-K tags rows
    for tag_info in top_k_tags:
        rank = tag_info.get("rank", 0)
        present = tag_info.get("present", False)
        hash01 = tag_info.get("hash01")
        length = tag_info.get("length")
        row_class = "tag-row present" if present else "tag-row"
        
        html_content += f"""
                <tr class="{row_class}">
                    <td>{rank}</td>
                    <td>{'Да' if present else 'Нет'}</td>
                    <td>{hash01 if hash01 is not None else 'N/A'}</td>
                    <td>{length if length is not None else 'N/A'}</td>
                    <td>Privacy-safe хэш и длина хэштега (raw текст не сохраняется)</td>
                </tr>
"""
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>Статистика и настройки</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Извлечение включено</div>
                    <div class="feature-value">{'Да' if statistics.get("group_extract_enabled") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Мутация текстов</div>
                    <div class="feature-value">{'Да' if statistics.get("group_mutate_clean_texts_enabled") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Мутация хэштегов</div>
                    <div class="feature-value">{'Да' if statistics.get("group_mutate_hashtags_enabled") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Требуется заголовок</div>
                    <div class="feature-value">{'Да' if statistics.get("require_title_enabled") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Слотов Top-K</div>
                    <div class="feature-value">{statistics.get("topk_slots", "N/A")}</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // График распределения хэштегов
        var hashtagCounts = {json.dumps(hashtag_counts)};
        if (Object.keys(hashtagCounts).length > 0) {{
            var countsTrace = {{
                x: Object.keys(hashtagCounts),
                y: Object.values(hashtagCounts),
                type: 'bar',
                marker: {{ color: '#4CAF50' }}
            }};
            var countsLayout = {{
                title: 'Распределение хэштегов',
                xaxis: {{ title: 'Категория' }},
                yaxis: {{ title: 'Количество' }}
            }};
            Plotly.newPlot('hashtag-counts-plot', [countsTrace], countsLayout);
        }}
        
        // График длин топ-K хэштегов
        var topKLengths = {json.dumps(top_k_lengths)};
        if (Object.keys(topKLengths).length > 0) {{
            var lengthsTrace = {{
                x: Object.keys(topKLengths),
                y: Object.values(topKLengths),
                type: 'bar',
                marker: {{ color: '#2196F3' }}
            }};
            var lengthsLayout = {{
                title: 'Длины топ-K хэштегов',
                xaxis: {{ title: 'Ранг' }},
                yaxis: {{ title: 'Длина (символов)' }}
            }};
            Plotly.newPlot('top-k-lengths-plot', [lengthsTrace], lengthsLayout);
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
    
    logger.info(f"Tags extractor HTML render saved to {output_path}")
    return output_path


__all__ = ["render_tags_extractor", "render_tags_extractor_html"]

