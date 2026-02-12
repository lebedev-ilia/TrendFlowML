"""
Renderer для lexico_static_features extractor: генерация render-context JSON.

Генерирует human-friendly представление лексических статических фич для визуализации и LLM.
"""

import logging
import math
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_lexico_static_features(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для lexico_static_features extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "lexico_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "lexico_static_features",
        "summary": {},
        "title_features": {},
        "description_features": {},
        "transcript_features": {},
        "statistics": {},
    }
    
    # Extract title features (prefix: tp_lex_title_*)
    # Clean NaN values for JSON compatibility
    def _clean_value(v: Any) -> Any:
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        return v
    
    title_features = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_lex_title_"):
            feature_name = key.replace("tp_lex_title_", "")
            title_features[feature_name] = _clean_value(value)
    
    # Extract description features (prefix: tp_lex_description_*)
    description_features = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_lex_description_"):
            feature_name = key.replace("tp_lex_description_", "")
            description_features[feature_name] = _clean_value(value)
    
    # Extract transcript features (prefix: tp_lex_transcript_*)
    transcript_features = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_lex_transcript_"):
            feature_name = key.replace("tp_lex_transcript_", "")
            transcript_features[feature_name] = _clean_value(value)
    
    # Extract general statistics (tp_lex_* but not title/description/transcript specific)
    stats = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_lex_") and not any(key.startswith(prefix) for prefix in ["tp_lex_title_", "tp_lex_description_", "tp_lex_transcript_"]):
            stats[key.replace("tp_lex_", "")] = _clean_value(value)
    
    render["title_features"] = title_features
    render["description_features"] = description_features
    render["transcript_features"] = transcript_features
    render["statistics"] = stats
    
    # Summary (clean NaN values)
    clickbait_score = title_features.get("clickbait_score")
    if clickbait_score is not None and not (isinstance(clickbait_score, (float, np.floating)) and (math.isnan(clickbait_score) or math.isinf(clickbait_score))):
        has_clickbait = float(clickbait_score) > 0.5
    else:
        has_clickbait = None
    
    render["summary"] = {
        "title_length_chars": _clean_value(title_features.get("len_chars", 0)),
        "title_length_words": _clean_value(title_features.get("len_words", 0)),
        "description_length_words": _clean_value(description_features.get("len_words", 0)),
        "transcript_length_words": _clean_value(transcript_features.get("len_words", 0)),
        "title_clickbait_score": _clean_value(clickbait_score),
        "has_clickbait": has_clickbait,
        "title_emoji_count": _clean_value(title_features.get("emoji_count", 0)),
        "description_emoji_count": _clean_value(description_features.get("emoji_count", 0)),
        "transcript_lexical_diversity": _clean_value(transcript_features.get("lexical_diversity", None)),
    }
    
    return render


def render_lexico_static_features_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага lexico_static_features результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости с AudioProcessor API)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    import os
    import json
    import math
    
    render = render_lexico_static_features(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    title_features = render.get("title_features", {})
    description_features = render.get("description_features", {})
    transcript_features = render.get("transcript_features", {})
    statistics = render.get("statistics", {})
    
    # Prepare data for visualization
    # Title features distribution (русские названия для графиков)
    title_numeric_features = {
        "Длина (символов)": title_features.get("len_chars"),
        "Длина (слов)": title_features.get("len_words"),
        "Средняя длина слова": title_features.get("avg_word_len"),
        "Восклицательные знаки": title_features.get("exclamation_count"),
        "Вопросительные знаки": title_features.get("question_count"),
        "Доля препинания": title_features.get("punctuation_ratio"),
        "Доля заглавных слов": title_features.get("capital_words_ratio"),
        "Оценка кликбейта": title_features.get("clickbait_score"),
    }
    # Filter out None/NaN values
    title_numeric_features = {k: v for k, v in title_numeric_features.items() 
                              if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    
    # Description features
    desc_numeric_features = {
        "Длина (слов)": description_features.get("len_words"),
        "Количество URL": description_features.get("num_urls"),
        "Количество упоминаний": description_features.get("num_mentions"),
    }
    desc_numeric_features = {k: v for k, v in desc_numeric_features.items() 
                            if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    
    # Transcript features
    transcript_numeric_features = {
        "Длина (слов)": transcript_features.get("len_words"),
        "Средняя длина предложения": transcript_features.get("avg_sentence_len"),
        "Лексическое разнообразие": transcript_features.get("lexical_diversity"),
        "Доля стоп-слов": transcript_features.get("stopword_ratio"),
        "Оценка читаемости": transcript_features.get("readability_score"),
    }
    transcript_numeric_features = {k: v for k, v in transcript_numeric_features.items() 
                                  if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Лексические статические признаки</title>
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
        <h1>Лексические статические признаки</h1>
        <p><strong>Компонент:</strong> {render.get("component", "lexico_static_features")}</p>
        <p><strong>Статус:</strong> <span style="color: green;">✓ OK</span></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Длина заголовка (символов)</div>
                    <div class="feature-value">{summary.get("title_length_chars", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Длина заголовка (слов)</div>
                    <div class="feature-value">{summary.get("title_length_words", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Длина описания (слов)</div>
                    <div class="feature-value">{summary.get("description_length_words", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Длина транскрипта (слов)</div>
                    <div class="feature-value">{summary.get("transcript_length_words", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Оценка кликбейта</div>
                    <div class="feature-value">{summary.get("title_clickbait_score", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Содержит кликбейт</div>
                    <div class="feature-value">{'Да' if summary.get("has_clickbait") else 'Нет' if summary.get("has_clickbait") is False else 'N/A'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Лексическое разнообразие</div>
                    <div class="feature-value">{summary.get("transcript_lexical_diversity", "N/A")}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Признаки заголовка</h2>
            <div class="plot-container">
                <div id="title-features-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Длина (символов)</td>
                    <td>{title_features.get("len_chars", "N/A")}</td>
                    <td>Общее количество символов в заголовке</td>
                </tr>
                <tr>
                    <td>Длина (слов)</td>
                    <td>{title_features.get("len_words", "N/A")}</td>
                    <td>Общее количество слов в заголовке</td>
                </tr>
                <tr>
                    <td>Средняя длина слова</td>
                    <td>{title_features.get("avg_word_len", "N/A")}</td>
                    <td>Среднее количество символов на слово</td>
                </tr>
                <tr>
                    <td>Количество восклицательных знаков</td>
                    <td>{title_features.get("exclamation_count", "N/A")}</td>
                    <td>Количество восклицательных знаков в заголовке</td>
                </tr>
                <tr>
                    <td>Количество вопросительных знаков</td>
                    <td>{title_features.get("question_count", "N/A")}</td>
                    <td>Количество вопросительных знаков в заголовке</td>
                </tr>
                <tr>
                    <td>Доля знаков препинания</td>
                    <td>{title_features.get("punctuation_ratio", "N/A")}</td>
                    <td>Отношение знаков препинания к общему количеству символов</td>
                </tr>
                <tr>
                    <td>Доля заглавных слов</td>
                    <td>{title_features.get("capital_words_ratio", "N/A")}</td>
                    <td>Отношение полностью заглавных слов к общему количеству слов</td>
                </tr>
                <tr>
                    <td>Оценка кликбейта</td>
                    <td>{title_features.get("clickbait_score", "N/A")}</td>
                    <td>Эвристическая оценка кликбейта (0-1), где выше значение - выше вероятность кликбейта</td>
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
                    <td>Длина (слов)</td>
                    <td>{description_features.get("len_words", "N/A")}</td>
                    <td>Общее количество слов в описании</td>
                </tr>
                <tr>
                    <td>Количество URL</td>
                    <td>{description_features.get("num_urls", "N/A")}</td>
                    <td>Количество найденных URL-адресов в описании</td>
                </tr>
                <tr>
                    <td>Количество упоминаний</td>
                    <td>{description_features.get("num_mentions", "N/A")}</td>
                    <td>Количество найденных @упоминаний в описании</td>
                </tr>
                <tr>
                    <td>Содержит временные метки</td>
                    <td>{'Да' if description_features.get("has_timestamps_flag") else 'Нет'}</td>
                    <td>Содержит ли описание паттерны временных меток (например, "01:23" или "1:02:03")</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Признаки транскрипта</h2>
            <div class="plot-container">
                <div id="transcript-features-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Длина (слов)</td>
                    <td>{transcript_features.get("len_words", "N/A")}</td>
                    <td>Общее количество слов в транскрипте</td>
                </tr>
                <tr>
                    <td>Средняя длина предложения</td>
                    <td>{transcript_features.get("avg_sentence_len", "N/A")}</td>
                    <td>Среднее количество слов в предложении</td>
                </tr>
                <tr>
                    <td>Лексическое разнообразие</td>
                    <td>{transcript_features.get("lexical_diversity", "N/A")}</td>
                    <td>Коэффициент лексического разнообразия (отношение уникальных слов к общему количеству слов). Чем выше значение, тем разнообразнее лексика</td>
                </tr>
                <tr>
                    <td>Доля стоп-слов</td>
                    <td>{transcript_features.get("stopword_ratio", "N/A")}</td>
                    <td>Отношение стоп-слов (служебных слов) к общему количеству слов</td>
                </tr>
                <tr>
                    <td>Оценка читаемости</td>
                    <td>{transcript_features.get("readability_score", "N/A")}</td>
                    <td>Простая метрика читаемости текста</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Статистика</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Включен</div>
                    <div class="feature-value">{'Да' if statistics.get("enabled") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Присутствует (любое поле)</div>
                    <div class="feature-value">{'Да' if statistics.get("present_any") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Доля специальных символов</div>
                    <div class="feature-value">{statistics.get("special_character_ratio", "N/A")}</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // График признаков заголовка
        var titleFeatures = {json.dumps(title_numeric_features)};
        if (Object.keys(titleFeatures).length > 0) {{
            var titleTrace = {{
                x: Object.keys(titleFeatures),
                y: Object.values(titleFeatures),
                type: 'bar',
                marker: {{ color: '#4CAF50' }}
            }};
            var titleLayout = {{
                title: 'Распределение признаков заголовка',
                xaxis: {{ title: 'Признак' }},
                yaxis: {{ title: 'Значение' }}
            }};
            Plotly.newPlot('title-features-plot', [titleTrace], titleLayout);
        }}
        
        // График признаков транскрипта
        var transcriptFeatures = {json.dumps(transcript_numeric_features)};
        if (Object.keys(transcriptFeatures).length > 0) {{
            var transcriptTrace = {{
                x: Object.keys(transcriptFeatures),
                y: Object.values(transcriptFeatures),
                type: 'bar',
                marker: {{ color: '#2196F3' }}
            }};
            var transcriptLayout = {{
                title: 'Распределение признаков транскрипта',
                xaxis: {{ title: 'Признак' }},
                yaxis: {{ title: 'Значение' }}
            }};
            Plotly.newPlot('transcript-features-plot', [transcriptTrace], transcriptLayout);
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
    
    logger.info(f"Lexico static features HTML render saved to {output_path}")
    return output_path


__all__ = ["render_lexico_static_features", "render_lexico_static_features_html"]

