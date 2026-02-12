"""
Renderer для asr_text_proxy_audio_features extractor: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление proxy-признаков аудио из ASR транскрипта для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_asr_text_proxy_audio_features(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для asr_text_proxy_audio_features extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_asrproxy_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "asr_text_proxy_audio_features",
        "summary": {},
        "presence": {},
        "audio_meta": {},
        "confidence": {},
        "noise": {},
        "rhythm": {},
        "intonation": {},
        "statistics": {},
    }
    
    # Clean NaN values for JSON compatibility
    def _clean_value(v: Any) -> Any:
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        return v
    
    # Extract presence/masks features
    presence = {}
    for key, value in extractor_features.items():
        if key in ["tp_asrproxy_present", "tp_asrproxy_has_confidence", "tp_asrproxy_confidence_present_rate"]:
            feature_name = key.replace("tp_asrproxy_", "")
            presence[feature_name] = _clean_value(value)
    
    # Extract audio meta features
    audio_meta = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_asrproxy_audio_duration") or key.startswith("tp_asrproxy_duration") or key.startswith("tp_asrproxy_segments_count"):
            feature_name = key.replace("tp_asrproxy_", "")
            audio_meta[feature_name] = _clean_value(value)
    
    # Extract confidence features
    confidence = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_asrproxy_confidence"):
            feature_name = key.replace("tp_asrproxy_confidence_", "")
            confidence[feature_name] = _clean_value(value)
    
    # Extract noise features
    noise = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_asrproxy_text_noise") or key.startswith("tp_asrproxy_noise_proxy"):
            feature_name = key.replace("tp_asrproxy_", "")
            noise[feature_name] = _clean_value(value)
    
    # Extract rhythm features
    rhythm = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_asrproxy_speech_rate") or key.startswith("tp_asrproxy_speech_char_density") or \
           key.startswith("tp_asrproxy_pause_density") or key.startswith("tp_asrproxy_filler_ratio"):
            feature_name = key.replace("tp_asrproxy_", "")
            rhythm[feature_name] = _clean_value(value)
    
    # Extract intonation features
    intonation = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_asrproxy_sentence_intonation"):
            feature_name = key.replace("tp_asrproxy_", "")
            intonation[feature_name] = _clean_value(value)
    
    # Extract text size features
    text_size = {}
    for key, value in extractor_features.items():
        if key in ["tp_asrproxy_text_chars", "tp_asrproxy_word_count", "tp_asrproxy_text_truncated_flag"]:
            feature_name = key.replace("tp_asrproxy_", "")
            text_size[feature_name] = _clean_value(value)
    
    # Extract general statistics (flags, enabled, thresholds)
    stats = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_asrproxy_") and not any(key.startswith(prefix) for prefix in [
            "tp_asrproxy_present", "tp_asrproxy_has_confidence", "tp_asrproxy_confidence",
            "tp_asrproxy_audio_duration", "tp_asrproxy_duration", "tp_asrproxy_segments_count",
            "tp_asrproxy_text_noise", "tp_asrproxy_noise_proxy",
            "tp_asrproxy_speech_rate", "tp_asrproxy_speech_char_density",
            "tp_asrproxy_pause_density", "tp_asrproxy_filler_ratio",
            "tp_asrproxy_sentence_intonation",
            "tp_asrproxy_text_chars", "tp_asrproxy_word_count", "tp_asrproxy_text_truncated"
        ]):
            stats[key.replace("tp_asrproxy_", "")] = _clean_value(value)
    
    render["presence"] = presence
    render["audio_meta"] = audio_meta
    render["confidence"] = confidence
    render["noise"] = noise
    render["rhythm"] = rhythm
    render["intonation"] = intonation
    render["text_size"] = text_size
    render["statistics"] = stats
    
    # Summary
    render["summary"] = {
        "present": bool(presence.get("present", 0) > 0.5),
        "has_confidence": bool(presence.get("has_confidence", 0) > 0.5),
        "segments_count": _clean_value(audio_meta.get("segments_count", 0)),
        "audio_duration_sec": _clean_value(audio_meta.get("audio_duration_sec")),
        "text_chars": _clean_value(text_size.get("text_chars", 0)),
        "word_count": _clean_value(text_size.get("word_count", 0)),
        "confidence_mean": _clean_value(confidence.get("mean")),
        "speech_rate_wpm": _clean_value(rhythm.get("speech_rate_wpm")),
        "noise_proxy": _clean_value(noise.get("noise_proxy")),
    }
    
    return render


def render_asr_text_proxy_audio_features_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага asr_text_proxy_audio_features результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_asr_text_proxy_audio_features(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    presence = render.get("presence", {})
    audio_meta = render.get("audio_meta", {})
    confidence = render.get("confidence", {})
    noise = render.get("noise", {})
    rhythm = render.get("rhythm", {})
    intonation = render.get("intonation", {})
    text_size = render.get("text_size", {})
    statistics = render.get("statistics", {})
    
    # Prepare data for visualization
    def _clean_value(v: Any) -> Any:
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        return v
    
    # Confidence metrics for bar chart
    confidence_metrics = {
        "Среднее": confidence.get("mean"),
        "Стандартное отклонение": confidence.get("std"),
        "Минимум": confidence.get("chunked_min"),
        "Доля низкой уверенности": confidence.get("low_conf_rate"),
    }
    confidence_metrics = {k: v for k, v in confidence_metrics.items() 
                         if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    
    # Rhythm metrics for bar chart
    rhythm_metrics = {
        "Скорость речи (WPM)": rhythm.get("speech_rate_wpm"),
        "Плотность символов": rhythm.get("speech_char_density"),
        "Плотность пауз": rhythm.get("pause_density"),
        "Доля слов-паразитов": rhythm.get("filler_ratio"),
    }
    rhythm_metrics = {k: v for k, v in rhythm_metrics.items() 
                     if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    
    # Noise metrics for bar chart
    noise_metrics = {
        "Редкие слова": noise.get("text_noise_rare_ratio"),
        "Неизвестные слова": noise.get("text_noise_oov_ratio"),
        "Общий proxy": noise.get("noise_proxy"),
    }
    noise_metrics = {k: v for k, v in noise_metrics.items() 
                    if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>ASR Proxy-признаки аудио</title>
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
        <h1>ASR Proxy-признаки аудио</h1>
        <p><strong>Компонент:</strong> {render.get("component", "asr_text_proxy_audio_features")}</p>
        <p><strong>Статус:</strong> <span style="color: green;">✓ OK</span></p>
        <p><em>Признаки извлекаются из ASR транскрипта без прямого анализа аудио сигнала</em></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Транскрипт присутствует</div>
                    <div class="feature-value">{'Да' if summary.get("present") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Есть confidence</div>
                    <div class="feature-value">{'Да' if summary.get("has_confidence") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Количество сегментов</div>
                    <div class="feature-value">{summary.get("segments_count", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Длительность аудио (сек)</div>
                    <div class="feature-value">{summary.get("audio_duration_sec", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Символов в тексте</div>
                    <div class="feature-value">{summary.get("text_chars", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Слов в тексте</div>
                    <div class="feature-value">{summary.get("word_count", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Средняя confidence</div>
                    <div class="feature-value">{summary.get("confidence_mean", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Скорость речи (WPM)</div>
                    <div class="feature-value">{summary.get("speech_rate_wpm", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Proxy шума</div>
                    <div class="feature-value">{summary.get("noise_proxy", "N/A")}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Присутствие и маски</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Транскрипт присутствует</td>
                    <td>{'Да' if presence.get("present") else 'Нет'}</td>
                    <td>Наличие непустого транскрипта в документе</td>
                </tr>
                <tr>
                    <td>Есть confidence</td>
                    <td>{'Да' if presence.get("has_confidence") else 'Нет'}</td>
                    <td>Наличие confidence хотя бы у одного сегмента ASR</td>
                </tr>
                <tr>
                    <td>Доля сегментов с confidence</td>
                    <td>{presence.get("confidence_present_rate", "N/A")}</td>
                    <td>Отношение сегментов с confidence к общему количеству сегментов [0..1]</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Метаданные аудио</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Длительность аудио (сек)</td>
                    <td>{audio_meta.get("audio_duration_sec", "N/A")}</td>
                    <td>Общая длительность аудио в секундах (из Segmenter/AudioProcessor)</td>
                </tr>
                <tr>
                    <td>Количество сегментов</td>
                    <td>{audio_meta.get("segments_count", "N/A")}</td>
                    <td>Количество сегментов ASR в payload</td>
                </tr>
                <tr>
                    <td>Длительность из payload</td>
                    <td>{'Да' if audio_meta.get("duration_from_payload_flag") else 'Нет'}</td>
                    <td>Была ли длительность взята из payload (иначе из audio_duration_sec)</td>
                </tr>
                <tr>
                    <td>Длительность невалидна</td>
                    <td>{'Да' if audio_meta.get("duration_invalid_flag") else 'Нет'}</td>
                    <td>Была ли обнаружена невалидная длительность (отрицательная или нулевая)</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Уверенность распознавания (Confidence)</h2>
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
                    <td>Среднее значение</td>
                    <td>{confidence.get("mean", "N/A")}</td>
                    <td>Средняя confidence по всем сегментам [0..1]</td>
                </tr>
                <tr>
                    <td>Стандартное отклонение</td>
                    <td>{confidence.get("std", "N/A")}</td>
                    <td>Стандартное отклонение confidence</td>
                </tr>
                <tr>
                    <td>Минимальное значение</td>
                    <td>{confidence.get("chunked_min", "N/A")}</td>
                    <td>Минимальная confidence среди всех сегментов</td>
                </tr>
                <tr>
                    <td>Доля низкой уверенности</td>
                    <td>{confidence.get("low_conf_rate", "N/A")}</td>
                    <td>Доля сегментов с confidence ниже порога (low_conf_threshold) [0..1]</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Proxy шума</h2>
            <div class="plot-container">
                <div id="noise-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Доля редких слов</td>
                    <td>{noise.get("text_noise_rare_ratio", "N/A")}</td>
                    <td>Отношение редких слов к общему количеству слов [0..1]. Выше значение - больше "шумности"</td>
                </tr>
                <tr>
                    <td>Доля неизвестных слов</td>
                    <td>{noise.get("text_noise_oov_ratio", "N/A")}</td>
                    <td>Отношение слов вне словаря (OOV) к общему количеству слов [0..1]</td>
                </tr>
                <tr>
                    <td>Общий proxy шума</td>
                    <td>{noise.get("noise_proxy", "N/A")}</td>
                    <td>Комбинированный proxy индикатор "шумности" текста [0..1]. Выше значение - больше шума</td>
                </tr>
                <tr>
                    <td>Proxy шума присутствует</td>
                    <td>{'Да' if noise.get("noise_proxy_present") else 'Нет'}</td>
                    <td>Был ли вычислен proxy шума (требует enable_noise=true)</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Ритм речи</h2>
            <div class="plot-container">
                <div id="rhythm-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Скорость речи (WPM)</td>
                    <td>{rhythm.get("speech_rate_wpm", "N/A")}</td>
                    <td>Скорость речи в словах в минуту (words per minute)</td>
                </tr>
                <tr>
                    <td>Плотность символов</td>
                    <td>{rhythm.get("speech_char_density", "N/A")}</td>
                    <td>Плотность символов речи на секунду аудио</td>
                </tr>
                <tr>
                    <td>Плотность пауз</td>
                    <td>{rhythm.get("pause_density", "N/A")}</td>
                    <td>Плотность пауз в речи (≥ 0, выше значение - больше пауз)</td>
                </tr>
                <tr>
                    <td>Доля слов-паразитов</td>
                    <td>{rhythm.get("filler_ratio", "N/A")}</td>
                    <td>Отношение слов-паразитов (например, "э-э", "мм") к общему количеству слов [0..1]</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Интонация</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Интонация предложений</td>
                    <td>{intonation.get("sentence_intonation", "N/A")}</td>
                    <td>Proxy индикатор интонации на основе знаков препинания [0..1]. Выше значение - более выраженная интонация</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Размер текста</h2>
            <table>
                <tr>
                    <th>Признак</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Символов</td>
                    <td>{text_size.get("text_chars", "N/A")}</td>
                    <td>Количество символов в транскрипте (после применения лимитов)</td>
                </tr>
                <tr>
                    <td>Слов</td>
                    <td>{text_size.get("word_count", "N/A")}</td>
                    <td>Количество слов в транскрипте</td>
                </tr>
                <tr>
                    <td>Текст обрезан</td>
                    <td>{'Да' if text_size.get("text_truncated_flag") else 'Нет'}</td>
                    <td>Был ли текст обрезан из-за превышения max_text_chars</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Настройки и флаги</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Включен</div>
                    <div class="feature-value">{'Да' if statistics.get("enabled") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Базовые признаки</div>
                    <div class="feature-value">{'Да' if statistics.get("basic_enabled") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Признаки шума</div>
                    <div class="feature-value">{'Да' if statistics.get("noise_enabled") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Признаки ритма</div>
                    <div class="feature-value">{'Да' if statistics.get("rhythm_enabled") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Признаки интонации</div>
                    <div class="feature-value">{'Да' if statistics.get("intonation_enabled") else 'Нет'}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Порог низкой уверенности</div>
                    <div class="feature-value">{statistics.get("low_conf_threshold", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Базовый WPM</div>
                    <div class="feature-value">{statistics.get("words_per_minute_baseline", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Макс. символов</div>
                    <div class="feature-value">{statistics.get("max_text_chars", "N/A")}</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // График метрик confidence
        var confidenceMetrics = {json.dumps(confidence_metrics)};
        if (Object.keys(confidenceMetrics).length > 0) {{
            var confTrace = {{
                x: Object.keys(confidenceMetrics),
                y: Object.values(confidenceMetrics),
                type: 'bar',
                marker: {{ color: '#4CAF50' }}
            }};
            var confLayout = {{
                title: 'Метрики уверенности распознавания',
                xaxis: {{ title: 'Метрика' }},
                yaxis: {{ title: 'Значение' }}
            }};
            Plotly.newPlot('confidence-plot', [confTrace], confLayout);
        }}
        
        // График метрик ритма
        var rhythmMetrics = {json.dumps(rhythm_metrics)};
        if (Object.keys(rhythmMetrics).length > 0) {{
            var rhythmTrace = {{
                x: Object.keys(rhythmMetrics),
                y: Object.values(rhythmMetrics),
                type: 'bar',
                marker: {{ color: '#2196F3' }}
            }};
            var rhythmLayout = {{
                title: 'Метрики ритма речи',
                xaxis: {{ title: 'Метрика' }},
                yaxis: {{ title: 'Значение' }}
            }};
            Plotly.newPlot('rhythm-plot', [rhythmTrace], rhythmLayout);
        }}
        
        // График метрик шума
        var noiseMetrics = {json.dumps(noise_metrics)};
        if (Object.keys(noiseMetrics).length > 0) {{
            var noiseTrace = {{
                x: Object.keys(noiseMetrics),
                y: Object.values(noiseMetrics),
                type: 'bar',
                marker: {{ color: '#FF9800' }}
            }};
            var noiseLayout = {{
                title: 'Proxy метрики шума',
                xaxis: {{ title: 'Метрика' }},
                yaxis: {{ title: 'Значение' }}
            }};
            Plotly.newPlot('noise-plot', [noiseTrace], noiseLayout);
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
    
    logger.info(f"ASR text proxy audio features HTML render saved to {output_path}")
    return output_path


__all__ = ["render_asr_text_proxy_audio_features", "render_asr_text_proxy_audio_features_html"]

