"""
Renderer для speaker_turn_embeddings_aggregator: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление агрегированных эмбеддингов спикеров для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_speaker_turn_embeddings_aggregator(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для speaker_turn_embeddings_aggregator.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_spkemb_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "speaker_turn_embeddings_aggregator",
        "summary": {},
        "speakers": {},
        "input": {},
        "configuration": {},
    }
    
    # Clean NaN values for JSON compatibility
    def _clean_value(v: Any) -> Any:
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        return v
    
    # Extract speaker metrics
    speakers = {}
    if "tp_spkemb_speakers_total" in extractor_features:
        speakers["total"] = _clean_value(extractor_features["tp_spkemb_speakers_total"])
    if "tp_spkemb_speakers_embedded" in extractor_features:
        speakers["embedded"] = _clean_value(extractor_features["tp_spkemb_speakers_embedded"])
    if "tp_spkemb_turns_total" in extractor_features:
        speakers["turns_total"] = _clean_value(extractor_features["tp_spkemb_turns_total"])
    
    # Extract input mode flags
    input_mode = {}
    if "tp_spkemb_input_present" in extractor_features:
        input_mode["present"] = bool(extractor_features["tp_spkemb_input_present"] > 0.5)
    if "tp_spkemb_input_mode_diar_asr" in extractor_features:
        input_mode["diar_asr"] = bool(extractor_features["tp_spkemb_input_mode_diar_asr"] > 0.5)
    if "tp_spkemb_input_mode_legacy_doc_speakers" in extractor_features:
        input_mode["legacy_doc_speakers"] = bool(extractor_features["tp_spkemb_input_mode_legacy_doc_speakers"] > 0.5)
    if "tp_spkemb_asr_present" in extractor_features:
        input_mode["asr_present"] = bool(extractor_features["tp_spkemb_asr_present"] > 0.5)
    if "tp_spkemb_diar_present" in extractor_features:
        input_mode["diar_present"] = bool(extractor_features["tp_spkemb_diar_present"] > 0.5)
    
    # Extract configuration
    configuration = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_spkemb_") and (key.endswith("_enabled") or key.startswith("tp_spkemb_compute") or 
                                              key.startswith("tp_spkemb_write") or key.startswith("tp_spkemb_max") or
                                              key.startswith("tp_spkemb_min") or key.startswith("tp_spkemb_batch")):
            feature_name = key.replace("tp_spkemb_", "")
            configuration[feature_name] = _clean_value(value)
    
    # Try to get speaker_embeddings_meta from payload if available
    speaker_embeddings_meta = {}
    if isinstance(payload, dict):
        speaker_embeddings_meta = payload.get("speaker_embeddings_meta", {})
    
    render["speakers"] = speakers
    render["input"] = input_mode
    render["configuration"] = configuration
    render["speaker_embeddings_meta"] = speaker_embeddings_meta
    
    # Summary
    render["summary"] = {
        "present": bool(extractor_features.get("tp_spkemb_present", 0) > 0.5),
        "speakers_total": _clean_value(speakers.get("total")),
        "speakers_embedded": _clean_value(speakers.get("embedded")),
        "turns_total": _clean_value(speakers.get("turns_total")),
        "input_present": bool(input_mode.get("present", False)),
        "input_mode": "diar_asr" if input_mode.get("diar_asr") else "legacy_doc_speakers" if input_mode.get("legacy_doc_speakers") else "unknown",
        "model_name": speaker_embeddings_meta.get("model_name") if isinstance(speaker_embeddings_meta, dict) else None,
        "model_version": speaker_embeddings_meta.get("model_version") if isinstance(speaker_embeddings_meta, dict) else None,
    }
    
    return render


def render_speaker_turn_embeddings_aggregator_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага speaker_turn_embeddings_aggregator результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_speaker_turn_embeddings_aggregator(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    speakers = render.get("speakers", {})
    input_mode = render.get("input", {})
    configuration = render.get("configuration", {})
    speaker_embeddings_meta = render.get("speaker_embeddings_meta", {})
    
    # Prepare data for visualization
    speaker_stats = {
        "Всего спикеров": speakers.get("total"),
        "Обработано спикеров": speakers.get("embedded"),
        "Всего реплик": speakers.get("turns_total"),
    }
    # Filter out None/NaN values
    speaker_stats = {k: v for k, v in speaker_stats.items() 
                    if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    
    input_mode_name = summary.get("input_mode", "unknown")
    input_mode_ru = {
        "diar_asr": "Diarization + ASR",
        "legacy_doc_speakers": "Legacy doc.speakers",
        "unknown": "Неизвестно",
    }.get(input_mode_name, input_mode_name)
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Агрегация эмбеддингов спикеров</title>
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
        <h1>Агрегация эмбеддингов спикеров</h1>
        <p><strong>Компонент:</strong> {render.get("component", "speaker_turn_embeddings_aggregator")}</p>
        <p><strong>Статус:</strong> <span style="color: {'green' if summary.get('present') else 'red'};">{{'✓ OK' if summary.get('present') else '✗ Нет данных'}}</span></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Присутствует</div>
                    <div class="feature-value">{{'Да' if summary.get('present') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Всего спикеров</div>
                    <div class="feature-value">{summary.get("speakers_total", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Обработано спикеров</div>
                    <div class="feature-value">{summary.get("speakers_embedded", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Всего реплик</div>
                    <div class="feature-value">{summary.get("turns_total", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Режим входа</div>
                    <div class="feature-value">{input_mode_ru}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Модель</div>
                    <div class="feature-value">{summary.get("model_name", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Версия модели</div>
                    <div class="feature-value">{summary.get("model_version", "N/A")}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Статистика спикеров</h2>
            <div class="plot-container">
                <div id="speakers-plot"></div>
            </div>
            <table>
                <tr>
                    <th>Метрика</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Всего спикеров</td>
                    <td>{speakers.get("total", "N/A")}</td>
                    <td>Общее количество уникальных спикеров в видео</td>
                </tr>
                <tr>
                    <td>Обработано спикеров</td>
                    <td>{speakers.get("embedded", "N/A")}</td>
                    <td>Количество спикеров, для которых были вычислены эмбеддинги</td>
                </tr>
                <tr>
                    <td>Всего реплик</td>
                    <td>{speakers.get("turns_total", "N/A")}</td>
                    <td>Общее количество реплик (speaker turns) всех спикеров</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Режим входа</h2>
            <table>
                <tr>
                    <th>Параметр</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Вход присутствует</td>
                    <td>{{'Да' if input_mode.get('present') else 'Нет'}}</td>
                    <td>Были ли доступны данные о спикерах</td>
                </tr>
                <tr>
                    <td>Режим: Diarization + ASR</td>
                    <td>{{'Да' if input_mode.get('diar_asr') else 'Нет'}}</td>
                    <td>Использовался ли режим с diarization и ASR сегментами</td>
                </tr>
                <tr>
                    <td>Режим: Legacy doc.speakers</td>
                    <td>{{'Да' if input_mode.get('legacy_doc_speakers') else 'Нет'}}</td>
                    <td>Использовался ли legacy режим с doc.speakers</td>
                </tr>
                <tr>
                    <td>ASR присутствует</td>
                    <td>{{'Да' if input_mode.get('asr_present') else 'Нет'}}</td>
                    <td>Были ли доступны ASR сегменты</td>
                </tr>
                <tr>
                    <td>Diarization присутствует</td>
                    <td>{{'Да' if input_mode.get('diar_present') else 'Нет'}}</td>
                    <td>Были ли доступны данные diarization</td>
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
                    <td>Вычисление среднего</td>
                    <td>{{'Да' if configuration.get('compute_mean', 0) > 0.5 else 'Нет'}}</td>
                    <td>Было ли вычислено среднее эмбеддингов реплик для каждого спикера</td>
                </tr>
                <tr>
                    <td>Вычисление максимума</td>
                    <td>{{'Да' if configuration.get('compute_max', 0) > 0.5 else 'Нет'}}</td>
                    <td>Был ли применен max pooling к эмбеддингам реплик</td>
                </tr>
                <tr>
                    <td>Запись артефактов</td>
                    <td>{{'Да' if configuration.get('write_artifacts', 0) > 0.5 else 'Нет'}}</td>
                    <td>Были ли записаны файлы с эмбеддингами спикеров</td>
                </tr>
"""
    
    if configuration.get("max_speakers") is not None:
        html_content += f"""
                <tr>
                    <td>Максимум спикеров</td>
                    <td>{configuration.get("max_speakers", "N/A")}</td>
                    <td>Максимальное количество спикеров для обработки</td>
                </tr>
"""
    
    if configuration.get("max_turns_per_speaker") is not None:
        html_content += f"""
                <tr>
                    <td>Максимум реплик на спикера</td>
                    <td>{configuration.get("max_turns_per_speaker", "N/A")}</td>
                    <td>Максимальное количество реплик для каждого спикера</td>
                </tr>
"""
    
    html_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>Метаданные модели</h2>
            <table>
                <tr>
                    <th>Поле</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Название модели</td>
                    <td>{speaker_embeddings_meta.get("model_name", "N/A") if isinstance(speaker_embeddings_meta, dict) else "N/A"}</td>
                    <td>Название модели, использованной для создания эмбеддингов</td>
                </tr>
                <tr>
                    <td>Версия модели</td>
                    <td>{speaker_embeddings_meta.get("model_version", "N/A") if isinstance(speaker_embeddings_meta, dict) else "N/A"}</td>
                    <td>Версия модели</td>
                </tr>
                <tr>
                    <td>Weights Digest</td>
                    <td>{speaker_embeddings_meta.get("weights_digest", "N/A") if isinstance(speaker_embeddings_meta, dict) else "N/A"}</td>
                    <td>Хеш весов модели для идентификации</td>
                </tr>
            </table>
        </div>
    </div>
    
    <script>
        // График статистики спикеров
        var speakerStats = {json.dumps(speaker_stats)};
        if (Object.keys(speakerStats).length > 0) {{
            var trace = {{
                x: Object.keys(speakerStats),
                y: Object.values(speakerStats),
                type: 'bar',
                marker: {{ color: '#4CAF50' }}
            }};
            var layout = {{
                title: 'Статистика спикеров',
                xaxis: {{ title: 'Метрика' }},
                yaxis: {{ title: 'Значение' }}
            }};
            Plotly.newPlot('speakers-plot', [trace], layout);
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
    
    logger.info(f"Speaker turn embeddings aggregator HTML render saved to {output_path}")
    return output_path


__all__ = ["render_speaker_turn_embeddings_aggregator", "render_speaker_turn_embeddings_aggregator_html"]

