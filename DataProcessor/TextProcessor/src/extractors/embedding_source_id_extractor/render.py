"""
Renderer для embedding_source_id_extractor: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление идентификатора источника эмбеддинга для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_embedding_source_id_extractor(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для embedding_source_id_extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_embid_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "embedding_source_id_extractor",
        "summary": {},
        "policy": {},
        "primary": {},
    }
    
    # Clean NaN values for JSON compatibility
    def _clean_value(v: Any) -> Any:
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        return v
    
    # Extract policy flags
    policy = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_embid_policy_"):
            policy_name = key.replace("tp_embid_policy_", "")
            policy[policy_name] = bool(value > 0.5)
    
    # Extract primary source flags
    primary = {}
    for key, value in extractor_features.items():
        if key.startswith("tp_embid_primary_is_"):
            source_name = key.replace("tp_embid_primary_is_", "")
            primary[source_name] = bool(value > 0.5)
    
    # Try to get embedding_source_id from payload if available
    embedding_source_id = {}
    if isinstance(payload, dict):
        embedding_source_id = payload.get("embedding_source_id", {})
    
    render["policy"] = policy
    render["primary"] = primary
    render["embedding_source_id"] = embedding_source_id
    
    # Determine active policy
    active_policy = None
    for policy_name, is_active in policy.items():
        if is_active:
            active_policy = policy_name
            break
    
    # Determine primary source
    primary_source = None
    for source_name, is_primary in primary.items():
        if is_primary:
            primary_source = source_name
            break
    
    # Summary
    render["summary"] = {
        "present": bool(extractor_features.get("tp_embid_present", 0) > 0.5),
        "active_policy": active_policy,
        "primary_source": primary_source,
        "vector_id": embedding_source_id.get("vector_id") if isinstance(embedding_source_id, dict) else None,
        "vector_store_uri": embedding_source_id.get("vector_store_uri") if isinstance(embedding_source_id, dict) else None,
        "model_name": embedding_source_id.get("model_name") if isinstance(embedding_source_id, dict) else None,
        "model_version": embedding_source_id.get("model_version") if isinstance(embedding_source_id, dict) else None,
    }
    
    return render


def render_embedding_source_id_extractor_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага embedding_source_id_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_embedding_source_id_extractor(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    policy = render.get("policy", {})
    primary = render.get("primary", {})
    embedding_source_id = render.get("embedding_source_id", {})
    
    # Map policy names to Russian
    policy_names_ru = {
        "transcript_first": "Транскрипт первый",
        "title_first": "Заголовок первый",
        "description_first": "Описание первый",
        "title_only": "Только заголовок",
        "transcript_only": "Только транскрипт",
    }
    
    # Map primary source names to Russian
    primary_names_ru = {
        "transcript": "Транскрипт",
        "title": "Заголовок",
        "description": "Описание",
    }
    
    active_policy_ru = policy_names_ru.get(summary.get("active_policy", ""), summary.get("active_policy", "N/A"))
    primary_source_ru = primary_names_ru.get(summary.get("primary_source", ""), summary.get("primary_source", "N/A"))
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Идентификатор источника эмбеддинга</title>
    <meta charset="UTF-8">
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
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .code {{ font-family: monospace; background-color: #f0f0f0; padding: 2px 6px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Идентификатор источника эмбеддинга</h1>
        <p><strong>Компонент:</strong> {render.get("component", "embedding_source_id_extractor")}</p>
        <p><strong>Статус:</strong> <span style="color: {'green' if summary.get('present') else 'red'};">{{'✓ OK' if summary.get('present') else '✗ Нет данных'}}</span></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Присутствует</div>
                    <div class="feature-value">{{'Да' if summary.get('present') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Активная политика</div>
                    <div class="feature-value">{active_policy_ru}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Основной источник</div>
                    <div class="feature-value">{primary_source_ru}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Vector ID</div>
                    <div class="feature-value"><span class="code">{summary.get("vector_id", "N/A")}</span></div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Vector Store URI</div>
                    <div class="feature-value"><span class="code">{summary.get("vector_store_uri", "N/A")}</span></div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Имя модели</div>
                    <div class="feature-value">{summary.get("model_name", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Версия модели</div>
                    <div class="feature-value">{summary.get("model_version", "N/A")}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Политика выбора</h2>
            <table>
                <tr>
                    <th>Политика</th>
                    <th>Активна</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Транскрипт первый</td>
                    <td>{{'Да' if policy.get('transcript_first') else 'Нет'}}</td>
                    <td>Сначала пытается использовать транскрипт, затем заголовок/описание</td>
                </tr>
                <tr>
                    <td>Заголовок первый</td>
                    <td>{{'Да' if policy.get('title_first') else 'Нет'}}</td>
                    <td>Сначала пытается использовать заголовок, затем другие источники</td>
                </tr>
                <tr>
                    <td>Описание первый</td>
                    <td>{{'Да' if policy.get('description_first') else 'Нет'}}</td>
                    <td>Сначала пытается использовать описание, затем другие источники</td>
                </tr>
                <tr>
                    <td>Только заголовок</td>
                    <td>{{'Да' if policy.get('title_only') else 'Нет'}}</td>
                    <td>Использует только заголовок</td>
                </tr>
                <tr>
                    <td>Только транскрипт</td>
                    <td>{{'Да' if policy.get('transcript_only') else 'Нет'}}</td>
                    <td>Использует только транскрипт</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Основной источник</h2>
            <table>
                <tr>
                    <th>Источник</th>
                    <th>Используется</th>
                </tr>
                <tr>
                    <td>Транскрипт</td>
                    <td>{{'Да' if primary.get('transcript') else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Заголовок</td>
                    <td>{{'Да' if primary.get('title') else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Описание</td>
                    <td>{{'Да' if primary.get('description') else 'Нет'}}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Метаданные эмбеддинга</h2>
            <table>
                <tr>
                    <th>Поле</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>Vector ID</td>
                    <td><span class="code">{embedding_source_id.get("vector_id", "N/A") if isinstance(embedding_source_id, dict) else "N/A"}</span></td>
                    <td>Переносимый стабильный идентификатор вектора (sha256 по значениям)</td>
                </tr>
                <tr>
                    <td>Vector Store URI</td>
                    <td><span class="code">{embedding_source_id.get("vector_store_uri", "N/A") if isinstance(embedding_source_id, dict) else "N/A"}</span></td>
                    <td>URI хранилища векторов (например, faiss://semantic_titles_v1)</td>
                </tr>
                <tr>
                    <td>Имя модели</td>
                    <td>{embedding_source_id.get("model_name", "N/A") if isinstance(embedding_source_id, dict) else "N/A"}</td>
                    <td>Логическое имя модели из upstream meta (может отсутствовать для transcript mean)</td>
                </tr>
                <tr>
                    <td>Версия модели</td>
                    <td>{embedding_source_id.get("model_version", "N/A") if isinstance(embedding_source_id, dict) else "N/A"}</td>
                    <td>Версия/идентификатор модели (из meta или конфига; не смешивается с именем)</td>
                </tr>
                <tr>
                    <td>Weights Digest</td>
                    <td><span class="code">{embedding_source_id.get("weights_digest", "N/A") if isinstance(embedding_source_id, dict) else "N/A"}</span></td>
                    <td>Хеш весов модели для идентификации</td>
                </tr>
                <tr>
                    <td>Primary Source</td>
                    <td>{embedding_source_id.get("primary_source", "N/A") if isinstance(embedding_source_id, dict) else "N/A"}</td>
                    <td>Источник эмбеддинга (title, description, transcript_combined и т.д.)</td>
                </tr>
            </table>
        </div>
    </div>
</body>
</html>
"""
    
    # Atomic write
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    os.replace(tmp_path, output_path)
    
    logger.info(f"Embedding source ID extractor HTML render saved to {output_path}")
    return output_path


__all__ = ["render_embedding_source_id_extractor", "render_embedding_source_id_extractor_html"]

