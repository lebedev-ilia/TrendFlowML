"""
Renderer для semantic_cluster_extractor: генерация render-context JSON и HTML debug страницы.

Генерирует human-friendly представление семантического кластера для визуализации и LLM.
"""

import logging
import math
import os
import json
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def render_semantic_cluster_extractor(
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Генерировать render-context для semantic_cluster_extractor.
    
    Args:
        extractor_features: Словарь фич этого extractor'а (префикс "tp_semclust_")
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Dict с render-context данными
    """
    render = {
        "component": "semantic_cluster_extractor",
        "summary": {},
        "cluster": {},
        "source": {},
        "presence": {},
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

    def _bool_flag(key: str) -> bool:
        v = extractor_features.get(key, 0.0)
        try:
            return float(v) > 0.5
        except (TypeError, ValueError):
            return False

    # Extract cluster metrics
    cluster = {}
    if "tp_semclust_id" in extractor_features:
        cluster["id"] = _clean_value(extractor_features["tp_semclust_id"])
    if "tp_semclust_similarity" in extractor_features:
        cluster["similarity"] = _clean_value(extractor_features["tp_semclust_similarity"])
    if "tp_semclust_distance" in extractor_features:
        cluster["distance"] = _clean_value(extractor_features["tp_semclust_distance"])
    
    # Extract source information
    source = {}
    if "tp_semclust_source_title" in extractor_features:
        source["title"] = bool(extractor_features["tp_semclust_source_title"] > 0.5)
    if "tp_semclust_source_description" in extractor_features:
        source["description"] = bool(extractor_features["tp_semclust_source_description"] > 0.5)
    if "tp_semclust_source_hashtag" in extractor_features:
        source["hashtag"] = bool(extractor_features["tp_semclust_source_hashtag"] > 0.5)
    if "tp_semclust_fallback_used" in extractor_features:
        source["fallback_used"] = bool(extractor_features["tp_semclust_fallback_used"] > 0.5)
    
    # Extract presence flags
    presence = {}
    if "tp_semclust_title_present" in extractor_features:
        presence["title"] = bool(extractor_features["tp_semclust_title_present"] > 0.5)
    if "tp_semclust_description_present" in extractor_features:
        presence["description"] = bool(extractor_features["tp_semclust_description_present"] > 0.5)
    if "tp_semclust_hashtag_present" in extractor_features:
        presence["hashtag"] = bool(extractor_features["tp_semclust_hashtag_present"] > 0.5)
    
    # Extract safety / diagnostics flags
    safety = {
        "dim_mismatch_flag": _bool_flag("tp_semclust_dim_mismatch_flag"),
        "backend_faiss": _bool_flag("tp_semclust_backend_faiss"),
        "unsafe_relpath": _bool_flag("tp_semclust_unsafe_relpath_flag"),
        "title_embed_missing": _bool_flag("tp_semclust_title_embed_missing_flag"),
        "description_embed_missing": _bool_flag("tp_semclust_description_embed_missing_flag"),
        "hashtag_embed_missing": _bool_flag("tp_semclust_hashtag_embed_missing_flag"),
    }

    configuration = {
        "require_primary_source": _bool_flag("tp_semclust_require_primary_source_enabled"),
        "require_embedding": _bool_flag("tp_semclust_require_embedding_enabled"),
        "use_faiss": _bool_flag("tp_semclust_use_faiss_enabled"),
        "require_faiss": _bool_flag("tp_semclust_require_faiss_enabled"),
        "emit_extra_metrics": _bool_flag("tp_semclust_emit_extra_metrics_enabled"),
        "config_primary_title": _bool_flag("tp_semclust_config_primary_title"),
        "config_primary_description": _bool_flag("tp_semclust_config_primary_description"),
        "config_primary_hashtag": _bool_flag("tp_semclust_config_primary_hashtag"),
    }
    
    # Try to get cluster metadata from payload
    cluster_meta = {}
    if isinstance(payload, dict):
        semantic_cluster_meta = payload.get("semantic_cluster_meta", {})
        if isinstance(semantic_cluster_meta, dict):
            cluster_meta = semantic_cluster_meta
    
    # Extract extra metrics if available
    extra_metrics = {}
    if "tp_semclust_n_clusters" in extractor_features:
        extra_metrics["n_clusters"] = _clean_value(extractor_features["tp_semclust_n_clusters"])
    if "tp_semclust_model_orig_dim" in extractor_features:
        extra_metrics["model_orig_dim"] = _clean_value(extractor_features["tp_semclust_model_orig_dim"])
    if "tp_semclust_model_reduced_dim" in extractor_features:
        extra_metrics["model_reduced_dim"] = _clean_value(extractor_features["tp_semclust_model_reduced_dim"])
    if "tp_semclust_embedding_dim" in extractor_features:
        extra_metrics["embedding_dim"] = _clean_value(extractor_features["tp_semclust_embedding_dim"])
    if "tp_semclust_margin_top2" in extractor_features:
        extra_metrics["margin_top2"] = _clean_value(extractor_features["tp_semclust_margin_top2"])
    if "tp_semclust_compute_ms" in extractor_features:
        extra_metrics["compute_ms"] = _clean_value(extractor_features["tp_semclust_compute_ms"])
    
    render["cluster"] = cluster
    render["source"] = source
    render["presence"] = presence
    render["configuration"] = configuration
    render["safety"] = safety
    render["cluster_meta"] = cluster_meta
    render["extra_metrics"] = extra_metrics
    
    # Summary
    render["summary"] = {
        "present": bool(extractor_features.get("tp_semclust_present", 0) > 0.5),
        "cluster_id": _clean_value(cluster.get("id")),
        "similarity": _clean_value(cluster.get("similarity")),
        "distance": _clean_value(cluster.get("distance")),
        "source_used": next((k for k, v in source.items() if v and k != "fallback_used"), None),
        "fallback_used": bool(source.get("fallback_used", False)),
    }
    
    return render


def render_semantic_cluster_extractor_html(
    npz_path: str,
    output_path: str,
    extractor_features: Dict[str, Any],
    payload: Dict[str, Any],
    meta: Dict[str, Any]
) -> str:
    """
    Генерировать HTML страницу для дебага semantic_cluster_extractor результатов.
    
    Args:
        npz_path: Путь к NPZ файлу (для совместимости)
        output_path: Путь для сохранения HTML файла
        extractor_features: Словарь фич этого extractor'а
        payload: Полный payload из NPZ
        meta: Meta информация из NPZ
    
    Returns:
        Путь к сохраненному HTML файлу
    """
    render = render_semantic_cluster_extractor(extractor_features, payload, meta)
    
    summary = render.get("summary", {})
    cluster = render.get("cluster", {})
    source = render.get("source", {})
    presence = render.get("presence", {})
    configuration = render.get("configuration", {})
    safety = render.get("safety", {})
    cluster_meta = render.get("cluster_meta", {})
    extra_metrics = render.get("extra_metrics", {})
    
    source_used = summary.get("source_used", "N/A")
    if source_used:
        source_used = source_used.capitalize()
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Семантический кластер</title>
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
    </style>
</head>
<body>
    <div class="container">
        <h1>Семантический кластер</h1>
        <p><strong>Компонент:</strong> {render.get("component", "semantic_cluster_extractor")}</p>
        <p><strong>Статус:</strong> <span style="color: {'green' if summary.get('present') else 'red'};">{{'✓ OK' if summary.get('present') else '✗ Нет данных'}}</span></p>
        
        <div class="section">
            <h2>Сводка</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-name">Присутствует</div>
                    <div class="feature-value">{{'Да' if summary.get('present') else 'Нет'}}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">ID кластера</div>
                    <div class="feature-value">{summary.get("cluster_id", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Косинусное сходство</div>
                    <div class="feature-value">{summary.get("similarity", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Расстояние</div>
                    <div class="feature-value">{summary.get("distance", "N/A")}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Источник эмбеддинга</div>
                    <div class="feature-value">{source_used}</div>
                </div>
                <div class="feature-card">
                    <div class="feature-name">Использован fallback</div>
                    <div class="feature-value">{{'Да' if summary.get('fallback_used') else 'Нет'}}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Конфигурация (зеркала features_flat)</h2>
            <table>
                <tr><th>Параметр</th><th>Включено</th></tr>
                <tr><td>require_primary_source</td><td>{{'Да' if configuration.get('require_primary_source') else 'Нет'}}</td></tr>
                <tr><td>require_embedding</td><td>{{'Да' if configuration.get('require_embedding') else 'Нет'}}</td></tr>
                <tr><td>use_faiss</td><td>{{'Да' if configuration.get('use_faiss') else 'Нет'}}</td></tr>
                <tr><td>require_faiss</td><td>{{'Да' if configuration.get('require_faiss') else 'Нет'}}</td></tr>
                <tr><td>emit_extra_metrics</td><td>{{'Да' if configuration.get('emit_extra_metrics') else 'Нет'}}</td></tr>
                <tr><td>primary в конфиге: title</td><td>{{'Да' if configuration.get('config_primary_title') else 'Нет'}}</td></tr>
                <tr><td>primary в конфиге: description</td><td>{{'Да' if configuration.get('config_primary_description') else 'Нет'}}</td></tr>
                <tr><td>primary в конфиге: hashtag</td><td>{{'Да' if configuration.get('config_primary_hashtag') else 'Нет'}}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Кластер</h2>
            <table>
                <tr>
                    <th>Метрика</th>
                    <th>Значение</th>
                    <th>Описание</th>
                </tr>
                <tr>
                    <td>ID кластера</td>
                    <td>{cluster.get("id", "N/A")}</td>
                    <td>Идентификатор ближайшего семантического кластера</td>
                </tr>
                <tr>
                    <td>Косинусное сходство</td>
                    <td>{cluster.get("similarity", "N/A")}</td>
                    <td>Косинусное сходство эмбеддинга к центроиду кластера</td>
                </tr>
                <tr>
                    <td>Расстояние</td>
                    <td>{cluster.get("distance", "N/A")}</td>
                    <td>Расстояние до центроида кластера</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Источник эмбеддинга</h2>
            <table>
                <tr>
                    <th>Источник</th>
                    <th>Использован</th>
                    <th>Присутствует</th>
                </tr>
                <tr>
                    <td>Заголовок</td>
                    <td>{{'Да' if source.get('title') else 'Нет'}}</td>
                    <td>{{'Да' if presence.get('title') else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Описание</td>
                    <td>{{'Да' if source.get('description') else 'Нет'}}</td>
                    <td>{{'Да' if presence.get('description') else 'Нет'}}</td>
                </tr>
                <tr>
                    <td>Хэштеги</td>
                    <td>{{'Да' if source.get('hashtag') else 'Нет'}}</td>
                    <td>{{'Да' if presence.get('hashtag') else 'Нет'}}</td>
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
        if "embedding_dim" in extra_metrics:
            html_content += f"""
                <tr>
                    <td>Размерность эмбеддинга</td>
                    <td>{extra_metrics.get("embedding_dim", "N/A")}</td>
                    <td>Размерность входного эмбеддинга</td>
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
                    <td>Время вычисления кластера в миллисекундах</td>
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
                    <td>Несоответствие размерности</td>
                    <td>{{'Да' if safety.get('dim_mismatch_flag') else 'Нет'}}</td>
                    <td>Соответствует ли размерность эмбеддинга размерности модели</td>
                </tr>
                <tr>
                    <td>Backend FAISS</td>
                    <td>{{'Да' if safety.get('backend_faiss') else 'Нет'}}</td>
                    <td>Использовался ли FAISS для поиска</td>
                </tr>
                <tr>
                    <td>Небезопасный relpath</td>
                    <td>{{'Да' if safety.get('unsafe_relpath') else 'Нет'}}</td>
                    <td>Путь артефакта вне разрешённого join (unsafe)</td>
                </tr>
                <tr>
                    <td>Нет файла / ошибка title .npy</td>
                    <td>{{'Да' if safety.get('title_embed_missing') else 'Нет'}}</td>
                    <td>Безопасный путь, но файл отсутствует или невалиден</td>
                </tr>
                <tr>
                    <td>Нет файла / ошибка description .npy</td>
                    <td>{{'Да' if safety.get('description_embed_missing') else 'Нет'}}</td>
                    <td>Безопасный путь, но файл отсутствует или невалиден</td>
                </tr>
                <tr>
                    <td>Нет файла / ошибка hashtag .npy</td>
                    <td>{{'Да' if safety.get('hashtag_embed_missing') else 'Нет'}}</td>
                    <td>Безопасный путь, но файл отсутствует или невалиден</td>
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
    
    logger.info(f"Semantic cluster extractor HTML render saved to {output_path}")
    return output_path


__all__ = ["render_semantic_cluster_extractor", "render_semantic_cluster_extractor_html"]

