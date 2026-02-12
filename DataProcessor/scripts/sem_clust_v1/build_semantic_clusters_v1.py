#!/usr/bin/env python3
"""
Скрипт для создания semantic_clusters_v1 из эмбеддингов.

Генерирует:
- pca.npy: матрица PCA (orig_dim, reduced_dim)
- centroids.npy: центроиды кластеров (n_clusters, reduced_dim)
- clusters.jsonl: словарь кластеров (id → name/group)

Использование:
    python3 scripts/build_semantic_clusters_v1.py \
        --embeddings-path path/to/embeddings.npy \
        --output-dir dp_models/bundled_models/text/semantic_clusters_v1 \
        --n-clusters 32 \
        --reduced-dim 128 \
        --orig-dim 1024
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def load_embeddings(embeddings_path: str) -> np.ndarray:
    """Загрузить эмбеддинги из файла."""
    print(f"Загрузка эмбеддингов из {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    elif embeddings.ndim > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    
    print(f"  Форма: {embeddings.shape}")
    print(f"  Тип: {embeddings.dtype}")
    return embeddings


def train_pca(embeddings: np.ndarray, reduced_dim: int) -> np.ndarray:
    """Обучить PCA и вернуть матрицу преобразования."""
    print(f"\nОбучение PCA: {embeddings.shape[1]} -> {reduced_dim}...")
    
    pca = PCA(n_components=reduced_dim, random_state=42)
    pca.fit(embeddings)
    
    # Сохраняем матрицу преобразования (orig_dim, reduced_dim)
    pca_matrix = pca.components_.T  # (orig_dim, reduced_dim)
    
    print(f"  PCA матрица: {pca_matrix.shape}")
    print(f"  Объясненная дисперсия: {pca.explained_variance_ratio_.sum():.4f}")
    
    return pca_matrix.astype(np.float32)


def apply_pca(embeddings: np.ndarray, pca_matrix: np.ndarray) -> np.ndarray:
    """Применить PCA к эмбеддингам."""
    print(f"\nПрименение PCA...")
    reduced = embeddings @ pca_matrix
    print(f"  Форма после PCA: {reduced.shape}")
    return reduced.astype(np.float32)


def l2_normalize(vectors: np.ndarray, axis: int = 1) -> np.ndarray:
    """L2-нормализация векторов."""
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return (vectors / norms).astype(np.float32)


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """Кластеризовать эмбеддинги с помощью KMeans."""
    print(f"\nКластеризация KMeans: {n_clusters} кластеров...")
    
    # L2-нормализация перед кластеризацией (для косинусного расстояния)
    normalized = l2_normalize(embeddings)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
    kmeans.fit(normalized)
    
    # Центроиды уже нормализованы после fit на нормализованных данных
    centroids = kmeans.cluster_centers_.astype(np.float32)
    centroids = l2_normalize(centroids)  # Дополнительная нормализация для безопасности
    
    print(f"  Центроиды: {centroids.shape}")
    print(f"  Inertia: {kmeans.inertia_:.4f}")
    
    return centroids


def create_clusters_jsonl(n_clusters: int, output_path: str, cluster_names: List[str] = None, cluster_groups: List[str] = None) -> None:
    """Создать clusters.jsonl файл."""
    print(f"\nСоздание clusters.jsonl...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(n_clusters):
            cluster_data = {
                "cluster_id": i,
                "name": cluster_names[i] if cluster_names and i < len(cluster_names) else f"cluster_{i:03d}",
                "group": cluster_groups[i] if cluster_groups and i < len(cluster_groups) else "general",
            }
            f.write(json.dumps(cluster_data, ensure_ascii=False) + "\n")
    
    print(f"  Создано {n_clusters} записей в {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Создание semantic_clusters_v1 из эмбеддингов")
    parser.add_argument(
        "--embeddings-path",
        type=str,
        required=True,
        help="Путь к файлу с эмбеддингами (.npy, форма: (N, D))"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dp_models/bundled_models/text/semantic_clusters_v1",
        help="Директория для сохранения результатов"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=32,
        help="Количество кластеров"
    )
    parser.add_argument(
        "--reduced-dim",
        type=int,
        default=128,
        help="Размерность после PCA"
    )
    parser.add_argument(
        "--orig-dim",
        type=int,
        default=None,
        help="Исходная размерность (автоматически определяется из данных, если не указана)"
    )
    parser.add_argument(
        "--clusters-names",
        type=str,
        default=None,
        help="JSON файл с именами кластеров: [{'id': 0, 'name': '...', 'group': '...'}, ...]"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state для воспроизводимости"
    )
    
    args = parser.parse_args()
    
    # Загрузка эмбеддингов
    embeddings = load_embeddings(args.embeddings_path)
    orig_dim = args.orig_dim if args.orig_dim is not None else embeddings.shape[1]
    
    if embeddings.shape[1] != orig_dim:
        print(f"Предупреждение: размерность эмбеддингов ({embeddings.shape[1]}) не совпадает с --orig-dim ({orig_dim})")
        print(f"Используется размерность из данных: {embeddings.shape[1]}")
        orig_dim = embeddings.shape[1]
    
    # Обучение PCA
    pca_matrix = train_pca(embeddings, args.reduced_dim)
    
    # Применение PCA
    reduced_embeddings = apply_pca(embeddings, pca_matrix)
    
    # Кластеризация
    centroids = cluster_embeddings(reduced_embeddings, args.n_clusters, args.random_state)
    
    # Загрузка имен кластеров (если указаны)
    cluster_names = None
    cluster_groups = None
    if args.clusters_names:
        with open(args.clusters_names, "r", encoding="utf-8") as f:
            clusters_meta = json.load(f)
            cluster_names = [c.get("name", f"cluster_{i:03d}") for i, c in enumerate(clusters_meta)]
            cluster_groups = [c.get("group", "general") for i, c in enumerate(clusters_meta)]
    
    # Создание выходной директории
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохранение файлов
    print(f"\nСохранение файлов в {output_dir}...")
    
    pca_path = output_dir / "pca.npy"
    np.save(pca_path, pca_matrix)
    print(f"  ✓ pca.npy: {pca_matrix.shape}")
    
    centroids_path = output_dir / "centroids.npy"
    np.save(centroids_path, centroids)
    print(f"  ✓ centroids.npy: {centroids.shape}")
    
    clusters_jsonl_path = output_dir / "clusters.jsonl"
    create_clusters_jsonl(args.n_clusters, str(clusters_jsonl_path), cluster_names, cluster_groups)
    print(f"  ✓ clusters.jsonl")
    
    # Проверка совместимости
    print(f"\nПроверка совместимости...")
    assert pca_matrix.shape == (orig_dim, args.reduced_dim), \
        f"PCA форма должна быть ({orig_dim}, {args.reduced_dim}), получено {pca_matrix.shape}"
    assert centroids.shape == (args.n_clusters, args.reduced_dim), \
        f"Центроиды форма должна быть ({args.n_clusters}, {args.reduced_dim}), получено {centroids.shape}"
    assert pca_matrix.shape[1] == centroids.shape[1], \
        f"reduced_dim должен совпадать: PCA={pca_matrix.shape[1]}, Centroids={centroids.shape[1]}"
    
    print("\n✓ Все проверки пройдены!")
    print(f"\nРезультаты сохранены в: {output_dir}")
    print(f"\nСледующие шаги:")
    print(f"1. Проверьте, что spec файл существует: dp_models/spec_catalog/text/semantic_clusters_v1.yaml")
    print(f"2. Пересоберите manifest (если используется): python3 dp_models/bundled_models/_tools/build_db_manifest.py")


if __name__ == "__main__":
    main()

