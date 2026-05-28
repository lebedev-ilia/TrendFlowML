"""Configuration for Embedding Service"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Загрузить .env файл если существует
try:
    from dotenv import load_dotenv

    # Искать .env файл в нескольких местах:
    # 1. В корне embedding_service/
    # 2. В корне DataProcessor/
    # 3. В текущей рабочей директории
    env_paths = [
        Path(__file__).parent.parent / ".env",  # embedding_service/.env
        Path(__file__).parent.parent.parent / ".env",  # DataProcessor/.env
        Path.cwd() / ".env",  # Текущая директория/.env
    ]

    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)  # Не перезаписывать существующие переменные
            break
except ImportError:
    # python-dotenv не установлен, работаем только с переменными окружения
    pass


@dataclass
class EmbeddingServiceConfig:
    """Configuration for Embedding Service"""

    # Database
    postgres_host: str = os.environ.get("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.environ.get("POSTGRES_PORT", "5432"))
    postgres_db: str = os.environ.get("POSTGRES_DB", "embeddings")
    postgres_user: str = os.environ.get("POSTGRES_USER", "postgres")
    postgres_password: str = os.environ.get("POSTGRES_PASSWORD", "")

    # Triton
    triton_base_url: str = os.environ.get("TRITON_BASE_URL", "http://localhost:8000")
    triton_timeout_sec: float = float(os.environ.get("TRITON_TIMEOUT_SEC", "30.0"))

    # Storage
    storage_type: str = os.environ.get("STORAGE_TYPE", "local")  # local, s3
    storage_local_path: str = os.environ.get("STORAGE_LOCAL_PATH", "./embedding_storage")
    storage_s3_bucket: Optional[str] = os.environ.get("STORAGE_S3_BUCKET")
    storage_s3_region: Optional[str] = os.environ.get("STORAGE_S3_REGION", "us-east-1")

    # FAISS
    faiss_index_path: str = os.environ.get("FAISS_INDEX_PATH", "./faiss_indices")
    faiss_sync_interval_sec: float = float(os.environ.get("FAISS_SYNC_INTERVAL_SEC", "300.0"))

    # Server
    server_port: int = int(os.environ.get("EMBEDDING_SERVICE_PORT", "8005"))
    server_host: str = os.environ.get("EMBEDDING_SERVICE_HOST", "0.0.0.0")

    # Model assignments (category -> model_name)
    category_model_mapping: dict[str, str] = None

    def __post_init__(self):
        """Initialize default category-model mapping"""
        if self.category_model_mapping is None:
            self.category_model_mapping = {
                "face": "arcface",
                "face_semantic": "arcface",
                "brand": "clip_336",
                "brand_semantic": "clip_336",
                "car": "clip_336",
                "car_semantic": "clip_336",
                "place": "clip_448",
                "place_semantic": "clip_448",
                "person": "clip_224",
                "object": "clip_224",
                "logo": "clip_336",
                "franchise": "clip_224",  # Franchise recognition uses CLIP 224
            }
        # Локальная E2E / sync known_*: в triton/models_t_1 по умолчанию есть clip_image_224, но
        # может ещё не быть clip_image_336 / clip_image_448. Тогда: export EMBEDDING_DEV_MAP_CAR_BRAND_LOGO_TO_CLIP224=1
        if os.environ.get("EMBEDDING_DEV_MAP_CAR_BRAND_LOGO_TO_CLIP224", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            for k in ("brand", "brand_semantic", "car", "car_semantic", "logo"):
                self.category_model_mapping[k] = "clip_224"
        # В E2E репо часто нет clip_image_448, зато есть clip_image_336: export EMBEDDING_DEV_MAP_PLACE_TO_CLIP336=1
        if os.environ.get("EMBEDDING_DEV_MAP_PLACE_TO_CLIP336", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            self.category_model_mapping["place"] = "clip_336"
            self.category_model_mapping["place_semantic"] = "clip_336"

