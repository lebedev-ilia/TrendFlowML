#!/usr/bin/env python3
"""Создание бакетов MinIO/S3 для Fetcher (video-analytics-raw, video-analytics-processed, video-analytics-temp).

Запуск из корня Fetcher с теми же переменными окружения, что и воркер:
  PYTHONPATH=. python scripts/init_minio_buckets.py

Или после export переменных для run_worker_on_host.sh:
  export FETCHER_S3_ENDPOINT_URL=http://localhost:9000
  export FETCHER_S3_ACCESS_KEY=minioadmin
  export FETCHER_S3_SECRET_KEY=minioadmin123
  PYTHONPATH=. python scripts/init_minio_buckets.py
"""
from __future__ import annotations

import sys

import boto3
from botocore.exceptions import ClientError

# Настройки из fetcher.config (нужен PYTHONPATH=Fetcher root)
try:
    from fetcher.config import settings
except ImportError:
    print("Run from Fetcher root with PYTHONPATH=. or install fetcher.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    endpoint = str(settings.s3_endpoint_url) if settings.s3_endpoint_url else None
    use_ssl = settings.s3_use_ssl
    if endpoint and endpoint.startswith("http://"):
        use_ssl = False
    elif endpoint and endpoint.startswith("https://"):
        use_ssl = True

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region or "us-east-1",
        use_ssl=use_ssl,
        verify=settings.s3_verify_ssl,
    )

    buckets = [
        settings.bucket_raw,
        settings.bucket_processed,
        settings.bucket_temp,
    ]
    for name in buckets:
        try:
            client.head_bucket(Bucket=name)
            print(f"Bucket '{name}' already exists.")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                try:
                    client.create_bucket(Bucket=name)
                    print(f"Created bucket '{name}'.")
                except ClientError as create_err:
                    print(f"Failed to create bucket '{name}': {create_err}", file=sys.stderr)
                    sys.exit(1)
            else:
                print(f"Error checking bucket '{name}': {e}", file=sys.stderr)
                sys.exit(1)


if __name__ == "__main__":
    main()
