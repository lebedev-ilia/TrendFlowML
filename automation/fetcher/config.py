"""Конфигурация Fetcher-инфраструктуры (dataset collector на RunPod).

НАМЕРЕННО ОТДЕЛЬНАЯ от automation/runner/config.py — это независимая система (постоянные CPU-поды
для сбора датасета), не связанная с ML-валидацией компонентов. Общая точка соприкосновения — только
реестр `automation/runner/state/machines.json` (podmanager), куда эта система РЕГИСТРИРУЕТ свои поды
с `policy="persistent"`, чтобы ML-раннер их не трогал (см. automation/runner/runpod_api.py, шапка).

.env читается из automation/fetcher/.env (свой файл, отдельный от runner/.env).
"""
from __future__ import annotations
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except Exception:
    pass

FETCHER_DIR = Path(__file__).resolve().parent
AUTOMATION_DIR = FETCHER_DIR.parent
REPO_DIR = AUTOMATION_DIR.parent  # .../TrendFlowML
FETCHER_CODE_DIR = REPO_DIR / "Fetcher"

STATE_DIR = FETCHER_DIR / "state"
STATE_DIR.mkdir(exist_ok=True)

# Общий реестр машин ML-раннера — сюда РЕГИСТРИРУЕМСЯ (persistent), чтобы ML-агенты не трогали
# наши поды в terminate_all_and_bill()/stop_all_running(). См. automation/runner/podmanager.py.
RUNNER_MACHINES_REGISTRY = AUTOMATION_DIR / "runner" / "state" / "machines.json"

# --- RunPod ---
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_API = "https://rest.runpod.io/v1"
RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"

# --- VK (Третий бот — наблюдатель Fetcher) ---
VK_TOKEN3 = os.environ.get("VK_TOKEN3", "")
VK_OWNER_ID = int(os.environ.get("VK_OWNER_ID", "344881779") or 344881779)
VK_API_VERSION = os.environ.get("VK_API_VERSION", "5.199")
WATCHDOG_MODEL = os.environ.get("FETCHER_WATCHDOG_MODEL", "claude-haiku-4-5-20251001")

# --- Секреты кампании (НЕ коммитить, только через .env/секрет-хранилище) ---
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# --- Поды: имена, роли, спецификация ---
# Единая спецификация железа для всех трёх (подтверждено владельцем): 2 vCPU / 8GB RAM, ~$0.09/ч.
POD_VCPU_COUNT = int(os.environ.get("FETCHER_POD_VCPU", "2"))
POD_CONTAINER_DISK_GB = int(os.environ.get("FETCHER_POD_CONTAINER_DISK_GB", "20"))
POD_MAX_HOURLY_USD = float(os.environ.get("FETCHER_POD_MAX_HOURLY_USD", "0.12"))  # потолок, чуть выше наблюдаемых $0.09
NETWORK_VOLUME_GB = int(os.environ.get("FETCHER_VOLUME_GB", "15"))  # подтверждено владельцем: 15ГБ достаточно (видео удаляются после HF-аплоада)
PREFERRED_DATACENTERS = [d.strip() for d in os.environ.get(
    "FETCHER_PREFERRED_DATACENTERS", "EU-CZ-1,EU-RO-1,EU-SE-1,US-KS-2,US-IL-1").split(",") if d.strip()]
# Подтверждено живым тестом (2026-07-16): cpu3g@2vCPU -> ровно 8ГБ RAM, $0.08/ч (в рамках спеки
# владельца "2 CPU 8GB RAM", $0.09/ч). cpu3c той же вилки даёт только 4ГБ на 2vCPU — не подходит.
CPU_FLAVOR_CANDIDATES = ["cpu3g", "cpu5g"]  # general purpose — 4ГБ/vCPU, даёт 8ГБ на 2vCPU

PODS = {
    "fetcher-main": {
        "worker_id": "fetcher-main",
        "worker_shard_index": 0,
        "worker_shard_count": 3,
        "role": "main",  # discover (непрерывно) + workers shard 0/3
        "volume_name": "fetcher-main-vol",
    },
    "fetcher-worker-b": {
        "worker_id": "fetcher-worker-b",
        "worker_shard_index": 1,
        "worker_shard_count": 3,
        "role": "worker",  # workers shard 1/3
        "volume_name": "fetcher-worker-b-vol",
    },
    "fetcher-worker-c": {
        "worker_id": "fetcher-worker-c",
        "worker_shard_index": 2,
        "worker_shard_count": 3,
        "role": "worker",  # workers shard 2/3
        "volume_name": "fetcher-worker-c-vol",
    },
}

DEFAULT_IMAGE = os.environ.get("FETCHER_POD_IMAGE", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
CAMPAIGN_PROFILE = "100k-monthly"
HF_REPO_PREFIX = "Ilialebedev"
OUTPUT_DIR_ON_POD = "/workspace/dataset_runs/100k-monthly"

# --- Мониторинг ---
WATCHDOG_INTERVAL_SEC = int(os.environ.get("FETCHER_WATCHDOG_INTERVAL_SEC", str(3600)))
HOURLY_REPORT_INTERVAL_SEC = int(os.environ.get("FETCHER_REPORT_INTERVAL_SEC", str(3600)))

SSH_KEY_PATH = FETCHER_DIR / "ssh" / "id_ed25519"


def require(name: str, value) -> None:
    if not value:
        raise SystemExit(f"[fetcher/config] Не задана переменная окружения {name}. Заполни automation/fetcher/.env.")
