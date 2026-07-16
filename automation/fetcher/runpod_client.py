"""Минимальный REST/GraphQL-клиент RunPod для Fetcher-инфраструктуры.

НАМЕРЕННО отдельный от automation/runner/runpod_api.py (та же платформа, но независимая система —
см. config.py шапку). Здесь только то, что нужно для постоянных CPU-подов + Network Volume:
создание, список, статус, SSH-эндпоинт. НЕТ "погасить всё" — Fetcher-поды никогда не гасятся
автоматикой (см. AGENT_CONTEXT.md / TZ задачи), удаление — только явным вызовом человеком/агентом
с полным пониманием последствий.
"""
from __future__ import annotations
import time
import requests

import config

HEADERS = lambda: {"Authorization": f"Bearer {config.RUNPOD_API_KEY}"}


def list_pods() -> list[dict]:
    r = requests.get(f"{config.RUNPOD_API}/pods", headers=HEADERS(), timeout=40)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        return data.get("pods", data.get("data", []))
    return data


def get_pod(pod_id: str) -> dict:
    r = requests.get(f"{config.RUNPOD_API}/pods/{pod_id}", headers=HEADERS(), timeout=40)
    r.raise_for_status()
    return r.json()


def pod_endpoint(pod: dict) -> str | None:
    """SSH host:port (см. runner/runpod_api.py — та же логика, продублировано намеренно для
    независимости модулей)."""
    pm = pod.get("portMappings")
    ip = pod.get("publicIp") or pod.get("ip")
    if isinstance(pm, dict):
        port = pm.get("22") or pm.get(22)
        if port and ip:
            return f"{ip}:{port}"
    elif isinstance(pm, list):
        for pr in pm:
            if isinstance(pr, dict) and str(pr.get("privatePort")) == "22":
                return f"{pr.get('ip')}:{pr.get('publicPort')}"
    rt = pod.get("runtime") or {}
    for pr in rt.get("ports", []) or []:
        if isinstance(pr, dict) and pr.get("privatePort") == 22 and pr.get("isIpPublic"):
            return f"{pr.get('ip')}:{pr.get('publicPort')}"
    return None


def wait_running(pod_id: str, timeout: int = 180, interval: int = 8) -> str | None:
    waited = 0
    while waited < timeout:
        try:
            p = get_pod(pod_id)
        except requests.RequestException:
            p = {}
        ep = pod_endpoint(p) if p else None
        if ep:
            return ep
        time.sleep(interval)
        waited += interval
    return None


def delete_pod(pod_id: str) -> bool:
    """Терминация пода. Данные на Network Volume сохраняются. НЕ вызывается автоматически нигде
    в watchdog/report — только явно, человеком или намеренным кодом провижининга."""
    r = requests.delete(f"{config.RUNPOD_API}/pods/{pod_id}", headers=HEADERS(), timeout=60)
    return r.status_code < 400


def start_pod(pod_id: str) -> tuple[bool, str]:
    r = requests.post(f"{config.RUNPOD_API}/pods/{pod_id}/start", headers=HEADERS(), timeout=60)
    if r.status_code >= 400:
        return False, f"HTTP {r.status_code}: {r.text[:300]}"
    return True, (r.text[:300] if r.text else "started")


def restart_pod(pod_id: str) -> tuple[bool, str]:
    r = requests.post(f"{config.RUNPOD_API}/pods/{pod_id}/restart", headers=HEADERS(), timeout=60)
    if r.status_code >= 400:
        return False, f"HTTP {r.status_code}: {r.text[:300]}"
    return True, "restarted"


# ------------------------------------------------------------------ Network Volume
def list_network_volumes() -> list[dict]:
    r = requests.get(f"{config.RUNPOD_API}/networkvolumes", headers=HEADERS(), timeout=30)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        return data.get("networkVolumes", data.get("data", []))
    return data


def create_network_volume(name: str, size_gb: int, data_center_id: str) -> dict:
    """POST /networkvolumes — required: name, size, dataCenterId (схема подтверждена
    2026-07-16 через живой openapi.json)."""
    body = {"name": name, "size": size_gb, "dataCenterId": data_center_id}
    r = requests.post(f"{config.RUNPOD_API}/networkvolumes", headers=HEADERS(), json=body, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"create_network_volume HTTP {r.status_code}: {r.text[:400]}")
    return r.json() if r.text else {}


# ------------------------------------------------------------------ CPU Pod
def cpu_flavors() -> list[dict]:
    """Доступные CPU-флейворы (cpu3c/cpu3g/cpu3m/cpu5c/cpu5g/cpu5m) через GraphQL (REST не отдаёт)."""
    q = "query { cpuFlavors { id displayName groupId minVcpu maxVcpu } }"
    r = requests.post(f"{config.RUNPOD_GRAPHQL}?api_key={config.RUNPOD_API_KEY}",
                      json={"query": q}, timeout=30)
    r.raise_for_status()
    data = r.json()
    return ((data.get("data") or {}).get("cpuFlavors")) or []


def create_cpu_pod(*, name: str, network_volume_id: str, data_center_id: str,
                   vcpu_count: int, container_disk_gb: int, image_name: str,
                   volume_mount_path: str = "/workspace",
                   cpu_flavor_ids: list[str] | None = None,
                   env: dict | None = None) -> dict:
    """POST /pods с computeType=CPU (схема подтверждена 2026-07-16 через живой openapi.json).
    cpuFlavorPriority=availability — RunPod сам берёт самый доступный флейвор из списка-приоритета."""
    body = {
        "name": name,
        "computeType": "CPU",
        "cpuFlavorIds": cpu_flavor_ids or list(config.CPU_FLAVOR_CANDIDATES),
        "cpuFlavorPriority": "availability",
        "vcpuCount": vcpu_count,
        "containerDiskInGb": container_disk_gb,
        "dataCenterIds": [data_center_id],
        "networkVolumeId": network_volume_id,
        "volumeMountPath": volume_mount_path,
        "imageName": image_name,
        "cloudType": "SECURE",
        "ports": ["22/tcp"],
        "locked": True,  # запрет случайного stop/reset (2026-07-16: поды исчезли без явной причины
                         # в первый раз — locked как дополнительная защита сверх persistent-реестра)
    }
    if env:
        body["env"] = env
    r = requests.post(f"{config.RUNPOD_API}/pods", headers=HEADERS(), json=body, timeout=90)
    if r.status_code >= 400:
        raise RuntimeError(f"create_cpu_pod HTTP {r.status_code}: {r.text[:600]}")
    return r.json() if r.text else {}


def account_balance() -> float | None:
    try:
        r = requests.post(f"{config.RUNPOD_GRAPHQL}?api_key={config.RUNPOD_API_KEY}",
                          json={"query": "query { myself { clientBalance } }"}, timeout=20)
        r.raise_for_status()
        data = r.json()
        bal = ((data.get("data") or {}).get("myself") or {}).get("clientBalance")
        return float(bal) if bal is not None else None
    except Exception:
        return None


def status_summary() -> str:
    pods = list_pods()
    if not pods:
        return "Fetcher-подов нет."
    lines = []
    for p in pods:
        st = p.get("desiredStatus") or p.get("status")
        ep = pod_endpoint(p) or "-"
        cost = p.get("costPerHr") or p.get("adjustedCostPerHr") or "?"
        lines.append(f"{p.get('id')} [{p.get('name')}]: {st} ${cost}/ч ssh={ep}")
    return "\n".join(lines)
