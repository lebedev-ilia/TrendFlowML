"""Управление подами RunPod через REST API.

Важный урок (2026-07-05): pod-id МЕНЯЕТСЯ при миграции. Поэтому останавливаем НЕ по
захардкоженному id, а перечисляя все RUNNING-поды и гася каждый.

КРИТИЧНО (2026-07-16): этот аккаунт RunPod держит поды ДВУХ независимых систем — ML-валидация
(этот раннер, эфемерные GPU-поды) и Fetcher dataset collector (постоянные CPU-поды, отдельная
инфраструктура, см. automation/fetcher/). Массовые операции (`stop_all_running`,
`terminate_all_and_bill`) должны трогать ТОЛЬКО поды ЭТОГО раннера — иначе рабочий лимит/простой/
крэш ML-агента случайно погасит вечные Fetcher-поды. Защита — через реестр `state/machines.json`
(podmanager.py): под с `policy=="persistent"` или `kind` не GPU-шный считается ЧУЖИМ и исключается
из `own_pods()`/массовых операций. Не создавай Fetcher-поды и другую постоянную инфраструктуру без
немедленной регистрации через `podmanager.register(..., policy="persistent")` — иначе она не защищена.
"""
from __future__ import annotations
import json
import time
import requests

import config
import budget

HEADERS = lambda: {"Authorization": f"Bearer {config.RUNPOD_API_KEY}"}

# Виды подов, которые массовые GPU-операции ЭТОГО раннера считают "своими". Всё остальное
# (persistent policy, kind="fetcher"/"cpu"/... не входящий сюда) — чужая инфраструктура, не трогаем.
_OWN_POD_KINDS = {"gpu", None}


def _registry() -> dict:
    try:
        reg_path = config.STATE_DIR / "machines.json"
        if reg_path.exists():
            return json.loads(reg_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _protected_pod_ids() -> set[str]:
    """ID подов из реестра podmanager, которые НЕ принадлежат этому раннеру (persistent policy —
    напр. Fetcher — или kind вне _OWN_POD_KINDS). См. предупреждение в шапке модуля."""
    reg = _registry()
    out = set()
    for mid, m in reg.items():
        if not isinstance(m, dict):
            continue
        if m.get("policy") == "persistent" or m.get("kind") not in _OWN_POD_KINDS:
            out.add(mid)
    return out


def list_pods() -> list[dict]:
    """ВСЕ поды аккаунта (включая чужую инфраструктуру) — для чтения/статуса. Для массовых
    деструктивных операций используй own_pods(), не эту функцию напрямую."""
    r = requests.get(f"{config.RUNPOD_API}/pods", headers=HEADERS(), timeout=40)
    r.raise_for_status()
    data = r.json()
    # API может вернуть список или {"pods": [...]}
    if isinstance(data, dict):
        return data.get("pods", data.get("data", []))
    return data


def own_pods() -> list[dict]:
    """Поды ЭТОГО раннера — с исключением защищённой чужой инфраструктуры (Fetcher и т.п.).
    Используй это, а не list_pods(), для start/stop/terminate/clone-логики ML-валидации."""
    protected = _protected_pod_ids()
    return [p for p in list_pods() if p.get("id") not in protected]


def running_pods() -> list[dict]:
    """RUNNING-поды ЭТОГО раннера (own_pods — чужая защищённая инфраструктура исключена)."""
    out = []
    for p in own_pods():
        status = p.get("desiredStatus") or p.get("status") or ""
        if str(status).upper() == "RUNNING":
            out.append(p)
    return out


def pod_endpoint(pod: dict) -> str | None:
    """SSH host:port, если под уже опубликовал порты (иначе None).

    RunPod REST отдаёт portMappings как dict {"22": <publicPort>} + publicIp на верхнем
    уровне (актуальная схема). Историческая схема (список dict с privatePort) и runtime.ports
    поддержаны как fallback.
    """
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
    # альтернативная схема (runtime.ports)
    rt = pod.get("runtime") or {}
    for pr in rt.get("ports", []) or []:
        if isinstance(pr, dict) and pr.get("privatePort") == 22 and pr.get("isIpPublic"):
            return f"{pr.get('ip')}:{pr.get('publicPort')}"
    return None


def get_pod(pod_id: str) -> dict:
    r = requests.get(f"{config.RUNPOD_API}/pods/{pod_id}", headers=HEADERS(), timeout=40)
    r.raise_for_status()
    return r.json()


def start_pod(pod_id: str) -> tuple[bool, str]:
    """Пытается запустить под. Возвращает (ok, message). ok=False обычно = нет свободного GPU."""
    r = requests.post(f"{config.RUNPOD_API}/pods/{pod_id}/start", headers=HEADERS(), timeout=60)
    if r.status_code >= 400:
        # RunPod часто отвечает ошибкой, когда в дата-центре нет свободного GPU нужного типа.
        return False, f"HTTP {r.status_code}: {r.text[:300]}"
    return True, (r.text[:300] if r.text else "started")


def wait_running(pod_id: str, timeout: int = 120, interval: int = 8) -> str | None:
    """Ждёт, пока под реально поднимется (появится SSH-эндпоинт). Возвращает host:port или None."""
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
    """Терминация пода (compute). Данные на Network Volume при этом сохраняются."""
    r = requests.delete(f"{config.RUNPOD_API}/pods/{pod_id}", headers=HEADERS(), timeout=60)
    return r.status_code < 400


def clone_spec(src_pod: dict, name: str | None = None) -> dict:
    """Собрать тело для создания нового пода по образцу существующего (best-effort)."""
    body: dict = {}
    body["name"] = name or (str(src_pod.get("name") or "trendflow") + "-mig")
    for src_key, dst_key in [
        ("imageName", "imageName"), ("image", "imageName"),
        ("gpuTypeIds", "gpuTypeIds"), ("gpuTypeId", "gpuTypeIds"),
        ("gpuCount", "gpuCount"),
        ("networkVolumeId", "networkVolumeId"),
        ("volumeMountPath", "volumeMountPath"),
        ("containerDiskInGb", "containerDiskInGb"),
        ("ports", "ports"), ("env", "env"), ("cloudType", "cloudType"),
    ]:
        if src_key in src_pod and src_pod[src_key] is not None and dst_key not in body:
            val = src_pod[src_key]
            if dst_key == "gpuTypeIds" and isinstance(val, str):
                val = [val]
            body[dst_key] = val
    nv = network_volume_id(src_pod)
    if nv and "networkVolumeId" not in body:
        body["networkVolumeId"] = nv
    body.setdefault("gpuCount", 1)
    body.setdefault("volumeMountPath", "/workspace")
    return body


def create_pod(body: dict) -> dict:
    r = requests.post(f"{config.RUNPOD_API}/pods", headers=HEADERS(), json=body, timeout=90)
    if r.status_code >= 400:
        raise RuntimeError(f"create_pod HTTP {r.status_code}: {r.text[:400]}")
    return r.json() if r.text else {}


GRAPHQL = "https://api.runpod.io/graphql"


def account_balance() -> float | None:
    """Баланс аккаунта RunPod в $. None, если не удалось получить (НЕ считать как $0 —
    отсутствие данных отличается от реально пустого баланса)."""
    try:
        r = requests.post(f"{GRAPHQL}?api_key={config.RUNPOD_API_KEY}",
                          json={"query": "query { myself { clientBalance } }"}, timeout=20)
        r.raise_for_status()
        data = r.json()
        bal = ((data.get("data") or {}).get("myself") or {}).get("clientBalance")
        return float(bal) if bal is not None else None
    except Exception:
        return None


def volume_info() -> dict:
    """Наш Network Volume: {'id','dc','name'}. Ищем по RUNPOD_VOLUME_ID, иначе по имени, иначе первый."""
    try:
        r = requests.get(f"{config.RUNPOD_API}/networkvolumes", headers=HEADERS(), timeout=30)
        vols = r.json()
        vols = vols.get("networkVolumes", vols) if isinstance(vols, dict) else vols
        vols = vols or []
        target = None
        for v in vols:
            if config.RUNPOD_VOLUME_ID and v.get("id") == config.RUNPOD_VOLUME_ID:
                target = v; break
        if not target and config.RUNPOD_VOLUME_NAME:
            for v in vols:
                if config.RUNPOD_VOLUME_NAME.lower() in str(v.get("name", "")).lower():
                    target = v; break
        target = target or (vols[0] if vols else None)
        if not target:
            return {}
        return {"id": target.get("id"), "dc": target.get("dataCenterId"), "name": target.get("name")}
    except Exception:
        return {}


def volume_datacenter() -> str | None:
    return volume_info().get("dc")


# Карта: ключевое слово из UI-имени -> gpuTypeId для create_pod (RunPod REST).
GPU_ID_MAP = {
    "a4500": "NVIDIA RTX A4500", "a4000": "NVIDIA RTX A4000", "a5000": "NVIDIA RTX A5000",
    "a6000": "NVIDIA RTX A6000", "a40": "NVIDIA A40", "a30": "NVIDIA A30",
    "4000 ada": "NVIDIA RTX 4000 Ada Generation", "2000 ada": "NVIDIA RTX 2000 Ada Generation",
    "5000 ada": "NVIDIA RTX 5000 Ada Generation", "6000 ada": "NVIDIA RTX 6000 Ada Generation",
    "4090": "NVIDIA GeForce RTX 4090", "5090": "NVIDIA GeForce RTX 5090",
    "l4": "NVIDIA L4", "l40s": "NVIDIA L40S", "l40": "NVIDIA L40",
    "pro 4500": "NVIDIA RTX PRO 4500 Blackwell", "pro 6000": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
}


def _match_gpu_id(name: str) -> str | None:
    n = (name or "").lower().replace("rtx", "").strip()
    for key, gid in GPU_ID_MAP.items():
        if key in n:
            return gid
    return None


def _scraped_gpus(max_age_sec: int = 300) -> list[dict]:
    """Точные цены/наличие из Chrome-парсера console.runpod.io. Обновляет, если устарело."""
    import json as _j, time as _t, subprocess as _s, sys as _sys
    fresh = (config.RUNPOD_GPUS_JSON.exists()
             and _t.time() - _j.loads(config.RUNPOD_GPUS_JSON.read_text()).get("ts", 0) < max_age_sec)
    if not fresh and config.CLAUDE_CDP_PORT:
        try:
            _s.run([_sys.executable, str(config.RUNNER_DIR / "runpod_gpu_scraper.py"),
                    "--cdp", config.CLAUDE_CDP_PORT], capture_output=True, text=True, timeout=90)
        except Exception:
            pass
    if config.RUNPOD_GPUS_JSON.exists():
        try:
            return _j.loads(config.RUNPOD_GPUS_JSON.read_text()).get("gpus", [])
        except Exception:
            return []
    return []


def list_gpu_types() -> list[dict]:
    """ТОЧНЫЕ цена+наличие GPU для нашего Network Volume. Приоритет — Chrome-парсер console.runpod.io
    (после выбора volume); GraphQL — фолбэк."""
    scraped = _scraped_gpus()
    if scraped:
        out = []
        for s in scraped:
            stock = str(s.get("stock", "")).lower()
            if stock in ("unavailable", "out of stock", "none"):
                continue
            out.append({"id": _match_gpu_id(s["name"]), "displayName": s["name"],
                        "memoryInGb": s.get("memoryInGb", ""), "price": s.get("price"),
                        "stock": s.get("stock"), "cloud": "SECURE"})
        out = [x for x in out if x["price"] is not None and x["id"]]
        out.sort(key=lambda x: x["price"])
        if out:
            return out
    # --- фолбэк: GraphQL ---
    dc = volume_datacenter()
    inp = {"gpuCount": 1}
    if dc:
        inp["dataCenterId"] = dc
    query = ("query($i:GpuLowestPriceInput){gpuTypes{id displayName memoryInGb "
             "lowestPrice(input:$i){uninterruptablePrice stockStatus}}}")
    r = requests.post(f"{GRAPHQL}?api_key={config.RUNPOD_API_KEY}",
                      json={"query": query, "variables": {"i": inp}}, timeout=40)
    r.raise_for_status()
    data = r.json()
    gpus = ((data.get("data") or {}).get("gpuTypes")) or []
    out = []
    for g in gpus:
        lp = g.get("lowestPrice") or {}
        out.append({
            "id": g.get("id"),
            "displayName": g.get("displayName"),
            "memoryInGb": g.get("memoryInGb"),
            "price": lp.get("uninterruptablePrice"),   # on-demand $/час в НАШЕМ датацентре
            "stock": lp.get("stockStatus"),            # High/Medium/Low/None
            "cloud": "SECURE",
            "datacenter": dc,
        })
    # Доступные = цена есть И stock не None/пусто. Сортировка по цене.
    avail = [x for x in out if x["price"] is not None and x.get("stock") and str(x["stock"]).lower() != "none"]
    avail.sort(key=lambda x: x["price"])
    return avail


def stop_pod(pod_id: str) -> dict:
    r = requests.post(f"{config.RUNPOD_API}/pods/{pod_id}/stop", headers=HEADERS(), timeout=60)
    r.raise_for_status()
    return r.json() if r.text else {}


def stop_all_running() -> list[str]:
    """Погасить ВСЕ RUNNING-поды ЭТОГО раннера (own_pods — чужая инфра исключена). Идемпотентно."""
    stopped = []
    for p in running_pods():
        pid = p.get("id")
        if pid:
            try:
                stop_pod(pid)
                stopped.append(pid)
            except requests.HTTPError:
                pass
    return stopped


def network_volume_id(pod: dict) -> str | None:
    return pod.get("networkVolumeId") or (pod.get("networkVolume") or {}).get("id")


def terminate_all_and_bill() -> list[str]:
    """УДАЛИТЬ (terminate) ВСЕ поды ЭТОГО раннера (own_pods — чужая инфра, напр. Fetcher,
    исключена) и закрыть в бюджете. Данные на Network Volume сохраняются.
    Жизненный цикл: создать → поработать → удалить (не останавливать)."""
    ids = []
    for p in own_pods():
        pid = p.get("id")
        if not pid:
            continue
        try:
            budget.pod_close(pid)
            delete_pod(pid)
            ids.append(pid)
        except Exception:
            pass
    return ids


def stop_all_and_bill() -> list[str]:
    """Погасить все RUNNING-поды и закрыть их в бюджете (посчитать аренду)."""
    ids = stop_all_running()
    for pid in ids:
        try:
            budget.pod_close(pid)
        except Exception:
            pass
    return ids


def status_summary() -> str:
    pods = list_pods()
    if not pods:
        return "Подов нет."
    lines = []
    for p in pods:
        st = p.get("desiredStatus") or p.get("status")
        ep = pod_endpoint(p) or "-"
        vol = network_volume_id(p) or "-"
        pin = " <PINNED>" if config.RUNPOD_POD_ID and p.get("id") == config.RUNPOD_POD_ID else ""
        lines.append(f"{p.get('id')}: {st} gpu={p.get('gpuCount', '?')} netVol={vol} ssh={ep}{pin}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    config.require("RUNPOD_API_KEY", config.RUNPOD_API_KEY)
    if len(sys.argv) > 1 and sys.argv[1] == "gpus":
        print("Датацентр Network Volume:", volume_datacenter())
        for g in list_gpu_types():
            print(f"  {g['displayName']:<22} {g['memoryInGb']}GB  ${g['price']}/ч  stock={g['stock']}")
    else:
        print(status_summary())
