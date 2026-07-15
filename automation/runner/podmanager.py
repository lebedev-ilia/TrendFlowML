"""PodManager — единый учёт и управление всеми машинами (поды/ноды/серверы) через провайдеры.

Сейчас реально работает RunPodProvider; K8sProvider — интерфейс-заглушка (прод-облако не выбрано).
Реестр state/machines.json подхватывает и машины, поднятые в обход. Денежные действия (create/stop/
migrate) в интерактиве идут через инструмент `manager`/`pod_control` с подтверждением в VK.

CLI (для ассистента/владельца): python podmanager.py list|health <id>
"""
from __future__ import annotations
import json
import time

import config
import runpod_api

REG = config.STATE_DIR / "machines.json"


# ------------------------------------------------------------------ реестр
def _load() -> dict:
    if REG.exists():
        try:
            return json.loads(REG.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save(d: dict) -> None:
    REG.write_text(json.dumps(d, ensure_ascii=False, indent=2))


def upsert(m: dict) -> None:
    d = _load()
    cur = d.get(m["id"], {})
    cur.update(m)
    cur["last_seen"] = time.time()
    cur.setdefault("created_at", time.time())
    d[m["id"]] = cur
    _save(d)


def register(id: str, kind: str = "other", purpose: str = "", policy: str = "persistent",
             provider: str = "manual", ssh: str = "", created_by: str = "owner") -> None:
    upsert({"id": id, "kind": kind, "purpose": purpose, "policy": policy,
            "provider": provider, "ssh": ssh, "created_by": created_by, "status": "registered"})


def forget(id: str) -> bool:
    d = _load()
    ok = d.pop(id, None) is not None
    _save(d)
    return ok


# --------------------------------------------------------------- провайдеры
class Provider:
    name = "base"

    def discover(self) -> list[dict]:
        return []


class RunPodProvider(Provider):
    name = "runpod"

    def discover(self) -> list[dict]:
        out = []
        for p in runpod_api.list_pods():
            out.append({
                "id": p.get("id"),
                "provider": "runpod",
                "kind": "gpu",
                "policy": "ephemeral",
                "status": p.get("desiredStatus") or p.get("status"),
                "ssh": runpod_api.pod_endpoint(p) or "",
                "gpu": (p.get("machine") or {}).get("gpuType") or p.get("gpuTypeIds"),
                "netVol": runpod_api.network_volume_id(p),
            })
        return out


class K8sProvider(Provider):
    name = "k8s"

    def discover(self) -> list[dict]:
        # Заглушка: прод-облако не выбрано. Позже — kubectl get nodes/pods через kubeconfig.
        return []


PROVIDERS = [RunPodProvider(), K8sProvider()]


# ------------------------------------------------------------------ сводка
def list_machines() -> list[dict]:
    """Все машины: из провайдеров (актуальный статус) + persistent/manual из реестра."""
    d = _load()
    seen = set()
    result = []
    for prov in PROVIDERS:
        try:
            for m in prov.discover():
                if not m.get("id"):
                    continue
                upsert(m)  # синхронизируем реестр
                seen.add(m["id"])
                result.append(_load()[m["id"]])
        except Exception as e:
            result.append({"id": f"[{prov.name} error]", "status": str(e)[:120]})
    for mid, m in d.items():
        if mid not in seen:
            # runpod-под, которого больше нет в API → устарел, чистим (не показываем как RUNNING).
            if m.get("provider") == "runpod":
                d2 = _load(); d2.pop(mid, None); _save(d2)
                continue
            result.append(m)  # оставляем только manual/persistent записи
    return result


def summary_text() -> str:
    ms = list_machines()
    if not ms:
        return "Машин нет."
    lines = [f"🖥️ Машин: {len(ms)}"]
    for m in ms:
        lines.append(f"• {m.get('id')} [{m.get('provider','?')}/{m.get('kind','?')}/{m.get('policy','?')}] "
                     f"{m.get('status','?')} ssh={m.get('ssh') or '-'}"
                     + (f" purpose={m['purpose']}" if m.get('purpose') else ""))
    return "\n".join(lines)


def health(mid: str) -> str:
    for m in list_machines():
        if m.get("id") == mid:
            return f"{mid}: {m.get('status')} ssh={m.get('ssh') or '-'} provider={m.get('provider')}"
    return f"{mid}: не найдена в реестре/провайдерах."


# ------------------------------------------------------------ управление подами
def terminate_pod(pod_id: str) -> tuple[bool, str]:
    """Полностью удалить под (прекратить счёт). Данные на Network Volume сохраняются.

    Используется:
    - Ручное удаление неработающих подов
    - Self-recovery: если 2 пода EXITED, удалить 1 и поднять новый
    """
    try:
        if runpod_api.delete_pod(pod_id):
            forget(pod_id)
            return True, f"✓ Pod {pod_id} terminated (delete_pod API)"
        else:
            return False, f"✗ delete_pod failed for {pod_id}"
    except Exception as e:
        return False, f"✗ Exception in terminate_pod: {str(e)[:200]}"


def self_recovery_check() -> str:
    """Проверить и восстановить поды:
    - Если 2+ пода со статусом EXITED, удалить 1, поднять новый по образцу другого.

    Вызывается в ночном режиме (agent сам).
    """
    machines = list_machines()
    exited = [m for m in machines if m.get("status", "").upper() == "EXITED"
              and m.get("provider") == "runpod"]

    if len(exited) < 2:
        return f"Self-recovery: {len(exited)} EXITED pod(s), no action needed"

    # Есть 2+ EXITED. Удаляем первый, пытаемся поднять новый.
    to_delete = exited[0]
    to_clone = exited[1]  # или RUNNING если есть

    # Ищем рабочий под для клонирования
    running = [m for m in machines if m.get("status", "").upper() == "RUNNING"
               and m.get("provider") == "runpod"]
    if running:
        to_clone = running[0]

    delete_ok, delete_msg = terminate_pod(to_delete["id"])
    if not delete_ok:
        return f"Self-recovery failed on delete: {delete_msg}"

    # Пытаемся получить спец пода для клонирования и создать новый
    try:
        clone_spec_dict = runpod_api.clone_spec(to_clone)
        new_pod = runpod_api.create_pod(clone_spec_dict)
        new_id = new_pod.get("id")
        if new_id:
            upsert({"id": new_id, "provider": "runpod", "kind": "gpu",
                   "policy": "ephemeral", "status": "CREATED", "created_by": "self-recovery"})
            return f"✓ Self-recovery: deleted {to_delete['id']}, created new pod {new_id}"
        else:
            return f"✗ Self-recovery create_pod returned no id: {str(new_pod)[:200]}"
    except Exception as e:
        return f"✗ Self-recovery create_pod failed: {str(e)[:200]}"


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "health" and len(sys.argv) > 2:
        print(health(sys.argv[2]))
    elif cmd == "terminate" and len(sys.argv) > 2:
        ok, msg = terminate_pod(sys.argv[2])
        print(msg)
        exit(0 if ok else 1)
    elif cmd == "self-recovery":
        print(self_recovery_check())
    else:
        print(summary_text())
