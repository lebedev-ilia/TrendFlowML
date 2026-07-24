"""SSH-хелперы для деплоя/запуска/перезапуска ролей на Fetcher-подах.

Используется: (1) один раз вручную для первого запуска (после того как будут секреты —
HF_TOKEN, youtube keys.txt, cookies), (2) агентом-наблюдателем (watchdog.py) для перезапуска
процесса после фикса кода.
"""
from __future__ import annotations
import json
import subprocess

import config
import runpod_client as rc


def _pod_ssh_endpoint(pod_id: str) -> tuple[str, str] | None:
    p = rc.get_pod(pod_id)
    ep = rc.pod_endpoint(p)
    if not ep:
        # REST API /v1/pods/{id} для CPU-подов не возвращает runtime.ports (баг RunPod, 2026-07-20):
        # publicIp всегда пустая строка, portMappings/runtime отсутствуют. GraphQL-fallback.
        ep = rc.get_pod_endpoint_gql(pod_id)
    if not ep:
        return None
    host, port = ep.split(":")
    return host, port


def _ssh_base(host: str, port: str) -> list[str]:
    return ["ssh", "-F", "/dev/null", "-i", str(config.SSH_KEY_PATH), "-p", port,
            f"root@{host}", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=15"]


def ssh_run(pod_name: str, remote_cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    info = _provision_info()[pod_name]
    host, port = _pod_ssh_endpoint(info["pod_id"])
    return subprocess.run(_ssh_base(host, port) + [remote_cmd],
                          capture_output=True, text=True, timeout=timeout)


def scp_to_pod(pod_name: str, local_path: str, remote_path: str, timeout: int = 60) -> subprocess.CompletedProcess:
    info = _provision_info()[pod_name]
    host, port = _pod_ssh_endpoint(info["pod_id"])
    return subprocess.run(
        ["scp", "-F", "/dev/null", "-i", str(config.SSH_KEY_PATH), "-P", port,
         "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
         local_path, f"root@{host}:{remote_path}"],
        capture_output=True, text=True, timeout=timeout,
    )


def _provision_info() -> dict:
    return json.loads((config.STATE_DIR / "provision_result.json").read_text(encoding="utf-8"))


def deploy_secrets(pod_name: str, *, hf_token: str,
                   youtube_keys_local: str | None = None,
                   cookies_dir_local: str | None = None) -> None:
    """Копирует секреты на под: HF_TOKEN (через launch_role.sh env), keys.txt, cookies/*.txt."""
    remote_fetcher = "/workspace/TrendFlowML/Fetcher"
    if youtube_keys_local:
        ssh_run(pod_name, f"mkdir -p {remote_fetcher}/fetcher/dataset_collector/keys")
        scp_to_pod(pod_name, youtube_keys_local, f"{remote_fetcher}/fetcher/dataset_collector/keys/keys.txt")
    if cookies_dir_local:
        import os
        ssh_run(pod_name, f"mkdir -p {remote_fetcher}/fetcher/dataset_collector/cookies")
        for fn in os.listdir(cookies_dir_local):
            if fn.endswith(".txt"):
                scp_to_pod(pod_name, os.path.join(cookies_dir_local, fn),
                          f"{remote_fetcher}/fetcher/dataset_collector/cookies/{fn}")
    scp_to_pod(pod_name, str(config.FETCHER_DIR / "launch_role.sh"), "/workspace/launch_role.sh")


def kill_processes(pod_name: str) -> str:
    """Убить текущие discover/workers процессы на поде (идемпотентность launch() + перед restart).

    ВАЖНО (баг найден и исправлен 2026-07-16): `pkill -f pattern` матчит по ПОЛНОЙ командной строке
    процесса, включая командную строку САМОГО pkill/shell, который его вызывает — если паттерн
    буквально встречается где-то в переданной по SSH команде (например, в этой же строке или в
    последующей проверке через grep), pkill убивает СЕБЯ/родительский shell сигналом, что рвёт SSH-
    сессию (`Exit status -1`, код возврата 255) ДО того, как реально убьёт целевые процессы. Фикс —
    классический bracket-trick: `[d]ataset_collector` матчит "dataset_collector" в ЧУЖИХ процессах,
    но не совпадает буквально со своей же командной строкой `pkill -f [d]ataset_collector`. НЕ убирай
    скобки при правке этой функции — это не опечатка.

    ВАЖНО #2 (баг найден 2026-07-16): раньше сразу слали SIGKILL (-9). download-воркер (yt-dlp/
    pytubefix) дозаписывает cookie-файл ПОСЛЕ каждого запроса НЕ атомарно — SIGKILL посреди этой
    записи оставляет cookie-файл обрезанным до 0 байт (реально случилось с bon_cookie.txt на
    fetcher-worker-b, сломало download на несколько часов, пока не заметили и не перезалили файл).
    Демоны (_queue_worker_daemon в run_workers.py) уже ловят SIGTERM и штатно завершаются между
    единицами работы (should_stop()) — даём им на это несколько секунд ПЕРЕД добиванием -9,
    вместо того чтобы убивать сразу самым грубым способом."""
    cmd = ("pkill -f '[f]etcher.dataset_collector' ; "
          "pkill -f '[c]olab_20k_bootstrap' ; "
          "pkill -f '[l]aunch_role.sh' ; "
          "pkill -f 'while tr[u]e' ; "
          "sleep 5; "
          "pkill -9 -f '[f]etcher.dataset_collector' ; "
          "pkill -9 -f '[c]olab_20k_bootstrap' ; "
          "pkill -9 -f '[l]aunch_role.sh' ; "
          "pkill -9 -f 'while tr[u]e' ; "
          "sleep 1; echo killed")
    r = ssh_run(pod_name, cmd, timeout=30)
    return (r.stdout or "") + (r.stderr or "")


def launch(pod_name: str, hf_token: str, *, kill_first: bool = True, wait: bool = False) -> str:
    """Запустить роль на поде. kill_first=True (дефолт) — сначала гасит старые discover/workers
    процессы, чтобы повторный вызов НЕ плодил дубли (баг 2026-07-16: launch без kill_first привёл
    к двум параллельным discover-циклам на одном поде).

    Первый запуск строит venv на Network Volume (pip install тяжёлых пакетов — TikTokApi/playwright/
    и т.д. — может занять несколько минут), поэтому весь launch_role.sh оборачиваем в nohup СНАРУЖИ —
    SSH-вызов возвращается сразу, реальный запуск (venv + процессы) идёт в фоне на поде. Смотри
    прогресс через tail_logs()/venv_ready()."""
    out = []
    if kill_first:
        out.append(kill_processes(pod_name))
    spec = config.PODS[pod_name]
    role = "main" if spec["role"] == "main" else "worker"
    inner = (
        f"HF_TOKEN={hf_token} WORKER_ID={spec['worker_id']} "
        f"WORKER_SHARD_INDEX={spec['worker_shard_index']} WORKER_SHARD_COUNT={spec['worker_shard_count']} "
        f"ROLE={role} bash /workspace/launch_role.sh"
    )
    # setsid — иначе ssh иногда виснет на ожидании закрытия канала, даже с nohup+redirect+disown
    # (известная особенность: без полного отсоединения от сессии/pty ssh не возвращается сразу).
    cmd = (
        f"mkdir -p /workspace/logs && "
        f"setsid nohup bash -c '{inner}' > /workspace/logs/launch_role.log 2>&1 < /dev/null & "
        f"disown; echo launcher_started"
    )
    r = ssh_run(pod_name, cmd, timeout=20)
    out.append((r.stdout or "") + (r.stderr or ""))
    if wait:
        import time as _t
        for _ in range(60):  # до ~10 минут (60×10с)
            if venv_ready(pod_name):
                out.append("venv готов")
                break
            _t.sleep(10)
        else:
            out.append("ТАЙМАУТ ожидания venv (>10 мин) — проверь launch_role.log вручную")
    return "\n".join(out)


def venv_ready(pod_name: str) -> bool:
    r = ssh_run(pod_name, "test -f /workspace/venv/bin/activate && echo yes || echo no", timeout=20)
    return "yes" in (r.stdout or "")


def pull_latest_code(pod_name: str) -> str:
    r = ssh_run(pod_name, "cd /workspace/TrendFlowML && git pull", timeout=60)
    return (r.stdout or "") + (r.stderr or "")


def tail_logs(pod_name: str, n: int = 100) -> str:
    cmd = f"tail -n {n} /workspace/logs/discover.log /workspace/logs/workers_*.log 2>/dev/null"
    r = ssh_run(pod_name, cmd, timeout=30)
    return (r.stdout or "") + (r.stderr or "")


def read_inventory_summary(pod_name: str) -> dict | None:
    cmd = "cat /workspace/dataset_runs/100k-monthly/state/inventory/summary.json 2>/dev/null"
    r = ssh_run(pod_name, cmd, timeout=30)
    try:
        return json.loads(r.stdout)
    except Exception:
        return None


def read_snapshot_status(pod_name: str) -> dict | None:
    """Разовый, побочно-безопасный (не запускает сбор, только читает state) статус снапшотов —
    см. Fetcher/scripts/snapshot_status.py. Добавлено 2026-07-24 (владелец: hourly_report.py
    должен показывать прогресс по снапшотам)."""
    cmd = (
        "cd /workspace/TrendFlowML/Fetcher && /workspace/venv/bin/python3 scripts/snapshot_status.py "
        "/workspace/dataset_runs/100k-monthly/runtime_dataset_campaign_20k.json 2>/dev/null"
    )
    r = ssh_run(pod_name, cmd, timeout=30)
    try:
        return json.loads(r.stdout)
    except Exception:
        return None
