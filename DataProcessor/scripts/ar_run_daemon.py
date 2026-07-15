#!/usr/bin/env python3
"""
Локальный демон-воркер прогонов на ТВОЁМ ПК (с GPU). Запускается ОДИН РАЗ тобой; дальше Claude
кладёт заявки в `automation/queue/*.json`, демон исполняет их на GPU и пишет результаты в
`automation/results/<id>/` + маркер `DONE`. Claude читает результаты (общая workspace-папка) и
сам пишет REPORT. Cursor не нужен.

БЕЗОПАСНОСТЬ: демон НЕ исполняет произвольные команды. Он принимает только фиксированную заявку
(видео + параметры) и запускает единственный доверенный скрипт `run_ar_local.py`. Видео должно
лежать внутри репозитория (иначе заявка отклоняется).

Запуск (один раз):
  DataProcessor/VisualProcessor/.vp_venv/bin/python DataProcessor/scripts/ar_run_daemon.py
  # или в фоне: nohup ... &   / systemd (см. automation/README.md)
"""
from __future__ import annotations
import json, os, subprocess, sys, time, shutil
from pathlib import Path

DP = Path(__file__).resolve().parents[1]
ROOT = DP.parent
AUTO = ROOT / "automation"
Q = AUTO / "queue"; RES = AUTO / "results"; PROC = AUTO / "processed"; LOGS = AUTO / "logs"
RUNNER = DP / "scripts" / "run_ar_local.py"
POLL_S = float(os.environ.get("AR_DAEMON_POLL_S", "5"))


def _log(msg: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOGS / "daemon.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _detect_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch  # noqa
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    _log("GPU недоступен → fallback device=cpu (медленно; только smoke)")
    return "cpu"


def _safe_video(path_str: str) -> Path | None:
    """Разрешаем только видео внутри репозитория (защита от произвольных путей)."""
    try:
        p = Path(path_str).resolve()
        p.relative_to(ROOT.resolve())  # бросит ValueError, если вне репо
        return p if p.is_file() else None
    except Exception:
        return None


def process(req_path: Path) -> None:
    try:
        req = json.loads(req_path.read_text(encoding="utf-8"))
    except Exception as e:
        _log(f"битая заявка {req_path.name}: {e}"); req_path.rename(PROC / req_path.name); return
    rid = str(req.get("id") or req_path.stem)
    video = _safe_video(str(req.get("video", "")))
    if video is None:
        _log(f"[{rid}] отклонено: video вне репозитория / не найдено ({req.get('video')})")
        (RES / rid).mkdir(parents=True, exist_ok=True)
        (RES / rid / "DONE").write_text(json.dumps({"rc": 90, "error": "invalid video path"}))
        req_path.rename(PROC / req_path.name); return

    workdir = RES / rid; workdir.mkdir(parents=True, exist_ok=True)
    device = _detect_device(str(req.get("device", "cuda")))
    cmd = [
        sys.executable, str(RUNNER),
        "--video", str(video),
        "--video-id", str(req.get("video_id", rid)),
        "--seconds", str(int(req.get("seconds", 0))),
        "--fps", str(req.get("fps", 25)),
        "--device", device,
        "--workdir", str(workdir),
    ]
    if req.get("width"):
        cmd += ["--width", str(int(req["width"]))]
    _log(f"[{rid}] старт device={device} video={video.name}")
    t = time.time()
    rc = 1
    try:
        with open(workdir / "daemon_run.log", "w", encoding="utf-8") as lg:
            rc = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=str(DP)).returncode
    except Exception as e:
        _log(f"[{rid}] исключение: {e}")
    dur = round(time.time() - t, 1)
    (workdir / "DONE").write_text(json.dumps({"rc": rc, "seconds": dur, "device": device}, ensure_ascii=False))
    _log(f"[{rid}] готово rc={rc} за {dur}s → {workdir}")
    req_path.rename(PROC / req_path.name)


def main() -> int:
    for d in (Q, RES, PROC, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    _log(f"daemon старт. queue={Q} poll={POLL_S}s runner={RUNNER}")
    if not RUNNER.exists():
        _log(f"ОШИБКА: нет {RUNNER}"); return 2
    while True:
        try:
            reqs = sorted(Q.glob("*.json"))
            for r in reqs:
                process(r)
        except KeyboardInterrupt:
            _log("остановка (Ctrl-C)"); return 0
        except Exception as e:
            _log(f"loop error: {e}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
