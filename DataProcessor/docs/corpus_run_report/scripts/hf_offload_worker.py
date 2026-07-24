#!/usr/bin/env python3
"""HF-offload воркер для больших прогонов (10k): выгружает завершённые видео в HF dataset и удаляет
локально, чтобы не переполнить том. Резюмируемо, идемпотентно, с backpressure по диску.

Запуск (на поде, рядом с батчем):
  export $(grep -E '^HF_TOKEN=' automation/fetcher/.env)
  <venv>/bin/python hf_offload_worker.py \
      --root /workspace/corpus_smoke --repo <hf_user>/<dataset> \
      --disk-high-gb 90 --disk-low-gb 60 --keep-manifest

Логика:
- Видео «готово к выгрузке» = есть metrics.jsonl с '_summary' (пайплайн завершён).
- Выгружает <video>/ (rs/ + metrics + логи) в HF под path_in_repo=corpus/<video_id>/.
- После успешной выгрузки — помечает .offloaded и (если диск высокий или --always-delete) удаляет rs/.
- Backpressure: если использование тома > disk-high-gb — удаляет уже выгруженные локально до disk-low-gb.
- Идемпотентно: пропускает уже .offloaded; повторный upload_folder перезапишет (delete_patterns не трогаем).
"""
from __future__ import annotations
import argparse, os, time, json, shutil, glob, sys

def log(m): print("%s %s" % (time.strftime("%H:%M:%S"), m), flush=True)

def disk_used_gb(path):
    try:
        st = os.statvfs(path); return (st.f_blocks - st.f_bfree) * st.f_frsize / 1e9
    except Exception:
        return 0.0

def has_summary(vdir):
    m = os.path.join(vdir, "metrics.jsonl")
    try:
        return os.path.exists(m) and "_summary" in open(m).read()
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="corpus_smoke root")
    ap.add_argument("--repo", required=True, help="HF dataset repo id (user/name)")
    ap.add_argument("--path-prefix", default="corpus", help="path_in_repo prefix")
    ap.add_argument("--disk-high-gb", type=float, default=90.0, help="выше — удалять выгруженное")
    ap.add_argument("--disk-low-gb", type=float, default=60.0, help="удалять до этого уровня")
    ap.add_argument("--always-delete", action="store_true", help="удалять локально сразу после выгрузки")
    ap.add_argument("--poll-sec", type=float, default=30.0)
    ap.add_argument("--stop-file", default=None, help="если файл существует — корректно выйти")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN", "")
    api = HfApi(token=token)
    try:
        api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    except Exception as e:
        log("create_repo warn: %s" % str(e)[:80])

    log("offload start root=%s repo=%s disk_high=%.0f low=%.0f" % (args.root, args.repo, args.disk_high_gb, args.disk_low_gb))
    offloaded_local = []  # выгруженные, ещё на диске (кандидаты на удаление при backpressure)
    while True:
        if args.stop_file and os.path.exists(args.stop_file):
            log("stop-file -> exit"); break
        vids = [d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d)]
        for vdir in vids:
            vid = os.path.basename(vdir)
            if os.path.exists(os.path.join(vdir, ".offloaded")):
                if vdir not in offloaded_local and os.path.isdir(os.path.join(vdir, "rs")):
                    offloaded_local.append(vdir)
                continue
            if not has_summary(vdir):
                continue
            try:
                api.upload_folder(
                    folder_path=vdir, repo_id=args.repo, repo_type="dataset",
                    path_in_repo="%s/%s" % (args.path_prefix, vid),
                    ignore_patterns=["**/.offloaded", "seg/**", "**/*.npy"],  # кадры не выгружаем
                )
                open(os.path.join(vdir, ".offloaded"), "w").write(str(time.time()))
                offloaded_local.append(vdir)
                log("uploaded %s" % vid)
                if args.always_delete:
                    shutil.rmtree(os.path.join(vdir, "rs"), ignore_errors=True)
            except Exception as e:
                log("upload FAIL %s: %s" % (vid, str(e)[:100]))
                time.sleep(5)
        # backpressure по диску
        used = disk_used_gb(args.root)
        if used > args.disk_high_gb:
            log("disk %.0fG > high %.0fG -> freeing uploaded" % (used, args.disk_high_gb))
            for vdir in list(offloaded_local):
                if disk_used_gb(args.root) <= args.disk_low_gb:
                    break
                rs = os.path.join(vdir, "rs")
                if os.path.isdir(rs):
                    shutil.rmtree(rs, ignore_errors=True)
                    offloaded_local.remove(vdir)
                    log("freed %s" % os.path.basename(vdir))
        time.sleep(args.poll_sec)

if __name__ == "__main__":
    main()
