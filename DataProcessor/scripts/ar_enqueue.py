#!/usr/bin/env python3
"""
Постановка заявки на прогон в очередь демона (этим пользуется Claude, но можно и вручную).
Пишет `automation/queue/<id>.json`. Демон (ar_run_daemon.py) её подхватит и исполнит на GPU.

Пример:
  python DataProcessor/scripts/ar_enqueue.py \
    --video DataProcessor/docs/component_reports/action_recognition/fixtures/ar_real_4m35_people.mp4 \
    --video-id ar_real_4m35_people --seconds 0 --fps 25 --device cuda
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

DP = Path(__file__).resolve().parents[1]
ROOT = DP.parent
Q = ROOT / "automation" / "queue"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--seconds", type=int, default=0, help="0 = полный клип (GPU)")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--id", default=None)
    args = ap.parse_args()

    Q.mkdir(parents=True, exist_ok=True)
    rid = args.id or f"{args.video_id or Path(args.video).stem}_{int(time.time())}"
    req = {
        "id": rid,
        "video": str(args.video),
        "video_id": args.video_id or Path(args.video).stem,
        "seconds": int(args.seconds),
        "fps": float(args.fps),
        "device": args.device,
    }
    if args.width:
        req["width"] = int(args.width)
    out = Q / f"{rid}.json"
    out.write_text(json.dumps(req, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"enqueued: {out}")
    print(f"результаты появятся в: automation/results/{rid}/ (маркер DONE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
