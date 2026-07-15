#!/usr/bin/env python3
"""
Inprocess-обход core_depth_midas (который Triton-only) для прогона shot_quality без Triton.
Считает depth через torch.hub MiDaS (публичный) на тех же frame_indices, что у shot_quality,
и пишет `<rs>/core_depth_midas/depth.npz` в формате, который читает shot_quality:
  frame_indices (N,), times_s (N,), depth_maps (N, out_h, out_w) float32, + meta.

frame_indices берутся из metadata[shot_quality].frame_indices (aligned sampling group).
Использование:
  python midas_depth_inprocess.py --frames-dir <fd> --rs-path <rs> [--out-hw 128 128] [--device cuda]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

DP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DP / "VisualProcessor"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--rs-path", required=True)
    ap.add_argument("--out-h", type=int, default=128)
    ap.add_argument("--out-w", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="MiDaS_small")
    a = ap.parse_args()
    import torch, cv2
    from utils.frame_manager import FrameManager

    meta = json.load(open(os.path.join(a.frames_dir, "metadata.json"), encoding="utf-8"))
    fi = meta.get("shot_quality", {}).get("frame_indices") or meta.get("core_depth_midas", {}).get("frame_indices")
    if not fi:
        print("нет shot_quality/core_depth_midas frame_indices в metadata", file=sys.stderr); return 2
    fi = [int(x) for x in fi]
    uts = meta.get("union_timestamps_sec") or []
    times_s = np.asarray([float(uts[i]) if i < len(uts) else 0.0 for i in fi], dtype=np.float32)

    fm = FrameManager(a.frames_dir, meta)
    midas = torch.hub.load("intel-isl/MiDaS", a.model, trust_repo=True).to(a.device).eval()
    tfm = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = tfm.small_transform if "small" in a.model.lower() else tfm.dpt_transform

    N = len(fi)
    depth_maps = np.full((N, a.out_h, a.out_w), np.nan, dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for k, idx in enumerate(fi):
            frame = fm.get(int(idx))  # RGB uint8
            inp = transform(frame).to(a.device)
            pred = midas(inp)
            pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(a.out_h, a.out_w),
                                                   mode="bicubic", align_corners=False).squeeze()
            depth_maps[k] = pred.detach().cpu().numpy().astype(np.float32)
    dt = time.time() - t0

    out_dir = Path(a.rs_path) / "core_depth_midas"; out_dir.mkdir(parents=True, exist_ok=True)
    meta_info = {
        "producer": "core_depth_midas", "producer_version": "inprocess_midas_bypass",
        "schema_version": "core_depth_midas_npz_v2", "status": "ok", "empty_reason": None,
        "impl": f"torch.hub:{a.model}", "device": a.device,
        "platform_id": meta.get("platform_id"), "video_id": meta.get("video_id"),
        "run_id": meta.get("run_id"), "sampling_policy_version": meta.get("sampling_policy_version"),
        "config_hash": meta.get("config_hash"), "dataprocessor_version": meta.get("dataprocessor_version") or "inprocess",
    }
    np.savez_compressed(
        out_dir / "depth.npz",
        frame_indices=np.asarray(fi, dtype=np.int32), times_s=times_s,
        depth_maps=depth_maps,
        meta=np.asarray(meta_info, dtype=object),
        meta_json=np.array(json.dumps(meta_info, ensure_ascii=False), dtype="U"),
    )
    print(json.dumps({"ok": True, "N": N, "depth_shape": [N, a.out_h, a.out_w], "seconds": round(dt, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
