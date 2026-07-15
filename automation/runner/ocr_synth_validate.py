#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Изолированный валидатор ocr_extractor на синтетической фикстуре.

Настоящей 41-классовой YOLO с классом `text_region` в репо нет (симлинк-заглушка на COCO
yolo11l; обученный детектор — зона владельца, YOLO-датасет). Поэтому OCR валидируем В ИЗОЛЯЦИИ:
- синтезируем frames_dir (FrameManager-формат: .npy чанки + metadata.json) с известным текстом;
- синтезируем core_object_detections/detections.npz с боксами класса text_region;
- гоняем НАСТОЯЩИЙ ocr_extractor/main.py (subprocess, engine=ppocr_rec_onnx);
- считаем метрики под критерии U1-U6 / C1-C4.

Запуск на поде:
  DP_MODELS_ROOT=/workspace/TrendFlowML/DataProcessor/dp_models \
  /workspace/venv/bin/python ocr_synth_validate.py --workdir /workspace/ocr_synth
"""
from __future__ import annotations
import argparse, os, sys, json, time, subprocess, hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DP = None  # set in main
TEXT_REGION_ID = 34  # произвольный id класса в синтетической таксономии
MAXB = 5             # max боксов на кадр в detections
W, H = 480, 270
FPS = 6.0

def _font(size=30):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

FONT = None

def _render_frame(lines):
    """lines: list of (text, x, y). Возвращает (rgb uint8 HxWx3, [bbox...])."""
    img = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(img)
    boxes = []
    for text, x, y in lines:
        bb = d.textbbox((x, y), text, font=FONT)
        d.text((x, y), text, fill=(10, 10, 10), font=FONT)
        # чуть расширим бокс
        boxes.append([float(bb[0] - 3), float(bb[1] - 3), float(bb[2] + 3), float(bb[3] + 3)])
    return np.asarray(img, dtype=np.uint8), boxes


def build_video(root: Path, video_id: str, n_frames: int, text_map: dict,
                has_text_region_class: bool = True):
    """text_map: {local_frame_idx: [(text,x,y),...]}. Строит frames_dir + detections.npz.
    Возвращает dict с ожидаемыми данными."""
    vdir = root / video_id
    fdir = vdir / "frames"
    rs = vdir / "rs"
    fdir.mkdir(parents=True, exist_ok=True)
    (rs / "core_object_detections").mkdir(parents=True, exist_ok=True)

    frames = np.zeros((n_frames, H, W, 3), dtype=np.uint8)
    per_frame_boxes = []  # list of list[bbox]
    expected_texts = []   # list of list[str]
    for i in range(n_frames):
        lines = text_map.get(i, [])
        rgb, boxes = _render_frame(lines)
        frames[i] = rgb
        per_frame_boxes.append(boxes)
        expected_texts.append([t for (t, _, _) in lines])

    # frames как один .npy чанк
    chunk_size = 32
    n_batches = (n_frames + chunk_size - 1) // chunk_size
    batches = []
    for b in range(n_batches):
        s, e = b * chunk_size, min((b + 1) * chunk_size, n_frames)
        np.save(str(fdir / f"chunk_{b}.npy"), frames[s:e])
        batches.append({"batch_index": b, "path": f"chunk_{b}.npy", "num_frames": int(e - s)})

    frame_indices = list(range(n_frames))
    uts = [round(i / FPS, 6) for i in range(n_frames)]
    meta = {
        "platform_id": "synth", "video_id": video_id, "run_id": f"{video_id}_run",
        "config_hash": "synthhash", "sampling_policy_version": "synth_v1",
        "dataprocessor_version": "synth_dp",
        "total_frames": n_frames, "chunk_size": chunk_size, "batch_size": chunk_size,
        "height": H, "width": W, "channels": 3, "fps": FPS, "color_space": "RGB",
        "batches": batches,
        "union_timestamps_sec": uts,
        "core_object_detections": {"frame_indices": frame_indices},
    }
    (fdir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False))

    # detections.npz
    N = n_frames
    boxes_arr = np.zeros((N, MAXB, 4), dtype=np.float32)
    scores = np.zeros((N, MAXB), dtype=np.float32)
    class_ids = np.zeros((N, MAXB), dtype=np.int32)
    valid = np.zeros((N, MAXB), dtype=bool)
    for i in range(N):
        for j, bb in enumerate(per_frame_boxes[i][:MAXB]):
            boxes_arr[i, j] = bb
            scores[i, j] = 0.95
            class_ids[i, j] = TEXT_REGION_ID
            valid[i, j] = True
    if has_text_region_class:
        class_names = np.asarray([f"{TEXT_REGION_ID}:text_region", "0:person"], dtype=object)
    else:
        class_names = np.asarray(["0:person", "2:car"], dtype=object)
    np.savez(str(rs / "core_object_detections" / "detections.npz"),
             frame_indices=np.asarray(frame_indices, dtype=np.int32),
             boxes=boxes_arr, scores=scores, class_ids=class_ids,
             valid_mask=valid, class_names=class_names)
    return {"video_id": video_id, "frames_dir": str(fdir), "rs_path": str(rs),
            "frame_indices": frame_indices, "expected_texts": expected_texts}


def run_ocr(info, retain=False, out_tag=""):
    """Запускает настоящий ocr_extractor/main.py в subprocess."""
    rs = Path(info["rs_path"])
    if out_tag:
        rs = rs.parent / f"rs_{out_tag}"
        # копируем detections
        (rs / "core_object_detections").mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(Path(info["rs_path"]) / "core_object_detections" / "detections.npz",
                    rs / "core_object_detections" / "detections.npz")
    main_py = DP / "VisualProcessor/core/model_process/ocr_extractor/main.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{DP/'VisualProcessor'}:{DP}:" + env.get("PYTHONPATH", "")
    env["DP_MODELS_ROOT"] = str(DP / "dp_models")
    cmd = [sys.executable, str(main_py), "--frames-dir", info["frames_dir"],
           "--rs-path", str(rs), "--engine", "ppocr_rec_onnx",
           "--rec-model-spec", "ppocr_rec_onnx_v1_inprocess", "--min-det-score", "0.5"]
    if retain:
        cmd.append("--retain-raw-ocr-text")
    t = time.time()
    p = subprocess.run(cmd, env=env, cwd=str(DP), capture_output=True, text=True, timeout=1200)
    dt = round(time.time() - t, 1)
    out_npz = rs / "ocr_extractor" / "ocr.npz"
    return {"rc": p.returncode, "npz": str(out_npz), "sec": dt,
            "stderr_tail": p.stderr[-800:] if p.returncode != 0 else ""}


def load_ocr(npz):
    d = np.load(npz, allow_pickle=True)
    meta = d["meta"].reshape(-1)[0]
    rows = list(d["ocr_raw"].tolist())
    return {"frame_indices": np.asarray(d["frame_indices"]).astype(int),
            "times_s": np.asarray(d["times_s"], dtype=np.float64),
            "rows": rows, "meta": meta,
            "status": meta.get("status"), "empty_reason": meta.get("empty_reason"),
            "retain": meta.get("retain_raw_ocr_text")}


def golden_key(rows):
    """Ключ для сравнения golden: сортированный набор (frame, bbox_rounded, sha)."""
    ks = []
    for r in rows:
        sha = r.get("text_sha256")
        if sha is None and r.get("text_norm") is not None:
            sha = hashlib.sha256(str(r["text_norm"]).encode()).hexdigest()
        bb = tuple(round(float(x), 2) for x in r.get("bbox", []))
        ks.append((int(r["frame"]), bb, sha))
    return sorted(ks)


def rec_confs(rows):
    return [float(r["rec_confidence"]) for r in rows if r.get("rec_confidence") is not None]


def main():
    global DP, FONT
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/workspace/ocr_synth")
    ap.add_argument("--dp", default="/workspace/TrendFlowML/DataProcessor")
    a = ap.parse_args()
    DP = Path(a.dp)
    FONT = _font(30)
    root = Path(a.workdir)
    root.mkdir(parents=True, exist_ok=True)

    report = {"builds": {}, "runs": {}, "criteria": {}}

    # --- Видео ---
    A = build_video(root, "vidA", 12, {
        2: [("TRENDFLOW", 40, 40)], 5: [("HELLO2026", 40, 120)],
        8: [("OCRTEST", 40, 40), ("LINE99", 40, 150)]})
    B = build_video(root, "vidB", 8, {
        1: [("DIFFERENT", 30, 60)], 4: [("WORLD777", 30, 130)]})
    Cblank = build_video(root, "vidCblank", 6, {})  # боксы text_region есть, но текста нет
    # для Cblank сделаем валидные боксы поверх пустого фона (нет текста)
    # переопределим detections: добавим бокс в центр каждого кадра
    _inject_blank_boxes(Cblank)
    Dnobox = build_video(root, "vidDnobox", 6, {
        1: [("SOMETEXT", 40, 40)]}, has_text_region_class=False)  # нет класса text_region
    E = build_video(root, "vidElong", 200, {
        10: [("LONGVID", 40, 40)], 100: [("FRAME100", 40, 120)], 190: [("ENDX", 40, 40)]})
    F = build_video(root, "vidFshort", 3, {1: [("SHORT3", 40, 60)]})

    # --- Прогоны ---
    R = {}
    R["A_r0"] = run_ocr(A, retain=False)
    R["A_r0b"] = run_ocr(A, retain=False, out_tag="golden2")  # golden повтор
    R["A_r1"] = run_ocr(A, retain=True, out_tag="retain")     # privacy retain=true
    R["B"] = run_ocr(B, retain=False)
    R["Cblank"] = run_ocr(Cblank, retain=False)
    R["Dnobox"] = run_ocr(Dnobox, retain=False)
    R["E"] = run_ocr(E, retain=False)
    R["F"] = run_ocr(F, retain=False)
    report["runs"] = R

    # --- Загрузка результатов ---
    def L(tag):
        if R[tag]["rc"] != 0:
            return None
        try:
            return load_ocr(R[tag]["npz"])
        except Exception as e:
            return {"load_error": str(e)}
    dA0, dA0b, dA1 = L("A_r0"), L("A_r0b"), L("A_r1")
    dB, dC, dD, dE, dF = L("B"), L("Cblank"), L("Dnobox"), L("E"), L("F")

    crit = report["criteria"]

    # U1 — валидатор rc=0
    val = DP / "VisualProcessor/core/model_process/ocr_extractor/utils/validate_ocr.py"
    env = os.environ.copy(); env["PYTHONPATH"] = f"{DP}:" + env.get("PYTHONPATH", "")
    vres = {}
    for tag in ["A_r0", "B", "Cblank", "Dnobox", "E", "F"]:
        if R[tag]["rc"] == 0:
            pv = subprocess.run([sys.executable, str(val), R[tag]["npz"], "--struct"],
                                env=env, cwd=str(DP), capture_output=True, text=True)
            vres[tag] = {"rc": pv.returncode, "tail": pv.stdout.strip().splitlines()[-3:]}
    crit["U1_validator"] = {"all_main_rc0": all(R[t]["rc"] == 0 for t in R),
                            "validate_ocr": vres}

    # U2 — ось времени: times_s == uts[frame_indices], возрастание, 0% NaN
    u2 = {}
    for tag, d, info in [("A", dA0, A), ("E", dE, E), ("F", dF, F)]:
        if d and "load_error" not in d:
            uts = np.asarray([round(i / FPS, 6) for i in range(len(info["frame_indices"]))])
            fi = d["frame_indices"]
            exp = uts[fi]
            u2[tag] = {"axis_match": bool(np.allclose(d["times_s"], exp, atol=1e-5)),
                       "monotonic": bool(np.all(np.diff(fi) > 0)),
                       "nan_pct": float(np.mean(~np.isfinite(d["times_s"])) * 100)}
    crit["U2_time_axis"] = u2

    # U3 — health: finite, dtype/range
    u3 = {}
    for tag, d in [("A", dA0), ("E", dE)]:
        if d and "load_error" not in d:
            fi = d["frame_indices"]
            u3[tag] = {"times_finite": bool(np.all(np.isfinite(d["times_s"]))),
                       "fi_in_range": bool(np.all((fi >= 0) & (fi < 10**6))),
                       "fi_dtype": str(np.asarray(d["frame_indices"]).dtype)}
    crit["U3_health"] = u3

    # U4 — expected-empty (Cblank: боксы без текста; Dnobox: нет класса text_region)
    crit["U4_empty"] = {
        "Cblank_status": dC.get("status") if dC else None,
        "Cblank_reason": dC.get("empty_reason") if dC else None,
        "Cblank_rows": len(dC["rows"]) if dC else None, "Cblank_rc": R["Cblank"]["rc"],
        "Dnobox_status": dD.get("status") if dD else None,
        "Dnobox_reason": dD.get("empty_reason") if dD else None,
        "Dnobox_rows": len(dD["rows"]) if dD else None, "Dnobox_rc": R["Dnobox"]["rc"]}

    # U5/C4 — golden: A_r0 vs A_r0b
    if dA0 and dA0b and "load_error" not in dA0:
        k0, k0b = golden_key(dA0["rows"]), golden_key(dA0b["rows"])
        c0 = {r_["frame"]: r_.get("rec_confidence") for r_ in []}  # placeholder
        # max|Δ rec_confidence| по совпадающим (frame,bbox)
        def conf_map(rows):
            m = {}
            for r in rows:
                bb = tuple(round(float(x), 2) for x in r.get("bbox", []))
                m[(int(r["frame"]), bb)] = r.get("rec_confidence")
            return m
        m0, m0b = conf_map(dA0["rows"]), conf_map(dA0b["rows"])
        deltas = [abs(float(m0[k]) - float(m0b[k])) for k in m0 if k in m0b and m0[k] is not None and m0b[k] is not None]
        crit["U5_C4_golden"] = {"key_identical": k0 == k0b,
                                "n_rows_r0": len(dA0["rows"]), "n_rows_r0b": len(dA0b["rows"]),
                                "max_abs_dconf": max(deltas) if deltas else None}

    # U6 — разные длины
    crit["U6_lengths"] = {tag: {"rc": R[rk]["rc"], "n": (len(d["rows"]) if d and "load_error" not in d else None),
                                "sec": R[rk]["sec"]}
                          for tag, rk, d in [("F", "F", dF), ("A", "A_r0", dA0), ("E", "E", dE)]}

    # C1 — frame каждой строки ∈ frame_indices (100%)
    c1 = {}
    for tag, d in [("A", dA0), ("B", dB), ("E", dE)]:
        if d and "load_error" not in d and d["rows"]:
            fis = set(int(x) for x in d["frame_indices"])
            inside = sum(1 for r in d["rows"] if int(r["frame"]) in fis)
            c1[tag] = {"rows": len(d["rows"]), "inside": inside,
                       "pct": round(100 * inside / len(d["rows"]), 2)}
    crit["C1_frame_binding"] = c1

    # C2 — privacy
    def has_raw(rows):
        return any(("text_raw" in r or "text_norm" in r) for r in rows)
    def has_sha(rows):
        return all(("text_sha256" in r and "text_len" in r) for r in rows) if rows else True
    crit["C2_privacy"] = {
        "retain_false_has_raw": (has_raw(dA0["rows"]) if dA0 else None),
        "retain_false_all_sha": (has_sha(dA0["rows"]) if dA0 else None),
        "retain_false_meta": (dA0["retain"] if dA0 else None),
        "retain_true_has_raw": (has_raw(dA1["rows"]) if dA1 else None),
        "retain_true_meta": (dA1["retain"] if dA1 else None)}

    # C3 — различимость: R>0 на A,B; rec_conf∈[0,1]; empty на C,D; R варьируется
    allconf = []
    for d in [dA0, dB, dE]:
        if d and "load_error" not in d:
            allconf += rec_confs(d["rows"])
    Rvals = {"A": len(dA0["rows"]) if dA0 else None, "B": len(dB["rows"]) if dB else None,
             "E": len(dE["rows"]) if dE else None,
             "Cblank": len(dC["rows"]) if dC else None, "Dnobox": len(dD["rows"]) if dD else None}
    crit["C3_discrimination"] = {
        "R_per_video": Rvals,
        "R_varies": len(set(v for v in [Rvals["A"], Rvals["B"], Rvals["E"]] if v is not None)) > 1,
        "conf_in_0_1": (all(0.0 <= c <= 1.0 for c in allconf) if allconf else None),
        "conf_min": (min(allconf) if allconf else None),
        "conf_max": (max(allconf) if allconf else None),
        "n_conf": len(allconf)}

    # Пример распознанного текста (retain=true) для sanity
    if dA1 and "load_error" not in dA1:
        crit["_recognized_sample_A_retain"] = [
            {"frame": int(r["frame"]), "text": r.get("text_raw") or r.get("text_norm"),
             "conf": r.get("rec_confidence")} for r in dA1["rows"]]

    out = root / "REPORT.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"criteria": report["criteria"],
                      "runs_rc": {k: v["rc"] for k, v in R.items()},
                      "stderr": {k: v["stderr_tail"] for k, v in R.items() if v["stderr_tail"]}},
                     ensure_ascii=False, indent=2))
    print(f"\nПОЛНЫЙ ОТЧЁТ: {out}")


def _inject_blank_boxes(info):
    """Для Cblank: перезаписываем detections боксом в центр каждого кадра (текста там нет)."""
    rs = Path(info["rs_path"])
    det = rs / "core_object_detections" / "detections.npz"
    d = dict(np.load(det, allow_pickle=True))
    fi = np.asarray(d["frame_indices"]).astype(int)
    N = len(fi)
    boxes = np.zeros((N, MAXB, 4), dtype=np.float32)
    scores = np.zeros((N, MAXB), dtype=np.float32)
    class_ids = np.zeros((N, MAXB), dtype=np.int32)
    valid = np.zeros((N, MAXB), dtype=bool)
    for i in range(N):
        boxes[i, 0] = [180, 110, 300, 160]
        scores[i, 0] = 0.9
        class_ids[i, 0] = TEXT_REGION_ID
        valid[i, 0] = True
    np.savez(str(det), frame_indices=fi.astype(np.int32), boxes=boxes, scores=scores,
             class_ids=class_ids, valid_mask=valid,
             class_names=np.asarray([f"{TEXT_REGION_ID}:text_region", "0:person"], dtype=object))


if __name__ == "__main__":
    main()
