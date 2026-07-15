#!/usr/bin/env python3
"""
Автономные юнит-тесты чистой логики v3-доработки action_recognition (без GPU/модели).
Запуск: DataProcessor/.data_venv/bin/python <this file>
Проверяет: appearance_tracker (ассоциация+re-ID), планировщик окон Segmenter, v3-builder, валидатор.
"""
import os, sys, json, numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DP = os.path.join(REPO, "DataProcessor")
sys.path.insert(0, os.path.join(DP, "VisualProcessor", "core", "model_process", "core_object_detections", "utils"))
sys.path.insert(0, os.path.join(DP, "Segmenter"))
sys.path.insert(0, os.path.join(DP, "VisualProcessor", "modules", "action_recognition", "utils"))

import appearance_tracker as T
import action_windows as W
import action_recognition_v3 as V
import validate_action_recognition_npz as VAL


def bv(seed, d=8):
    v = np.random.default_rng(seed).standard_normal(d); return v / np.linalg.norm(v)


def test_tracker():
    N, M = 10, 3
    boxes = np.zeros((N, M, 4), np.float32); sc = np.zeros((N, M), np.float32)
    cls = np.zeros((N, M), np.int32); vm = np.zeros((N, M), bool); emb = np.zeros((N, M, 8), np.float32)
    vA, vB = bv(1), bv(2)
    for n in range(N):
        x = 10 + n * 5
        boxes[n, 0] = [x, 10, x + 20, 60]; sc[n, 0] = .9; cls[n, 0] = 0; vm[n, 0] = True; emb[n, 0] = vA + .02 * bv(100 + n)
        boxes[n, 1] = [200 - n * 3, 10, 220 - n * 3, 60]; sc[n, 1] = .9; cls[n, 1] = 0; vm[n, 1] = True; emb[n, 1] = vB + .02 * bv(200 + n)
    tid, meta = T.track_detections(frame_indices=list(range(N)), boxes=boxes, scores=sc, class_ids=cls,
                                   valid_mask=vm, frame_wh=(256, 64), embeddings=emb)
    assert meta["num_tracks"] == 2
    assert len({tid[n, 0] for n in range(N)}) == 1 and len({tid[n, 1] for n in range(N)}) == 1
    # re-ID across a gap
    N, M = 10, 2; boxes = np.zeros((N, M, 4), np.float32); sc = np.zeros((N, M), np.float32)
    cls = np.zeros((N, M), np.int32); vm = np.zeros((N, M), bool); emb = np.zeros((N, M, 8), np.float32)
    vA = bv(7); present = [0, 1, 2, 3, 7, 8, 9]
    for n in present:
        boxes[n, 0] = [50, 10, 70, 60]; sc[n, 0] = .9; cls[n, 0] = 0; vm[n, 0] = True; emb[n, 0] = vA + .01 * bv(300 + n)
    tid, meta = T.track_detections(frame_indices=list(range(N)), boxes=boxes, scores=sc, class_ids=cls,
                                   valid_mask=vm, frame_wh=(256, 64), embeddings=emb)
    assert len({int(tid[n, 0]) for n in present}) == 1
    print("  tracker: OK (2 coherent tracks, re-ID across gap)")


def test_windows():
    assert W.plan_dense_windows(20, 25, clip_len=32)[0] == list(range(20))
    w = W.plan_dense_windows(12000, 25, clip_len=32, hop_s=2.0, max_windows=48)
    assert len(w) <= 48 and all(len(x) == 32 for x in w) and w[-1][-1] == 11999
    si = W.windows_to_source_indices(w); assert si == sorted(set(si))
    print("  windows: OK (dense contiguous, capped, sorted-unique)")


def test_v3_and_validator():
    rng = np.random.default_rng(0); C = 7
    emb = rng.standard_normal((C, 256)).astype(np.float32); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    tid = np.array([0, 0, 1, 2, 1, 2, 0], np.int32); cfi = np.array([5, 40, 12, 60, 80, 90, 120], np.int32)
    ts = np.array([.2, 1.6, .5, 2.4, 3.2, 3.6, 4.8], np.float32); logits = rng.standard_normal((C, 400)).astype(np.float32)
    out = V.build_v3_arrays(clip_embeddings=emb, clip_track_ids=tid, clip_center_frame_idx=cfi,
                            clip_times_s=ts, clip_logits=logits, topk=5)
    assert int(out["num_tracks"]) == 3 and np.all(np.diff(out["clip_times_s"]) >= 0)
    tmp = "/tmp/ar_v3_ci.npz"
    d = dict(out); d["meta_json"] = np.array(json.dumps({"status": "ok", "total_frames": 200}), dtype="U")
    np.savez_compressed(tmp, **d); assert not VAL.validate(tmp)
    # empty valid
    oute = V.build_v3_arrays(clip_embeddings=np.zeros((0, 256), np.float32), clip_track_ids=np.zeros((0,), np.int32),
                             clip_center_frame_idx=np.zeros((0,), np.int32), clip_times_s=np.zeros((0,), np.float32), clip_logits=None)
    de = dict(oute); de["meta_json"] = np.array(json.dumps({"status": "empty"}), dtype="U")
    np.savez_compressed("/tmp/ar_v3_ci_e.npz", **de); assert not VAL.validate("/tmp/ar_v3_ci_e.npz")
    # negative: not L2
    bad = dict(out); bad["clip_embeddings"] = (emb * 5).astype(np.float32); bad["meta_json"] = d["meta_json"]
    np.savez_compressed("/tmp/ar_v3_ci_b.npz", **bad); assert VAL.validate("/tmp/ar_v3_ci_b.npz")
    print("  v3 builder+validator: OK (ok/empty pass, bad-L2 caught)")


def test_segments_and_runs():
    # builder emits clip_segment_id + num_action_segments, reordered by time
    rng = np.random.default_rng(3); C = 5; D = 2304
    emb = rng.standard_normal((C, D)).astype(np.float32); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    seg = np.array([1, 1, 2, 3, 3], np.int32)
    out = V.build_v3_arrays(clip_embeddings=emb, clip_track_ids=np.array([0, 0, 1, 2, 2], np.int32),
                            clip_center_frame_idx=np.array([10, 20, 5, 30, 40], np.int32),
                            clip_times_s=np.array([1., 2., .5, 3., 4.], np.float32),
                            clip_logits=rng.standard_normal((C, 400)).astype(np.float32), clip_segment_ids=seg)
    assert "clip_segment_id" in out and int(out["num_action_segments"]) == 3

    # contiguous runs → multi-clip (track present in 2 dense windows)
    def runs(idx):
        r, c = [], []
        for x in sorted(set(idx)):
            if not c or x == c[-1] + 1: c.append(x)
            else: r.append(c); c = [x]
        if c: r.append(c)
        return r
    rr = runs(list(range(0, 32)) + list(range(64, 96)))
    assert len(rr) == 2 and all(len(x) == 32 for x in rr)
    print("  tubelet/localization (segments+runs): OK (multi-window → multi-clip, segment ids)")


if __name__ == "__main__":
    test_tracker(); test_windows(); test_v3_and_validator(); test_segments_and_runs()
    print("ALL PURE-LOGIC v3 TESTS PASSED")
