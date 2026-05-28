#!/usr/bin/env python3
"""
content_domain (v1 semantic head)

CLIP text-retrieval over core_clip frame embeddings to classify content domain:
game/anime/cartoon/live_action/screen_recording/etc.

Constraints:
- no-network (offline db + ModelManager (clip_text_triton))
- sampling group = core_clip.frame_indices (Segmenter-owned)
- top-K is never gated; thresholds only produce is_confident flags
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_vp_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
# Must be first: local subdirectory "utils/" would shadow VisualProcessor/utils (e.g. utils.logger).
if _vp_root not in sys.path:
    sys.path.insert(0, _vp_root)
elif sys.path[0] != _vp_root:
    try:
        sys.path.remove(_vp_root)
    except ValueError:
        pass
    sys.path.insert(0, _vp_root)

from utils.logger import get_logger  # type: ignore  # noqa: E402
from utils.utilites import load_metadata  # type: ignore  # noqa: E402
from utils.meta_builder import apply_models_meta  # type: ignore  # noqa: E402

NAME = "content_domain"
VERSION = "0.2"
SCHEMA_VERSION = "content_domain_npz_v2"
ARTIFACT_FILENAME = "content_domain.npz"
LOGGER = get_logger(NAME)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _atomic_save_npz(out_path: str, **kwargs: Any) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = out_path + ".tmp.npz"
    np.savez_compressed(tmp_path, **kwargs)
    os.replace(tmp_path, out_path)


def _is_triton_or_embedding_failure(exc: BaseException) -> bool:
    msg = str(exc).lower()
    ec = str(getattr(exc, "error_code", "") or "").lower()
    cls = type(exc).__name__.lower()
    return (
        "triton" in cls
        or "tritonerror" in cls
        or "triton" in msg
        or "triton_unavailable" in ec
        or "modelmanager" in cls
        or "connection" in msg
        or "refused" in msg
        or "timed out" in msg
        or "timeout" in msg
        or "errno 111" in msg
        or "http" in msg and ("502" in msg or "503" in msg or "504" in msg)
    )


def _load_npz(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise RuntimeError(f"{NAME} | required artifact not found: {path}")
    z = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {}
    for k in z.files:
        v = z[k]
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            try:
                out[k] = v.item()
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"{NAME} | json must be a dict: {path}")
    return obj


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise RuntimeError(f"{NAME} | jsonl row must be a dict at {path}:{ln}")
            rows.append(obj)
    return rows


def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
    return x / n


def _load_triton_spec_via_model_manager(model_spec_name: str, triton_http_url: Optional[str] = None) -> dict:
    from dp_models import get_global_model_manager  # type: ignore
    from dp_models.errors import ModelManagerError  # type: ignore

    # Try to get from parameter or environment first
    if not triton_http_url:
        triton_http_url = os.environ.get("TRITON_HTTP_URL")
    
    mm = get_global_model_manager()
    try:
        rm = mm.get(model_name=str(model_spec_name))
        rp = rm.spec.runtime_params or {}
        handle = rm.handle or {}
        client = None
        if isinstance(handle, dict):
            client = handle.get("client")
        
        # If client is None or rp doesn't have triton_http_url, try to create from args/env
        if client is None or not rp.get("triton_http_url"):
            if triton_http_url:
                from dp_triton import TritonHttpClient, TritonError
                client = TritonHttpClient(base_url=str(triton_http_url), timeout_sec=60.0)
                if not client.ready():
                    raise TritonError(
                        f"{NAME} | Triton is not ready at {triton_http_url}",
                        error_code="triton_unavailable",
                    )
                # Update runtime_params with triton_http_url and ensure default params are set
                if not isinstance(rp, dict):
                    rp = {}
                rp["triton_http_url"] = str(triton_http_url)
                # Ensure default parameters are set if missing (from clip_text_triton.yaml spec)
                if not rp.get("triton_model_name"):
                    rp["triton_model_name"] = "clip_text"
                if not rp.get("triton_model_version"):
                    rp["triton_model_version"] = "1"
                if not rp.get("triton_input_name"):
                    rp["triton_input_name"] = "INPUT__0"
                if not rp.get("triton_output_name"):
                    rp["triton_output_name"] = "OUTPUT__0"
                if not rp.get("triton_input_datatype"):
                    rp["triton_input_datatype"] = "INT64"
                # Update models_used_entry if available
                models_used_entry = rm.models_used_entry if hasattr(rm, 'models_used_entry') else None
            else:
                raise RuntimeError(f"{NAME} | ModelManager returned empty Triton client handle for: {model_spec_name} and triton_http_url not provided (set --triton-http-url or TRITON_HTTP_URL env var)")
        else:
            # Use triton_http_url from runtime_params if available
            if not triton_http_url and rp.get("triton_http_url"):
                triton_http_url = str(rp.get("triton_http_url"))
            models_used_entry = rm.models_used_entry if hasattr(rm, 'models_used_entry') else None
        
        if client is None:
            raise RuntimeError(f"{NAME} | ModelManager returned empty Triton client handle for: {model_spec_name} and triton_http_url not provided")
        if not isinstance(rp, dict) or not rp:
            raise RuntimeError(f"{NAME} | ModelManager returned empty runtime_params for: {model_spec_name}")
        return {"client": client, "rp": rp, "models_used_entry": models_used_entry}
    except ModelManagerError as e:
        # If ModelManager fails but we have triton_http_url, create client directly with default params
        if triton_http_url:
            # This is expected when spec uses ${TRITON_HTTP_URL} - ModelManager doesn't expand env vars during validation
            # We handle it gracefully with fallback
            LOGGER.debug(f"{NAME} | ModelManager spec validation failed for {model_spec_name}: {e}, using provided triton_http_url with default clip_text parameters")
            from dp_triton import TritonHttpClient, TritonError
            client = TritonHttpClient(base_url=str(triton_http_url), timeout_sec=60.0)
            if not client.ready():
                raise TritonError(
                    f"{NAME} | Triton is not ready at {triton_http_url}",
                    error_code="triton_unavailable",
                )
            # Use default parameters for clip_text model (from spec_catalog/vision/clip_text_triton.yaml)
            rp = {
                "triton_http_url": str(triton_http_url),
                "triton_model_name": "clip_text",
                "triton_model_version": "1",
                "triton_input_name": "INPUT__0",
                "triton_output_name": "OUTPUT__0",
                "triton_input_datatype": "INT64",
            }
            return {"client": client, "rp": rp, "models_used_entry": None}
        raise RuntimeError(f"{NAME} | ModelManager failed for {model_spec_name}: {e} and triton_http_url not provided")


def _triton_infer(*, client, model_name: str, model_version: Optional[str], input_name: str, input_tensor: np.ndarray, output_name: str, datatype: str) -> np.ndarray:
    res = client.infer(
        model_name=str(model_name),
        model_version=str(model_version) if model_version else None,
        input_name=str(input_name),
        input_tensor=input_tensor,
        output_name=str(output_name),
        datatype=str(datatype),
    )
    return np.asarray(res.output)


def _compute_text_label_embeddings_triton(*, prompts_per_label: List[List[str]], txt_mm: dict) -> np.ndarray:
    """
    Returns label embeddings: (A,512) float32 normalized.
    For each label, embed each prompt and average (then normalize).
    """
    import clip  # type: ignore

    rp = txt_mm["rp"]
    client = txt_mm["client"]
    # Extract parameters with defaults (matching clip_text_triton.yaml spec)
    triton_model_name = str(rp.get("triton_model_name") or "clip_text")
    triton_model_version = str(rp.get("triton_model_version") or "1") or None
    triton_input_name = str(rp.get("triton_input_name") or "INPUT__0")
    triton_output_name = str(rp.get("triton_output_name") or "OUTPUT__0")
    triton_datatype = str(rp.get("triton_input_datatype") or "INT64")

    # Flatten prompts to run Triton in batches (avoid 1 request per prompt).
    flat_prompts: List[str] = []
    prompt_slices: List[Tuple[int, int]] = []
    for prompts in prompts_per_label:
        start = len(flat_prompts)
        ps = [str(p) for p in (prompts or []) if str(p).strip()]
        flat_prompts.extend(ps)
        end = len(flat_prompts)
        prompt_slices.append((start, end))

    if not flat_prompts:
        # All labels empty -> return zeros
        return np.zeros((len(prompts_per_label), 512), dtype=np.float32)

    toks = clip.tokenize(flat_prompts)  # (P,77)
    toks_np = toks.detach().cpu().numpy().astype(np.int64)

    # Triton batch inference
    bs = 64
    embs: List[np.ndarray] = []
    for i in range(0, int(toks_np.shape[0]), bs):
        batch = toks_np[i : i + bs, :]
        seq = _triton_infer(
            client=client,
            model_name=triton_model_name,
            model_version=triton_model_version,
            input_name=triton_input_name,
            input_tensor=batch,
            output_name=triton_output_name,
            datatype=triton_datatype,
        )
        arr = np.asarray(seq, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == batch.shape[0] and arr.shape[1] == 512:
            embs.append(arr)
        else:
            if arr.ndim != 3 or arr.shape[0] != batch.shape[0] or arr.shape[1] != 77 or arr.shape[2] != 512:
                raise RuntimeError(f"{NAME} | clip_text output has invalid shape: {arr.shape}")
            # EOT position per row: OpenAI CLIP uses a large token id for EOT (argmax works).
            eot_pos = np.argmax(batch, axis=1).astype(np.int64)
            eot_pos = np.clip(eot_pos, 0, 76)
            out = np.zeros((batch.shape[0], 512), dtype=np.float32)
            for r in range(int(batch.shape[0])):
                out[r, :] = arr[r, int(eot_pos[r]), :]
            embs.append(out)

    prompt_emb = _l2norm_rows(np.concatenate(embs, axis=0))  # (P,512)
    if prompt_emb.shape[0] != len(flat_prompts):
        raise RuntimeError(f"{NAME} | internal error: prompt_emb rows mismatch")

    # Aggregate prompt embeddings per label
    label_embs: List[np.ndarray] = []
    for (a, b) in prompt_slices:
        if b <= a:
            label_embs.append(np.zeros((512,), dtype=np.float32))
            continue
        m = np.mean(prompt_emb[a:b, :], axis=0).astype(np.float32)
        label_embs.append(m)
    return _l2norm_rows(np.stack(label_embs, axis=0))


def _require_frame_indices(meta: dict) -> List[int]:
    block = meta.get("core_clip")
    if not isinstance(block, dict) or "frame_indices" not in block:
        raise RuntimeError(f"{NAME} | frames metadata missing core_clip.frame_indices (no-fallback)")
    frame_indices = block.get("frame_indices")
    if not isinstance(frame_indices, list) or not frame_indices:
        raise RuntimeError(f"{NAME} | core_clip.frame_indices empty/invalid (no-fallback)")
    return [int(x) for x in frame_indices]


def _load_domain_db(*, db_dir: str, threshold_global: float, thresholds_json: Optional[str]) -> Tuple[List[int], List[str], List[List[str]], Dict[str, Any], Dict[int, float]]:
    """
    Expected package layout:
      <db_dir>/manifest.json
      <db_dir>/domains.jsonl
      optional: <db_dir>/thresholds.json
    """
    pkg = os.path.abspath(str(db_dir))
    if not os.path.isdir(pkg):
        raise RuntimeError(f"{NAME} | domain db dir not found: {pkg}")
    manifest_path = os.path.join(pkg, "manifest.json")
    domains_path = os.path.join(pkg, "domains.jsonl")
    if not os.path.isfile(manifest_path):
        raise RuntimeError(f"{NAME} | domain db missing manifest.json: {manifest_path}")
    if not os.path.isfile(domains_path):
        raise RuntimeError(f"{NAME} | domain db missing domains.jsonl: {domains_path}")

    manifest = _read_json(manifest_path)
    db_meta = {
        "db_name": str(manifest.get("db_name") or "content_domain"),
        "db_version": str(manifest.get("db_version") or "v1"),
        "db_digest": str(manifest.get("db_digest") or ""),
        "db_path": pkg,
    }
    rows = _read_jsonl(domains_path)
    if not rows:
        raise RuntimeError(f"{NAME} | domains.jsonl is empty: {domains_path}")

    label_ids: List[int] = []
    label_names: List[str] = []
    prompts_per_label: List[List[str]] = []
    seen: set[int] = set()
    for it in rows:
        lid = int(it.get("id"))
        if lid in seen:
            raise RuntimeError(f"{NAME} | duplicate domain id in domains.jsonl: {lid}")
        seen.add(lid)
        name_en = str(it.get("name_en") or it.get("name") or f"domain_{lid}").strip() or f"domain_{lid}"
        pr_en = it.get("prompts_en") or []
        pr_ru = it.get("prompts_ru") or []
        if not isinstance(pr_en, list):
            pr_en = []
        if not isinstance(pr_ru, list):
            pr_ru = []
        pr = [str(x).strip() for x in (list(pr_en) + list(pr_ru)) if str(x).strip()]
        if not pr:
            # safe default prompt: use the name itself
            pr = [name_en]
        label_ids.append(lid)
        label_names.append(name_en)
        prompts_per_label.append(pr)

    threshold_per_label: Dict[int, float] = {}
    thresholds_path = os.path.abspath(str(thresholds_json)) if thresholds_json else os.path.join(pkg, "thresholds.json")
    if os.path.isfile(thresholds_path):
        thr = _read_json(thresholds_path)
        m = thr.get("threshold_per_label") if isinstance(thr, dict) else None
        if isinstance(m, dict):
            for k, v in m.items():
                try:
                    threshold_per_label[int(k)] = float(v)
                except Exception:
                    continue
        if "threshold_global" in thr:
            try:
                threshold_global = float(thr["threshold_global"])
            except Exception:
                pass
    db_meta["threshold_global"] = float(threshold_global)

    # Ensure db_digest is always present (reproducibility)
    if not str(db_meta.get("db_digest") or "").strip():
        canon = {
            "db_name": db_meta.get("db_name"),
            "db_version": db_meta.get("db_version"),
            "domains": [{"id": int(i), "name": str(n), "prompts": list(p)} for i, n, p in zip(label_ids, label_names, prompts_per_label)],
            "threshold_global": float(threshold_global),
            "threshold_per_label": {str(int(k)): float(v) for k, v in sorted(threshold_per_label.items(), key=lambda kv: int(kv[0]))},
        }
        db_meta["db_digest"] = _sha256_hex(
            json.dumps(canon, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    return label_ids, label_names, prompts_per_label, db_meta, threshold_per_label


def _append_state_event_if_possible(*, rs_path: str, event: Dict[str, Any]) -> None:
    """
    Best-effort writer for `state_events.jsonl` (backend tails this file).
    """
    try:
        run_rs = Path(rs_path).resolve()
        rs_base = run_rs.parents[2]  # <rs_base>/<platform>/<video>/<run>
        runs_root = rs_base.parent
        platform_id = str(event.get("platform_id") or "")
        video_id = str(event.get("video_id") or "")
        run_id = str(event.get("run_id") or "")
        if not (platform_id and video_id and run_id):
            return
        p = runs_root / "state" / platform_id / video_id / run_id / "state_events.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        event["platform_id"] = platform_id
        event["video_id"] = video_id
        event["run_id"] = run_id
        line = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        with open(p, "ab") as f:
            f.write(line)
    except Exception:
        return


def _emit_stage(*, rs_path: str, platform_id: str, video_id: str, run_id: str, stage: str) -> None:
    """Emit stage event to state_events.jsonl."""
    _append_state_event_if_possible(
        rs_path=rs_path,
        event={
            "ts": datetime.utcnow().isoformat() + "Z",
            "scope": "progress",
            "processor": "visual",
            "component": NAME,
            "status": "running",
            "stage": str(stage),
            "platform_id": platform_id,
            "video_id": video_id,
            "run_id": run_id,
        },
    )


def _emit_progress(
    *,
    rs_path: str,
    platform_id: str,
    video_id: str,
    run_id: str,
    done: int,
    total: int,
    stage: str,
) -> None:
    """Emit progress event to state_events.jsonl."""
    if total <= 0:
        return
    progress = float(done) / float(total)
    _append_state_event_if_possible(
        rs_path=rs_path,
        event={
            "ts": datetime.utcnow().isoformat() + "Z",
            "scope": "progress",
            "processor": "visual",
            "component": NAME,
            "status": "running",
            "progress": progress,
            "done": int(done),
            "total": int(total),
            "stage": str(stage),
            "platform_id": platform_id,
            "video_id": video_id,
            "run_id": run_id,
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser(NAME)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--rs-path", required=True)
    ap.add_argument("--domain-db-dir", default="dp_models/bundled_models/semantics/content_domain/v1")
    ap.add_argument("--clip-text-model-spec", default="clip_text_triton")
    ap.add_argument("--triton-http-url", default=None, help="Triton HTTP URL (can also be set via TRITON_HTTP_URL env var)")
    ap.add_argument("--topk", type=int, default=5, help="Must be 5 (contract).")
    ap.add_argument(
        "--threshold-global",
        type=float,
        default=0.23,
        help=(
            "DEPRECATED name (kept for backward compatibility). "
            "Used ONLY as fallback for --confidence-threshold-top1."
        ),
    )
    ap.add_argument(
        "--confidence-threshold-top1",
        type=float,
        default=None,
        help="Confidence threshold for *_is_confident_top1 flags (does NOT gate top-K).",
    )
    ap.add_argument("--thresholds-json", default=None)
    args = ap.parse_args()

    # Contract: fixed K=5
    if int(args.topk) != 5:
        raise RuntimeError(f"{NAME} | topk must be 5 (contract), got {args.topk}")

    confidence_threshold_top1 = (
        float(args.confidence_threshold_top1)
        if args.confidence_threshold_top1 is not None
        else float(args.threshold_global)
    )
    if not (0.0 <= confidence_threshold_top1 <= 1.0):
        raise RuntimeError(
            f"{NAME} | confidence_threshold_top1 out of range [0,1]: {confidence_threshold_top1}"
        )

    # Initialize timing dictionary
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    meta = load_metadata(os.path.join(args.frames_dir, "metadata.json"), NAME)
    
    # Extract run identity for state_events
    platform_id = str(meta.get("platform_id") or "")
    video_id = str(meta.get("video_id") or "")
    run_id = str(meta.get("run_id") or "")
    
    # Baseline contract: emit start stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="start",
    )
    
    t_load_deps = time.perf_counter()
    timings["initialization"] = t_load_deps - t0
    
    frame_indices = _require_frame_indices(meta)

    uts = meta.get("union_timestamps_sec")
    if uts is None:
        raise RuntimeError(f"{NAME} | metadata.json missing union_timestamps_sec (contract)")
    uts_arr = np.asarray(uts, dtype=np.float32).reshape(-1)
    fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    if fi_np.size == 0:
        raise RuntimeError(f"{NAME} | frame_indices is empty (no-fallback)")
    if np.any(fi_np < 0) or np.any(fi_np >= int(uts_arr.shape[0])):
        raise RuntimeError(f"{NAME} | frame_indices out of range for union_timestamps_sec")
    times_s = uts_arr[fi_np].astype(np.float32)

    # Baseline contract: emit load_deps stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="load_deps",
    )
    
    # Load core_clip embeddings (no-fallback: full coverage for required frame_indices)
    core_clip_path = os.path.join(str(args.rs_path), "core_clip", "embeddings.npz")
    clip_npz = _load_npz(core_clip_path)
    
    t_load_deps_end = time.perf_counter()
    timings["load_deps"] = t_load_deps_end - t_load_deps
    clip_meta = clip_npz.get("meta")
    upstream_models_used: List[Dict[str, Any]] = []
    upstream_model_signature: Any = None
    if isinstance(clip_meta, dict):
        if isinstance(clip_meta.get("models_used"), list):
            upstream_models_used = clip_meta.get("models_used") or []
        upstream_model_signature = clip_meta.get("model_signature")

    clip_fi = np.asarray(clip_npz.get("frame_indices"), dtype=np.int32).reshape(-1)
    clip_emb = np.asarray(clip_npz.get("frame_embeddings"), dtype=np.float32)
    if clip_fi.size == 0 or clip_emb.size == 0:
        raise RuntimeError(f"{NAME} | core_clip embeddings.npz missing frame_indices/frame_embeddings (no-fallback)")
    if clip_emb.ndim != 2 or clip_emb.shape[0] != clip_fi.shape[0]:
        raise RuntimeError(f"{NAME} | core_clip frame_embeddings invalid shape: {clip_emb.shape}")
    clip_emb = _l2norm_rows(clip_emb)
    clip_map: Dict[int, int] = {int(u): int(i) for i, u in enumerate(clip_fi.tolist())}
    sel_rows: List[int] = []
    for u in frame_indices:
        if int(u) not in clip_map:
            raise RuntimeError(f"{NAME} | core_clip embeddings do not cover required frame_index={u} (no-fallback)")
        sel_rows.append(int(clip_map[int(u)]))
    frame_emb = clip_emb[np.asarray(sel_rows, dtype=np.int32)]  # (N,D)

    # Resolve domain_db_dir path (try DP_MODELS_ROOT if relative)
    domain_db_dir = str(args.domain_db_dir)
    if not os.path.isabs(domain_db_dir):
        mr = os.environ.get("DP_MODELS_ROOT")
        if mr:
            candidate = os.path.join(str(mr), domain_db_dir)
            if os.path.isdir(candidate):
                domain_db_dir = candidate
                LOGGER.info(f"{NAME} | Resolved domain_db_dir via DP_MODELS_ROOT: {domain_db_dir}")
    
    # Baseline contract: emit process_frames stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="process_frames",
    )
    
    t_process_start = time.perf_counter()

    # Load domain db + compute text embeddings (A,512) — fail-fast if missing/invalid
    label_ids, label_names, prompts_per_label, db_meta, threshold_per_label = _load_domain_db(
        db_dir=domain_db_dir,
        threshold_global=float(confidence_threshold_top1),
        thresholds_json=str(args.thresholds_json) if args.thresholds_json else None,
    )
    K = 5

    # Get triton_http_url from args or environment
    triton_http_url = args.triton_http_url
    if not triton_http_url:
        triton_http_url = os.environ.get("TRITON_HTTP_URL")
    
    try:
        txt_mm = _load_triton_spec_via_model_manager(
            str(args.clip_text_model_spec), triton_http_url=triton_http_url
        )
        label_emb = _compute_text_label_embeddings_triton(
            prompts_per_label=prompts_per_label, txt_mm=txt_mm
        )  # (A,512)
    except Exception as e:
        if _is_triton_or_embedding_failure(e):
            url_hint = triton_http_url or os.environ.get("TRITON_HTTP_URL") or "(TRITON_HTTP_URL not set)"
            raise RuntimeError(
                f"{NAME} | CLIP text embeddings require a reachable Triton instance (spec="
                f"{args.clip_text_model_spec!r}, url={url_hint!r}). "
                f"Same class of failure as Embedding Service / inference offline. Detail: {e}"
            ) from e
        raise
    if label_emb.ndim != 2 or label_emb.shape[0] != len(label_ids) or label_emb.shape[1] != frame_emb.shape[1]:
        raise RuntimeError(f"{NAME} | label embeddings shape mismatch: {label_emb.shape} vs D={frame_emb.shape[1]}")

    # Baseline contract: granular progress (>=10 updates)
    n_frames = len(frame_indices)
    _emit_progress(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        done=n_frames // 2,
        total=n_frames,
        stage="process_frames",
    )

    # cosine sims: (N,A)
    sims = np.matmul(frame_emb, label_emb.T).astype(np.float32)
    A = int(label_emb.shape[0])
    if A <= 0:
        raise RuntimeError(f"{NAME} | domain db has 0 labels (fail-fast)")

    label_ids_np = np.asarray(label_ids, dtype=np.int32)
    k_eff = int(min(K, A))
    order = np.argsort(-sims, axis=1)[:, :k_eff]  # (N,k_eff) indices into A
    frame_topk_ids = np.full((sims.shape[0], K), -1, dtype=np.int32)
    frame_topk_scores = np.full((sims.shape[0], K), np.nan, dtype=np.float32)
    frame_topk_scores[:, :k_eff] = np.take_along_axis(sims, order, axis=1).astype(np.float32)
    frame_topk_ids[:, :k_eff] = label_ids_np[order].astype(np.int32)

    # thresholds array aligned to semantic_label_names
    threshold_global = float(db_meta.get("threshold_global") or float(confidence_threshold_top1))
    threshold_per_label_arr = np.full((A,), np.nan, dtype=np.float32)
    for i, lid in enumerate(label_ids):
        if int(lid) in threshold_per_label:
            threshold_per_label_arr[i] = float(threshold_per_label[int(lid)])

    frame_is_confident_top1 = np.zeros((len(frame_indices),), dtype=np.bool_)
    for i in range(int(frame_topk_ids.shape[0])):
        lid = int(frame_topk_ids[i, 0])
        sc = float(frame_topk_scores[i, 0])
        thr = float(threshold_per_label.get(lid, threshold_global))
        frame_is_confident_top1[i] = bool((lid >= 0) and np.isfinite(sc) and sc >= thr)

    # video aggregate (track=1): max over time
    track_ids = np.asarray([0], dtype=np.int32)
    track_present_mask = np.asarray([True], dtype=np.bool_)
    max_scores = np.max(sims, axis=0)  # (A,)
    top_vid = np.argsort(-max_scores)[:k_eff]
    track_topk_scores = np.full((1, K), np.nan, dtype=np.float32)
    track_topk_ids = np.full((1, K), -1, dtype=np.int32)
    track_topk_scores[0, :k_eff] = np.asarray(max_scores[top_vid], dtype=np.float32).reshape(1, k_eff)
    track_topk_ids[0, :k_eff] = np.asarray(label_ids_np[top_vid], dtype=np.int32).reshape(1, k_eff)
    top1_lid = int(track_topk_ids[0, 0])
    top1_sc = float(track_topk_scores[0, 0])
    track_is_confident_top1 = np.asarray(
        [bool((top1_lid >= 0) and np.isfinite(top1_sc) and top1_sc >= float(threshold_per_label.get(top1_lid, threshold_global)))],
        dtype=np.bool_,
    )

    semantic_label_names = np.asarray([f"{int(i)}:{str(n)}" for i, n in zip(label_ids, label_names)], dtype="U")

    t_process_end = time.perf_counter()
    timings["process_frames"] = t_process_end - t_process_start

    # meta
    required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
    missing = [k for k in required_run_keys if not meta.get(k)]
    if missing:
        raise RuntimeError(f"{NAME} | frames metadata missing required run identity keys: {missing}")
    
    meta_out: Dict[str, Any] = {
        "producer": NAME,
        "producer_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": "ok",
        "empty_reason": None,
    }
    
    # Required run identity fields
    for k in required_run_keys:
        meta_out[k] = meta.get(k)
    
    # Required by contract (baseline may use "unknown")
    meta_out["dataprocessor_version"] = str(meta.get("dataprocessor_version") or "unknown")
    
    meta_out.update({
        # thresholds + db
        "threshold_global": threshold_global,
        "threshold_per_label": threshold_per_label,
        **db_meta,
        # provenance chaining
        "core_clip_model_signature": upstream_model_signature,
        # model system
        "models_used": [],
    })
    # Attach model provenance (clip_text only) + upstream (core_clip) for reproducibility
    meta_out["models_used"].extend(upstream_models_used)
    # Avoid duplicating entries already present in core_clip.models_used (model_signature must be stable).
    try:
        txt_entry = txt_mm.get("models_used_entry")
        if isinstance(txt_entry, dict):
            existing_names = {str(m.get("model_name") or "") for m in meta_out.get("models_used") or [] if isinstance(m, dict)}
            if str(txt_entry.get("model_name") or "") not in existing_names:
                meta_out["models_used"].append(txt_entry)
    except Exception:
        pass
    meta_out = apply_models_meta(meta_out, models_used=meta_out.get("models_used"))
    
    # Baseline contract: stage_timings_ms in meta
    timings["saving"] = 0.0  # Will be updated after save
    timings["total"] = time.perf_counter() - t0
    stage_timings_ms: Dict[str, float] = {}
    for key, value in timings.items():
        stage_timings_ms[key] = float(value) * 1000.0
    meta_out["stage_timings_ms"] = stage_timings_ms

    # Baseline contract: emit save stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="save",
    )
    
    out_dir = os.path.join(str(args.rs_path), NAME)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, ARTIFACT_FILENAME)

    # Enrich meta with config highlights
    meta_out["top_k"] = int(K)
    meta_out["confidence_threshold_top1"] = float(confidence_threshold_top1)
    meta_out["clip_text_model_spec"] = str(args.clip_text_model_spec)
    meta_out["domain_db_dir"] = str(domain_db_dir)

    def _build_npz_payload(meta_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "frame_indices": np.asarray(frame_indices, dtype=np.int32),
            "times_s": times_s,
            "semantic_label_names": semantic_label_names,
            "threshold_per_label_arr": threshold_per_label_arr.astype(np.float32),
            "track_ids": track_ids,
            "track_present_mask": track_present_mask,
            "track_topk_ids": track_topk_ids,
            "track_topk_scores": track_topk_scores,
            "track_is_confident_top1": track_is_confident_top1,
            "frame_topk_ids": frame_topk_ids,
            "frame_topk_scores": frame_topk_scores,
            "frame_is_confident_top1": frame_is_confident_top1,
            "meta": np.asarray(meta_dict, dtype=object),
            "meta_json": np.asarray(
                json.dumps(meta_dict, ensure_ascii=False, sort_keys=True),
                dtype="U",
            ),
        }

    # Two-pass write: measure saving time and persist final meta.stage_timings_ms
    t_save_start = time.perf_counter()
    _atomic_save_npz(out_path, **_build_npz_payload(meta_out))
    timings["saving"] = time.perf_counter() - t_save_start
    timings["total"] = time.perf_counter() - t0
    meta_out["stage_timings_ms"] = {k: float(v) * 1000.0 for k, v in timings.items()}
    _atomic_save_npz(out_path, **_build_npz_payload(meta_out))

    # Validate artifact (meta + schema if known)
    from utils.artifact_validator import validate_npz  # type: ignore

    ok, issues, _ = validate_npz(out_path)
    if not ok:
        try:
            os.remove(out_path)
        except Exception:
            pass
        msgs = "; ".join([f"{i.level}:{i.message}" for i in issues if getattr(i, "level", "") == "error"])
        raise RuntimeError(f"{NAME} | saved artifact failed validation: {msgs}")

    # Baseline contract: emit done stage
    _emit_stage(
        rs_path=args.rs_path,
        platform_id=platform_id,
        video_id=video_id,
        run_id=run_id,
        stage="done",
    )
    
    LOGGER.info("%s | wrote %s", NAME, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


