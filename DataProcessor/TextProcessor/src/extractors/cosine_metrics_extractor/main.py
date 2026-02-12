from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes


_EPS_NORM = 1e-8


class CosineMetricsExtractor(BaseExtractor):
    VERSION = "1.2.0"

    def __init__(
        self,
        artifacts_dir: str | None = None,
        *,
        transcript_source_priority: Sequence[str] | str = ("combined", "whisper", "youtube_auto"),
        comments_mode: str = "aggregates",
        compute_title_desc: bool = True,
        compute_title_transcript: bool = True,
        compute_desc_transcript: bool = True,
        compute_transcript_comments_mean: bool = True,
        compute_transcript_comments_median: bool = True,
        require_any_metric: bool = False,
        require_title: bool = False,
        require_description: bool = False,
        require_transcript: bool = False,
        require_comments_for_tc: bool = False,
        emit_extra_metrics: bool = False,
    ) -> None:
        from src.core.path_utils import default_artifacts_dir  # local import

        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.comments_mode = str(comments_mode or "aggregates").strip().lower()
        self.compute_title_desc = bool(compute_title_desc)
        self.compute_title_transcript = bool(compute_title_transcript)
        self.compute_desc_transcript = bool(compute_desc_transcript)
        self.compute_transcript_comments_mean = bool(compute_transcript_comments_mean)
        self.compute_transcript_comments_median = bool(compute_transcript_comments_median)
        self.require_any_metric = bool(require_any_metric)
        self.require_title = bool(require_title)
        self.require_description = bool(require_description)
        self.require_transcript = bool(require_transcript)
        self.require_comments_for_tc = bool(require_comments_for_tc)
        self.emit_extra_metrics = bool(emit_extra_metrics)
        if isinstance(transcript_source_priority, str):
            pr = [p.strip() for p in transcript_source_priority.split(",") if p.strip()]
        else:
            pr = [str(p).strip() for p in transcript_source_priority if str(p).strip()]
        self.transcript_source_priority = pr or ["combined", "whisper", "youtube_auto"]

    @staticmethod
    def _safe_join_artifacts_dir(base_dir: Path, relpath: str) -> Path:
        """
        Join artifacts_dir with relpath and forbid path traversal.
        """
        base = base_dir.expanduser().resolve()
        cand = (base / relpath).resolve()
        if not (cand == base or base in cand.parents):
            raise RuntimeError("CosineMetricsExtractor: relpath escapes artifacts_dir")
        return cand

    def _load_rel_vector(self, relpath: str, *, unsafe_flag: Dict[str, float]) -> Optional[np.ndarray]:
        try:
            p = self._safe_join_artifacts_dir(self.artifacts_dir, relpath)
        except Exception:
            unsafe_flag["tp_cos_unsafe_relpath_flag"] = 1.0
            return None
        if not p.exists():
            return None
        try:
            v = np.load(p)
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            return v if v.size > 0 else None
        except Exception:
            return None

    def _load_rel_matrix(self, relpath: str, *, unsafe_flag: Dict[str, float]) -> Optional[np.ndarray]:
        try:
            p = self._safe_join_artifacts_dir(self.artifacts_dir, relpath)
        except Exception:
            unsafe_flag["tp_cos_unsafe_relpath_flag"] = 1.0
            return None
        if not p.exists():
            return None
        try:
            m = np.load(p)
            m = np.asarray(m, dtype=np.float32)
            if m.ndim == 1:
                m = m.reshape(1, -1)
            if m.ndim != 2:
                return None
            return m if m.size > 0 else None
        except Exception:
            return None

    @staticmethod
    def _tp_get_rel(tp: Any, *path: str) -> Optional[str]:
        """
        Safe nested access for tp_artifacts.
        Example: _tp_get_rel(tp, "embeddings", "title", "relpath")
        """
        cur: Any = tp
        for k in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        return cur if isinstance(cur, str) and cur else None

    def _pick_transcript_agg_mean_relpath(self, tp: Any) -> Optional[str]:
        # canonical: tp_artifacts.transcripts.{source}.agg_mean_relpath
        tr = tp.get("transcripts") if isinstance(tp, dict) else None
        if isinstance(tr, dict):
            for key in self.transcript_source_priority:
                d = tr.get(str(key))
                if isinstance(d, dict) and isinstance(d.get("agg_mean_relpath"), str) and d.get("agg_mean_relpath"):
                    return str(d.get("agg_mean_relpath"))
        # legacy: tp_artifacts.transcript_aggregates.{source}.agg_mean_relpath
        t_aggs = tp.get("transcript_aggregates") if isinstance(tp, dict) else None
        if isinstance(t_aggs, dict):
            for key in self.transcript_source_priority:
                d = t_aggs.get(str(key))
                if isinstance(d, dict) and isinstance(d.get("agg_mean_relpath"), str) and d.get("agg_mean_relpath"):
                    return str(d.get("agg_mean_relpath"))
        return None

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        tp = getattr(doc, "tp_artifacts", None)
        if self.require_any_metric and not (
            self.compute_title_desc
            or self.compute_title_transcript
            or self.compute_desc_transcript
            or self.compute_transcript_comments_mean
            or self.compute_transcript_comments_median
        ):
            raise RuntimeError("CosineMetricsExtractor: require_any_metric=True but all compute_* flags are disabled")

        t_load0 = time.perf_counter()
        unsafe_flag: Dict[str, float] = {"tp_cos_unsafe_relpath_flag": 0.0}

        title_rel = self._tp_get_rel(tp, "embeddings", "title", "relpath")
        desc_rel = self._tp_get_rel(tp, "embeddings", "description", "relpath")

        # Comments sources:
        # - preferred: comments aggregates in tp_artifacts["comments"]
        # - legacy alias: embeddings["comments_agg_*"]["relpath"]
        comments_mean_rel = self._tp_get_rel(tp, "comments", "agg_mean_relpath") or self._tp_get_rel(tp, "embeddings", "comments_agg_mean", "relpath")
        comments_median_rel = self._tp_get_rel(tp, "comments", "agg_median_relpath") or self._tp_get_rel(tp, "embeddings", "comments_agg_median", "relpath")
        comments_matrix_rel = self._tp_get_rel(tp, "embeddings", "comments", "relpath")

        transcript_rel = self._pick_transcript_agg_mean_relpath(tp)

        title = self._load_rel_vector(title_rel, unsafe_flag=unsafe_flag) if isinstance(title_rel, str) and title_rel else None
        desc = self._load_rel_vector(desc_rel, unsafe_flag=unsafe_flag) if isinstance(desc_rel, str) and desc_rel else None
        transcript = self._load_rel_vector(transcript_rel, unsafe_flag=unsafe_flag) if isinstance(transcript_rel, str) and transcript_rel else None
        comments_mean = self._load_rel_vector(comments_mean_rel, unsafe_flag=unsafe_flag) if isinstance(comments_mean_rel, str) and comments_mean_rel else None
        comments_median = self._load_rel_vector(comments_median_rel, unsafe_flag=unsafe_flag) if isinstance(comments_median_rel, str) and comments_median_rel else None
        comments_matrix = self._load_rel_matrix(comments_matrix_rel, unsafe_flag=unsafe_flag) if isinstance(comments_matrix_rel, str) and comments_matrix_rel else None

        load_s = time.perf_counter() - t_load0

        # presence flags for UI/analytics
        title_present = float(title is not None and isinstance(title, np.ndarray) and title.size > 0)
        desc_present = float(desc is not None and isinstance(desc, np.ndarray) and desc.size > 0)
        transcript_present = float(transcript is not None and isinstance(transcript, np.ndarray) and transcript.size > 0)
        comments_present = float(
            (comments_matrix is not None and isinstance(comments_matrix, np.ndarray) and comments_matrix.size > 0)
            or (comments_mean is not None and isinstance(comments_mean, np.ndarray) and comments_mean.size > 0)
            or (comments_median is not None and isinstance(comments_median, np.ndarray) and comments_median.size > 0)
        )

        # fail-fast policies (A-policy): only when required flags are enabled
        if self.require_title and title_present == 0.0:
            raise RuntimeError("CosineMetricsExtractor: required title embedding missing")
        if self.require_description and desc_present == 0.0:
            raise RuntimeError("CosineMetricsExtractor: required description embedding missing")
        if self.require_transcript and transcript_present == 0.0:
            raise RuntimeError("CosineMetricsExtractor: required transcript aggregate embedding missing")
        if self.require_comments_for_tc and (self.compute_transcript_comments_mean or self.compute_transcript_comments_median) and comments_present == 0.0:
            raise RuntimeError("CosineMetricsExtractor: required comments embeddings missing for transcript↔comments metrics")

        dim_mismatch_flag = 0.0
        pair_dim_mismatch_flag = 0.0
        tc_dim_mismatch_flag = 0.0
        zero_norm_flag = 0.0
        t_comp0 = time.perf_counter()

        def _safe_cos(a: Optional[np.ndarray], b: Optional[np.ndarray], *, mismatch_bucket: str) -> float:
            nonlocal dim_mismatch_flag, pair_dim_mismatch_flag, tc_dim_mismatch_flag, zero_norm_flag
            if a is None or b is None:
                return float("nan")
            try:
                a = np.asarray(a, dtype=np.float32).reshape(-1)
                b = np.asarray(b, dtype=np.float32).reshape(-1)
                if a.size == 0 or b.size == 0:
                    return float("nan")
                if int(a.shape[0]) != int(b.shape[0]):
                    dim_mismatch_flag = 1.0
                    if mismatch_bucket == "pair":
                        pair_dim_mismatch_flag = 1.0
                    else:
                        tc_dim_mismatch_flag = 1.0
                    return float("nan")
                na = float(np.linalg.norm(a))
                nb = float(np.linalg.norm(b))
                if na < _EPS_NORM or nb < _EPS_NORM:
                    zero_norm_flag = 1.0
                    return float("nan")
                sim = float(np.dot(a, b) / (na * nb))
                return sim
            except Exception:
                dim_mismatch_flag = 1.0
                if mismatch_bucket == "pair":
                    pair_dim_mismatch_flag = 1.0
                else:
                    tc_dim_mismatch_flag = 1.0
                return float("nan")

        td = float("nan")
        tt = float("nan")
        dt = float("nan")
        if self.compute_title_desc:
            td = _safe_cos(title, desc, mismatch_bucket="pair")
        if self.compute_title_transcript:
            tt = _safe_cos(title, transcript, mismatch_bucket="pair")
        if self.compute_desc_transcript:
            dt = _safe_cos(desc, transcript, mismatch_bucket="pair")

        tcm = float("nan")
        tcmd = float("nan")
        sims_std = float("nan")
        sims_p95 = float("nan")
        n_comments_used = float("nan")

        if transcript is not None and transcript.size > 0 and (self.compute_transcript_comments_mean or self.compute_transcript_comments_median):
            if self.comments_mode == "aggregates":
                if self.compute_transcript_comments_mean:
                    tcm = _safe_cos(transcript, comments_mean, mismatch_bucket="tc")
                if self.compute_transcript_comments_median:
                    tcmd = _safe_cos(transcript, comments_median, mismatch_bucket="tc")
            elif self.comments_mode == "matrix":
                if comments_matrix is not None and comments_matrix.size > 0:
                    try:
                        t = np.asarray(transcript, dtype=np.float32).reshape(1, -1)
                        c = np.asarray(comments_matrix, dtype=np.float32)
                        if t.size == 0 or c.size == 0:
                            raise ValueError("empty arrays")
                        if int(c.shape[1]) != int(t.shape[1]):
                            dim_mismatch_flag = 1.0
                            tc_dim_mismatch_flag = 1.0
                        else:
                            t_norm = float(np.linalg.norm(t))
                            if t_norm < _EPS_NORM:
                                zero_norm_flag = 1.0
                            else:
                                c_norms = np.linalg.norm(c, axis=1)
                                bad = c_norms < _EPS_NORM
                                if bool(np.any(bad)):
                                    zero_norm_flag = 1.0
                                # compute sims, marking bad rows as NaN
                                sims = (c @ t.reshape(-1)) / (c_norms * t_norm)
                                sims = sims.astype(np.float32, copy=False)
                                if bool(np.any(bad)):
                                    sims = sims.astype(np.float32, copy=True)
                                    sims[bad] = np.nan

                                n_comments_used = float(int(sims.shape[0]))
                                if self.compute_transcript_comments_mean:
                                    tcm = float(np.nanmean(sims)) if bool(np.any(np.isfinite(sims))) else float("nan")
                                if self.compute_transcript_comments_median:
                                    tcmd = float(np.nanmedian(sims)) if bool(np.any(np.isfinite(sims))) else float("nan")
                                if self.emit_extra_metrics and sims.size > 0:
                                    if bool(np.any(np.isfinite(sims))):
                                        sims_std = float(np.nanstd(sims))
                                        sims_p95 = float(np.nanpercentile(sims, 95))
                    except Exception:
                        dim_mismatch_flag = 1.0
                        tc_dim_mismatch_flag = 1.0
                        # keep NaNs
            else:
                # unknown comments mode → do nothing (NaNs)
                pass

        compute_s = time.perf_counter() - t_comp0

        # empty reasons (privacy-safe): only when enabled metrics require missing inputs
        enabled_pair_needs_title = bool(self.compute_title_desc or self.compute_title_transcript)
        enabled_pair_needs_desc = bool(self.compute_title_desc or self.compute_desc_transcript)
        enabled_pair_needs_transcript = bool(self.compute_title_transcript or self.compute_desc_transcript)
        enabled_tc = bool(self.compute_transcript_comments_mean or self.compute_transcript_comments_median)

        empty_no_title = float(enabled_pair_needs_title and title_present == 0.0)
        empty_no_desc = float(enabled_pair_needs_desc and desc_present == 0.0)
        empty_no_transcript = float((enabled_pair_needs_transcript or enabled_tc) and transcript_present == 0.0)
        empty_no_comments = float(enabled_tc and comments_present == 0.0)

        features_flat: Dict[str, float] = {
            "tp_cos_title_present": float(title_present),
            "tp_cos_desc_present": float(desc_present),
            "tp_cos_transcript_present": float(transcript_present),
            "tp_cos_comments_present": float(comments_present),
            "tp_cos_title_desc_enabled": float(self.compute_title_desc),
            "tp_cos_title_transcript_enabled": float(self.compute_title_transcript),
            "tp_cos_desc_transcript_enabled": float(self.compute_desc_transcript),
            "tp_cos_transcript_comments_mean_enabled": float(self.compute_transcript_comments_mean),
            "tp_cos_transcript_comments_median_enabled": float(self.compute_transcript_comments_median),
            "tp_cos_empty_no_title": float(empty_no_title),
            "tp_cos_empty_no_desc": float(empty_no_desc),
            "tp_cos_empty_no_transcript": float(empty_no_transcript),
            "tp_cos_empty_no_comments": float(empty_no_comments),
            "tp_cos_zero_norm_flag": float(zero_norm_flag),
            "tp_cos_dim_mismatch_flag": float(dim_mismatch_flag),
            "tp_cos_pair_dim_mismatch_flag": float(pair_dim_mismatch_flag),
            "tp_cos_tc_dim_mismatch_flag": float(tc_dim_mismatch_flag),
            "tp_cos_unsafe_relpath_flag": float(unsafe_flag["tp_cos_unsafe_relpath_flag"]),
            "tp_cos_title_desc": float(td),
            "tp_cos_title_transcript": float(tt),
            "tp_cos_desc_transcript": float(dt),
            "tp_cos_transcript_comments_mean": float(tcm),
            "tp_cos_transcript_comments_median": float(tcmd),
        }
        if self.emit_extra_metrics:
            features_flat.update(
                {
                    "tp_cos_load_ms": float(round(load_s * 1000.0, 3)),
                    "tp_cos_compute_ms": float(round(compute_s * 1000.0, 3)),
                    "tp_cos_comments_mode_aggregates": float(self.comments_mode == "aggregates"),
                    "tp_cos_comments_mode_matrix": float(self.comments_mode == "matrix"),
                    "tp_cos_tc_n_comments_used": float(n_comments_used),
                    "tp_cos_tc_sims_std": float(sims_std),
                    "tp_cos_tc_sims_p95": float(sims_p95),
                }
            )

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        return {
            "device": "cpu",
            "version": self.VERSION,
            "system": {
                "pre_init": sys_before,
                "post_init": sys_before,
                "post_process": sys_after,
                "peaks": {
                    "ram_peak_mb": int(max(mem_before, mem_after) / 1024 / 1024),
                    "gpu_peak_mb": 0,
                },
            },
            "timings_s": {"total": round(total_s, 3)},
            "result": {"features_flat": features_flat},
            "error": None,
        }


