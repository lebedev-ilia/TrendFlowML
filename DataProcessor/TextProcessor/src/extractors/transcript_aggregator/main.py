from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.schemas.models import VideoDocument
from src.core.text_utils import normalize_whitespace


class TranscriptAggregatorExtractor(BaseExtractor):
    """
    A-policy transcript chunk embeddings aggregator.

    - Reads chunk embeddings deterministically via doc.tp_artifacts (canonical + legacy)
    - Writes per-run fixed-name aggregate artifacts (no content hashes)
    - Valid empty semantics: no fake vectors; missing optional inputs -> NaNs + *_present flags
    """

    VERSION = "1.2.0"

    def __init__(
        self,
        artifacts_dir: Optional[str] = None,
        # model_name kept for legacy compatibility/metadata only (aggregator does not run the model)
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = "cpu",
        decay_rate: float = 0.01,
        compute_std: bool = False,
        compute_mean: bool = True,
        compute_max: bool = True,
        compute_combined: bool = True,
        write_artifacts: bool = True,
        require_chunks: bool = False,
        sources: Optional[List[str]] = None,  # e.g. ["whisper","youtube_auto"]
        emit_extra_metrics: bool = False,
    ) -> None:
        from src.core.path_utils import default_artifacts_dir  # local import to avoid cycles

        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.device = str(device or "cpu")
        self.decay_rate = decay_rate
        self.compute_std = bool(compute_std)
        self.compute_mean = bool(compute_mean)
        self.compute_max = bool(compute_max)
        self.compute_combined = bool(compute_combined)
        self.write_artifacts = bool(write_artifacts)
        self.require_chunks = bool(require_chunks)
        self.emit_extra_metrics = bool(emit_extra_metrics)
        self.sources = [str(x) for x in (sources or ["whisper", "youtube_auto"]) if str(x).strip()]
        if not self.sources:
            raise RuntimeError("TranscriptAggregatorExtractor: sources list must be non-empty")
        if not self.compute_mean and not self.compute_max:
            raise RuntimeError("TranscriptAggregatorExtractor: at least one of compute_mean/compute_max must be True")
        # no heavy model; pure tensor ops

    @staticmethod
    def _normalize(vec: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(vec, p=2, dim=-1)

    @staticmethod
    def _position_decay(n_chunks: int, decay_rate: float = 0.01) -> torch.Tensor:
        positions = torch.arange(n_chunks, dtype=torch.float32)
        weights = torch.exp(-decay_rate * positions)
        return weights

    def _aggregate_mean_streaming(
        self,
        m: np.ndarray,
        *,
        confidence: Optional[List[float]] = None,
        decay_rate: float = 0.01,
    ) -> Dict[str, Any]:
        if m.size == 0:
            return {"embedding": None, "count": 0, "std": float("nan")}

        if m.ndim == 1:
            m = m.reshape(1, -1)
        n_chunks = int(m.shape[0])
        d = int(m.shape[1])
        weights = self._position_decay(n_chunks, decay_rate)
        if confidence is not None and len(confidence) == n_chunks:
            weights = weights * torch.tensor(confidence, dtype=torch.float32)
        weights = weights / (weights.sum() + 1e-8)

        acc = torch.zeros((d,), dtype=torch.float32)
        # optional streaming scalar std across all elements
        mean = 0.0
        m2 = 0.0
        cnt = 0
        for i in range(n_chunks):
            v = np.asarray(m[i], dtype=np.float32)
            w = float(weights[i].item())
            acc += torch.tensor(v) * w
            if self.compute_std:
                for x in v.tolist():
                    cnt += 1
                    delta = x - mean
                    mean += delta / cnt
                    m2 += delta * (x - mean)
        normed = self._normalize(acc)
        std_val = float((m2 / max(1, cnt - 1)) ** 0.5) if (self.compute_std and cnt > 1) else float("nan")
        return {"embedding": normed.cpu().numpy(), "count": n_chunks, "std": std_val}

    def _aggregate_maxpool_streaming(self, m: np.ndarray) -> Dict[str, Any]:
        if m.size == 0:
            return {"embedding": None, "count": 0, "std": float("nan")}
        if m.ndim == 1:
            m = m.reshape(1, -1)
        n_chunks = int(m.shape[0])
        d = int(m.shape[1])
        maxv = torch.full((d,), -1e9, dtype=torch.float32)
        mean = 0.0
        m2 = 0.0
        cnt = 0
        for i in range(n_chunks):
            v = np.asarray(m[i], dtype=np.float32)
            maxv = torch.maximum(maxv, torch.tensor(v))
            if self.compute_std:
                for x in v.tolist():
                    cnt += 1
                    delta = x - mean
                    mean += delta / cnt
                    m2 += delta * (x - mean)
        normed = self._normalize(maxv)
        std_val = float((m2 / max(1, cnt - 1)) ** 0.5) if (self.compute_std and cnt > 1) else float("nan")
        return {"embedding": normed.cpu().numpy(), "count": n_chunks, "std": std_val}

    @staticmethod
    def _safe_join_artifacts_dir(base_dir: Path, relpath: str) -> Path:
        base = base_dir.expanduser().resolve()
        cand = (base / relpath).resolve()
        if not (cand == base or base in cand.parents):
            raise RuntimeError("transcript_aggregator: relpath escapes artifacts_dir")
        return cand

    @staticmethod
    def _read_chunk_relpath(tp_art: Dict[str, Any], source: str) -> Optional[str]:
        # Canonical
        tr = tp_art.get("transcripts")
        if isinstance(tr, dict) and isinstance(tr.get(source), dict):
            rel = tr[source].get("chunk_embeddings_relpath")
            if isinstance(rel, str) and rel:
                return rel
        # Legacy
        tchunks = tp_art.get("transcript_chunks")
        if isinstance(tchunks, dict) and isinstance(tchunks.get(source), dict):
            rel = tchunks[source].get("embeddings_relpath") or tchunks[source].get("embeddings_path")
            if isinstance(rel, str) and rel:
                return rel
        return None

    @staticmethod
    def _read_chunk_confidence(tp_art: Dict[str, Any], source: str) -> Optional[List[float]]:
        # Optional hook (if upstream ever provides per-chunk weights in-memory)
        tr = tp_art.get("transcripts")
        if isinstance(tr, dict) and isinstance(tr.get(source), dict):
            conf = tr[source].get("chunk_confidence")
            if isinstance(conf, list) and conf:
                out: List[float] = []
                for x in conf:
                    if isinstance(x, (int, float)):
                        out.append(float(x))
                return out if out else None
        return None

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        # Dependency: requires transcript chunk embeddings created earlier in this run.
        tp_art = getattr(doc, "tp_artifacts", None)
        if not isinstance(tp_art, dict):
            raise RuntimeError("TranscriptAggregatorExtractor requires TranscriptChunkEmbedder outputs (doc.tp_artifacts missing)")

        load_s = 0.0
        agg_s = 0.0

        # Load embeddings for each source if available
        t_load = time.perf_counter()
        emb_by_source: Dict[str, Any] = {}
        for source in self.sources:
            rel = self._read_chunk_relpath(tp_art, source)
            if not isinstance(rel, str) or not rel:
                continue
            p = self._safe_join_artifacts_dir(self.artifacts_dir, rel)
            if not p.exists():
                if self.require_chunks:
                    raise RuntimeError(f"TranscriptAggregatorExtractor: chunk embeddings file missing for source={source!r}")
                continue
            arr = np.load(p, mmap_mode="r")
            emb_by_source[source] = arr
        load_s = time.perf_counter() - t_load

        results: Dict[str, Any] = {}
        t_agg = time.perf_counter()
        # Aggregate per source
        for source, emb in emb_by_source.items():
            if emb is None:
                continue
            conf = self._read_chunk_confidence(tp_art, source)
            mean_res = self._aggregate_mean_streaming(emb, confidence=conf, decay_rate=self.decay_rate) if self.compute_mean else {"embedding": None, "count": int(getattr(emb, "shape", [0])[0] or 0), "std": float("nan")}
            max_res = self._aggregate_maxpool_streaming(emb) if self.compute_max else {"embedding": None, "count": int(getattr(emb, "shape", [0])[0] or 0), "std": float("nan")}

            # Write artifacts (fixed per-run names)
            mean_rel = None
            max_rel = None
            if self.write_artifacts:
                if self.compute_mean and mean_res.get("embedding") is not None:
                    mean_path = self.artifacts_dir / f"transcript_{source}_agg_mean.npy"
                    tmp = mean_path.with_suffix(".tmp.npy")
                    np.save(tmp, np.asarray(mean_res["embedding"], dtype=np.float32))
                    tmp.replace(mean_path)
                    mean_rel = mean_path.name
                if self.compute_max and max_res.get("embedding") is not None:
                    max_path = self.artifacts_dir / f"transcript_{source}_agg_max.npy"
                    tmp = max_path.with_suffix(".tmp.npy")
                    np.save(tmp, np.asarray(max_res["embedding"], dtype=np.float32))
                    tmp.replace(max_path)
                    max_rel = max_path.name

            results[source] = {
                "present": True,
                "n_chunks": int(mean_res.get("count") or max_res.get("count") or 0),
                "mean": {"present": bool(self.compute_mean and mean_res.get("embedding") is not None), "std": float(mean_res.get("std", float("nan"))), "relpath": mean_rel},
                "max": {"present": bool(self.compute_max and max_res.get("embedding") is not None), "std": float(max_res.get("std", float("nan"))), "relpath": max_rel},
                "conf_used": True if (conf is not None and self.compute_mean) else False,
            }

        # Combined over all sources (if both present)
        if self.compute_combined and emb_by_source:
            # Deterministic concat order = self.sources order.
            parts: List[np.ndarray] = []
            for s in self.sources:
                if s in emb_by_source and emb_by_source[s] is not None:
                    parts.append(np.asarray(emb_by_source[s], dtype=np.float32))
            if parts:
                combined = np.vstack(parts)
                mean_res = self._aggregate_mean_streaming(combined, confidence=None, decay_rate=self.decay_rate) if self.compute_mean else {"embedding": None, "count": int(combined.shape[0]), "std": float("nan")}
                max_res = self._aggregate_maxpool_streaming(combined) if self.compute_max else {"embedding": None, "count": int(combined.shape[0]), "std": float("nan")}

                mean_rel = None
                max_rel = None
                if self.write_artifacts:
                    if self.compute_mean and mean_res.get("embedding") is not None:
                        mean_path = self.artifacts_dir / "transcript_combined_agg_mean.npy"
                        tmp = mean_path.with_suffix(".tmp.npy")
                        np.save(tmp, np.asarray(mean_res["embedding"], dtype=np.float32))
                        tmp.replace(mean_path)
                        mean_rel = mean_path.name
                    if self.compute_max and max_res.get("embedding") is not None:
                        max_path = self.artifacts_dir / "transcript_combined_agg_max.npy"
                        tmp = max_path.with_suffix(".tmp.npy")
                        np.save(tmp, np.asarray(max_res["embedding"], dtype=np.float32))
                        tmp.replace(max_path)
                        max_rel = max_path.name

                results["combined"] = {
                    "present": True,
                    "n_chunks": int(mean_res.get("count") or max_res.get("count") or 0),
                    "mean": {"present": bool(self.compute_mean and mean_res.get("embedding") is not None), "std": float(mean_res.get("std", float("nan"))), "relpath": mean_rel},
                    "max": {"present": bool(self.compute_max and max_res.get("embedding") is not None), "std": float(max_res.get("std", float("nan"))), "relpath": max_rel},
                    "conf_used": False,
                }
        agg_s = time.perf_counter() - t_agg

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        # In-memory registry for downstream selection (no absolute paths in result).
        tp_art.setdefault("transcript_aggregates", {})  # legacy alias bucket
        tp_art.setdefault("transcripts", {})
        for k, v in results.items():
            if not isinstance(v, dict):
                continue
            mean_rel = (v.get("mean") or {}).get("relpath") if isinstance(v.get("mean"), dict) else None
            max_rel = (v.get("max") or {}).get("relpath") if isinstance(v.get("max"), dict) else None
            if isinstance(mean_rel, str) and mean_rel:
                tp_art["transcript_aggregates"].setdefault(k, {})
                tp_art["transcript_aggregates"][k]["agg_mean_relpath"] = mean_rel
                tp_art["transcripts"].setdefault(k, {})
                tp_art["transcripts"][k]["agg_mean_relpath"] = mean_rel
            if isinstance(max_rel, str) and max_rel:
                tp_art["transcript_aggregates"].setdefault(k, {})
                tp_art["transcript_aggregates"][k]["agg_max_relpath"] = max_rel
                tp_art["transcripts"].setdefault(k, {})
                tp_art["transcripts"][k]["agg_max_relpath"] = max_rel

        features_flat: Dict[str, float] = {
            "tp_tragg_present": 1.0 if bool(results) else 0.0,
            "tp_tragg_present_whisper": 1.0 if ("whisper" in results) else 0.0,
            "tp_tragg_present_youtube": 1.0 if ("youtube_auto" in results) else 0.0,
            "tp_tragg_present_combined": 1.0 if ("combined" in results) else 0.0,
            "tp_tragg_decay_rate": float(self.decay_rate),
            "tp_tragg_compute_std": 1.0 if self.compute_std else 0.0,
            "tp_tragg_compute_mean": 1.0 if self.compute_mean else 0.0,
            "tp_tragg_compute_max": 1.0 if self.compute_max else 0.0,
            "tp_tragg_compute_combined": 1.0 if self.compute_combined else 0.0,
            "tp_tragg_write_artifacts": 1.0 if self.write_artifacts else 0.0,
        }
        if self.emit_extra_metrics:
            # counts per source
            for src in ("whisper", "youtube_auto", "combined"):
                n = float("nan")
                if isinstance(results.get(src), dict):
                    n = float(int(results[src].get("n_chunks", 0) or 0))
                features_flat[f"tp_tragg_{src}_n_chunks"] = float(n)
                if self.compute_std:
                    ms = (results.get(src) or {}).get("mean") if isinstance(results.get(src), dict) else None
                    xs = (results.get(src) or {}).get("max") if isinstance(results.get(src), dict) else None
                    mstd = float(ms.get("std")) if isinstance(ms, dict) and ms.get("present") else float("nan")
                    xstd = float(xs.get("std")) if isinstance(xs, dict) and xs.get("present") else float("nan")
                    features_flat[f"tp_tragg_{src}_mean_std"] = float(mstd)
                    features_flat[f"tp_tragg_{src}_max_std"] = float(xstd)

        return {
            "device": self.device,
            "version": self.VERSION,
            "system": {
                "pre_init": sys_before,
                "post_init": sys_before,  # aggregator has no heavy init
                "post_process": sys_after,
                "peaks": {
                    "ram_peak_mb": int(max(mem_before, mem_after) / 1024 / 1024),
                    "gpu_peak_mb": 0,
                },
            },
            "timings_s": {"load": round(load_s, 3), "aggregate": round(agg_s, 3), "total": round(total_s, 3)},
            "result": {"features_flat": features_flat},
            "error": None,
        }


