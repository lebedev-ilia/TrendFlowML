from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.model_registry import get_model
from src.core.path_utils import default_artifacts_dir
from src.core.text_utils import normalize_whitespace


def _l2n(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 0:
        return v / n
    return v


class SpeakerTurnEmbeddingsAggregatorExtractor(BaseExtractor):
    """
    A-policy speaker-turn embeddings aggregator.

    Privacy:
    - No raw speaker names/texts in result.
    - No raw-derived hashes in filenames.

    Determinism:
    - Per-run fixed artifact names.
    - Stable speaker_id assignment.
    """

    VERSION = "1.2.0"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        artifacts_dir: str | None = None,
        device: Optional[str] = "cpu",
        fp16: bool = True,
        batch_size: int = 64,
        # A-policy behavior controls
        compute_mean: bool = True,
        compute_max: bool = True,
        write_artifacts: bool = True,
        require_input: bool = False,
        max_speakers: int = 16,
        max_turns_per_speaker: int = 64,
        min_chars_per_turn: int = 5,
        max_chars_per_turn: int = 600,
        dedup_turn_texts: bool = True,
        emit_extra_metrics: bool = False,
    ) -> None:
        self.model_name = model_name
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.device = str(device or "cpu")
        self.fp16 = fp16 and ("cuda" in self.device)
        self.batch_size = batch_size
        self.compute_mean = bool(compute_mean)
        self.compute_max = bool(compute_max)
        self.write_artifacts = bool(write_artifacts)
        self.require_input = bool(require_input)
        self.max_speakers = int(max_speakers)
        self.max_turns_per_speaker = int(max_turns_per_speaker)
        self.min_chars_per_turn = int(min_chars_per_turn)
        self.max_chars_per_turn = int(max_chars_per_turn)
        self.dedup_turn_texts = bool(dedup_turn_texts)
        self.emit_extra_metrics = bool(emit_extra_metrics)

        if not self.compute_mean and not self.compute_max:
            raise RuntimeError("speaker_turn_embeddings_aggregator: at least one of compute_mean/compute_max must be True")
        if self.max_speakers <= 0:
            raise RuntimeError("speaker_turn_embeddings_aggregator: max_speakers must be > 0")
        if self.max_turns_per_speaker <= 0:
            raise RuntimeError("speaker_turn_embeddings_aggregator: max_turns_per_speaker must be > 0")
        if self.max_chars_per_turn <= 0:
            raise RuntimeError("speaker_turn_embeddings_aggregator: max_chars_per_turn must be > 0")
        if self.min_chars_per_turn < 0:
            raise RuntimeError("speaker_turn_embeddings_aggregator: min_chars_per_turn must be >= 0")
        if self.min_chars_per_turn > self.max_chars_per_turn:
            raise RuntimeError("speaker_turn_embeddings_aggregator: min_chars_per_turn must be <= max_chars_per_turn")

        # Resolve model metadata via dp_models (fail-fast).
        try:
            from dp_models import get_global_model_manager  # type: ignore

            mm = get_global_model_manager()
            spec = mm.get_spec(model_name=str(self.model_name))
            _d, _p, _rt, _eng, weights_digest, _arts = mm.resolve(spec)
            self.weights_digest = str(weights_digest or "unknown")
            self.model_version = str(getattr(spec, "model_version", "unknown") or "unknown")
        except Exception as e:
            raise RuntimeError(f"speaker_turn_embeddings_aggregator: dp_models resolve failed for model_name={self.model_name!r}: {e}") from e

        self._model = get_model(self.model_name, self.device, self.fp16)

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if torch is None:
            raise RuntimeError("speaker_turn_embeddings_aggregator: torch is required to run sentence-transformers in-process")
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch = [normalize_whitespace(t) for t in batch]
            with torch.no_grad():
                raw = self._model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
            raw = np.asarray(raw, dtype=np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out.append(raw / norms)
        return np.vstack(out)

    def _select_turn_texts(self, texts: List[str]) -> List[str]:
        """
        Deterministic selection/cleanup for speaker turn texts.
        """
        out: List[str] = []
        seen: set[str] = set()
        for t in texts:
            tt = normalize_whitespace(str(t or "")).strip()
            if not tt:
                continue
            if self.max_chars_per_turn and len(tt) > self.max_chars_per_turn:
                tt = tt[: self.max_chars_per_turn].rstrip()
            if len(tt) < self.min_chars_per_turn:
                continue
            if self.dedup_turn_texts:
                key = tt.casefold()
                if key in seen:
                    continue
                seen.add(key)
            out.append(tt)
            if len(out) >= self.max_turns_per_speaker:
                break
        return out

    def _group_speakers(self, doc: Any) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
        """
        Returns (speaker_id -> texts, meta_flags as float dict).

        Supported inputs:
        - Preferred: doc.speaker_diarization + doc.asr (NOT yet wired in orchestrator, but supported for future).
          speaker_diarization format expected (minimal):
            {"speaker_segments": [{"speaker_id": int|str, "start_sec": float, "end_sec": float}], ...}
          asr format expected:
            {"segments": [{"text": str, "start_sec": float|None, "end_sec": float|None}], ...}
        - Legacy: doc.speakers as Dict[str, Dict], where each value may contain:
            {"name": str, "description": str}
        """
        flags: Dict[str, float] = {
            "tp_spkemb_input_present": 0.0,
            "tp_spkemb_input_mode_diar_asr": 0.0,
            "tp_spkemb_input_mode_legacy_doc_speakers": 0.0,
            "tp_spkemb_asr_present": 0.0,
            "tp_spkemb_diar_present": 0.0,
        }

        # Preferred mode: diarization + ASR
        diar = getattr(doc, "speaker_diarization", None)
        asr = getattr(doc, "asr", None)
        diar_segments = None
        asr_segments = None
        if isinstance(diar, dict):
            diar_segments = diar.get("speaker_segments")
        if isinstance(asr, dict):
            asr_segments = asr.get("segments")
        if isinstance(diar_segments, list) and isinstance(asr_segments, list):
            flags["tp_spkemb_input_present"] = 1.0
            flags["tp_spkemb_input_mode_diar_asr"] = 1.0
            flags["tp_spkemb_asr_present"] = 1.0
            flags["tp_spkemb_diar_present"] = 1.0

            # Build per-speaker texts by assigning ASR segments to diar segments by time overlap.
            by_spk: Dict[str, List[str]] = {}
            for seg in diar_segments:
                if not isinstance(seg, dict):
                    continue
                sid = seg.get("speaker_id")
                if sid is None:
                    continue
                spk_id = str(sid)
                start = seg.get("start_sec")
                end = seg.get("end_sec")
                try:
                    s0 = float(start)
                    s1 = float(end)
                except Exception:
                    continue
                if not (s1 > s0):
                    continue

                texts: List[str] = []
                for a in asr_segments:
                    if not isinstance(a, dict):
                        continue
                    t = a.get("text")
                    if not isinstance(t, str) or not t.strip():
                        continue
                    a0 = a.get("start_sec")
                    a1 = a.get("end_sec")
                    try:
                        aa0 = float(a0) if a0 is not None else None
                        aa1 = float(a1) if a1 is not None else None
                    except Exception:
                        aa0, aa1 = None, None
                    # If no timestamps, we can't align reliably; skip.
                    if aa0 is None or aa1 is None:
                        continue
                    # Overlap check
                    if aa1 <= s0 or aa0 >= s1:
                        continue
                    texts.append(t)
                if texts:
                    by_spk.setdefault(spk_id, []).extend(texts)

            # Deterministic speaker ordering: sort by speaker_id string.
            keys = sorted(by_spk.keys())[: self.max_speakers]
            out = {f"spk{idx:03d}": self._select_turn_texts(by_spk[k]) for idx, k in enumerate(keys)}
            # Drop empties after selection
            out = {k: v for k, v in out.items() if v}
            return out, flags

        # Legacy mode: doc.speakers dict with name/description fields
        speakers: Dict[str, Dict[str, Any]] = getattr(doc, "speakers", {}) or {}
        if isinstance(speakers, dict) and speakers:
            flags["tp_spkemb_input_present"] = 1.0
            flags["tp_spkemb_input_mode_legacy_doc_speakers"] = 1.0
            # group texts per "name" field but never persist that name
            name_to_texts: Dict[str, List[str]] = {}
            for _ts, turn in speakers.items():
                if not isinstance(turn, dict):
                    continue
                name = str(turn.get("name", "")).strip() or "unknown"
                text = str(turn.get("description", "")).strip()
                if not text:
                    continue
                name_to_texts.setdefault(name, []).append(text)

            # Deterministic speaker_id assignment: sort by normalized name, then enumerate.
            names_sorted = sorted(name_to_texts.keys(), key=lambda s: normalize_whitespace(s).casefold())
            names_sorted = names_sorted[: self.max_speakers]
            out = {f"spk{idx:03d}": self._select_turn_texts(name_to_texts[n]) for idx, n in enumerate(names_sorted)}
            out = {k: v for k, v in out.items() if v}
            return out, flags

        return {}, flags

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        speaker_texts_by_id, flags = self._group_speakers(doc)
        if not speaker_texts_by_id:
            if self.require_input and flags.get("tp_spkemb_input_present", 0.0) <= 0.0:
                raise RuntimeError("speaker_turn_embeddings_aggregator: required input is missing (no speaker diarization / no legacy doc.speakers)")
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return {
                "device": self.device,
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
                "result": {
                    "features_flat": {
                        "tp_spkemb_present": 0.0,
                        "tp_spkemb_speakers_total": 0.0,
                        "tp_spkemb_speakers_embedded": 0.0,
                        "tp_spkemb_turns_total": 0.0,
                        "tp_spkemb_write_artifacts": 1.0 if self.write_artifacts else 0.0,
                        "tp_spkemb_compute_mean": 1.0 if self.compute_mean else 0.0,
                        "tp_spkemb_compute_max": 1.0 if self.compute_max else 0.0,
                        **flags,
                    },
                    "speaker_embeddings_meta": {
                        "model_name": str(self.model_name),
                        "model_version": str(self.model_version),
                        "weights_digest": str(self.weights_digest),
                    },
                },
                "error": None,
            }

        results: Dict[str, Any] = {}
        n_saved = 0
        turns_total = 0
        for speaker_id, texts in speaker_texts_by_id.items():
            turns_total += int(len(texts))
            embs = self._encode_texts(texts)
            if embs.size == 0:
                continue

            mean_emb = _l2n(np.mean(embs, axis=0)) if self.compute_mean else None
            max_emb = _l2n(np.max(embs, axis=0)) if self.compute_max else None

            # save per-run fixed artifacts (no content-derived hashes)
            spk_entry: Dict[str, Any] = {"count_turns": int(len(texts))}
            if self.write_artifacts:
                if mean_emb is not None:
                    mean_path = self.artifacts_dir / f"speaker_{speaker_id}_mean.npy"
                    tmp_m = mean_path.with_suffix(".tmp.npy")
                    np.save(tmp_m, mean_emb.astype(np.float32))
                    tmp_m.replace(mean_path)
                    spk_entry["mean_relpath"] = mean_path.name
                if max_emb is not None:
                    max_path = self.artifacts_dir / f"speaker_{speaker_id}_max.npy"
                    tmp_x = max_path.with_suffix(".tmp.npy")
                    np.save(tmp_x, max_emb.astype(np.float32))
                    tmp_x.replace(max_path)
                    spk_entry["max_relpath"] = max_path.name

            if ("mean_relpath" in spk_entry) or ("max_relpath" in spk_entry):
                n_saved += 1
            results[str(speaker_id)] = spk_entry

        # In-memory registry for downstream (if any)
        try:
            tp = getattr(doc, "tp_artifacts", None)
            if not isinstance(tp, dict):
                tp = {}
                setattr(doc, "tp_artifacts", tp)
            # Canonical
            tp.setdefault("speakers", {})
            tp["speakers"]["embeddings"] = results
            # Legacy alias
            tp.setdefault("speaker_embeddings", {})
            tp["speaker_embeddings"].update(results)
        except Exception:
            pass

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        features_flat: Dict[str, float] = {
            "tp_spkemb_present": 1.0 if n_saved > 0 else 0.0,
            "tp_spkemb_speakers_total": float(int(len(speaker_texts_by_id))),
            "tp_spkemb_speakers_embedded": float(int(n_saved)),
            "tp_spkemb_turns_total": float(int(turns_total)),
            "tp_spkemb_write_artifacts": 1.0 if self.write_artifacts else 0.0,
            "tp_spkemb_compute_mean": 1.0 if self.compute_mean else 0.0,
            "tp_spkemb_compute_max": 1.0 if self.compute_max else 0.0,
            **flags,
        }
        if self.emit_extra_metrics:
            features_flat["tp_spkemb_batch_size"] = float(int(self.batch_size))
            features_flat["tp_spkemb_max_speakers"] = float(int(self.max_speakers))
            features_flat["tp_spkemb_max_turns_per_speaker"] = float(int(self.max_turns_per_speaker))
            features_flat["tp_spkemb_min_chars_per_turn"] = float(int(self.min_chars_per_turn))
            features_flat["tp_spkemb_max_chars_per_turn"] = float(int(self.max_chars_per_turn))

        return {
            "device": self.device,
            "version": self.VERSION,
            "model_version": str(self.model_version),
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
            "result": {
                "features_flat": features_flat,
                "speaker_embeddings_meta": {
                    "model_name": str(self.model_name),
                    "model_version": str(self.model_version),
                    "weights_digest": str(self.weights_digest),
                },
            },
            "error": None,
        }


