#!/usr/bin/env python3
"""
Проверка качества и контрактности артефактов E2E (§0.2), поверх оркестрации §0.1.

§0.1 (e2e_validate_full_green.py) проверяет только статусы пайплайна.
Этот скрипт проверяет NPZ/manifest: контракт, сигналы на mock-видео, деградацию данных.

Usage:
  python backend/scripts/e2e_validate_output_quality.py --latest-e2e-artifact
  python backend/scripts/e2e_validate_output_quality.py \\
    --run-id 38baa7e5-d6c9-4244-bd18-45f9498ef579 \\
    --platform-id youtube --video-id -Q6fnPIybEI
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Reuse run resolution from §0.1 validator
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from e2e_validate_full_green import (  # noqa: E402
    _find_run_id_in_dir,
    _latest_e2e_artifact,
    _load_json,
    _repo_root,
    _storage_root,
)

# Text sub-extractors write into text_processor/text_features.npz (no per-extractor NPZ in manifest).
TEXT_BUNDLED_EXTRACTORS: Set[str] = {
    "ASRTextProxyExtractor",
    "CommentsAggregationExtractor",
    "CommentsEmbedder",
    "CosineMetricsExtractor",
    "DescriptionEmbedder",
    "EmbeddingPairTopKExtractor",
    "EmbeddingShiftIndicatorExtractor",
    "EmbeddingSourceIdExtractor",
    "EmbeddingStatsExtractor",
    "HashtagEmbedder",
    "LexicalStatsExtractor",
    "QAEmbeddingPairsExtractor",
    "SemanticClusterExtractor",
    "SemanticTopicExtractor",
    "SpeakerTurnEmbeddingsAggregatorExtractor",
    "TagsExtractor",
    "TitleEmbedder",
    "TitleEmbeddingClusterEntropyExtractor",
    "TitleToHashtagCosineExtractor",
    "TopKSimilarCorpusTitlesExtractor",
    "TranscriptAggregatorExtractor",
    "TranscriptChunkEmbedder",
}

# На mock 3s без лиц/логотипов — ожидаемый empty (не ошибка качества).
MOCK_EXPECTED_EMPTY: Set[str] = {
    "action_recognition",
    "behavioral",
    "brand_semantics",
    "car_semantics",
    "core_face_landmarks",
    "core_object_detections",
    "detalize_face",
    "emotion_face",
    "frames_composition",
    "micro_emotion",
    "ocr_extractor",
    "place_semantics",
    "text_scoring",
    "emotion_diarization_extractor",
    "source_separation_extractor",
    "speech_analysis_extractor",
    "voice_quality_extractor",
}

# producer folder name may differ from manifest component name
COMPONENT_DIR_ALIASES: Dict[str, str] = {
    "face_identity": "core_face_identity",
}

CORE_IDENTITY_COMPONENTS: Set[str] = {
    "brand_semantics",
    "car_semantics",
    "content_domain",
    "face_identity",
    "place_semantics",
    "franchise_recognition",
    "core_face_identity",
}

# NPZ meta.status=empty с этими причинами — валидный контракт (L6: manifest может оставаться ok).
VALID_NPZ_EMPTY_REASONS: Set[str] = {
    "no_faces_in_video",
    "no_faces_processed",
    "embedding_service_unavailable",
    "no_logo_proposals",
    "no_car_proposals",
    "no_places_detected",
    "no_franchise_matches",
    "no_detections_above_threshold",
    "dependency_missing",
    "no_text_available",
}


@dataclass
class Finding:
    level: str  # error | warning | info
    code: str
    message: str
    component: Optional[str] = None


@dataclass
class QualityReport:
    errors: List[Finding] = field(default_factory=list)
    warnings: List[Finding] = field(default_factory=list)
    info: List[Finding] = field(default_factory=list)

    def add(self, level: str, code: str, message: str, component: Optional[str] = None) -> None:
        f = Finding(level=level, code=code, message=message, component=component)
        if level == "error":
            self.errors.append(f)
        elif level == "warning":
            self.warnings.append(f)
        else:
            self.info.append(f)

    @property
    def ok(self) -> bool:
        return not self.errors


def _meta_from_npz(npz: np.lib.npyio.NpzFile) -> Optional[Dict[str, Any]]:
    if "meta" not in npz.files:
        return None
    meta = npz["meta"]
    if isinstance(meta, np.ndarray) and meta.dtype == object:
        try:
            meta = meta.item()
        except Exception:
            return None
    return meta if isinstance(meta, dict) else None


def _finite_ratio(arr: np.ndarray) -> float:
    if not isinstance(arr, np.ndarray) or arr.size == 0:
        return 0.0
    if not np.issubdtype(arr.dtype, np.number):
        return 1.0
    return float(np.isfinite(arr).sum()) / float(arr.size)


def _find_primary_npz(run_dir: Path, component: str) -> Optional[Path]:
    folder = COMPONENT_DIR_ALIASES.get(component, component)
    base = run_dir / folder
    if not base.is_dir():
        return None
    candidates = sorted(base.glob("*.npz"))
    return candidates[0] if candidates else None


def _asr_token_count(run_dir: Path) -> int:
    npz_path = run_dir / "asr_extractor" / "asr_extractor_features.npz"
    if not npz_path.is_file():
        return 0
    try:
        z = np.load(npz_path, allow_pickle=True)
        tok = z.get("token_ids_by_segment")
        if tok is None and "payload" in z.files:
            payload = z["payload"]
            if isinstance(payload, np.ndarray) and payload.dtype == object:
                payload = payload.item()
            if isinstance(payload, dict):
                tok = payload.get("token_ids_by_segment")
        if tok is None:
            return 0
        if isinstance(tok, np.ndarray):
            if tok.dtype == object:
                tok = tok.tolist()
            elif tok.ndim == 1:
                return int(tok.size)
        if isinstance(tok, list):
            total = 0
            for x in tok:
                if isinstance(x, np.ndarray):
                    total += int(x.size)
                elif isinstance(x, (list, tuple)):
                    total += len(x)
            return total
    except Exception:
        return 0
    return 0


def _text_input_has_asr_tokens(run_dir: Path) -> bool:
    autogen = run_dir / "_tmp" / "text_input_autogen.json"
    if not autogen.is_file():
        return False
    try:
        doc = _load_json(autogen)
        tti = doc.get("transcripts_token_ids") or {}
        whisper = tti.get("whisper") if isinstance(tti, dict) else None
        if isinstance(whisper, list) and whisper:
            return True
        asr = doc.get("asr") or {}
        toks = asr.get("token_ids_by_segment") if isinstance(asr, dict) else None
        return isinstance(toks, list) and bool(toks)
    except Exception:
        return False


def _validate_npz_contract(npz_path: Path) -> Tuple[bool, List[str]]:
    repo = _repo_root()
    vp_root = repo / "DataProcessor" / "VisualProcessor"
    if str(vp_root) not in sys.path:
        sys.path.insert(0, str(vp_root))
    try:
        from utils.artifact_validator import validate_npz as _validate  # type: ignore
    except Exception as e:
        return False, [f"artifact_validator unavailable: {e}"]
    ok, issues, _ = _validate(str(npz_path), validate_schema=True, require_known_schema=False)
    return ok, [i.message for i in issues if i.level == "error"]


def validate_output_quality(
    *,
    run_dir: Path,
    mock_offline: bool = True,
    min_text_finite_ratio: float = 0.55,
    strict_manifest_contract: bool = True,
    skip_core_identity: bool = False,
) -> QualityReport:
    report = QualityReport()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        report.add("error", "manifest_missing", f"missing {manifest_path}")
        return report

    manifest = _load_json(manifest_path)
    if (manifest.get("run") or {}).get("status") != "success":
        report.add(
            "error",
            "run_status",
            f"manifest run.status={(manifest.get('run') or {}).get('status')!r}",
        )

    components = [c for c in (manifest.get("components") or []) if isinstance(c, dict)]
    by_name = {c.get("name"): c for c in components if c.get("name")}

    text_npz = run_dir / "text_processor" / "text_features.npz"
    if not text_npz.is_file():
        report.add("error", "text_features_missing", f"missing {text_npz}", "text_processor")
    else:
        try:
            z = np.load(text_npz, allow_pickle=True)
            meta = _meta_from_npz(z)
            if meta and meta.get("status") == "error":
                report.add("error", "text_meta_error", str(meta.get("error")), "text_processor")
            if "feature_values" in z.files:
                fr = _finite_ratio(z["feature_values"])
                if fr < min_text_finite_ratio:
                    report.add(
                        "error",
                        "text_low_finite",
                        f"text_features finite ratio {fr:.1%} < {min_text_finite_ratio:.0%}",
                        "text_processor",
                    )
                elif fr < 0.75:
                    report.add(
                        "warning",
                        "text_moderate_nan",
                        f"text_features finite ratio {fr:.1%} (many NaN slots expected for disabled extractors)",
                        "text_processor",
                    )
            payload = z.get("payload")
            if payload is not None:
                if isinstance(payload, np.ndarray) and payload.dtype == object:
                    payload = payload.item()
                if isinstance(payload, dict):
                    inp = payload.get("input_stats") or {}
                    if mock_offline and int(inp.get("transcript_len_chars") or 0) == 0:
                        report.add(
                            "warning",
                            "no_transcript",
                            "text payload transcript_len_chars=0 (ASR not propagated to TextProcessor on mock)",
                            "text_processor",
                        )
                    elif not mock_offline:
                        asr_st = str((by_name.get("asr_extractor") or {}).get("status") or "").lower()
                        asr_tokens = _asr_token_count(run_dir)
                        wired = _text_input_has_asr_tokens(run_dir)
                        if asr_st == "ok" and asr_tokens > 0:
                            if wired:
                                report.add(
                                    "info",
                                    "asr_wired_to_text",
                                    f"ASR autogen: {asr_tokens} token(s) in NPZ → text_input_autogen.json",
                                    "text_processor",
                                )
                            else:
                                report.add(
                                    "error",
                                    "asr_not_wired_to_text",
                                    f"asr_extractor ok with {asr_tokens} token(s) but _tmp/text_input_autogen.json missing",
                                    "text_processor",
                                )
                        elif asr_st == "ok" and asr_tokens == 0:
                            report.add(
                                "warning",
                                "asr_no_speech_tokens",
                                "asr_extractor ok but no token_ids (silent/short clip?)",
                                "asr_extractor",
                            )
        except Exception as e:
            report.add("error", "text_load_failed", str(e), "text_processor")

    for comp in components:
        name = str(comp.get("name") or "").strip()
        status = str(comp.get("status") or "").strip().lower()
        empty_reason = comp.get("empty_reason")
        artifacts = comp.get("artifacts") or []

        if not name:
            report.add("error", "component_name_missing", "manifest component without name")
            continue

        if skip_core_identity and name in CORE_IDENTITY_COMPONENTS:
            continue

        if status == "error":
            report.add("error", "component_error", comp.get("error") or "status=error", name)
            continue

        if status == "empty":
            if strict_manifest_contract and not str(empty_reason or "").strip():
                npz_path = _find_primary_npz(run_dir, name)
                npz_reason = None
                if npz_path and npz_path.is_file():
                    try:
                        z = np.load(npz_path, allow_pickle=True)
                        meta = _meta_from_npz(z)
                        npz_reason = (meta or {}).get("empty_reason")
                    except Exception:
                        pass
                if npz_reason:
                    report.add(
                        "warning",
                        "manifest_empty_reason_missing",
                        f"manifest empty_reason missing (NPZ has {npz_reason!r})",
                        name,
                    )
                else:
                    report.add("error", "empty_reason_missing", "empty component without empty_reason", name)
            if name not in MOCK_EXPECTED_EMPTY and mock_offline:
                report.add(
                    "warning",
                    "unexpected_empty",
                    f"status=empty on mock video (not in expected-empty set)",
                    name,
                )
            continue

        if status != "ok":
            report.add("error", "bad_status", f"unexpected status={status!r}", name)
            continue

        # status == ok
        if name in TEXT_BUNDLED_EXTRACTORS:
            if not text_npz.is_file():
                report.add("error", "bundled_text_missing", "text_features.npz required for bundled extractors", name)
            continue

        npz_path = _find_primary_npz(run_dir, name)
        manifest_npz = None
        for a in artifacts:
            if isinstance(a, dict) and str(a.get("type", "")).lower() == "npz":
                rel = a.get("path")
                if rel:
                    manifest_npz = run_dir / rel
                    break

        effective_npz = manifest_npz if manifest_npz and manifest_npz.is_file() else npz_path
        if effective_npz is None or not effective_npz.is_file():
            report.add("error", "ok_without_npz", "status=ok but no NPZ on disk", name)
            continue

        try:
            z = np.load(effective_npz, allow_pickle=True)
        except Exception as e:
            report.add("error", "npz_load_failed", str(e), name)
            continue

        meta = _meta_from_npz(z)
        meta_status = str((meta or {}).get("status") or "ok").lower()
        if meta_status == "error":
            report.add("error", "npz_meta_error", str((meta or {}).get("error")), name)
            continue
        if meta_status == "empty":
            empty_reason_npz = str((meta or {}).get("empty_reason") or "").strip()
            if empty_reason_npz in VALID_NPZ_EMPTY_REASONS:
                report.add(
                    "warning",
                    "manifest_ok_npz_valid_empty",
                    f"manifest ok but NPZ meta.status=empty ({empty_reason_npz}) — sync L6",
                    name,
                )
            else:
                report.add(
                    "error",
                    "ok_manifest_empty_npz",
                    f"manifest ok but NPZ meta.status=empty ({empty_reason_npz or 'no reason'})",
                    name,
                )
            continue

        ok_contract, contract_errs = _validate_npz_contract(effective_npz)
        if not ok_contract:
            for msg in contract_errs[:5]:
                report.add("error", "npz_contract", msg, name)
            if len(contract_errs) > 5:
                report.add(
                    "error",
                    "npz_contract_more",
                    f"... and {len(contract_errs) - 5} more contract errors",
                    name,
                )

        if "feature_values" in z.files:
            fr = _finite_ratio(z["feature_values"])
            # similarity_metrics uses NaN for optional/missing modalities by design.
            low_finite_is_error = fr < 0.5 and name not in {"similarity_metrics"}
            if low_finite_is_error:
                report.add(
                    "error",
                    "low_finite_features",
                    f"feature_values finite ratio {fr:.1%} (<50%) for status=ok",
                    name,
                )
            elif fr < 0.5:
                report.add(
                    "warning",
                    "sparse_features",
                    f"feature_values finite ratio {fr:.1%} (expected for {name})",
                    name,
                )

    # Mock offline signal checks (tone + title/description fixtures)
    if mock_offline:
        _check_mock_signals(run_dir, report)

    return report


def _check_mock_signals(run_dir: Path, report: QualityReport) -> None:
    clap = run_dir / "clap_extractor" / "clap_extractor_features.npz"
    if clap.is_file():
        z = np.load(clap, allow_pickle=True)
        emb = z.get("embedding")
        if isinstance(emb, np.ndarray) and emb.size:
            std = float(np.std(emb[np.isfinite(emb)])) if np.isfinite(emb).any() else 0.0
            if std < 0.005:
                report.add("error", "clap_degenerate", f"CLAP embedding std={std:.5f} (near-constant)", "clap_extractor")
            else:
                report.add("info", "clap_ok", f"CLAP embedding std={std:.4f}", "clap_extractor")

    tempo = run_dir / "tempo_extractor" / "tempo_extractor_features.npz"
    if tempo.is_file():
        z = np.load(tempo, allow_pickle=True)
        names = list(z.get("feature_names", []))
        vals = z.get("feature_values")
        if vals is not None and len(names) == len(vals):
            d = dict(zip([str(n) for n in names], vals))
            bpm = float(d.get("tempo_bpm", float("nan")))
            if not (40 <= bpm <= 220):
                report.add("warning", "tempo_out_of_range", f"tempo_bpm={bpm}", "tempo_extractor")
            else:
                report.add("info", "tempo_ok", f"tempo_bpm={bpm:.1f}", "tempo_extractor")

    asr = run_dir / "asr_extractor" / "asr_extractor_features.npz"
    if asr.is_file():
        z = np.load(asr, allow_pickle=True)
        tok = z.get("token_counts")
        n_tok = int(tok[0]) if isinstance(tok, np.ndarray) and tok.size else 0
        if n_tok < 5:
            report.add(
                "warning",
                "asr_short",
                f"ASR token_counts={n_tok} on 440Hz tone mock (transcript not meaningful)",
                "asr_extractor",
            )

    spk = run_dir / "speaker_diarization_extractor" / "speaker_diarization_extractor_features.npz"
    if spk.is_file():
        z = np.load(spk, allow_pickle=True)
        names = list(z.get("feature_names", []))
        vals = z.get("feature_values")
        if vals is not None and len(names) == len(vals):
            d = dict(zip([str(n) for n in names], vals))
            sc = int(d.get("speaker_count", 0))
            if sc == 0:
                report.add(
                    "warning",
                    "no_speakers",
                    "speaker_diarization ok but speaker_count=0 (expected on non-speech tone)",
                    "speaker_diarization_extractor",
                )

    clip = run_dir / "core_clip" / "embeddings.npz"
    if clip.is_file():
        z = np.load(clip, allow_pickle=True)
        emb = z.get("embeddings")
        if isinstance(emb, np.ndarray) and emb.size:
            fr = _finite_ratio(emb)
            if fr < 0.9:
                report.add("error", "clip_degenerate", f"core_clip finite ratio {fr:.1%}", "core_clip")
            else:
                report.add("info", "clip_ok", f"core_clip embeddings finite {fr:.1%}", "core_clip")


def _print_report(report: QualityReport) -> None:
    def _block(title: str, items: Sequence[Finding]) -> None:
        if not items:
            return
        print(f"\n{title} ({len(items)}):")
        for f in items:
            where = f" [{f.component}]" if f.component else ""
            print(f"  [{f.level.upper()}:{f.code}]{where} {f.message}")

    print("=== E2E output quality (§0.2) ===")
    _block("ERRORS", report.errors)
    _block("WARNINGS", report.warnings)
    _block("INFO", report.info)
    print()
    if report.ok:
        if report.warnings:
            print(f"PASS with {len(report.warnings)} warning(s): output quality §0.2")
        else:
            print("PASS: output quality §0.2")
    else:
        print(f"FAIL: output quality §0.2 ({len(report.errors)} error(s), {len(report.warnings)} warning(s))")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate E2E output quality (§0.2)")
    ap.add_argument("--run-id")
    ap.add_argument("--platform-id", default="youtube")
    ap.add_argument("--video-id", default="-Q6fnPIybEI")
    ap.add_argument("--latest-e2e-artifact", action="store_true")
    ap.add_argument("--e2e-artifact-dir", type=Path, default=None)
    ap.add_argument("--no-mock-offline", action="store_true", help="Disable mock-video signal checks")
    ap.add_argument("--real-video", action="store_true", help="Alias: real HF/local video (no mock tone checks)")
    ap.add_argument(
        "--skip-core-identity",
        action="store_true",
        help="Ignore core_identity semantic heads in quality checks",
    )
    ap.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = ap.parse_args()

    repo = _repo_root()
    if args.latest_e2e_artifact:
        artifact_dir = args.e2e_artifact_dir or _latest_e2e_artifact(repo)
        run_id = args.run_id or _find_run_id_in_dir(artifact_dir)
        if not run_id:
            print(f"FAIL: could not find run_id in {artifact_dir}", file=sys.stderr)
            return 2
    else:
        run_id = args.run_id
        if not run_id:
            print("FAIL: --run-id or --latest-e2e-artifact required", file=sys.stderr)
            return 2

    run_dir = _storage_root(repo) / "result_store" / args.platform_id / args.video_id / run_id
    if not run_dir.is_dir():
        print(f"FAIL: run dir not found: {run_dir}", file=sys.stderr)
        return 2

    print(f"Quality check run_id={run_id} dir={run_dir}")
    report = validate_output_quality(
        run_dir=run_dir,
        mock_offline=not (args.no_mock_offline or args.real_video),
        skip_core_identity=bool(args.skip_core_identity),
    )
    _print_report(report)

    if args.strict and report.warnings:
        return 1
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
