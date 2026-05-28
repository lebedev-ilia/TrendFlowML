#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple


def load_and_validate_segments(
    frames_dir: Optional[str],
    extractor_keys: List[str],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Load and validate segments.json from frames_dir.
    
    Args:
        frames_dir: Path to frames directory (or None)
        extractor_keys: List of extractor keys to validate
        
    Returns:
        Tuple of (audio_path, segments_payload)
        - audio_path: Path to audio.wav file (or None)
        - segments_payload: Parsed segments.json content (or None)
        
    Raises:
        RuntimeError: If required files are missing or invalid
    """
    if not frames_dir:
        return None, None
    
    frames_dir = os.path.abspath(frames_dir)
    audio_path = os.path.join(frames_dir, "audio", "audio.wav")
    segments_json = os.path.join(frames_dir, "audio", "segments.json")
    
    if not os.path.exists(segments_json):
        raise RuntimeError(f"AudioProcessor | missing required segments.json: {segments_json}")
    
    with open(segments_json, "r", encoding="utf-8") as f:
        segments_payload = json.load(f) or {}

    if str(segments_payload.get("schema_version")) != "audio_segments_v1":
        raise RuntimeError(f"AudioProcessor | unsupported segments.json schema: {segments_payload.get('schema_version')}")

    # Valid empty: video has no audio stream. Segmenter writes segments.json with audio_present=false.
    # In this case AudioProcessor must NOT require audio.wav and must NOT validate segment families.
    if segments_payload.get("audio_present") is False:
        return None, segments_payload

    if not os.path.exists(audio_path):
        raise RuntimeError(f"AudioProcessor | missing required audio file: {audio_path}")

    families = segments_payload.get("families") or {}
    primary = ((families.get("primary") or {}) if isinstance(families, dict) else {}) or {}
    clap_f = ((families.get("clap") or {}) if isinstance(families, dict) else {}) or {}
    tempo_f = ((families.get("tempo") or {}) if isinstance(families, dict) else {}) or {}
    asr_f = ((families.get("asr") or {}) if isinstance(families, dict) else {}) or {}
    diar_f = ((families.get("diarization") or {}) if isinstance(families, dict) else {}) or {}
    emo_f = ((families.get("emotion") or {}) if isinstance(families, dict) else {}) or {}
    sep_f = ((families.get("source_separation") or {}) if isinstance(families, dict) else {}) or {}
    # speech_analysis uses both ASR and diarization families
    primary_segments = primary.get("segments") or []
    clap_segments = clap_f.get("segments") or []
    tempo_segments = tempo_f.get("segments") or []
    asr_segments = asr_f.get("segments") or []
    diar_segments = diar_f.get("segments") or []
    emo_segments = emo_f.get("segments") or []
    sep_segments = sep_f.get("segments") or []
    # NOTE(Audit v3): unified sampling policy. Some extractors intentionally share a family.
    # We treat `spectral` as the required family for:
    # - pitch
    # - band_energy
    # - spectral_entropy
    #
    # This is NOT a runtime fallback: it is the declared sampling requirement.
    # Segmenter remains the single owner of sampling (Segmenter-only policy).
    pitch_segments: List[Any] = []
    spectral_f = families.get("spectral", {})
    spectral_segments = spectral_f.get("segments") or []
    quality_f = families.get("quality", {})
    quality_segments = quality_f.get("segments") or []
    mfcc_f = families.get("mfcc", {})
    mfcc_segments = mfcc_f.get("segments") or []
    mel_f = families.get("mel", {})
    mel_segments = mel_f.get("segments") or []
    onset_f = families.get("onset", {})
    onset_segments = onset_f.get("segments") or []
    chroma_f = families.get("chroma", {})
    chroma_segments = chroma_f.get("segments") or []
    rhythmic_f = families.get("rhythmic", {})
    rhythmic_segments = rhythmic_f.get("segments") or []
    voice_quality_f = families.get("voice_quality", {})
    voice_quality_segments = voice_quality_f.get("segments") or []
    hpss_f = families.get("hpss", {})
    hpss_segments = hpss_f.get("segments") or []
    key_f = families.get("key", {})
    key_segments = key_f.get("segments") or []
    # Shared-family extractors (Audit v3): require spectral family
    band_energy_segments = spectral_segments
    spectral_entropy_segments = spectral_segments
    pitch_segments = spectral_segments
    # Validate required segments for each extractor
    if ("loudness" in extractor_keys) and (not isinstance(primary_segments, list) or not primary_segments):
        raise RuntimeError("AudioProcessor | segments.json missing families.primary.segments (no-fallback)")
    if ("clap" in extractor_keys) and (not isinstance(clap_segments, list) or not clap_segments):
        raise RuntimeError("AudioProcessor | segments.json missing families.clap.segments (no-fallback)")
    if ("tempo" in extractor_keys) and (not isinstance(tempo_segments, list) or not tempo_segments):
        raise RuntimeError("AudioProcessor | segments.json missing families.tempo.segments (no-fallback)")
    if ("asr" in extractor_keys) and (not isinstance(asr_segments, list) or not asr_segments):
        raise RuntimeError("AudioProcessor | segments.json missing families.asr.segments (no-fallback)")
    if ("speaker_diarization" in extractor_keys) and (not isinstance(diar_segments, list) or not diar_segments):
        raise RuntimeError("AudioProcessor | segments.json missing families.diarization.segments (no-fallback)")
    if ("emotion_diarization" in extractor_keys) and (not isinstance(emo_segments, list) or not emo_segments):
        raise RuntimeError("AudioProcessor | segments.json missing families.emotion.segments (no-fallback)")
    if ("source_separation" in extractor_keys) and (not isinstance(sep_segments, list) or not sep_segments):
        raise RuntimeError("AudioProcessor | segments.json missing families.source_separation.segments (no-fallback)")
    # Shared-family extractors must have spectral segments present.
    if (("pitch" in extractor_keys) or ("band_energy" in extractor_keys) or ("spectral_entropy" in extractor_keys)) and (
        not isinstance(spectral_segments, list) or not spectral_segments
    ):
        raise RuntimeError("AudioProcessor | segments.json missing families.spectral.segments (required for pitch/band_energy/spectral_entropy)")
    if ("spectral" in extractor_keys) and (not isinstance(spectral_segments, list) or not spectral_segments):
        raise RuntimeError("AudioProcessor | segments.json missing families.spectral.segments (no-fallback)")
    if ("quality" in extractor_keys) and (not isinstance(quality_segments, list) or not quality_segments):
        raise RuntimeError("AudioProcessor | segments.json missing families.quality.segments (no-fallback)")
    if ("mfcc" in extractor_keys) and (not isinstance(mfcc_segments, list) or not mfcc_segments):
        raise RuntimeError("AudioProcessor | segments.json missing families.mfcc.segments (no-fallback)")
    if ("mel" in extractor_keys) and (not isinstance(mel_segments, list) or not mel_segments):
        raise RuntimeError("AudioProcessor | segments.json missing families.mel.segments (no-fallback)")
    if ("onset" in extractor_keys) and (not isinstance(onset_segments, list) or not onset_segments):
        raise RuntimeError("AudioProcessor | segments.json missing families.onset.segments (no-fallback)")
    if ("chroma" in extractor_keys) and (not isinstance(chroma_segments, list) or not chroma_segments):
        raise RuntimeError("AudioProcessor | segments.json missing families.chroma.segments (no-fallback)")
    # Audit v3 (rhythmic): unified sampling policy — rhythmic_extractor requires `families.tempo`.
    # Migration support: accept legacy `families.rhythmic` if tempo is absent (explicitly documented).
    if "rhythmic" in extractor_keys:
        tempo_ok = isinstance(tempo_segments, list) and bool(tempo_segments)
        rhythmic_ok = isinstance(rhythmic_segments, list) and bool(rhythmic_segments)
        if not tempo_ok and not rhythmic_ok:
            raise RuntimeError("AudioProcessor | segments.json missing families.tempo.segments (required for rhythmic_extractor)")
    if ("voice_quality" in extractor_keys) and (not isinstance(voice_quality_segments, list) or not voice_quality_segments):
        raise RuntimeError("AudioProcessor | segments.json missing families.voice_quality.segments (no-fallback)")
    if ("hpss" in extractor_keys) and (not isinstance(hpss_segments, list) or not hpss_segments):
        raise RuntimeError("AudioProcessor | segments.json missing families.hpss.segments (no-fallback)")
    if ("key" in extractor_keys) and (not isinstance(key_segments, list) or not key_segments):
        raise RuntimeError("AudioProcessor | segments.json missing families.key.segments (no-fallback)")
    # NOTE: band_energy/spectral_entropy are shared-family extractors validated above via spectral family.
    
    return audio_path, segments_payload

