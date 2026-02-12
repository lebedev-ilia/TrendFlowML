from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import unicodedata
import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.schemas.models import VideoDocument
from src.core.text_utils import normalize_whitespace


def _tokenize(text: str) -> List[str]:
    t = normalize_whitespace(text or "")
    if not t:
        return []
    return [tok for tok in t.split() if tok]


class ASRTextProxyExtractor(BaseExtractor):
    VERSION = "1.1.0"

    def __init__(
        self,
        *,
        enabled: bool = True,
        enable_basic: bool = True,
        enable_noise: bool = True,
        enable_rhythm: bool = True,
        enable_intonation: bool = True,
        low_conf_threshold: float = 0.5,
        words_per_minute_baseline: float = 160.0,
        max_text_chars: int = 200_000,
    ) -> None:
        """
        Production policy:
        - ASR transcript is expected to come from AudioProcessor as structured payload (doc.asr).
        - If transcript is missing, this extractor produces valid-empty (NaN + masks), not heuristic fallbacks.
        - audio_duration_sec must be present (Segmenter/AudioProcessor contract); if missing -> fail-fast.
        """
        self.enabled = bool(enabled)
        self.enable_basic = bool(enable_basic)
        self.enable_noise = bool(enable_noise)
        self.enable_rhythm = bool(enable_rhythm)
        self.enable_intonation = bool(enable_intonation)
        self.low_conf_threshold = float(low_conf_threshold)
        self.words_per_minute_baseline = float(words_per_minute_baseline)
        self.max_text_chars = int(max(0, max_text_chars))

    def _extract_asr_payload(self, doc: VideoDocument) -> Tuple[List[Dict[str, Any]], Optional[float]]:
        """
        Returns: (segments, total_audio_duration_sec_from_payload)
        segments item schema (best-effort):
          - text: str
          - confidence: float|None
          - start_sec/end_sec: float|None
        """
        # Preferred: doc.asr (AudioProcessor-owned)
        asr = getattr(doc, "asr", None)
        if isinstance(asr, dict):
            segs = asr.get("segments") or []
            if isinstance(segs, list):
                items = [x for x in segs if isinstance(x, dict)]
                dur = asr.get("total_audio_duration_sec")
                if dur is None:
                    dur = asr.get("total_audio_duration")
                return items, (float(dur) if dur is not None else None)

        # Legacy alias: doc.transcripts_meta (deprecated; will be removed after AudioProcessor audit)
        tm = getattr(doc, "transcripts_meta", None)
        if isinstance(tm, dict):
            segs = tm.get("with_confidence") or []
            if isinstance(segs, list):
                items = [x for x in segs if isinstance(x, dict)]
                dur = tm.get("total_audio_duration")
                return items, (float(dur) if dur is not None else None)

        return [], None

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        # Stable feature schema (keys always present)
        def _stable_template() -> Dict[str, float]:
            return {
                "tp_asrproxy_present": 0.0,
                "tp_asrproxy_has_confidence": 0.0,
                "tp_asrproxy_enabled": float(bool(self.enabled)),
                "tp_asrproxy_basic_enabled": float(bool(self.enable_basic)),
                "tp_asrproxy_noise_enabled": float(bool(self.enable_noise)),
                "tp_asrproxy_rhythm_enabled": float(bool(self.enable_rhythm)),
                "tp_asrproxy_intonation_enabled": float(bool(self.enable_intonation)),
                "tp_asrproxy_low_conf_threshold": float(self.low_conf_threshold),
                "tp_asrproxy_words_per_minute_baseline": float(self.words_per_minute_baseline),
                "tp_asrproxy_max_text_chars": float(int(self.max_text_chars)),
                "tp_asrproxy_text_truncated_flag": 0.0,
                "tp_asrproxy_asr_schema_invalid_flag": 0.0,
                "tp_asrproxy_conf_invalid_flag": 0.0,
                "tp_asrproxy_duration_from_payload_flag": 0.0,
                "tp_asrproxy_duration_invalid_flag": 0.0,
                "tp_asrproxy_segments_count": 0.0,
                "tp_asrproxy_text_chars": 0.0,
                "tp_asrproxy_word_count": 0.0,
                "tp_asrproxy_confidence_present_rate": float("nan"),
                # required meta
                "tp_asrproxy_audio_duration_sec": float("nan"),
                # confidence
                "tp_asrproxy_confidence_mean": float("nan"),
                "tp_asrproxy_confidence_std": float("nan"),
                "tp_asrproxy_confidence_chunked_min": float("nan"),
                "tp_asrproxy_low_conf_rate": float("nan"),
                # noise
                "tp_asrproxy_text_noise_rare_ratio": float("nan"),
                "tp_asrproxy_text_noise_oov_ratio": float("nan"),
                "tp_asrproxy_noise_proxy": float("nan"),
                "tp_asrproxy_noise_proxy_present": 0.0,
                # rhythm
                "tp_asrproxy_speech_rate_wpm": float("nan"),
                "tp_asrproxy_speech_char_density": float("nan"),
                "tp_asrproxy_pause_density": float("nan"),
                "tp_asrproxy_filler_ratio": float("nan"),
                # intonation
                "tp_asrproxy_sentence_intonation": float("nan"),
            }

        features_flat = _stable_template()

        if not self.enabled:
            # valid empty by policy (duration may still be required, but do not force evaluation here)
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
                "result": {"asr_text_proxy": {"metrics": features_flat}, "features_flat": features_flat},
                "error": None,
            }

        # Contract: audio duration must exist (Segmenter extracts audio; AudioProcessor provides duration to TextProcessor).
        duration_sec = getattr(doc, "audio_duration_sec", None)
        items, payload_duration = self._extract_asr_payload(doc)
        if duration_sec is None and payload_duration is not None:
            duration_sec = payload_duration
            features_flat["tp_asrproxy_duration_from_payload_flag"] = 1.0
        if duration_sec is None:
            raise RuntimeError("ASRTextProxyExtractor requires audio_duration_sec (missing in VideoDocument)")
        duration_sec = float(duration_sec)
        if not (duration_sec > 0.0):
            features_flat["tp_asrproxy_duration_invalid_flag"] = 1.0
            raise RuntimeError(f"ASRTextProxyExtractor invalid audio_duration_sec={duration_sec}")
        features_flat["tp_asrproxy_audio_duration_sec"] = float(duration_sec)

        # Best-effort ASR schema checks (no PII)
        # segments_count: number of dict segments, even if text is empty
        features_flat["tp_asrproxy_segments_count"] = float(int(len(items)))
        if not all(isinstance(s, dict) for s in items):
            features_flat["tp_asrproxy_asr_schema_invalid_flag"] = 1.0

        texts = [normalize_whitespace(str(s.get("text", ""))) for s in items]
        full_text = " ".join(texts).strip()
        if self.max_text_chars and len(full_text) > self.max_text_chars:
            full_text = full_text[: self.max_text_chars].rstrip()
            features_flat["tp_asrproxy_text_truncated_flag"] = 1.0

        # Confidence extraction + validation
        confidences_raw = [s.get("confidence") for s in items if isinstance(s, dict)]
        n_conf_present = 0
        confidences: List[float] = []
        for c in confidences_raw:
            if c is None:
                continue
            n_conf_present += 1
            try:
                cf = float(c)
            except Exception:
                features_flat["tp_asrproxy_conf_invalid_flag"] = 1.0
                continue
            if not (0.0 <= cf <= 1.0):
                features_flat["tp_asrproxy_conf_invalid_flag"] = 1.0
                continue
            confidences.append(cf)

        duration_min = max(1e-6, duration_sec / 60.0)
        has_transcript = bool(full_text)
        has_confidence = bool(confidences)
        features_flat["tp_asrproxy_present"] = 1.0 if has_transcript else 0.0
        features_flat["tp_asrproxy_has_confidence"] = 1.0 if has_confidence else 0.0
        features_flat["tp_asrproxy_text_chars"] = float(int(len(full_text)))

        # 1) ASR confidence
        if confidences and self.enable_basic:
            asr_conf_mean = float(np.mean(confidences))
            asr_conf_std = float(np.std(confidences))
            # chunked means in blocks of ~10 segments (stable, bounded)
            block = max(1, int(round(len(confidences) / 10.0)))
            chunk_means = [float(np.mean(confidences[i : i + block])) for i in range(0, len(confidences), block)]
            asr_conf_chunked_min = float(np.min(chunk_means)) if chunk_means else asr_conf_mean
        else:
            asr_conf_mean = float("nan")
            asr_conf_std = float("nan")
            asr_conf_chunked_min = float("nan")

        # Tokens
        tokens = _tokenize(full_text)
        total_words = len(tokens)
        total_chars = len(full_text)
        features_flat["tp_asrproxy_word_count"] = float(int(total_words))
        if int(len(items)) > 0:
            features_flat["tp_asrproxy_confidence_present_rate"] = float(n_conf_present / max(1, int(len(items))))

        # 2) Errors and rarity proxies
        # rare word: length > 12 or contains digits/symbols heavily
        def _is_rare(tok: str) -> bool:
            if len(tok) > 12:
                return True
            has_digit = any(ch.isdigit() for ch in tok)
            sym_ratio = sum(1 for ch in tok if not (unicodedata.category(ch)[0] in ("L", "M") or ch.isdigit())) / max(1, len(tok))
            return has_digit or sym_ratio > 0.4

        rare_word_ratio = float(sum(1 for t in tokens if _is_rare(t)) / max(1, total_words)) if (tokens and self.enable_noise) else float("nan")
        low_conf_rate = (
            float(sum(1 for c in confidences if c < self.low_conf_threshold) / max(1, len(confidences)))
            if (confidences and self.enable_basic)
            else float("nan")
        )
        # Keep as "noise proxy" (not a claim of WER/ASR error).
        # Avoid fake "0.0" when both components are missing.
        if not (self.enable_noise or self.enable_basic):
            noise_proxy = float("nan")
            noise_proxy_present = False
        else:
            have_rr = bool(np.isfinite(rare_word_ratio))
            have_lc = bool(np.isfinite(low_conf_rate))
            if not (have_rr or have_lc):
                noise_proxy = float("nan")
                noise_proxy_present = False
            else:
                rr = float(rare_word_ratio) if have_rr else float("nan")
                lc = float(low_conf_rate) if have_lc else float("nan")
                # average only over present terms
                vals = [v for v in [rr, lc] if np.isfinite(v)]
                noise_proxy = float(min(1.0, float(np.mean(np.asarray(vals, dtype=np.float32))))) if vals else float("nan")
                noise_proxy_present = bool(noise_proxy == noise_proxy)

        # oov proxy: tokens with many non-letter marks
        def _is_oov(tok: str) -> bool:
            letters = sum(1 for ch in tok if unicodedata.category(ch)[0] in ("L", "M"))
            return letters < max(1, len(tok) // 2)

        oov_rate_asr_tokens = float(sum(1 for t in tokens if _is_oov(t)) / max(1, total_words)) if (tokens and self.enable_noise) else float("nan")

        # 3) Speech rhythm
        if has_transcript and self.enable_rhythm:
            speech_rate_wpm = float(total_words / duration_min)
            speech_character_density = float(total_chars / max(1e-6, duration_sec))
            # pauses approximation: comma/semicolon/colon per sentence; sentences by .?!
            n_sent = max(1, full_text.count(".") + full_text.count("?") + full_text.count("!"))
            pauses = full_text.count(",") + full_text.count(";") + full_text.count(":")
            pause_density_proxy = float(pauses / n_sent)
            filler_lexicon = {"ээ", "мм", "ну", "типа", "короче", "значит", "э", "эээ", "mmm", "uh", "um"}
            filler_word_ratio = float(sum(1 for w in tokens if w.lower() in filler_lexicon) / max(1, total_words))
        else:
            speech_rate_wpm = float("nan")
            speech_character_density = float("nan")
            pause_density_proxy = float("nan")
            filler_word_ratio = float("nan")

        # 4) Structure and emotions
        if has_transcript and self.enable_intonation:
            n_sent = max(1, full_text.count(".") + full_text.count("?") + full_text.count("!"))
            sentence_intonation_proxy = float((full_text.count("!") + full_text.count("?")) / n_sent)
        else:
            sentence_intonation_proxy = float("nan")

        # Fill stable features (numeric scalars only).
        features_flat["tp_asrproxy_confidence_mean"] = float(asr_conf_mean)
        features_flat["tp_asrproxy_confidence_std"] = float(asr_conf_std)
        features_flat["tp_asrproxy_confidence_chunked_min"] = float(asr_conf_chunked_min)
        features_flat["tp_asrproxy_low_conf_rate"] = float(low_conf_rate)
        features_flat["tp_asrproxy_text_noise_rare_ratio"] = float(rare_word_ratio)
        features_flat["tp_asrproxy_text_noise_oov_ratio"] = float(oov_rate_asr_tokens)
        features_flat["tp_asrproxy_noise_proxy"] = float(noise_proxy)
        features_flat["tp_asrproxy_noise_proxy_present"] = 1.0 if noise_proxy_present else 0.0
        features_flat["tp_asrproxy_speech_rate_wpm"] = float(speech_rate_wpm)
        features_flat["tp_asrproxy_speech_char_density"] = float(speech_character_density)
        features_flat["tp_asrproxy_pause_density"] = float(pause_density_proxy)
        features_flat["tp_asrproxy_filler_ratio"] = float(filler_word_ratio)
        features_flat["tp_asrproxy_sentence_intonation"] = float(sentence_intonation_proxy)

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
            "result": {
                # Human/debug grouping (safe, numeric only). Downstream should use features_flat.
                "asr_text_proxy": {"metrics": features_flat},
                "features_flat": features_flat,
            },
            "error": None,
        }


