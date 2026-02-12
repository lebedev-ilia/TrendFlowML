from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Tuple
import unicodedata

import numpy as np
try:
    import emoji as _emoji  # type: ignore
except Exception:
    _emoji = None  # type: ignore

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.schemas.models import VideoDocument
from src.core.text_utils import normalize_whitespace


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<!\w)@([A-Za-z0-9_\.]+)")
_TIMESTAMP_RE = re.compile(r"\b(?:\d{1,2}:){1,2}\d{2}\b")  # 01:23 or 1:02:03
_QUESTION_PREFIX_RE = re.compile(r"^(кто|что|где|когда|почему|зачем|как|who|what|where|when|why|how)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+")
_DATE_TIME_RE = re.compile(r"\b(\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?|\d{1,2}:\d{2})\b")
# Python's re doesn't support \p{..}. Use Unicode categories instead.
def _is_punct_symbol(ch: str) -> bool:
    try:
        cat = unicodedata.category(ch)
        # P* → punctuation, S* → symbols
        return len(cat) > 0 and cat[0] in ("P", "S")
    except Exception:
        return False

# Production policy:
# - This extractor must not use spaCy/langdetect directly (no-network + ModelManager packaging).
# - If language detection / POS / NER are needed, implement as a separate extractor backed by dp_models.


def _tokenize(text: str) -> List[str]:
    text = normalize_whitespace(text or "")
    if not text:
        return []
    return re.findall(r"\w+", text, flags=re.UNICODE)


def _sentences(text: str) -> List[str]:
    text = normalize_whitespace(text or "")
    if not text:
        return []
    parts = re.split(r"[.!?]+\s+", text)
    return [p for p in parts if p]


def _truncate_text(text: str, *, max_chars: Optional[int]) -> Tuple[str, int, int, bool]:
    s = text or ""
    used = len(s)
    if max_chars is None:
        return s, used, used, False
    try:
        m = int(max_chars)
    except Exception:
        m = -1
    if m <= 0:
        # Treat non-positive max_chars as "disable text" (hard stop)
        return "", used, 0, bool(s)
    if len(s) <= m:
        return s, used, len(s), False
    return s[:m], used, m, True


class LexicalStatsExtractor(BaseExtractor):
    VERSION = "1.1.0"

    def __init__(
        self,
        *,
        enabled: bool = True,
        enable_title: bool = True,
        enable_description: bool = True,
        enable_transcript: bool = True,
        enable_emoji: bool = False,
        enable_clickbait_heuristic: bool = True,
        transcript_source_policy: Literal["asr_only", "asr_then_legacy", "legacy_only"] = "asr_only",
        allow_legacy_transcripts: bool = False,  # legacy alias (deprecated)
        emoji_policy: Literal["required", "optional"] = "required",
        max_title_chars: Optional[int] = None,
        max_description_chars: Optional[int] = None,
        max_transcript_chars: Optional[int] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.enable_title = bool(enable_title)
        self.enable_description = bool(enable_description)
        self.enable_transcript = bool(enable_transcript)
        self.enable_emoji = bool(enable_emoji)
        self.enable_clickbait_heuristic = bool(enable_clickbait_heuristic)
        self.emoji_policy = str(emoji_policy)
        self.max_title_chars = max_title_chars
        self.max_description_chars = max_description_chars
        self.max_transcript_chars = max_transcript_chars

        # Backwards compatibility: allow_legacy_transcripts implies asr_then_legacy (unless user explicitly sets legacy_only).
        tsp = str(transcript_source_policy)
        if allow_legacy_transcripts and tsp == "asr_only":
            tsp = "asr_then_legacy"
        if tsp not in ("asr_only", "asr_then_legacy", "legacy_only"):
            raise RuntimeError(f"LexicalStatsExtractor: unknown transcript_source_policy='{tsp}'")
        self.transcript_source_policy = tsp
        self.allow_legacy_transcripts = bool(allow_legacy_transcripts)

        if self.enable_emoji and _emoji is None and self.emoji_policy == "required":
            raise RuntimeError("LexicalStatsExtractor: enable_emoji=true but 'emoji' package is not installed (emoji_policy=required)")

    @staticmethod
    def _join_asr_text(doc: VideoDocument) -> str:
        asr = getattr(doc, "asr", None)
        if isinstance(asr, dict):
            segs = asr.get("segments") or []
            if isinstance(segs, list):
                parts: List[str] = []
                for s in segs:
                    if isinstance(s, dict):
                        t = normalize_whitespace(s.get("text"))
                        if t:
                            parts.append(t)
                return " ".join(parts).strip()
        return ""

    @staticmethod
    def _join_legacy_transcripts(doc: VideoDocument) -> str:
        transcripts = getattr(doc, "transcripts", {}) or {}
        if not isinstance(transcripts, dict):
            return ""
        return " ".join([str(transcripts.get(k, "")) for k in ("whisper", "youtube_auto") if transcripts.get(k)]).strip()

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        # Stable schema: always return these keys, even if disabled/empty.
        # Use NaN for "valid empty" where inputs are absent or group disabled.
        if not self.enabled:
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            features_flat: Dict[str, Any] = {
                "tp_lex_enabled": 0.0,
                "tp_lex_disabled_by_policy": 1.0,
                "tp_lex_present_title": 0.0,
                "tp_lex_present_description": 0.0,
                "tp_lex_present_transcript": 0.0,
                "tp_lex_present_any": 0.0,
                "tp_lex_group_title_enabled": float(self.enable_title),
                "tp_lex_group_description_enabled": float(self.enable_description),
                "tp_lex_group_transcript_enabled": float(self.enable_transcript),
                "tp_lex_group_emoji_enabled": float(self.enable_emoji),
                "tp_lex_group_clickbait_enabled": float(self.enable_clickbait_heuristic),
                "tp_lex_has_emoji_lib": float(_emoji is not None),
                "tp_lex_emoji_dependency_missing_flag": float(self.enable_emoji and _emoji is None),
                "tp_lex_transcript_source_policy_asr_only": float(self.transcript_source_policy == "asr_only"),
                "tp_lex_transcript_source_policy_asr_then_legacy": float(self.transcript_source_policy == "asr_then_legacy"),
                "tp_lex_transcript_source_policy_legacy_only": float(self.transcript_source_policy == "legacy_only"),
                "tp_lex_transcript_source_used_asr": 0.0,
                "tp_lex_transcript_source_used_legacy": 0.0,
                "tp_lex_transcript_source_used_none": 1.0,
                "tp_lex_allow_legacy_transcripts": float(self.allow_legacy_transcripts),
                "tp_lex_title_chars_used": float("nan"),
                "tp_lex_title_chars_kept": float("nan"),
                "tp_lex_title_truncated_flag": float("nan"),
                "tp_lex_description_chars_used": float("nan"),
                "tp_lex_description_chars_kept": float("nan"),
                "tp_lex_description_truncated_flag": float("nan"),
                "tp_lex_transcript_chars_used": float("nan"),
                "tp_lex_transcript_chars_kept": float("nan"),
                "tp_lex_transcript_truncated_flag": float("nan"),
                "tp_lex_load_ms": 0.0,
                "tp_lex_compute_ms": float(round(total_s * 1000.0, 3)),
                # Title
                "tp_lex_title_len_words": float("nan"),
                "tp_lex_title_len_chars": float("nan"),
                "tp_lex_title_avg_word_len": float("nan"),
                "tp_lex_title_exclamation_count": float("nan"),
                "tp_lex_title_question_count": float("nan"),
                "tp_lex_title_emoji_count": float("nan"),
                "tp_lex_title_stopword_ratio": float("nan"),
                "tp_lex_title_type_token_ratio": float("nan"),
                "tp_lex_title_punctuation_ratio": float("nan"),
                "tp_lex_title_capital_words_ratio": float("nan"),
                "tp_lex_title_question_prefix_flag": float("nan"),
                "tp_lex_title_number_presence": float("nan"),
                "tp_lex_title_time_mention_flag": float("nan"),
                "tp_lex_title_clickbait_score": float("nan"),
                # Description
                "tp_lex_description_len_words": float("nan"),
                "tp_lex_description_num_urls": float("nan"),
                "tp_lex_description_num_mentions": float("nan"),
                "tp_lex_description_has_timestamps_flag": float("nan"),
                "tp_lex_description_emoji_count": float("nan"),
                # Transcript
                "tp_lex_transcript_len_words": float("nan"),
                "tp_lex_transcript_avg_sentence_len": float("nan"),
                "tp_lex_transcript_question_ratio": float("nan"),
                "tp_lex_transcript_lexical_diversity": float("nan"),
                "tp_lex_transcript_rare_word_ratio": float("nan"),
                "tp_lex_transcript_stopword_ratio": float("nan"),
                "tp_lex_transcript_readability_score": float("nan"),
                "tp_lex_transcript_orthographic_error_rate": float("nan"),
                "tp_lex_transcript_avg_token_frequency_percentile": float("nan"),
                # Combined
                "tp_lex_emoji_diversity": float("nan"),
                "tp_lex_punctuation_entropy": float("nan"),
                "tp_lex_punctuation_entropy_present": 0.0,
                "tp_lex_special_character_ratio": float("nan"),
                "tp_lex_upper_lower_ratio_title": float("nan"),
                "tp_lex_named_entity_density": float("nan"),
                "tp_lex_named_entity_density_enabled": 0.0,
            }

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
                    "lexical_stats": {"metrics": features_flat},
                    "features_flat": features_flat,
                },
                "error": None,
            }

        title_raw = str(getattr(doc, "title", "") or "") if self.enable_title else ""
        description_raw = str(getattr(doc, "description", "") or "") if self.enable_description else ""
        title_raw = normalize_whitespace(title_raw)
        description_raw = normalize_whitespace(description_raw)

        title, title_chars_used, title_chars_kept, title_trunc = _truncate_text(title_raw, max_chars=self.max_title_chars)
        description, desc_chars_used, desc_chars_kept, desc_trunc = _truncate_text(description_raw, max_chars=self.max_description_chars)

        # Transcript source-of-truth: AudioProcessor payload in doc.asr
        transcript_used_asr = False
        transcript_used_legacy = False
        transcript = ""
        if self.enable_transcript:
            if self.transcript_source_policy in ("asr_only", "asr_then_legacy"):
                transcript = self._join_asr_text(doc)
                transcript_used_asr = bool(transcript.strip())
            if (not transcript.strip()) and self.transcript_source_policy in ("asr_then_legacy", "legacy_only"):
                transcript = self._join_legacy_transcripts(doc)
                transcript_used_legacy = bool(transcript.strip())
            transcript = normalize_whitespace(transcript)

        transcript, trans_chars_used, trans_chars_kept, trans_trunc = _truncate_text(transcript, max_chars=self.max_transcript_chars)

        title_tokens = _tokenize(title)
        desc_tokens = _tokenize(description)
        trans_tokens = _tokenize(transcript)

        has_title = bool(title.strip())
        has_description = bool(description.strip())
        has_transcript = bool(transcript.strip())
        present_any = bool(has_title or has_description or has_transcript)

        def _is_emoji(ch: str) -> bool:
            return (_emoji is not None) and (ch in _emoji.EMOJI_DATA)

        # Title features
        title_len_words = float(len(title_tokens)) if has_title else float("nan")
        title_len_chars = float(len(title)) if has_title else float("nan")
        title_avg_word_len = float(np.mean([len(t) for t in title_tokens])) if title_tokens else float("nan")
        title_exclamation_count = float(title.count("!")) if has_title else float("nan")
        title_question_count = float(title.count("?")) if has_title else float("nan")
        emoji_count_title = float(sum(1 for c in title if _is_emoji(c))) if (has_title and self.enable_emoji) else float("nan")
        title_type_token_ratio = float(len(set(map(str.lower, title_tokens))) / max(1, len(title_tokens))) if title_tokens else float("nan")
        title_punct_ratio = float(sum(1 for c in title if _is_punct_symbol(c)) / max(1, len(title))) if has_title else float("nan")
        title_capital_words_ratio = float(sum(1 for t in title_tokens if t.isupper()) / max(1, len(title_tokens))) if title_tokens else float("nan")
        title_question_prefix_flag = float(bool(_QUESTION_PREFIX_RE.search(title.strip()))) if has_title else float("nan")
        title_number_presence = float(bool(_NUMBER_RE.search(title))) if has_title else float("nan")
        title_time_mention_flag = float(bool(_DATE_TIME_RE.search(title))) if has_title else float("nan")

        # Description features
        description_len_words = float(len(desc_tokens)) if has_description else float("nan")
        description_num_urls = float(len(_URL_RE.findall(description))) if has_description else float("nan")
        description_num_mentions = float(len(_MENTION_RE.findall(description))) if has_description else float("nan")
        description_has_timestamps_flag = float(bool(_TIMESTAMP_RE.search(description))) if has_description else float("nan")
        emoji_count_description = float(sum(1 for c in description if _is_emoji(c))) if (has_description and self.enable_emoji) else float("nan")
        # emoji diversity (по всем полям)
        all_text = (title or "") + "\n" + (description or "") + "\n" + (transcript or "")
        all_emojis = [c for c in all_text if _is_emoji(c)] if self.enable_emoji else []
        emoji_diversity = float(len(set(all_emojis)) / max(1, len(all_emojis))) if all_emojis else float("nan")

        # Transcript features
        transcript_len_words = float(len(trans_tokens)) if has_transcript else float("nan")
        sents = _sentences(transcript)
        transcript_avg_sentence_len = float(np.mean([len(_tokenize(s)) for s in sents])) if sents else float("nan")
        # доля вопросительных предложений
        if sents and has_transcript:
            _q = sum(1 for s in sents if "?" in s)
            question_ratio_transcript = float(_q / max(1, len(sents)))
        else:
            question_ratio_transcript = float("nan")
        lexical_diversity_transcript = float(len(set(map(str.lower, trans_tokens))) / max(1, len(trans_tokens))) if trans_tokens else float("nan")
        # rare_word_ratio_transcript proxy: words longer than 12 chars
        rare_word_ratio_transcript = float(sum(1 for t in trans_tokens if len(t) > 12) / max(1, len(trans_tokens))) if trans_tokens else float("nan")
        # stopword_ratio_transcript proxy: simple list for ru/en
        stopwords = set([
            "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так","его","но","да","ты","к","у",
            "the","a","an","and","or","but","if","in","on","with","for","to","of","is","are","was","were","be","been","it",
        ])
        stopword_ratio_transcript = float(sum(1 for t in map(str.lower, trans_tokens) if t in stopwords) / max(1, len(trans_tokens))) if trans_tokens else float("nan")
        # 33) title_stopword_ratio
        title_stopword_ratio = float(sum(1 for t in map(str.lower, title_tokens) if t in stopwords) / max(1, len(title_tokens))) if title_tokens else float("nan")

        # 48) Readability proxy for transcript (simple): avg_sentence_len / avg_word_len
        avg_word_len_transcript = float(np.mean([len(t) for t in trans_tokens])) if trans_tokens else float("nan")
        readability_score_transcript = (
            float(transcript_avg_sentence_len / max(1e-6, avg_word_len_transcript))
            if (np.isfinite(transcript_avg_sentence_len) and np.isfinite(avg_word_len_transcript) and avg_word_len_transcript > 0)
            else float("nan")
        )

        # 49) title_clickbait_score (rule-based): keywords + punct signals
        clickbait_words = {
            # RU
            "шок","срочно","невероятно","топ","лучшие","секрет","удивит","скандал","честно","разоблачение",
            # EN
            "shocking","urgent","incredible","top","best","secret","you won't believe","scandal","honest","exposed",
        }
        tl = title.lower()
        cb_hits = 0
        for w in clickbait_words:
            if w in tl:
                cb_hits += 1
        if self.enable_clickbait_heuristic and has_title:
            cb_signal = cb_hits + (1 if title.count("!") > 0 else 0) + (1 if bool(_QUESTION_PREFIX_RE.search(title.strip())) else 0)
            title_clickbait_score = float(min(1.0, cb_signal / 3.0))
        else:
            title_clickbait_score = float("nan")

        # Language detection removed from this extractor (dedicated component via dp_models if needed).
        text_language = None
        language_confidence = float("nan")

        # 59) orthographic_error_rate proxy: share of tokens not matching simple alpha pattern or too few vowels
        def _is_wellformed(tok: str) -> bool:
            t = tok.lower()
            if not t:
                return False
            # Allow only letters and marks (Unicode categories starting with 'L' or 'M')
            for ch in t:
                cat = unicodedata.category(ch)
                if not (cat and cat[0] in ("L", "M")):
                    return False
            vowels = "аеёиоуыэюяaeiouy"
            return any(ch in vowels for ch in t)
        if trans_tokens:
            ortho_bad = sum(1 for t in trans_tokens if not _is_wellformed(t))
            orthographic_error_rate = float(ortho_bad / max(1, len(trans_tokens)))
        else:
            orthographic_error_rate = float("nan")

        # 60) avg_token_frequency_percentile proxy: inverse normalized word length (shorter words → higher percentile)
        if trans_tokens:
            lens = np.array([len(t) for t in trans_tokens], dtype=np.float32)
            freq_proxy = 1.0 - np.clip(lens / 20.0, 0.0, 1.0)
            avg_token_frequency_percentile = float(freq_proxy.mean())
        else:
            avg_token_frequency_percentile = float("nan")

        # POS/NER removed from this extractor (dedicated component required).
        named_entity_density = float("nan")
        named_entity_density_enabled = 0.0

        # punctuation entropy (title + description combined)
        def _entropy_from_counts(counts: Dict[str, int]) -> float:
            total = sum(counts.values())
            if total <= 0:
                return 0.0
            probs = np.array([c / total for c in counts.values()], dtype=np.float32)
            return float(-np.sum(probs * np.log(probs + 1e-9)))

        punctuation_entropy_present = float(has_title or has_description)
        if punctuation_entropy_present:
            puncts_text = title + " " + description
            punct_counts: Dict[str, int] = {}
            for ch in puncts_text:
                if _is_punct_symbol(ch):
                    punct_counts[ch] = punct_counts.get(ch, 0) + 1
            punctuation_entropy = _entropy_from_counts(punct_counts)
        else:
            punctuation_entropy = float("nan")

        special_character_ratio = (
            float(sum(1 for ch in (title + description) if not ch.isalnum() and not ch.isspace()) / max(1, len(title + description)))
            if (has_title or has_description)
            else float("nan")
        )
        upper_lower_ratio_title = (
            float(sum(1 for c in title if c.isupper()) / max(1, sum(1 for c in title if c.islower()) or 1))
            if has_title
            else float("nan")
        )

        # Stable flat features for dataset/UI (preferred).
        features_flat: Dict[str, Any] = {
            "tp_lex_enabled": 1.0,
            "tp_lex_disabled_by_policy": 0.0,
            "tp_lex_present_title": float(has_title),
            "tp_lex_present_description": float(has_description),
            "tp_lex_present_transcript": float(has_transcript),
            "tp_lex_present_any": float(present_any),
            "tp_lex_group_title_enabled": float(self.enable_title),
            "tp_lex_group_description_enabled": float(self.enable_description),
            "tp_lex_group_transcript_enabled": float(self.enable_transcript),
            "tp_lex_group_emoji_enabled": float(self.enable_emoji),
            "tp_lex_group_clickbait_enabled": float(self.enable_clickbait_heuristic),
            "tp_lex_has_emoji_lib": float(_emoji is not None),
            "tp_lex_emoji_dependency_missing_flag": float(self.enable_emoji and _emoji is None),

            "tp_lex_transcript_source_policy_asr_only": float(self.transcript_source_policy == "asr_only"),
            "tp_lex_transcript_source_policy_asr_then_legacy": float(self.transcript_source_policy == "asr_then_legacy"),
            "tp_lex_transcript_source_policy_legacy_only": float(self.transcript_source_policy == "legacy_only"),
            "tp_lex_transcript_source_used_asr": float(transcript_used_asr),
            "tp_lex_transcript_source_used_legacy": float(transcript_used_legacy),
            "tp_lex_transcript_source_used_none": float((not transcript_used_asr) and (not transcript_used_legacy)),
            "tp_lex_allow_legacy_transcripts": float(self.allow_legacy_transcripts),

            "tp_lex_title_chars_used": float(title_chars_used),
            "tp_lex_title_chars_kept": float(title_chars_kept),
            "tp_lex_title_truncated_flag": float(title_trunc),
            "tp_lex_description_chars_used": float(desc_chars_used),
            "tp_lex_description_chars_kept": float(desc_chars_kept),
            "tp_lex_description_truncated_flag": float(desc_trunc),
            "tp_lex_transcript_chars_used": float(trans_chars_used),
            "tp_lex_transcript_chars_kept": float(trans_chars_kept),
            "tp_lex_transcript_truncated_flag": float(trans_trunc),

            # Title
            "tp_lex_title_len_words": float(title_len_words),
            "tp_lex_title_len_chars": float(title_len_chars),
            "tp_lex_title_avg_word_len": float(title_avg_word_len),
            "tp_lex_title_exclamation_count": float(title_exclamation_count),
            "tp_lex_title_question_count": float(title_question_count),
            "tp_lex_title_emoji_count": float(emoji_count_title),
            "tp_lex_title_stopword_ratio": float(title_stopword_ratio),
            "tp_lex_title_type_token_ratio": float(title_type_token_ratio),
            "tp_lex_title_punctuation_ratio": float(title_punct_ratio),
            "tp_lex_title_capital_words_ratio": float(title_capital_words_ratio),
            "tp_lex_title_question_prefix_flag": float(title_question_prefix_flag),
            "tp_lex_title_number_presence": float(title_number_presence),
            "tp_lex_title_time_mention_flag": float(title_time_mention_flag),
            "tp_lex_title_clickbait_score": float(title_clickbait_score),

            # Description
            "tp_lex_description_len_words": float(description_len_words),
            "tp_lex_description_num_urls": float(description_num_urls),
            "tp_lex_description_num_mentions": float(description_num_mentions),
            "tp_lex_description_has_timestamps_flag": float(description_has_timestamps_flag),
            "tp_lex_description_emoji_count": float(emoji_count_description),

            # Transcript
            "tp_lex_transcript_len_words": float(transcript_len_words),
            "tp_lex_transcript_avg_sentence_len": float(transcript_avg_sentence_len),
            "tp_lex_transcript_question_ratio": float(question_ratio_transcript),
            "tp_lex_transcript_lexical_diversity": float(lexical_diversity_transcript),
            "tp_lex_transcript_rare_word_ratio": float(rare_word_ratio_transcript),
            "tp_lex_transcript_stopword_ratio": float(stopword_ratio_transcript),
            "tp_lex_transcript_readability_score": float(readability_score_transcript),
            "tp_lex_transcript_orthographic_error_rate": float(orthographic_error_rate),
            "tp_lex_transcript_avg_token_frequency_percentile": float(avg_token_frequency_percentile),

            # Combined
            "tp_lex_emoji_diversity": float(emoji_diversity),
            "tp_lex_punctuation_entropy": float(punctuation_entropy),
            "tp_lex_punctuation_entropy_present": float(punctuation_entropy_present),
            "tp_lex_special_character_ratio": float(special_character_ratio),
            "tp_lex_upper_lower_ratio_title": float(upper_lower_ratio_title),
            "tp_lex_named_entity_density": float(named_entity_density),
            "tp_lex_named_entity_density_enabled": float(named_entity_density_enabled),
        }

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        # Observability for dataset/UI: timings as scalars.
        features_flat["tp_lex_load_ms"] = 0.0
        features_flat["tp_lex_compute_ms"] = float(round(total_s * 1000.0, 3))

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
                "lexical_stats": {"metrics": features_flat},
                "features_flat": features_flat,
            },
            "error": None,
        }


