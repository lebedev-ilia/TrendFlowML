from __future__ import annotations

import hashlib
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.text_utils import normalize_whitespace
from src.schemas.models import VideoDocument


def _norm01_from_sha256_u64(text: str) -> float:
    """
    Privacy-safe stable numeric fingerprint in [0, 1).
    Stored as scalar in features_flat.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    u64 = int.from_bytes(h[:8], "big", signed=False)
    return float(u64 / (2**64))


def _is_hashtag_char(ch: str) -> bool:
    """
    Unicode-aware hashtag chars:
    - letters (L*), marks (M*) and numbers (N*)
    - underscore and hyphen
    """
    if ch in ("_", "-"):
        return True
    cat = unicodedata.category(ch)
    return bool(cat) and cat[0] in ("L", "M", "N")


def _is_hashtag_first_char(ch: str) -> bool:
    """
    First char must be letter/mark/number (no '_'/'-' as first).
    """
    cat = unicodedata.category(ch)
    return bool(cat) and cat[0] in ("L", "M", "N")


def _is_hashtag_boundary(prev_ch: Optional[str]) -> bool:
    """
    Require a boundary before '#': start-of-string or a non-hashtag char.
    Avoid matching inside words like 'abc#tag'.
    """
    if prev_ch is None:
        return True
    return not _is_hashtag_char(prev_ch)


def _extract_hashtags(
    text: str,
    *,
    max_tag_len: int = 64,
    max_tags_total: int = 64,
) -> Tuple[str, List[str], int, float]:
    """
    Returns (cleaned_text, unique_tags_preserve_order, total_tags_found, tags_truncated_flag).
    Removes only the "#<tag>" token, keeps surrounding punctuation as-is.
    """
    if not text:
        return "", [], 0, 0.0

    tags_all: List[str] = []
    out_chars: List[str] = []

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != "#":
            out_chars.append(ch)
            i += 1
            continue

        prev = text[i - 1] if i > 0 else None
        if not _is_hashtag_boundary(prev):
            out_chars.append(ch)
            i += 1
            continue

        # Try parse tag after '#'
        j = i + 1
        if j >= n or not _is_hashtag_first_char(text[j]):
            out_chars.append(ch)
            i += 1
            continue

        tag_chars: List[str] = [text[j]]
        j += 1
        while j < n and _is_hashtag_char(text[j]) and len(tag_chars) < max_tag_len:
            tag_chars.append(text[j])
            j += 1

        if not tag_chars:
            # Just a '#' not followed by a valid tag char
            out_chars.append(ch)
            i += 1
            continue

        raw_tag = "".join(tag_chars)
        tag = raw_tag.casefold()
        tags_all.append(tag)

        # Skip "#<tag>" from output
        i = j

    cleaned = normalize_whitespace("".join(out_chars))

    seen = set()
    uniq: List[str] = []
    tags_truncated_flag = 0.0
    for t in tags_all:
        if t in seen:
            continue
        seen.add(t)
        if len(uniq) < int(max(1, max_tags_total)):
            uniq.append(t)
        else:
            tags_truncated_flag = 1.0

    return cleaned, uniq, len(tags_all), float(tags_truncated_flag)


class TagsExtractor(BaseExtractor):
    VERSION = "1.1.0"

    def __init__(
        self,
        *,
        enable_extract_hashtags: bool = True,
        mutate_doc_clean_texts: bool = True,
        mutate_doc_hashtags: bool = True,
        require_title: bool = False,
        unicode_normalization: str = "NFKC",
        max_text_chars: int = 5000,
        max_tags_total: int = 64,
        top_k_slots: int = 5,
        export_cleaned_texts_mode: str = "none",
        export_hashtags_mode: str = "none",
        # Back-compat booleans (deprecated): if True → mode="raw"
        export_cleaned_texts: bool = False,
        export_hashtags: bool = False,
        max_tag_len: int = 64,
    ) -> None:
        # Groups / gating
        self.enable_extract_hashtags = bool(enable_extract_hashtags)
        self.mutate_doc_clean_texts = bool(mutate_doc_clean_texts)
        self.mutate_doc_hashtags = bool(mutate_doc_hashtags)
        self.require_title = bool(require_title)
        self.unicode_normalization = str(unicode_normalization or "NFKC").strip().upper()
        if self.unicode_normalization not in ("NFKC", "NFC", "NFKD", "NFD", "NONE"):
            raise RuntimeError("TagsExtractor: unicode_normalization must be one of: NONE|NFKC|NFC|NFKD|NFD")
        self.max_text_chars = int(max(1, max_text_chars))
        self.max_tags_total = int(max(1, max_tags_total))
        self.top_k_slots = int(max(1, top_k_slots))

        # Privacy: raw outputs must be explicitly enabled
        m_ct = str(export_cleaned_texts_mode or "none").strip().lower()
        m_ht = str(export_hashtags_mode or "none").strip().lower()
        if bool(export_cleaned_texts):
            m_ct = "raw"
        if bool(export_hashtags):
            m_ht = "raw"
        if m_ct not in ("none", "raw"):
            raise RuntimeError("TagsExtractor: export_cleaned_texts_mode must be one of: none|raw")
        if m_ht not in ("none", "raw", "hashed"):
            raise RuntimeError("TagsExtractor: export_hashtags_mode must be one of: none|raw|hashed")
        self.export_cleaned_texts_mode = m_ct
        self.export_hashtags_mode = m_ht
        self.max_tag_len = int(max(1, max_tag_len))

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        def _prep_text(val: Any) -> Tuple[str, float]:
            raw = normalize_whitespace("" if val is None else str(val))
            if self.unicode_normalization != "NONE":
                try:
                    raw = unicodedata.normalize(self.unicode_normalization, raw)
                except Exception:
                    pass
            truncated_flag = 0.0
            if len(raw) > self.max_text_chars:
                raw = raw[: self.max_text_chars]
                truncated_flag = 1.0
            return raw, truncated_flag

        # Contract: title is optional by default (valid empty), but can be required.
        title_val = getattr(doc, "title", None)
        title_raw, title_truncated_flag = _prep_text(title_val)
        title_present = float(bool(title_raw))
        if self.require_title and title_present == 0.0:
            raise RuntimeError("TagsExtractor requires non-empty doc.title (require_title=true)")

        desc_val = getattr(doc, "description", None)
        desc_raw, desc_truncated_flag = _prep_text(desc_val)
        desc_present = float(bool(desc_raw))

        if self.enable_extract_hashtags:
            title_clean, title_tags, title_tags_found, title_tags_trunc = _extract_hashtags(
                title_raw, max_tag_len=self.max_tag_len, max_tags_total=self.max_tags_total
            )
            desc_clean, desc_tags, desc_tags_found, desc_tags_trunc = _extract_hashtags(
                desc_raw, max_tag_len=self.max_tag_len, max_tags_total=self.max_tags_total
            )
        else:
            title_clean, title_tags, title_tags_found, title_tags_trunc = title_raw, [], 0, 0.0
            desc_clean, desc_tags, desc_tags_found, desc_tags_trunc = desc_raw, [], 0, 0.0

        # merge and uniquify tags
        merged_tags: List[str] = []
        seen = set()
        for t in title_tags + desc_tags:
            if t not in seen:
                seen.add(t)
                merged_tags.append(t)

        # In-memory mutations for downstream extractors (preferred; not persisted by default)
        mutations: Dict[str, Any] = {}
        if self.mutate_doc_clean_texts:
            try:
                doc.title = title_clean  # type: ignore[attr-defined]
                doc.description = desc_clean  # type: ignore[attr-defined]
                mutations["cleaned_texts"] = {"title": title_clean, "description": desc_clean}
            except Exception:
                # Do not fail run if mutation fails; downstream just won't see cleaned text.
                pass
        hashtags_disabled_by_policy = float(not self.enable_extract_hashtags)
        # Privacy-safe in-memory marker for downstream extractors.
        # NOTE: This marker is NOT a persisted contract; it must not include raw text/tags.
        try:
            tp = getattr(doc, "tp_artifacts", None)
            if not isinstance(tp, dict):
                tp = {}
                setattr(doc, "tp_artifacts", tp)
            tp.setdefault("tags", {})
            if isinstance(tp.get("tags"), dict):
                tp["tags"]["hashtags_disabled_by_policy"] = float(hashtags_disabled_by_policy)
        except Exception:
            pass
        if self.mutate_doc_hashtags and hashtags_disabled_by_policy == 0.0:
            try:
                doc.hashtags = list(merged_tags)
                mutations["hashtags"] = list(merged_tags)
            except Exception:
                pass

        # Safe flat scalar features for dataset/UI.
        def _dens(count: int, denom: int) -> float:
            return float(count / denom) if denom > 0 else float("nan")

        all_tags = list(merged_tags)
        lens = [len(t) for t in all_tags] if all_tags else []

        features_flat: Dict[str, Any] = {
            "tp_tags_title_present": float(title_present),
            "tp_tags_description_present": float(desc_present),
            "tp_tags_group_extract_enabled": float(self.enable_extract_hashtags),
            "tp_tags_group_mutate_clean_texts_enabled": float(self.mutate_doc_clean_texts),
            "tp_tags_group_mutate_hashtags_enabled": float(self.mutate_doc_hashtags),
            "tp_tags_require_title_enabled": float(self.require_title),
            "tp_tags_hashtags_disabled_by_policy": float(hashtags_disabled_by_policy),
            "tp_tags_export_cleaned_texts_mode_none": float(self.export_cleaned_texts_mode == "none"),
            "tp_tags_export_cleaned_texts_mode_raw": float(self.export_cleaned_texts_mode == "raw"),
            "tp_tags_export_hashtags_mode_none": float(self.export_hashtags_mode == "none"),
            "tp_tags_export_hashtags_mode_raw": float(self.export_hashtags_mode == "raw"),
            "tp_tags_export_hashtags_mode_hashed": float(self.export_hashtags_mode == "hashed"),
            "tp_tags_title_truncated_flag": float(title_truncated_flag),
            "tp_tags_description_truncated_flag": float(desc_truncated_flag),
            "tp_tags_hashtags_truncated_flag": float(bool(title_tags_trunc or desc_tags_trunc)),
            "tp_tags_title_hashtag_found_count": float(title_tags_found),
            "tp_tags_description_hashtag_found_count": float(desc_tags_found),
            "tp_tags_hashtag_total_found_count": float(title_tags_found + desc_tags_found),
            "tp_tags_hashtag_unique_count": float(len(all_tags)),
            "tp_tags_hashtag_avg_len": float(sum(lens) / len(lens)) if lens else float("nan"),
            "tp_tags_hashtag_max_len": float(max(lens)) if lens else float("nan"),
            "tp_tags_title_hashtag_density_per_char": _dens(title_tags_found, len(title_raw)),
            "tp_tags_description_hashtag_density_per_char": _dens(desc_tags_found, len(desc_raw)),
            "tp_tags_topk_slots": float(self.top_k_slots),
        }

        for i in range(1, self.top_k_slots + 1):
            if i <= len(all_tags):
                tag = all_tags[i - 1]
                features_flat[f"tp_tags_top{i}_present"] = 1.0
                features_flat[f"tp_tags_top{i}_hash01"] = float(_norm01_from_sha256_u64(tag))
                features_flat[f"tp_tags_top{i}_len"] = float(len(tag))
            else:
                features_flat[f"tp_tags_top{i}_present"] = 0.0
                features_flat[f"tp_tags_top{i}_hash01"] = float("nan")
                features_flat[f"tp_tags_top{i}_len"] = float("nan")

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        # Privacy: do not include raw texts / tag strings in `result` unless explicitly enabled.
        result: Dict[str, Any] = {"features_flat": features_flat}
        if self.export_cleaned_texts_mode == "raw":
            result["cleaned_texts"] = {"title": title_clean, "description": desc_clean}
        if self.export_hashtags_mode == "raw":
            result["hashtags"] = list(merged_tags)
        if self.export_hashtags_mode == "hashed":
            result["hashtags_hashed"] = [hashlib.sha256(t.encode("utf-8")).hexdigest()[:24] for t in merged_tags]

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
            "result": result,
            "mutations": mutations,
            "error": None,
        }


