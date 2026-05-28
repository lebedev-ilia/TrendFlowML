from __future__ import annotations

import hashlib
import logging
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.text_utils import normalize_whitespace
from src.schemas.models import VideoDocument

logger = logging.getLogger(__name__)


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


def _normalize_json_hashtag_entry(raw: str) -> str:
    """
    Platform/API hashtags may include leading '#'. Stored dedupe key is casefolded body.
    """
    s = normalize_whitespace("" if raw is None else str(raw))
    if s.startswith("#"):
        s = s[1:]
    s = normalize_whitespace(s)
    return s.casefold()


class TagsExtractor(BaseExtractor):
    VERSION = "1.2.0"

    def __init__(
        self,
        *,
        enable_extract_hashtags: bool = True,
        mutate_doc_clean_texts: bool = True,
        mutate_doc_hashtags: bool = True,
        merge_json_hashtags: bool = True,
        require_title: bool = False,
        unicode_normalization: str = "NFKC",
        max_text_chars: int = 5000,
        max_parse_chars: int = 200_000,
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
        self.merge_json_hashtags = bool(merge_json_hashtags)
        self.require_title = bool(require_title)
        self.unicode_normalization = str(unicode_normalization or "NFKC").strip().upper()
        if self.unicode_normalization not in ("NFKC", "NFC", "NFKD", "NFD", "NONE"):
            raise RuntimeError("TagsExtractor: unicode_normalization must be one of: NONE|NFKC|NFC|NFKD|NFD")
        self.max_text_chars = int(max(1, max_text_chars))
        self.max_parse_chars = int(max(int(self.max_text_chars), int(max_parse_chars)))
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

        incoming_json_hashtags: List[str] = []
        raw_ht = getattr(doc, "hashtags", None)
        if isinstance(raw_ht, list):
            incoming_json_hashtags = [x for x in raw_ht if isinstance(x, str)]

        def _normalize_full(val: Any) -> str:
            raw = normalize_whitespace("" if val is None else str(val))
            if self.unicode_normalization != "NONE":
                try:
                    raw = unicodedata.normalize(self.unicode_normalization, raw)
                except Exception:
                    pass
            return raw

        # Contract: title is optional by default (valid empty), but can be required.
        title_val = getattr(doc, "title", None)
        title_full = _normalize_full(title_val)
        title_present = float(bool(title_full))
        if self.require_title and title_present == 0.0:
            raise RuntimeError("TagsExtractor requires non-empty doc.title (require_title=true)")

        desc_val = getattr(doc, "description", None)
        desc_full = _normalize_full(desc_val)
        desc_present = float(bool(desc_full))

        title_parse_capped = 0.0
        desc_parse_capped = 0.0
        title_for_extract = title_full
        if len(title_for_extract) > self.max_parse_chars:
            title_for_extract = title_for_extract[: self.max_parse_chars]
            title_parse_capped = 1.0
        desc_for_extract = desc_full
        if len(desc_for_extract) > self.max_parse_chars:
            desc_for_extract = desc_for_extract[: self.max_parse_chars]
            desc_parse_capped = 1.0

        if self.enable_extract_hashtags:
            title_clean_long, title_tags, title_tags_found, title_tags_trunc = _extract_hashtags(
                title_for_extract, max_tag_len=self.max_tag_len, max_tags_total=self.max_tags_total
            )
            desc_clean_long, desc_tags, desc_tags_found, desc_tags_trunc = _extract_hashtags(
                desc_for_extract, max_tag_len=self.max_tag_len, max_tags_total=self.max_tags_total
            )
        else:
            title_clean_long, title_tags, title_tags_found, title_tags_trunc = title_for_extract, [], 0, 0.0
            desc_clean_long, desc_tags, desc_tags_found, desc_tags_trunc = desc_for_extract, [], 0, 0.0

        title_storage_trunc = 0.0
        desc_storage_trunc = 0.0
        title_clean = title_clean_long
        if len(title_clean) > self.max_text_chars:
            title_clean = title_clean[: self.max_text_chars]
            title_storage_trunc = 1.0
        desc_clean = desc_clean_long
        if len(desc_clean) > self.max_text_chars:
            desc_clean = desc_clean[: self.max_text_chars]
            desc_storage_trunc = 1.0

        # merge and uniquify tags (inline: title then description), then optional JSON platform tags
        merged_tags: List[str] = []
        seen: set[str] = set()
        for t in title_tags + desc_tags:
            if t not in seen:
                seen.add(t)
                merged_tags.append(t)

        json_merged_extra = 0
        if self.merge_json_hashtags and incoming_json_hashtags:
            for raw_h in incoming_json_hashtags:
                key = _normalize_json_hashtag_entry(raw_h)
                if not key:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                merged_tags.append(key)
                json_merged_extra += 1

        # In-memory mutations for downstream extractors (preferred; not persisted by default)
        mutations: Dict[str, Any] = {}
        if self.mutate_doc_clean_texts:
            try:
                doc.title = title_clean  # type: ignore[attr-defined]
                doc.description = desc_clean  # type: ignore[attr-defined]
                mutations["cleaned_texts"] = {"title": title_clean, "description": desc_clean}
            except Exception as e:
                logger.exception("TagsExtractor: mutate_doc_clean_texts failed")
                raise RuntimeError(f"TagsExtractor: failed to mutate title/description: {e}") from e
        hashtags_disabled_by_policy = float(not self.enable_extract_hashtags)
        # Privacy-safe in-memory marker for downstream extractors.
        # NOTE: This marker is NOT a persisted contract; it must not include raw/text/tags.
        try:
            tp = getattr(doc, "tp_artifacts", None)
            if not isinstance(tp, dict):
                tp = {}
                setattr(doc, "tp_artifacts", tp)
            tp.setdefault("tags", {})
            if isinstance(tp.get("tags"), dict):
                tp["tags"]["hashtags_disabled_by_policy"] = float(hashtags_disabled_by_policy)
        except Exception as e:
            logger.exception("TagsExtractor: failed to update doc.tp_artifacts tags marker")
            raise RuntimeError(f"TagsExtractor: failed to update tp_artifacts: {e}") from e
        # Write hashtags when inline extraction is on, or when merging normalizes/extends JSON-only lists.
        should_write_hashtags = self.mutate_doc_hashtags and (
            self.enable_extract_hashtags or (self.merge_json_hashtags and bool(incoming_json_hashtags))
        )
        if should_write_hashtags:
            try:
                doc.hashtags = list(merged_tags)
                mutations["hashtags"] = list(merged_tags)
            except Exception as e:
                logger.exception("TagsExtractor: mutate_doc_hashtags failed")
                raise RuntimeError(f"TagsExtractor: failed to mutate doc.hashtags: {e}") from e

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
            "tp_tags_group_merge_json_hashtags_enabled": float(self.merge_json_hashtags),
            "tp_tags_require_title_enabled": float(self.require_title),
            "tp_tags_hashtags_disabled_by_policy": float(hashtags_disabled_by_policy),
            "tp_tags_export_cleaned_texts_mode_none": float(self.export_cleaned_texts_mode == "none"),
            "tp_tags_export_cleaned_texts_mode_raw": float(self.export_cleaned_texts_mode == "raw"),
            "tp_tags_export_hashtags_mode_none": float(self.export_hashtags_mode == "none"),
            "tp_tags_export_hashtags_mode_raw": float(self.export_hashtags_mode == "raw"),
            "tp_tags_export_hashtags_mode_hashed": float(self.export_hashtags_mode == "hashed"),
            "tp_tags_title_parse_capped_flag": float(title_parse_capped),
            "tp_tags_description_parse_capped_flag": float(desc_parse_capped),
            "tp_tags_title_truncated_flag": float(title_storage_trunc),
            "tp_tags_description_truncated_flag": float(desc_storage_trunc),
            "tp_tags_json_hashtag_merged_count": float(json_merged_extra),
            "tp_tags_hashtags_truncated_flag": float(bool(title_tags_trunc or desc_tags_trunc)),
            "tp_tags_title_hashtag_found_count": float(title_tags_found),
            "tp_tags_description_hashtag_found_count": float(desc_tags_found),
            "tp_tags_hashtag_total_found_count": float(title_tags_found + desc_tags_found),
            "tp_tags_hashtag_unique_count": float(len(all_tags)),
            "tp_tags_hashtag_avg_len": float(sum(lens) / len(lens)) if lens else float("nan"),
            "tp_tags_hashtag_max_len": float(max(lens)) if lens else float("nan"),
            "tp_tags_title_hashtag_density_per_char": _dens(title_tags_found, len(title_for_extract)),
            "tp_tags_description_hashtag_density_per_char": _dens(desc_tags_found, len(desc_for_extract)),
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


