from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class Comment:
    text: str


@dataclass
class VideoDocument:
    title: str
    description: str
    # Hashtags: loaded from JSON if present; after TagsExtractor with mutate_doc_hashtags, replaced by
    # merged list (inline #tags from title/description + optional JSON entries, deduped casefold).
    # Downstream (e.g. HashtagEmbedder) reads this field in-process.
    hashtags: List[str] = field(default_factory=list)
    transcripts: Dict[str, str] = field(default_factory=dict)
    # Optional: tokenized transcripts (shared tokenizer under dp_models).
    # Example:
    #   {"whisper": [101, 2023, ...], "youtube_auto": [...]}.
    transcripts_token_ids: Dict[str, List[int]] = field(default_factory=dict)
    # Audio duration (sec). Must be present when AudioProcessor/Segmenter audio pipeline is enabled.
    # TextProcessor extractors that compute time-normalized metrics should fail-fast if missing.
    audio_duration_sec: Optional[float] = None
    # ASR payload from AudioProcessor (preferred source-of-truth for transcript in prod).
    # Expected: {"schema_version": "...", "segments": [{"text": str, "confidence": float|None, "start_sec": float|None, "end_sec": float|None}], ...}
    asr: Optional[Dict[str, Any]] = None
    # Legacy alias used by older extractors. Prefer `asr`.
    transcripts_meta: Optional[Dict[str, Any]] = None
    video_description_by_neuro: Optional[str] = None
    trend_words: Optional[str] = None
    comments: List[Comment] = field(default_factory=list)
    speakers: Optional[Dict[str, Dict[str, Any]]] = None
    # In-memory intra-TextProcessor artifacts registry (NOT a persisted contract; used for passing paths/ids between extractors).
    tp_artifacts: Dict[str, Any] = field(default_factory=dict)


def video_document_from_dict(data: Dict) -> VideoDocument:
    comments_raw = data.get("comments") or []
    comments: List[Comment] = []
    for c in comments_raw:
        if isinstance(c, dict) and "text" in c:
            comments.append(Comment(text=str(c.get("text", ""))))
        else:
            comments.append(Comment(text=str(c)))

    hashtags_raw = data.get("hashtags")
    hashtags: List[str] = []
    if isinstance(hashtags_raw, list):
        for x in hashtags_raw:
            if isinstance(x, str) and x.strip():
                hashtags.append(x.strip())

    doc = VideoDocument(
        title=str(data.get("title", "")),
        description=str(data.get("description", "")),
        hashtags=hashtags,
        transcripts=dict(data.get("transcripts") or {}),
        transcripts_token_ids=dict(data.get("transcripts_token_ids") or {}),
        audio_duration_sec=(float(data.get("audio_duration_sec")) if data.get("audio_duration_sec") is not None else None),
        asr=(data.get("asr") if isinstance(data.get("asr"), dict) else None),
        transcripts_meta=(data.get("transcripts_meta") if isinstance(data.get("transcripts_meta"), dict) else None),
        video_description_by_neuro=data.get("video_description_by_neuro"),
        trend_words=data.get("trend_words"),
        comments=comments,
        speakers=data.get("speakers"),
        tp_artifacts={},
    )

    # If raw transcripts are missing but token IDs are provided, decode to text using shared tokenizer.
    # This keeps artifacts free of raw transcript text while allowing TextProcessor to operate.
    try:
        if (not doc.transcripts) and isinstance(doc.transcripts_token_ids, dict) and doc.transcripts_token_ids:
            token_ids = doc.transcripts_token_ids.get("whisper") or None
            if isinstance(token_ids, list) and token_ids:
                from dp_models import get_global_model_manager  # type: ignore

                mm = get_global_model_manager()
                tok_spec = mm.get_spec(model_name="shared_tokenizer_v1")
                _, _, _, _, _wd, artifacts = mm.resolve(tok_spec)
                tok_path = list(artifacts.values())[0] if artifacts else None
                if not tok_path:
                    raise RuntimeError("shared_tokenizer_v1 artifacts are empty")
                from tokenizers import Tokenizer  # type: ignore

                tok = Tokenizer.from_file(tok_path)
                text = tok.decode([int(x) for x in token_ids], skip_special_tokens=True)
                if isinstance(text, str) and text.strip():
                    doc.transcripts["whisper"] = text.strip()
    except Exception:
        # Fail-fast is enforced at pipeline level when transcript is required; schema remains flexible.
        pass

    return doc


