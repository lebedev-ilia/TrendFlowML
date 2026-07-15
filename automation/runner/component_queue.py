"""Очередь компонентов. Источник истины по штампам — COMPONENT_VALIDATION_CHECKLIST.md
(символ ✅). Порядок работы задаётся PRIORITY ниже; claim.json хранит текущий взятый компонент.
"""
from __future__ import annotations
import json
import re
import datetime as dt

import config

# Имя компонента в чеклисте иногда идёт с уточняющим суффиксом в скобках, чтобы отличить от
# похожего имени (напр. "optical_flow (module)" vs "core_optical_flow"). Срезаем ТАКОЙ суффикс
# ПЕРЕД проверкой на "мусорность" строки — иначе легитимный штамп теряется (баг от 2026-07-16:
# "optical_flow (module)" не распознавался как done из-за пробела/скобок и штамповался заново).
_PAREN_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")

# Приоритет валидации оставшихся компонентов (правь свободно).
# Сначала — визуальное ядро (Triton-заменяемое), потом модули, потом аудио/текст.
PRIORITY = [
    "core_depth_midas",
    "core_optical_flow",
    "color_light",
    "frames_composition",
    "video_pacing",
    "optical_flow",
    "ocr_extractor",
    "high_level_semantic",
    "story_structure",
    "emotion_face",
    "micro_emotion",
    "detalize_face",
    "behavioral",
    "text_scoring",
    "core_identity/place_semantics",
    "core_identity/brand_semantics",
    "core_identity/car_semantics",
    "core_identity/content_domain",
    "core_identity/face_identity",
    "core_identity/franchise_recognition",
    # аудио
    "clap_extractor",
    "asr_extractor",
    "speaker_diarization_extractor",
    "loudness_extractor",
    "spectral_extractor",
    "mel_extractor",
    "mfcc_extractor",
    "chroma_extractor",
    "tempo_extractor",
    "onset_extractor",
    # текст
    "title_embedder",
    "description_embedder",
    "hashtag_embedder",
    "transcript_chunk_embedder",
    "comments_embedder",
    "semantic_cluster_extractor",
]


def stamped_components() -> set[str]:
    """Компоненты со статусом ✅ в чеклисте (по имени в первой ячейке таблицы)."""
    done = set()
    if not config.CHECKLIST.exists():
        return done
    for line in config.CHECKLIST.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and "✅" in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not cells:
                continue
            raw_name = cells[0].replace("`", "").strip()
            if not raw_name or raw_name.lower() in ("компонент",):
                continue
            # Срезаем уточняющий суффикс в скобках (напр. "optical_flow (module)" -> "optical_flow").
            name = _PAREN_SUFFIX.sub("", raw_name).strip()
            # После срезки суффикса имена компонентов — без пробелов/скобок/угловых/запятых (это
            # отсекает настоящие строки фич-леджера, а не легитимные уточнения в скобках).
            if name and not any(ch in name for ch in " ()<>,"):
                done.add(name)
    return done


def _load_done() -> list[str]:
    if config.DONE_FILE.exists():
        try:
            return json.loads(config.DONE_FILE.read_text())
        except json.JSONDecodeError:
            return []
    return []


def mark_done(component: str) -> None:
    """Пометить компонент закрытым раннером — очередь двинется дальше, не дожидаясь ручного
    штампа владельца (владелец может отдельно заштамповать/верифицировать позже)."""
    done = _load_done()
    if component and component not in done:
        done.append(component)
        config.DONE_FILE.write_text(json.dumps(done, ensure_ascii=False, indent=2))


def load_claim() -> dict | None:
    if config.CLAIM_FILE.exists():
        try:
            return json.loads(config.CLAIM_FILE.read_text())
        except json.JSONDecodeError:
            return None
    return None


def set_claim(component: str) -> None:
    config.CLAIM_FILE.write_text(json.dumps(
        {"component": component, "claimed_at": dt.datetime.now().isoformat()},
        ensure_ascii=False, indent=2))


def clear_claim() -> None:
    if config.CLAIM_FILE.exists():
        config.CLAIM_FILE.unlink()


def next_component() -> str | None:
    """Следующий незаштампованный компонент по приоритету.
    Если есть незакрытый claim — вернуть его (продолжить прерванный)."""
    claim = load_claim()
    done = stamped_components() | set(_load_done())
    if claim and claim.get("component") and claim["component"] not in done:
        return claim["component"]
    for comp in PRIORITY:
        short = comp.split("/")[-1]
        if comp not in done and short not in done:
            return comp
    return None
