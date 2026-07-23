"""Заполняет core.processing_configs системными пресетами.

Идемпотентен: пресеты сопоставляются по имени, повторный запуск обновляет
состав, а не плодит дубликаты.

Состав компонентов соответствует каталогу DataProcessor
(configs/global_config.yaml) и baseline-набору Models/docs/contracts/BASELINE_MODEL.md.

Запуск:
    cd backend && .venv/bin/python -m scripts.seed_processing_configs
"""

from __future__ import annotations

from app.dbv2.models import ProcessingConfig
from app.db import SessionLocal

BASELINE_VISUAL = [
    "cut_detection",
    "optical_flow",
    "scene_classification",
    "shot_quality",
    "story_structure",
    "uniqueness",
    "video_pacing",
    "core_clip",
    "core_object_detections",
    "core_face_landmarks",
]
BASELINE_AUDIO = ["clap_extractor", "loudness_extractor", "tempo_extractor"]
BASELINE_TEXT = ["title_embedder", "description_embedder"]

PRESETS = [
    {
        "name": "Быстрый анализ",
        "description": "Базовый набор компонентов: прогноз без глубокого разбора кадра.",
        "payload": {
            "components": BASELINE_VISUAL[:7] + BASELINE_AUDIO + BASELINE_TEXT,
            "params": {"segmenter.analysis_fps": "0.5", "segmenter.frame_budget": "min"},
            "disabled_outputs": [],
        },
        "estimated_cost_units": 50,
        "estimated_minutes": 5,
    },
    {
        "name": "Полный анализ",
        "description": "Все базовые компоненты трёх модальностей и подробный разбор.",
        "payload": {
            "components": BASELINE_VISUAL
            + BASELINE_AUDIO
            + BASELINE_TEXT
            + ["asr_extractor", "diarization", "emotion_face", "ocr_extractor"],
            "params": {"segmenter.analysis_fps": "1", "segmenter.frame_budget": "target"},
            "disabled_outputs": [],
        },
        "estimated_cost_units": 125,
        "estimated_minutes": 15,
    },
    {
        "name": "Только текст и аудио",
        "description": "Без визуальных компонентов: быстрее и дешевле для подкастов.",
        "payload": {
            "components": BASELINE_AUDIO + BASELINE_TEXT + ["asr_extractor", "diarization"],
            "params": {"segmenter.analysis_fps": "0.5", "segmenter.frame_budget": "min"},
            "disabled_outputs": [],
        },
        "estimated_cost_units": 70,
        "estimated_minutes": 8,
    },
]


def main() -> None:
    db = SessionLocal()
    try:
        created, updated = 0, 0
        for preset in PRESETS:
            existing = (
                db.query(ProcessingConfig)
                .filter(
                    ProcessingConfig.is_system.is_(True),
                    ProcessingConfig.name == preset["name"],
                )
                .first()
            )
            if existing:
                existing.description = preset["description"]
                existing.payload = preset["payload"]
                existing.estimated_cost_units = preset["estimated_cost_units"]
                existing.estimated_minutes = preset["estimated_minutes"]
                updated += 1
                continue

            db.add(
                ProcessingConfig(
                    workspace_id=None,
                    created_by_user_id=None,
                    is_system=True,
                    **preset,
                )
            )
            created += 1
        db.commit()
        print(f"Системные пресеты: создано {created}, обновлено {updated}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
