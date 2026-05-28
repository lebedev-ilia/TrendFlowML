"""manifest.json → БД (Predictions) и регистрация артефактов для analysis job."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from sqlalchemy.orm import Session

from ..dbv2 import models as v2_models
from ..models import Artifact
from ..services.dataprocessor_adapter import DataProcessorPayload


def _register_artifact(db: Session, run_id: str, component: str, path: Path) -> None:
    """Регистрирует артефакт в legacy таблице artifacts (для обратной совместимости)."""
    rel = path.as_posix()
    existing = (
        db.query(Artifact)
        .filter(Artifact.run_id == run_id, Artifact.object_key == rel)
        .first()
    )
    if existing:
        return
    db.add(
        Artifact(
            run_id=run_id,
            component_name=component,
            kind=path.suffix.lstrip("."),
            object_key=rel,
            size_bytes=path.stat().st_size if path.exists() else None,
            sha256_hex=None,
        )
    )


def _sync_from_manifest_v2(
    db: Session,
    analysis_job: v2_models.AnalysisJob,
    manifest_path: Path,
    payload: DataProcessorPayload,
) -> Dict[str, Any]:
    """
    Синхронизирует данные из manifest.json в AnalysisJob и создаёт Prediction записи.

    Args:
        db: SQLAlchemy session
        analysis_job: AnalysisJob для обновления
        manifest_path: Path к manifest.json
        payload: DataProcessorPayload (для получения video_id, platform_id)

    Returns:
        Parsed manifest data
    """
    if not manifest_path.exists():
        return {}

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_meta = data.get("run") or {}
    predictions_data = data.get("predictions") or []

    if isinstance(run_meta, dict):
        pass

    for pred_data in predictions_data:
        if not isinstance(pred_data, dict):
            continue

        existing = (
            db.query(v2_models.Prediction)
            .filter(
                v2_models.Prediction.analysis_job_id == analysis_job.id,
                v2_models.Prediction.horizon_days == pred_data.get("horizon_days"),
            )
            .first()
        )
        if existing:
            continue

        db.add(
            v2_models.Prediction(
                analysis_job_id=analysis_job.id,
                horizon_days=pred_data.get("horizon_days", 7),
                predicted_views=pred_data.get("predicted_views", 0.0),
                predicted_likes=pred_data.get("predicted_likes", 0.0),
                percentile_score=pred_data.get("percentile_score", 0.0),
                confidence_lower=pred_data.get("confidence_lower", 0.0),
                confidence_upper=pred_data.get("confidence_upper", 0.0),
                model_version_id=analysis_job.model_version_id,
            )
        )

    db.flush()
    return data


def _scan_and_register_artifacts(db: Session, run_id: str, run_rs_path: Path) -> None:
    """Сканирует директорию результатов и регистрирует артефакты."""
    if not run_rs_path.exists():
        return
    for component_dir in run_rs_path.iterdir():
        if not component_dir.is_dir():
            continue
        component = component_dir.name
        for path in component_dir.rglob("*"):
            if path.is_dir():
                continue
            if path.name == "manifest.json":
                continue
            if path.suffix.lower() not in {".npz", ".json", ".html"}:
                continue
            _register_artifact(db, run_id, component, path)
