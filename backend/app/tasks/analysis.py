"""Celery: AnalysisJob → DataProcessor (v2)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict
from uuid import UUID

from ..config import Settings
from ..db import session_scope
from ..dbv2 import enums
from ..dbv2 import models as v2_models
from ..services.dataprocessor import run_dataprocessor_async, wait_for_run_completion_hybrid
from ..services.dataprocessor_adapter import prepare_dataprocessor_payload, resolve_run_paths_v2
from ..services.quality import discover_quality_scripts, run_quality_reports
from ..services import billing
from ..worker import celery_app
from .events import _emit_component, _emit_stage, _emit_status, _publish, _utcnow
from .manifest import _register_artifact, _scan_and_register_artifacts, _sync_from_manifest_v2

logger = logging.getLogger(__name__)
settings = Settings()


def _refund_failed_analysis(db, analysis_job) -> None:
    """Возврат единиц за прерванный анализ.

    Обещание пользователю: за невыполненные этапы средства не списываются.
    Ключ идемпотентности внутри refund_for_analysis не даст вернуть дважды
    при повторной обработке задачи.
    """
    try:
        billing.refund_for_analysis(
            db,
            workspace_id=analysis_job.workspace_id,
            analysis_job_id=analysis_job.id,
        )
    except Exception:
        # Возврат не должен маскировать исходную причину сбоя анализа.
        logger.exception("refund failed for analysis_job_id=%s", analysis_job.id)



@celery_app.task(name="process_analysis_job")
def process_analysis_job(analysis_job_id: str) -> None:
    """
    Celery task для обработки AnalysisJob (v2).

    Использует адаптер для преобразования v2 → legacy формат для DataProcessor,
    затем обновляет AnalysisJob и создаёт Prediction записи.

    Args:
        analysis_job_id: UUID строка AnalysisJob
    """
    paths = settings.resolve_paths()
    scripts = discover_quality_scripts(paths.dataproc_root)

    with session_scope() as db:
        # Загружаем AnalysisJob
        analysis_job = (
            db.query(v2_models.AnalysisJob)
            .filter(v2_models.AnalysisJob.id == UUID(analysis_job_id))
            .first()
        )
        if not analysis_job:
            return

        if analysis_job.status == enums.AnalysisStatus.canceled:
            return

        # Используем адаптер для преобразования в legacy формат
        try:
            payload = prepare_dataprocessor_payload(db, analysis_job)
        except ValueError as e:
            analysis_job.status = enums.AnalysisStatus.failed
            _refund_failed_analysis(db, analysis_job)
            analysis_job.error_message = str(e)
            analysis_job.completed_at = _utcnow()
            db.flush()
            _emit_status(str(analysis_job.id), "failed", error_code="preparation_failed", error_message=str(e))
            return

        db.refresh(analysis_job)
        if analysis_job.status == enums.AnalysisStatus.canceled:
            return

        # Обновляем статус
        analysis_job.status = enums.AnalysisStatus.processing
        analysis_job.started_at = _utcnow()
        db.flush()

        run_id = str(analysis_job.id)  # Используем analysis_job.id как run_id для событий

        _emit_status(run_id, "running")
        _emit_stage(run_id, "segmenter")
        _emit_component(run_id, "segmenter", "running")

        # Определяем пути
        run_paths_dict = resolve_run_paths_v2(
            platform_id=payload.platform_id,
            video_id=payload.video_id,
            analysis_job_id=analysis_job.id,
            result_store_base=paths.result_store_base,
        )
        run_rs_path = run_paths_dict["run_rs_path"]
        manifest_path = run_paths_dict["manifest_path"]

        # Запускаем DataProcessor через HTTP API
        # Используем один event loop для всех async операций
        loop = None
        try:
            # В Celery-воркере своего event loop нет, а в Python 3.12+
            # asyncio.get_event_loop() без активного loop бросает RuntimeError.
            # Поэтому создаём свой loop и всегда закрываем его в finally.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Отправить запрос на обработку
            loop.run_until_complete(
                run_dataprocessor_async(
                    video_path=payload.video_path,
                    platform_id=payload.platform_id,
                    video_id=payload.video_id,
                    run_id=payload.run_id,
                    profile_config=payload.profile_config,
                    result_store_base=paths.result_store_base,
                    frames_dir_base=paths.frames_dir_base,
                    visual_cfg_default=paths.visual_cfg_default,
                )
            )

            # Hybrid подход: SSE listener с fallback на polling
            # Обработка прогресса в реальном времени через callback
            def progress_callback(event: Dict[str, Any]) -> None:
                """Callback для обработки прогресса из SSE событий."""
                event_type = event.get("type")
                event_data = event.get("data", {})

                # Обновить WebSocket события для UI
                if event_type == "progress":
                    progress = event_data.get("progress", {})
                    overall = progress.get("overall", 0)
                    current_component = progress.get("current_component")
                    current_processor = progress.get("current_processor")

                    # Отправить событие прогресса
                    _publish(
                        run_id,
                        {
                            "type": "run.progress",
                            "run_id": run_id,
                            "ts": datetime.utcnow().isoformat() + "Z",
                            "payload": {
                                "progress": overall,
                                "current_component": current_component,
                                "current_processor": current_processor,
                                "components": progress.get("components", {}),
                            },
                        },
                    )
                elif event_type == "stage":
                    stage = event_data.get("stage")
                    if stage:
                        _emit_stage(run_id, stage)
                elif event_type == "component_start":
                    component = event_data.get("component")
                    if component:
                        _emit_component(run_id, component, "running")
                elif event_type == "component_complete":
                    component = event_data.get("component")
                    status = event_data.get("status", "success")
                    if component:
                        _emit_component(run_id, component, status)

            # Hybrid подход с обработкой прогресса
            final_status = loop.run_until_complete(
                wait_for_run_completion_hybrid(
                    run_id=payload.run_id,
                    webhook_timeout=30,  # Ждать webhook 30 секунд
                    timeout_seconds=settings.dataprocessor_timeout_seconds,
                    poll_interval=settings.dataprocessor_poll_interval,
                    progress_callback=progress_callback,
                )
            )

            # Проверить финальный статус
            status = final_status.get("status")
            if status == "success":
                # Обработка завершена успешно
                pass
            elif status == "error":
                # Обработка завершена с ошибкой
                error_message = final_status.get("error", "Unknown error")
                error_code = final_status.get("error_code", "dataprocessor_error")
                analysis_job.status = enums.AnalysisStatus.failed
                _refund_failed_analysis(db, analysis_job)
                analysis_job.error_message = error_message
                analysis_job.completed_at = _utcnow()
                db.flush()
                _emit_status(run_id, "failed", error_code=error_code, error_message=error_message)
                return
            elif status == "cancelled":
                # Обработка отменена
                analysis_job.status = enums.AnalysisStatus.canceled
                analysis_job.completed_at = _utcnow()
                db.flush()
                _emit_status(run_id, "cancelled")
                return
            else:
                # Неожиданный статус
                analysis_job.status = enums.AnalysisStatus.failed
                _refund_failed_analysis(db, analysis_job)
                analysis_job.error_message = f"Unexpected final status: {status}"
                analysis_job.completed_at = _utcnow()
                db.flush()
                _emit_status(
                    run_id,
                    "failed",
                    error_code="unexpected_status",
                    error_message=f"status={status}",
                )
                return

        except TimeoutError as e:
            # Timeout при polling
            analysis_job.status = enums.AnalysisStatus.failed
            _refund_failed_analysis(db, analysis_job)
            analysis_job.error_message = f"Processing timeout: {str(e)}"
            analysis_job.completed_at = _utcnow()
            db.flush()
            _emit_status(run_id, "failed", error_code="timeout", error_message=str(e))
            return
        except ValueError as e:
            # Run не найден или другая ошибка валидации
            analysis_job.status = enums.AnalysisStatus.failed
            _refund_failed_analysis(db, analysis_job)
            analysis_job.error_message = f"Validation error: {str(e)}"
            analysis_job.completed_at = _utcnow()
            db.flush()
            _emit_status(run_id, "failed", error_code="validation_error", error_message=str(e))
            return
        except Exception as e:
            # Другие ошибки (HTTP, соединение, и т.д.)
            analysis_job.status = enums.AnalysisStatus.failed
            _refund_failed_analysis(db, analysis_job)
            analysis_job.error_message = f"DataProcessor API error: {str(e)}"
            analysis_job.completed_at = _utcnow()
            db.flush()
            _emit_status(run_id, "failed", error_code="api_error", error_message=str(e))
            return
        finally:
            # Свой loop всегда закрываем и снимаем как текущий.
            if loop is not None:
                try:
                    asyncio.set_event_loop(None)
                    loop.close()
                except Exception:
                    pass

        # Синхронизируем результаты из manifest
        manifest = _sync_from_manifest_v2(db, analysis_job, manifest_path, payload)

        # Генерируем quality reports
        frames_dir = paths.frames_dir_base / payload.video_id / "video"
        components = [c.get("name") for c in (manifest.get("components") or []) if isinstance(c, dict)]
        components = [c for c in components if c]

        generated = run_quality_reports(
            scripts,
            run_rs_path=run_rs_path,
            frames_dir=frames_dir if frames_dir.exists() else None,
            video_path=payload.video_path if payload.video_path.exists() else None,
            components=components,
        )
        for comp, html in generated:
            _register_artifact(db, run_id, comp, html)

        _scan_and_register_artifacts(db, run_id, run_rs_path)

        # Обновляем статус
        analysis_job.status = enums.AnalysisStatus.completed
        analysis_job.completed_at = _utcnow()
        db.flush()
        _emit_status(run_id, "succeeded")
