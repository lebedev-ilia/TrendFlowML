from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import Settings
from .db import session_scope
from .models import Artifact, Run, RunComponent, RunLog, Video, VideoFile, VideoSource, AnalysisProfile
from .services.dataprocessor import resolve_run_paths
from .services.events import publish_run_event
from .services.quality import discover_quality_scripts, run_quality_reports
from .worker import celery_app


settings = Settings()


def _utcnow() -> datetime:
    return datetime.utcnow()


def _append_log(db, run_id: str, level: str, message: str) -> None:
    db.add(RunLog(run_id=run_id, level=level, message=message))
    db.flush()


def _publish(run_id: str, payload: Dict[str, Any]) -> None:
    try:
        import asyncio

        asyncio.run(publish_run_event(run_id, payload))
    except Exception:
        pass


def _emit_status(run_id: str, status: str, error_code: Optional[str] = None, error_message: Optional[str] = None) -> None:
    _publish(
        run_id,
        {
            "type": "run.status_changed",
            "run_id": run_id,
            "ts": datetime.utcnow().isoformat() + "Z",
            "payload": {"status": status, "error_code": error_code, "error_message": error_message},
        },
    )


def _emit_stage(run_id: str, stage: str) -> None:
    _publish(
        run_id,
        {
            "type": "run.stage_changed",
            "run_id": run_id,
            "ts": datetime.utcnow().isoformat() + "Z",
            "payload": {"stage": stage},
        },
    )


def _emit_component(run_id: str, component_name: str, status: str, empty_reason: Optional[str] = None) -> None:
    payload: Dict[str, Any] = {
        "component_name": component_name,
        "status": status,
    }
    if empty_reason:
        payload["empty_reason"] = empty_reason
    _publish(
        run_id,
        {
            "type": "component.finished" if status in {"ok", "empty", "error"} else "component.started",
            "run_id": run_id,
            "ts": datetime.utcnow().isoformat() + "Z",
            "payload": payload,
        },
    )


def _tail_state_events(run_id: str, path: Path, stop_flag: Dict[str, bool]) -> None:
    offset = 0
    last_processor = None
    while not stop_flag.get("stop"):
        if not path.exists():
            time.sleep(0.5)
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                f.seek(offset)
                for line in f:
                    offset = f.tell()
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue
                    scope = event.get("scope")
                    if scope == "processor":
                        proc = event.get("processor")
                        status = event.get("status")
                        if proc and status:
                            if proc != last_processor and status == "running":
                                last_processor = proc
                                _emit_stage(run_id, proc)
                            _emit_component(run_id, proc, status)
                            with session_scope() as inner:
                                rc = (
                                    inner.query(RunComponent)
                                    .filter(RunComponent.run_id == run_id, RunComponent.component_name == proc)
                                    .first()
                                )
                                if not rc:
                                    rc = RunComponent(run_id=run_id, component_name=proc)
                                    inner.add(rc)
                                rc.status = status
                    elif scope == "component":
                        comp = event.get("component")
                        status = event.get("status")
                        if comp and status:
                            _emit_component(run_id, comp, status)
                            with session_scope() as inner:
                                rc = (
                                    inner.query(RunComponent)
                                    .filter(RunComponent.run_id == run_id, RunComponent.component_name == comp)
                                    .first()
                                )
                                if not rc:
                                    rc = RunComponent(run_id=run_id, component_name=comp)
                                    inner.add(rc)
                                rc.status = status
                    elif scope == "progress":
                        # Optional component progress (percentage + counters).
                        comp = event.get("component") or event.get("component_name")
                        prog = event.get("progress")
                        done = event.get("done")
                        total = event.get("total")
                        stage = event.get("stage")
                        if comp and prog is not None:
                            try:
                                prog_f = float(prog)
                            except Exception:
                                prog_f = None
                            if prog_f is not None:
                                _publish(
                                    run_id,
                                    {
                                        "type": "component.progress",
                                        "run_id": run_id,
                                        "ts": datetime.utcnow().isoformat() + "Z",
                                        "payload": {
                                            "component_name": str(comp),
                                            "progress": max(0.0, min(1.0, prog_f)),
                                            "done": int(done) if done is not None else None,
                                            "total": int(total) if total is not None else None,
                                            "stage": str(stage) if stage is not None else None,
                                        },
                                    },
                                )
        except Exception:
            time.sleep(0.5)
            continue
        time.sleep(0.2)


def _register_artifact(db, run_id: str, component: str, path: Path) -> None:
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


def _sync_from_manifest(db, run_id: str, manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_meta = data.get("run") or {}
    comps = data.get("components") or []

    run = db.query(Run).filter(Run.id == run_id).first()
    if run and isinstance(run_meta, dict):
        run.config_hash = run_meta.get("config_hash") or run.config_hash

    for comp in comps:
        name = comp.get("name")
        if not name:
            continue
        rc = db.query(RunComponent).filter(RunComponent.run_id == run_id, RunComponent.component_name == name).first()
        if not rc:
            rc = RunComponent(run_id=run_id, component_name=name)
            db.add(rc)
        rc.status = comp.get("status") or rc.status
        rc.schema_version = comp.get("schema_version")
        rc.producer_version = comp.get("producer_version")
        rc.device_used = comp.get("device_used")
        rc.error_code = comp.get("error_code")
        rc.error_message = comp.get("error")
        rc.empty_reason = comp.get("empty_reason")

        for art in comp.get("artifacts") or []:
            if not isinstance(art, dict):
                continue
            path = art.get("path")
            kind = art.get("kind")
            if not path or not kind:
                continue
            existing = (
                db.query(Artifact)
                .filter(Artifact.run_id == run_id, Artifact.object_key == str(path))
                .first()
            )
            if not existing:
                db.add(
                    Artifact(
                        run_id=run_id,
                        component_name=name,
                        kind=str(kind),
                        object_key=str(path),
                        size_bytes=art.get("size_bytes"),
                        sha256_hex=art.get("sha256"),
                    )
                )

    db.flush()
    return data


def _scan_and_register_artifacts(db, run_id: str, run_rs_path: Path) -> None:
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


@celery_app.task(name="process_run")
def process_run(run_id: str) -> None:
    paths = settings.resolve_paths()
    scripts = discover_quality_scripts(paths.dataproc_root)

    with session_scope() as db:
        run = db.query(Run).filter(Run.id == run_id).first()
        if not run:
            return
        video = db.query(Video).filter(Video.id == run.video_id).first()
        if not video:
            run.status = "failed"
            run.error_code = "video_not_found"
            db.flush()
            return
        source = db.query(VideoSource).filter(VideoSource.video_id == video.id).first()
        if not source or not source.uploaded_file_id:
            run.status = "failed"
            run.error_code = "video_source_missing"
            db.flush()
            return
        file_row = db.query(VideoFile).filter(VideoFile.id == source.uploaded_file_id).first()
        if not file_row:
            run.status = "failed"
            run.error_code = "video_file_missing"
            db.flush()
            return
        video_path = Path(file_row.object_key)

        profile = None
        if run.profile_id:
            profile = db.query(AnalysisProfile).filter(AnalysisProfile.id == run.profile_id).first()
        profile_config = profile.config_json if profile else {}
        if "visual" not in profile_config:
            profile_config["visual"] = {"cfg_path": str(paths.visual_cfg_default)}
        if isinstance(profile_config.get("visual"), dict) and not profile_config["visual"].get("cfg_path"):
            profile_config["visual"]["cfg_path"] = str(paths.visual_cfg_default)
        if "processors" not in profile_config:
            profile_config["processors"] = {
                "audio": {"enabled": False, "required": False},
                "text": {"enabled": False, "required": False},
            }

        run.status = "running"
        run.stage = "segmenter"
        run.started_at = _utcnow()
        db.flush()

        _emit_status(run_id, "running")
        _emit_stage(run_id, "segmenter")
        _emit_component(run_id, "segmenter", "running")

        # State events tailer
        run_paths = resolve_run_paths(
            platform_id=video.platform_id,
            video_id=video.video_id,
            run_id=run.id,
            result_store_base=paths.result_store_base,
        )
        stop_flag = {"stop": False}
        tail_thread = threading.Thread(
            target=_tail_state_events,
            args=(run_id, run_paths.state_events_path, stop_flag),
            daemon=True,
        )
        tail_thread.start()

        # Run DataProcessor
        dp_main = paths.dataproc_root / "main.py"
        cmd = [
            os.environ.get("PYTHON", "python3"),
            str(dp_main),
            "--video-path",
            str(video_path),
            "--output",
            str(paths.frames_dir_base),
            "--chunk-size",
            "64",
            "--visual-cfg-path",
            str(paths.visual_cfg_default),
            "--profile-path",
            str(paths.result_store_base.parent / "profiles_cache" / run.id / "profile.yaml"),
            "--dag-path",
            str(paths.dataproc_root / "docs" / "reference" / "component_graph.yaml"),
            "--dag-stage",
            "baseline",
            "--platform-id",
            video.platform_id,
            f"--video-id={video.video_id}",
            "--run-id",
            run.id,
            "--sampling-policy-version",
            "v1",
            "--dataprocessor-version",
            "dev",
            "--rs-base",
            str(paths.result_store_base),
        ]

        # Write profile config YAML before running.
        profile_dir = paths.result_store_base.parent / "profiles_cache" / run.id
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_yaml = profile_dir / "profile.yaml"
        with open(profile_yaml, "w", encoding="utf-8") as f:
            import yaml

            yaml.safe_dump(profile_config, f, sort_keys=False, allow_unicode=True)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def _drain(stream, level: str) -> None:
            if stream is None:
                return
            for line in stream:
                msg = line.rstrip("\n")
                if not msg:
                    continue
                with session_scope() as inner:
                    _append_log(inner, run_id, level, msg)
                _publish(
                    run_id,
                    {
                        "type": "log.line",
                        "run_id": run_id,
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "payload": {"level": level, "message": msg},
                    },
                )

        t_out = threading.Thread(target=_drain, args=(proc.stdout, "info"), daemon=True)
        t_err = threading.Thread(target=_drain, args=(proc.stderr, "error"), daemon=True)
        t_out.start()
        t_err.start()

        exit_code = proc.wait()
        stop_flag["stop"] = True
        tail_thread.join(timeout=2.0)

        if exit_code != 0:
            run.status = "failed"
            run.error_code = "dataprocessor_failed"
            run.error_message = f"exit={exit_code}"
            run.finished_at = _utcnow()
            db.flush()
            _emit_status(run_id, "failed", error_code="dataprocessor_failed", error_message=f"exit={exit_code}")
            return

        run.stage = "render"
        _emit_stage(run_id, "render")

        manifest = _sync_from_manifest(db, run_id, run_paths.manifest_path)

        run_rs_path = paths.result_store_base / video.platform_id / video.video_id / run.id
        frames_dir = paths.frames_dir_base / video.video_id / "video"
        components = [c.get("name") for c in (manifest.get("components") or []) if isinstance(c, dict)]
        components = [c for c in components if c]

        generated = run_quality_reports(
            scripts,
            run_rs_path=run_rs_path,
            frames_dir=frames_dir if frames_dir.exists() else None,
            video_path=video_path if video_path.exists() else None,
            components=components,
        )
        for comp, html in generated:
            _register_artifact(db, run_id, comp, html)

        _scan_and_register_artifacts(db, run_id, run_rs_path)

        run.status = "succeeded"
        run.stage = "render"
        run.finished_at = _utcnow()
        db.flush()
        _emit_status(run_id, "succeeded")

