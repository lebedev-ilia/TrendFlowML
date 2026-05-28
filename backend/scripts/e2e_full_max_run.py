#!/usr/bin/env python3
"""
Полный локальный E2E с максимальным DataProcessor (global_config.yaml):
Segmenter + AudioProcessor + TextProcessor + VisualProcessor (включая Triton-ядра).

Артефакты прогона: ``<repo>/storage/e2e_full_max/<run_tag>/`` (копия логов стека, summary.json, patched global_config_e2e.yaml).

Перед запуском поднимите стек (из корня репозитория)::

    ./backend/scripts/start_e2e_stack.sh --with-infra

Затем (рекомендуется — Triton в Docker на порту 8010, без конфликта с Fetcher:8000)::

    cd backend && source scripts/e2e_env.sh
    source .venv/bin/activate
    python scripts/e2e_full_max_run.py --with-triton-docker

Только часть процессоров (Segmenter всегда): ``--processors text``, ``--processors visual``,
``--processors audio,text`` и т.д. (по умолчанию ``audio,text,visual``). Без ``visual`` Triton
не обязателен; при ``--with-triton-docker`` контейнер не поднимается, если ``visual`` не в списке.

Либо задайте ``TRITON_HTTP_URL`` на уже запущенный сервер. По умолчанию в E2E включаются
все флаги ``processors.visual.inline_config.core_providers`` и ``.modules`` из шаблона
``global_config.yaml`` (полный профиль visual). Чтобы оставить флаги как в YAML без
принудительного включения, передайте ``--visual-minimal``. Полный visual без Triton
невозможен для ``core_depth_midas`` / ``core_optical_flow`` (только runtime=triton);
для машины без GPU используйте ``--local-visual-no-triton`` (CLIP in-process, depth/flow off).

Офлайн-серия по встроенному плану (mock mp4 из ``example/example_videos`` + тексты из ``example/``)::

    cd backend && source scripts/e2e_env.sh && source .venv/bin/activate
    python scripts/e2e_full_max_run.py --example-suite-7 --with-triton-docker

По умолчанию берутся первые **7** роликов; встроенный план сейчас **до 20** шагов (``--example-suite-count N``) —
7 исходных + plan 7 (audit) + 8–10 (audit 03–05) + 11–20 (audit 06–15 на mock из первых роликов, без новых ``*.mp4``).

Уже обработанные ролики (``storage/result_store/youtube/<id>/.../manifest.json`` с
``run.status == "success"``) по умолчанию не ставятся в очередь; прогнать выбранное число подряд —
``--example-suite-force-all``.

Кеш Fetcher (глобальный ``videos`` по platform + platform_video_id):

- По умолчанию **не** чистим — подходит для РФ / таймаутов YouTube (cache hit без сети).
- ``--cold-ingestion`` — удалить видео из кеша Fetcher; нужны YouTube/прокси или успешный yt-dlp/API.

Указатель ``storage/e2e_full_max/active_global_config`` задаёт путь к YAML для
``--global-config`` при ``process_ingestion_run``. Маркер снимается после **каждого** прогона
E2E (успех или ошибка), чтобы не тянуть путь к чужому YAML между роликами сьюта
(если не передан ``--keep-active-global-config``). В лог пишутся строки ``[host …]`` (RAM, load1, GPU)
каждые ``--e2e-resource-snapshot-sec`` сек (по умолчанию 5); GPU в этих строках — суммарно по карте
(nvidia-smi). Тот же таймер пишет развёрнутые снимки в ``orchestrator_events.jsonl`` в каталоге артефакта
(полная история без дублирования огромных таблиц в stdout; отключить: ``--no-e2e-events-jsonl``).
Между роликами example-suite — ``[suite-between]`` + ``gc`` и ``cuda.empty_cache`` только
в процессе оркестратора (не в Triton/DataProcessor), поэтому занятость VRAM между роликами часто не падает.

Перед каждым роликом ``--example-suite-7`` по умолчанию ждём свободный слот в DataProcessor
(``GET /api/v1/health`` → ``active_runs < max_concurrent_runs``); отключить: ``--no-dp-wait-capacity``.

При одновременном ``audio`` и ``visual`` в патче конфига для экстрактора ``source_separation``
задаётся ``device: cpu``, чтобы не конкурировать за VRAM с Triton/visual на типичных 6 GiB картах.

По умолчанию без смены ``device``: в патч YAML входят только снижение батчей (CLAP, separation, emotion) и согласованные
переменные окружения (см. ``e2e_env.sh``, ``PYTORCH_CUDA_ALLOC_CONF``). Родительский ``DataProcessor/main.py`` после
Audio/Text вызывает ``torch.cuda.empty_cache()`` в своём процессе.

Если на узкой VRAM всё ещё OOM: ``--e2e-low-vram`` — тогда ``source_separation`` на CPU и тяжёлые текстовые
экстракторы (embed/aggregator/semantic и т.д., а также ``device`` cuda/gpu/auto) при одновременном ``visual`` + ``text``.

Между роликами ``--example-suite-7``: опционально ``scripts/e2e_host_memory_scrub.sh`` (см. ``--no-e2e-host-scrub``).
"""

from __future__ import annotations

import argparse
import atexit
import copy
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from e2e_host_resources import host_resource_snapshot_line, parent_process_gpu_gc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _wait_dataprocessor_slot(
    *,
    dataprocessor_url: str,
    api_key: Optional[str],
    max_wait_sec: int,
    poll_sec: float,
    label: str,
) -> None:
    """Ждём, пока в DataProcessor active_runs < max_concurrent_runs (свободен слот под новый POST /process)."""
    if max_wait_sec <= 0:
        return
    headers: Dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key
    url = f"{dataprocessor_url.rstrip('/')}/api/v1/health"
    deadline = time.monotonic() + max_wait_sec
    last_printed = ""
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, headers=headers, timeout=20.0)
            data = r.json() if r.content else {}
            m = data.get("metrics") or {}
            active = m.get("active_runs")
            mx = m.get("max_concurrent_runs")
            if active is not None and mx is not None:
                a, mxc = int(active), int(mx)
                if a < mxc:
                    print(f"[dp-capacity] {label} active_runs={a}/{mxc} (slot free)", flush=True)
                    return
                msg = f"[dp-capacity] {label} waiting active_runs={a}/{mxc} …"
                if msg != last_printed:
                    print(msg, flush=True)
                    last_printed = msg
            else:
                print(
                    f"[dp-capacity] {label} health http={r.status_code} (no metrics in body)",
                    flush=True,
                )
        except Exception as e:
            print(f"[dp-capacity] {label} GET /health failed: {e}", flush=True)
        time.sleep(max(0.5, float(poll_sec)))
    raise RuntimeError(
        f"Timeout {max_wait_sec}s waiting for DataProcessor capacity (active_runs < max); label={label}"
    )


def _between_suite_videos_cleanup(
    plan_index: int,
    exit_code: int,
    video_id: str,
    *,
    run_host_scrub: bool,
) -> None:
    """Между роликами example-suite: gc, CUDA-кэш в оркестраторе, опционально host scrub, снимок хоста."""
    gc.collect()
    parent_process_gpu_gc()
    if run_host_scrub:
        scrub = _SCRIPT_DIR / "e2e_host_memory_scrub.sh"
        if scrub.is_file():
            try:
                r = subprocess.run(
                    ["/bin/bash", str(scrub)],
                    cwd=str(_repo_root()),
                    timeout=180,
                    capture_output=True,
                    text=True,
                )
                tail = (r.stdout or "").strip().splitlines()[-8:]
                for ln in tail:
                    print(f"[suite-scrub] {ln}", flush=True)
                if r.returncode != 0:
                    print(
                        f"[suite-scrub] exit={r.returncode} stderr={(r.stderr or '')[:400]}",
                        flush=True,
                    )
            except subprocess.TimeoutExpired:
                print("[suite-scrub] timeout, continuing", flush=True)
            except Exception as e:
                print(f"[suite-scrub] skipped: {e}", flush=True)
    print(
        f"[suite-between] after plan_item={plan_index} video_id={video_id} e2e_exit={exit_code} | "
        f"{host_resource_snapshot_line()}",
        flush=True,
    )


def _triton_e2e_http_port() -> int:
    return int(os.environ.get("TRITON_E2E_HTTP_PORT", "8010"), 10)


def _triton_container_name() -> str:
    return os.environ.get("TRITON_E2E_CONTAINER_NAME", "trendflow-e2e-triton")


def _start_triton_docker(root: Path) -> str:
    script = root / "backend" / "scripts" / "e2e_triton_docker.sh"
    if not script.is_file():
        raise FileNotFoundError(f"Triton helper not found: {script}")
    subprocess.run(["/bin/bash", str(script), "start"], check=True)
    subprocess.run(["/bin/bash", str(script), "wait"], check=True)
    port = _triton_e2e_http_port()
    return f"http://127.0.0.1:{port}"


def _stop_triton_container() -> None:
    name = _triton_container_name()
    subprocess.run(["docker", "stop", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Контейнер с --rm удаляется после stop


def _enable_full_visual_inline_config(cfg: Dict[str, Any]) -> None:
    """Включает все core_providers и modules в inline_config (ключи из шаблона YAML)."""
    vis = cfg.get("processors", {}).get("visual")
    if not isinstance(vis, dict):
        return
    ic = vis.get("inline_config")
    if not isinstance(ic, dict):
        return
    cp = ic.get("core_providers")
    if isinstance(cp, dict):
        for k in cp:
            cp[k] = True
    md = ic.get("modules")
    if isinstance(md, dict):
        for k in md:
            md[k] = True


def _parse_processors_csv(s: str) -> frozenset[str]:
    allowed = frozenset({"audio", "text", "visual"})
    parts = {x.strip().lower() for x in (s or "").split(",") if x.strip()}
    bad = parts - allowed
    if bad:
        raise ValueError(f"Invalid processor names {sorted(bad)}; allowed: {sorted(allowed)}")
    if not parts:
        raise ValueError("--processors must list at least one of: audio, text, visual")
    return frozenset(parts)


def _e2e_tighten_audio_batches_for_vram(audio: Dict[str, Any]) -> None:
    """Снижение пиков VRAM без смены device: батчи CLAP / separation / emotion."""
    ex = audio.get("extractors") or {}
    if not isinstance(ex, dict):
        return
    clap = ex.get("clap")
    if isinstance(clap, dict):
        par = clap.setdefault("parallelism", {})
        if isinstance(par, dict):
            bs = par.get("batch_size")
            try:
                cur = int(bs) if bs is not None else 16
            except (TypeError, ValueError):
                cur = 16
            par["batch_size"] = max(1, min(8, cur))
    sep = ex.get("source_separation")
    if isinstance(sep, dict) and sep.get("enabled", False):
        sep["batch_size"] = 1
    emo = ex.get("emotion_diarization")
    if isinstance(emo, dict) and emo.get("enabled", False):
        try:
            cur = int(emo.get("batch_size", 4))
        except (TypeError, ValueError):
            cur = 4
        emo["batch_size"] = max(1, min(2, cur))


def _patch_global_config_for_e2e(
    base: Dict[str, Any],
    *,
    text_input_json: Path,
    triton_http_url: Optional[str],
    visual_relaxed: bool,
    visual_minimal: bool,
    local_visual_no_triton: bool,
    processors: frozenset[str],
    e2e_low_vram: bool = False,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)

    procs = cfg.setdefault("processors", {})
    text = procs.setdefault("text", {})
    run_text = "text" in processors
    text["enabled"] = run_text
    text["required"] = False
    if run_text:
        text["input_json"] = str(text_input_json.resolve())

    audio = procs.setdefault("audio", {})
    audio["enabled"] = "audio" in processors
    if "audio" in processors:
        _e2e_tighten_audio_batches_for_vram(audio)

    if e2e_low_vram:
        if run_text and "visual" in processors:
            for ename, ext in (text.get("extractors") or {}).items():
                if not isinstance(ext, dict):
                    continue
                if not ext.get("enabled", False):
                    continue
                nlow = str(ename).lower()
                dev = ext.get("device")
                dev_l = str(dev).strip().lower() if isinstance(dev, str) else ""
                heavy_name = any(
                    k in nlow
                    for k in (
                        "embed",
                        "aggregator",
                        "semantic",
                        "cosine_metrics",
                        "similar",
                        "cluster_entropy",
                    )
                )
                if heavy_name or dev_l in ("cuda", "gpu", "auto"):
                    ext["device"] = "cpu"
                    ext["fp16"] = False
        if "audio" in processors and "visual" in processors:
            sep_cfg = (audio.get("extractors") or {}).get("source_separation")
            if isinstance(sep_cfg, dict) and sep_cfg.get("enabled", False):
                sep_cfg["device"] = "cpu"

    vis = procs.setdefault("visual", {})
    vis["enabled"] = "visual" in processors
    if visual_relaxed:
        vis["required"] = False

    if "visual" in processors and not visual_minimal:
        _enable_full_visual_inline_config(cfg)

    if triton_http_url and "visual" in processors:
        g = vis.setdefault("inline_config", {}).setdefault("global", {})
        g["triton_http_url"] = triton_http_url

    if local_visual_no_triton and "visual" in processors:
        ic = vis.setdefault("inline_config", {})
        cp = ic.setdefault("core_providers", {})
        clip = ic.setdefault("core_clip", {})
        clip["runtime"] = "inprocess"
        cp["core_depth_midas"] = False
        cp["core_optical_flow"] = False
        if not triton_http_url:
            g = ic.setdefault("global", {})
            # Иначе в global_config остаётся localhost:8000 — порт Fetcher, не Triton.
            g.pop("triton_http_url", None)

    # DataProcessor main.py: final run_audio/run_text/run_visual after profile merge
    cfg["orchestration"] = {
        "processors": {
            "audio": "audio" in processors,
            "text": "text" in processors,
            "visual": "visual" in processors,
        }
    }

    return cfg


def _clear_fetcher_video_cache(
    *,
    platform: str,
    platform_video_id: str,
    dsn: str,
) -> None:
    try:
        import psycopg2
    except ImportError as e:
        raise RuntimeError("psycopg2 required for --cold-ingestion (pip install psycopg2-binary)") from e

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM videos WHERE platform = %s AND platform_video_id = %s",
                (platform, platform_video_id),
            )
            rows = cur.fetchall()
            if not rows:
                return
            vid = rows[0][0]
            cur.execute("DELETE FROM artifacts WHERE video_id = %s", (vid,))
            cur.execute("DELETE FROM comments WHERE video_id = %s", (vid,))
            cur.execute("DELETE FROM video_snapshots WHERE video_id = %s", (vid,))
            cur.execute("DELETE FROM video_metadata WHERE video_id = %s", (vid,))
            cur.execute("DELETE FROM channel_metadata WHERE video_id = %s", (vid,))
            cur.execute("DELETE FROM videos WHERE id = %s", (vid,))
        conn.commit()
    finally:
        conn.close()


@dataclass(frozen=True)
class ExampleSuiteItem:
    """Один прогон офлайн-E2E: локальное видео ``example_videos/{platform_video_id}.mp4`` + текст."""

    platform_video_id: str
    text_json: Optional[Path] = None
    audit_scenario_id: Optional[str] = None

    def __post_init__(self) -> None:
        has_file = self.text_json is not None
        has_audit = self.audit_scenario_id is not None
        if has_file == has_audit:
            raise ValueError(
                "ExampleSuiteItem: set exactly one of text_json or audit_scenario_id "
                f"(video={self.platform_video_id!r})"
            )


def builtin_example_suite_items(root: Path) -> List[ExampleSuiteItem]:
    """
    Расширяемый план офлайн-E2E: первые 7 — как раньше; слоты 8–10 — снова id роликов 1–3
    с другими audit-сценариями (тот же ``audit_v3_20_scenarios.json``), без новых ``*.mp4``;
    слоты 11–20 — снова mock из первых роликов + ``audit_v3_scen_06``…``15`` (для пилота 15+ / 17 без
    отдельного оркестратора; см. ``--example-suite-count``).

    Разрешение mock-видео совпадает с Fetcher: ``{id}.mp4``, иначе ``sample_{hash(id)%N}.mp4``,
    иначе единственный ``*.mp4`` в каталоге (см. ``_resolve_example_suite_video_path``).
    """
    ex = root / "example" / "example_text_documents"
    return [
        ExampleSuiteItem("-Q6fnPIybEI", text_json=ex / "video_document_1.json"),
        ExampleSuiteItem("-5EYUqIlyJU", text_json=ex / "video_document_2.json"),
        ExampleSuiteItem("-7Ei8e05x30", text_json=ex / "video_document_3.json"),
        ExampleSuiteItem("-15jH8mtfJw", text_json=ex / "video_document_4.json"),
        ExampleSuiteItem("-Ga4edhrfog", text_json=ex / "video_document_5.json"),
        ExampleSuiteItem("-FOB4jpQIg8", text_json=ex / "video_document_6.json"),
        ExampleSuiteItem("-BXwIsW0t9w", audit_scenario_id="audit_v3_scen_02"),
        # 8–10: те же mock-mp4, что у1–3 (в каталоге часто только {id}.mp4 без sample_*).
        ExampleSuiteItem("-Q6fnPIybEI", audit_scenario_id="audit_v3_scen_03"),
        ExampleSuiteItem("-5EYUqIlyJU", audit_scenario_id="audit_v3_scen_04"),
        ExampleSuiteItem("-7Ei8e05x30", audit_scenario_id="audit_v3_scen_05"),
        # 11–15: снова id из 1–5 + audit scen_06..10
        ExampleSuiteItem("-Q6fnPIybEI", audit_scenario_id="audit_v3_scen_06"),
        ExampleSuiteItem("-5EYUqIlyJU", audit_scenario_id="audit_v3_scen_07"),
        ExampleSuiteItem("-7Ei8e05x30", audit_scenario_id="audit_v3_scen_08"),
        ExampleSuiteItem("-15jH8mtfJw", audit_scenario_id="audit_v3_scen_09"),
        ExampleSuiteItem("-Ga4edhrfog", audit_scenario_id="audit_v3_scen_10"),
        # 16–20: scen_11..15 (в т.ч. plan 6 id + вращение 1–4)
        ExampleSuiteItem("-FOB4jpQIg8", audit_scenario_id="audit_v3_scen_11"),
        ExampleSuiteItem("-Q6fnPIybEI", audit_scenario_id="audit_v3_scen_12"),
        ExampleSuiteItem("-5EYUqIlyJU", audit_scenario_id="audit_v3_scen_13"),
        ExampleSuiteItem("-7Ei8e05x30", audit_scenario_id="audit_v3_scen_14"),
        ExampleSuiteItem("-15jH8mtfJw", audit_scenario_id="audit_v3_scen_15"),
    ]


def builtin_example_suite_7(root: Path) -> List[ExampleSuiteItem]:
    """Обратная совместимость: ровно первые 7 элементов плана."""
    return builtin_example_suite_items(root)[:7]


def _resolve_example_suite_video_path(root: Path, platform_video_id: str) -> Optional[Path]:
    """Та же логика, что ``Fetcher..._resolve_mock_sample_video_path`` (E2E-каталог примеров)."""
    vdir = root / "example" / "example_videos"
    if not vdir.is_dir():
        return None
    named = vdir / f"{platform_video_id}.mp4"
    if named.is_file():
        return named
    try:
        sample_count = int(os.environ.get("FETCHER_YOUTUBE_MOCK_SAMPLE_VIDEO_COUNT", "8"))
    except ValueError:
        sample_count = 8
    index = int(hashlib.sha256(platform_video_id.encode("utf-8")).hexdigest(), 16) % max(sample_count, 1)
    legacy = vdir / f"sample_{index}.mp4"
    if legacy.is_file():
        return legacy
    mp4s = sorted(vdir.glob("*.mp4"))
    if len(mp4s) == 1:
        return mp4s[0]
    return None


def _validate_suite_videos(root: Path, items: Sequence[ExampleSuiteItem]) -> Optional[str]:
    missing: List[str] = []
    for it in items:
        if _resolve_example_suite_video_path(root, it.platform_video_id) is None:
            missing.append(
                f"{it.platform_video_id}: нет ни {it.platform_video_id}.mp4, ни sample_* под "
                f"FETCHER_YOUTUBE_MOCK_SAMPLE_VIDEO_COUNT, ни ровно одного *.mp4 в example/example_videos"
            )
    if missing:
        return "Missing mock video files (как у Fetcher mock download):\n  " + "\n  ".join(missing)
    return None


def _youtube_result_store_has_success(root: Path, platform_video_id: str) -> bool:
    """
    True, если для ``platform_video_id`` уже есть завершённый прогон
    (``storage/result_store/youtube/<id>/<run_uuid>/manifest.json`` с ``run.status == "success"``).
    """
    base = root / "storage" / "result_store" / "youtube" / platform_video_id
    if not base.is_dir():
        return False
    try:
        for child in base.iterdir():
            if not child.is_dir():
                continue
            mf = child / "manifest.json"
            if not mf.is_file():
                continue
            try:
                with open(mf, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            run = data.get("run") if isinstance(data, dict) else None
            if isinstance(run, dict) and run.get("status") == "success":
                return True
    except OSError:
        return False
    return False


def _extract_scenario_video_document(
    scenarios_json: Path, scenario_id: Optional[str]
) -> Dict[str, Any]:
    with open(scenarios_json, encoding="utf-8") as f:
        data = json.load(f)
    scenarios = data.get("scenarios") or []
    if not scenarios:
        raise ValueError(f"No scenarios in {scenarios_json}")

    def _doc_from(entry: Dict[str, Any]) -> Dict[str, Any]:
        vd = (entry.get("inference") or {}).get("video_document")
        if not isinstance(vd, dict):
            raise ValueError(
                f"Scenario {entry.get('id')!r} has no inference.video_document object"
            )
        return vd

    if scenario_id:
        for s in scenarios:
            if s.get("id") == scenario_id:
                return _doc_from(s)
        raise ValueError(f"Scenario id not found: {scenario_id!r}")
    return _doc_from(scenarios[0])


def _load_base_config(root: Path) -> Dict[str, Any]:
    global_src = root / "DataProcessor" / "configs" / "global_config.yaml"
    if not global_src.is_file():
        raise FileNotFoundError(f"missing {global_src}")
    with open(global_src, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_text_document_path(args: argparse.Namespace, proc_set: frozenset[str]) -> Path:
    """Возвращает путь к VideoDocument JSON для TextProcessor (в т.ч. временный для audit)."""
    td_arg = Path(args.text_document)
    if args.offline_example and "text" in proc_set:
        if bool(getattr(args, "offline_use_example_text_file", False)):
            if not td_arg.is_file():
                raise FileNotFoundError(f"text document not found: {td_arg}")
            return td_arg.resolve()
        audit_path = Path(args.audit_scenarios_json)
        if not audit_path.is_file():
            raise FileNotFoundError(f"audit scenarios not found: {audit_path}")
        vd = _extract_scenario_video_document(audit_path, args.audit_scenario_id)
        text_doc = Path(tempfile.mkdtemp(prefix="e2e_full_max_audit_doc_")) / "video_document.json"
        text_doc.write_text(json.dumps(vd, ensure_ascii=False, indent=2), encoding="utf-8")
        return text_doc
    if "text" in proc_set and not td_arg.is_file():
        raise FileNotFoundError(f"text document not found: {td_arg}")
    return td_arg.resolve()


def _start_triton_and_get_url(args: argparse.Namespace, root: Path) -> tuple[Optional[str], bool]:
    """Старт Triton по флагу; иначе URL из env. Возвращает (url, started_locally)."""
    if args.with_triton_docker:
        if args.local_visual_no_triton:
            print(
                "WARN: --with-triton-docker with --local-visual-no-triton — Triton is started but YAML uses in-process CLIP.",
                flush=True,
            )
        try:
            triton_url = _start_triton_docker(root)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(f"Triton docker start: {e}") from e
        os.environ["TRITON_HTTP_URL"] = triton_url
        if not args.keep_triton_docker:
            atexit.register(_stop_triton_container)
        return triton_url, True
    triton_url = os.environ.get("TRITON_HTTP_URL") or os.environ.get("TRITON_HTTP")
    return triton_url, False


def _run_one_e2e(
    root: Path,
    args: argparse.Namespace,
    proc_set: frozenset[str],
    base_cfg: Dict[str, Any],
    triton_url: Optional[str],
    *,
    suite_label: Optional[str] = None,
) -> int:
    text_doc = _resolve_text_document_path(args, proc_set)

    storage = root / "storage"
    e2e_base = storage / "e2e_full_max"
    e2e_base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S_utc")
    run_tag = f"{ts}__{suite_label}" if suite_label else ts
    out_dir = e2e_base / run_tag
    out_dir.mkdir(parents=True, exist_ok=False)

    cfg_e2e = _patch_global_config_for_e2e(
        base_cfg,
        text_input_json=text_doc,
        triton_http_url=triton_url,
        visual_relaxed=not bool(args.visual_strict),
        visual_minimal=bool(args.visual_minimal),
        local_visual_no_triton=bool(args.local_visual_no_triton),
        processors=proc_set,
        e2e_low_vram=bool(getattr(args, "e2e_low_vram", False)),
    )
    out_yaml = out_dir / "global_config_e2e.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_e2e, f, sort_keys=False, allow_unicode=True)

    shutil.copy2(text_doc, out_dir / "text_input_video_document.json")

    marker = e2e_base / "active_global_config"
    marker.write_text(f"{out_yaml.resolve()}\n", encoding="utf-8")

    if args.cold_ingestion:
        dsn = args.fetcher_dsn.replace("postgresql+psycopg2://", "postgresql://")
        if not dsn:
            print("FAIL: --cold-ingestion needs FETCHER_POSTGRES_DSN or --fetcher-dsn", file=sys.stderr)
            return 2
        try:
            _clear_fetcher_video_cache(
                platform="youtube",
                platform_video_id=args.platform_video_id,
                dsn=dsn,
            )
        except Exception as e:
            print(f"FAIL: Fetcher cache clear: {e}", file=sys.stderr)
            return 2

    backend_dir = root / "backend"
    e2e_py = backend_dir / "scripts" / "e2e_run_to_complete.py"
    cmd = [
        sys.executable,
        str(e2e_py),
        "--source-url",
        args.source_url,
        "--fetcher-url",
        "http://localhost:8000",
        "--dataprocessor-url",
        "http://localhost:8002",
        "--with-dataprocessor",
        "--timeout",
        str(args.timeout),
        "--processing-grace-seconds",
        "900",
        "--poll-interval",
        "5",
        "--progress-heartbeat",
        "5",
        "--resource-snapshot-sec",
        str(getattr(args, "e2e_resource_snapshot_sec", 5)),
    ]
    if getattr(args, "e2e_cli_verbose", False):
        cmd.append("--verbose")
    if not getattr(args, "no_e2e_events_jsonl", False):
        cmd.extend(["--e2e-events-jsonl", str(out_dir / "orchestrator_events.jsonl")])
    env = os.environ.copy()
    if triton_url:
        env["TRITON_HTTP_URL"] = triton_url
    prefix = f"[{suite_label}] " if suite_label else ""
    print(f"{prefix}E2E platform_video_id={args.platform_video_id} text={text_doc}", flush=True)
    proc = subprocess.run(cmd, cwd=str(backend_dir), env=env)
    exit_code = proc.returncode

    latest_logs = backend_dir / ".e2e" / "logs" / "latest"
    log_dest = out_dir / "e2e_stack_logs"
    if latest_logs.is_symlink() or latest_logs.exists():
        try:
            shutil.copytree(latest_logs.resolve(), log_dest, symlinks=True)
        except Exception as e:
            (out_dir / "copy_logs_error.txt").write_text(str(e), encoding="utf-8")

    summary = {
        "run_tag": run_tag,
        "suite_label": suite_label,
        "output_dir": str(out_dir),
        "global_config_e2e": str(out_yaml),
        "active_global_config_marker": str(marker),
        "processors": sorted(proc_set),
        "text_document": str(text_doc.resolve()) if text_doc.is_file() else None,
        "source_url": args.source_url,
        "cold_ingestion": bool(args.cold_ingestion),
        "platform_video_id": args.platform_video_id,
        "triton_http_url": triton_url,
        "with_subprocess_triton_docker": bool(args.with_triton_docker),
        "keep_triton_docker": bool(args.keep_triton_docker),
        "local_visual_no_triton": bool(args.local_visual_no_triton),
        "visual_minimal": bool(args.visual_minimal),
        "visual_full_profile": not bool(args.visual_minimal),
        "offline_use_example_text_file": bool(getattr(args, "offline_use_example_text_file", False)),
        "e2e_exit_code": exit_code,
        "orchestrator_events_jsonl": None
        if getattr(args, "no_e2e_events_jsonl", False)
        else str(out_dir / "orchestrator_events.jsonl"),
        "e2e_low_vram": bool(getattr(args, "e2e_low_vram", False)),
        "note": "Без --cold-ingestion кеш Fetcher сохраняется (рекомендуется при проблемах с YouTube).",
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not args.keep_active_global_config:
        try:
            marker.unlink()
        except OSError:
            pass

    print(f"{prefix}Artifacts: {out_dir}", flush=True)
    return exit_code


def main() -> int:
    root = _repo_root()
    p = argparse.ArgumentParser(description="Full E2E with max DataProcessor global_config + TextProcessor")
    p.add_argument(
        "--source-url",
        default="https://www.youtube.com/watch?v=DYor3e2effY",
        help="URL для POST /api/runs (должен совпадать с platform_video_id при cold ingest)",
    )
    p.add_argument(
        "--platform-video-id",
        default="DYor3e2effY",
        help="id ролика для сброса кеша Fetcher (--cold-ingestion)",
    )
    p.add_argument(
        "--text-document",
        type=Path,
        default=root / "example" / "example_text_documents" / "video_document_1.json",
        help="VideoDocument JSON для TextProcessor (processors.text.input_json)",
    )
    p.add_argument(
        "--offline-example",
        action="store_true",
        help=(
            "Локальный E2E без сети Fetcher: source_url = watch?v=--example-youtube-id, "
            "текст по умолчанию из text_audit_v3_smoke (см. --audit-scenarios-json), "
            "либо из --text-document при --offline-use-example-text-file. "
            "Нужны FETCHER_YOUTUBE_MOCK_VIDEO_DOWNLOAD и example/example_videos/{id}.mp4 (см. e2e_env.sh)."
        ),
    )
    p.add_argument(
        "--offline-use-example-text-file",
        action="store_true",
        help=(
            "Вместе с --offline-example и процессором text: брать VideoDocument из --text-document "
            "(example/example_text_documents/*.json), а не из audit scenarios."
        ),
    )
    p.add_argument(
        "--example-suite-7",
        action="store_true",
        help=(
            "Последовательные офлайн-прогоны по встроенному плану (по умолчанию первые 7 роликов; "
            "см. --example-suite-count). Mock-видео: как у Fetcher — {id}.mp4 или sample_{hash%%N}. "
            "Тексты: video_document_1..6.json + audit сценарии. "
            "По умолчанию пропускает platform_video_id с успешным result_store; см. --example-suite-force-all. "
            "Включает --offline-example; несовместимо с --cold-ingestion."
        ),
    )
    p.add_argument(
        "--example-suite-count",
        type=int,
        default=7,
        metavar="N",
        help=(
            "С --example-suite-7: взять первые N роликов из встроенного плана "
            f"(допустимо 1..{len(builtin_example_suite_items(root))}; слоты 8+ — audit + повторы mock-id, см. "
            "builtin_example_suite_items в коде)."
        ),
    )
    p.add_argument(
        "--example-suite-force-all",
        action="store_true",
        help=(
            "С --example-suite-7: запускать все выбранные примеры, даже если для platform_video_id "
            "уже есть успешный результат в storage/result_store/youtube/."
        ),
    )
    p.add_argument(
        "--example-youtube-id",
        default="-Q6fnPIybEI",
        help="Имя example/example_videos/{id}.mp4 и query v= для YouTube URL (только идентификатор кеша; видео с диска).",
    )
    p.add_argument(
        "--audit-scenarios-json",
        type=Path,
        default=root / "example" / "text_audit_v3_smoke" / "scenarios" / "audit_v3_20_scenarios.json",
        help="JSON со сценариями audit v3 (берётся inference.video_document).",
    )
    p.add_argument(
        "--audit-scenario-id",
        default=None,
        help="id сценария из audit JSON (по умолчанию — первый в списке).",
    )
    p.add_argument(
        "--cold-ingestion",
        action="store_true",
        help="Удалить запись videos/артефакты для --platform-video-id в БД Fetcher (нужен YouTube/сеть)",
    )
    p.add_argument(
        "--visual-strict",
        action="store_true",
        help="Не ослаблять visual.required (как в global_config)",
    )
    p.add_argument(
        "--visual-minimal",
        action="store_true",
        help="Не включать все visual core/modules: оставить булевы флаги как в global_config.yaml (быстрее).",
    )
    p.add_argument(
        "--visual-modules-max",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--with-triton-docker",
        action="store_true",
        help="Поднять nvcr.io/nvidia/tritonserver (см e2e_triton_docker.sh); HTTP на TRITON_E2E_HTTP_PORT (default 8010)",
    )
    p.add_argument(
        "--keep-triton-docker",
        action="store_true",
        help="После прогона не останавливать контейнер Triton",
    )
    p.add_argument(
        "--local-visual-no-triton",
        action="store_true",
        help="Урезанный visual: CLIP in-process, без MiDaS/RAFT (без Triton/GPU для этих ядер)",
    )
    p.add_argument(
        "--processors",
        default="audio,text,visual",
        help=(
            "Подмножество процессоров через запятую: audio, text, visual "
            "(Segmenter выполняется всегда). Пример: --processors text"
        ),
    )
    p.add_argument(
        "--keep-active-global-config",
        action="store_true",
        help="Не удалять storage/e2e_full_max/active_global_config после успеха",
    )
    p.add_argument("--timeout", type=int, default=7200, help="Таймаут e2e_run_to_complete (сек)")
    p.add_argument(
        "--e2e-resource-snapshot-sec",
        type=int,
        default=5,
        metavar="N",
        help="Каждые N сек в лог e2e_run_to_complete: RAM, load1, GPU (0=выкл).",
    )
    p.add_argument(
        "--dp-wait-capacity-sec",
        type=int,
        default=7200,
        help=(
            "С --example-suite-7: перед каждым роликом ждать, пока DP /health metrics "
            "active_runs < max_concurrent_runs (0 = не ждать)."
        ),
    )
    p.add_argument(
        "--dp-wait-poll-sec",
        type=float,
        default=5.0,
        help="Интервал опроса GET /api/v1/health при ожидании слота в DataProcessor.",
    )
    p.add_argument(
        "--no-dp-wait-capacity",
        action="store_true",
        help="Не ждать свободный слот в DataProcessor между роликами сьюта.",
    )
    p.add_argument(
        "--e2e-low-vram",
        action="store_true",
        help=(
            "Узкая VRAM: source_separation и тяжёлые текстовые экстракторы (embed/aggregator/semantic и т.д.) на CPU "
            "при одновременном visual+text. Также cuda/gpu/auto → cpu в text.extractors."
        ),
    )
    p.add_argument(
        "--e2e-cli-verbose",
        action="store_true",
        help="Печатать в stdout расширенные снимки Backend API (--verbose у e2e_run_to_complete); по умолчанию выкл.",
    )
    p.add_argument(
        "--no-e2e-events-jsonl",
        action="store_true",
        help="Не писать orchestrator_events.jsonl в каталог артефактов прогона.",
    )
    p.add_argument(
        "--no-e2e-host-scrub",
        action="store_true",
        help="С --example-suite-7: не вызывать scripts/e2e_host_memory_scrub.sh между роликами.",
    )
    p.add_argument("--fetcher-dsn", default=os.environ.get("FETCHER_POSTGRES_DSN", ""))
    args = p.parse_args()

    try:
        proc_set = _parse_processors_csv(args.processors)
    except ValueError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 2

    if args.example_suite_7:
        if args.cold_ingestion:
            print("FAIL: --example-suite-7 is not compatible with --cold-ingestion", file=sys.stderr)
            return 2
        args.offline_example = True

    if args.offline_example:
        args.source_url = f"https://www.youtube.com/watch?v={args.example_youtube_id}"
        args.platform_video_id = args.example_youtube_id

    try:
        base_cfg = _load_base_config(root)
    except FileNotFoundError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 2

    try:
        triton_url, started_triton = _start_triton_and_get_url(args, root)
    except RuntimeError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 2

    if not args.local_visual_no_triton and not triton_url:
        print(
            "FAIL: full visual needs a Triton HTTP endpoint (depth/RAFT are triton-only).\n"
            "  Either: python scripts/e2e_full_max_run.py --with-triton-docker\n"
            "  Or:      export TRITON_HTTP_URL=http://127.0.0.1:8010\n"
            "  Or:      --local-visual-no-triton (reduced visual).",
            file=sys.stderr,
        )
        if started_triton:
            _stop_triton_container()
        return 2

    if args.example_suite_7:
        plan_all = builtin_example_suite_items(root)
        n_plan = int(args.example_suite_count)
        if n_plan < 1 or n_plan > len(plan_all):
            print(
                f"FAIL: --example-suite-count must be between 1 and {len(plan_all)} (got {n_plan})",
                file=sys.stderr,
            )
            if started_triton and not args.keep_triton_docker:
                _stop_triton_container()
            return 2
        suite = plan_all[:n_plan]
        audit_path = Path(args.audit_scenarios_json)
        if not audit_path.is_file():
            print(f"FAIL: audit scenarios not found: {audit_path}", file=sys.stderr)
            if started_triton and not args.keep_triton_docker:
                _stop_triton_container()
            return 2

        skipped_plan_indices: set[int] = set()
        to_run: List[tuple[int, ExampleSuiteItem]] = []
        for plan_i, item in enumerate(suite, start=1):
            if not args.example_suite_force_all and _youtube_result_store_has_success(
                root, item.platform_video_id
            ):
                skipped_plan_indices.add(plan_i)
                print(
                    f"SKIP [{plan_i}/{len(suite)}] platform_video_id={item.platform_video_id} "
                    f"(result_store youtube: successful manifest already exists)",
                    flush=True,
                )
                continue
            to_run.append((plan_i, item))

        msg = _validate_suite_videos(root, [it for _, it in to_run])
        if msg:
            print(f"FAIL: {msg}", file=sys.stderr)
            if started_triton and not args.keep_triton_docker:
                _stop_triton_container()
            return 2
        for _, it in to_run:
            if it.text_json and not it.text_json.is_file():
                print(f"FAIL: text JSON not found: {it.text_json}", file=sys.stderr)
                if started_triton and not args.keep_triton_docker:
                    _stop_triton_container()
                return 2

        executed_codes: List[int] = []
        manifest_items: List[Dict[str, Any]] = []
        suite_root_dir = root / "storage" / "e2e_full_max" / (
            f"example_suite_{n_plan}_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S_utc')}"
        )
        suite_root_dir.mkdir(parents=True, exist_ok=True)
        run_ordinal = 0
        for plan_i, item in enumerate(suite, start=1):
            if plan_i in skipped_plan_indices:
                manifest_items.append(
                    {
                        "index": plan_i,
                        "platform_video_id": item.platform_video_id,
                        "text_json": str(item.text_json) if item.text_json else None,
                        "audit_scenario_id": item.audit_scenario_id,
                        "skipped": True,
                        "skip_reason": "result_store_success_exists",
                        "exit_code": None,
                    }
                )
                continue
            run_ordinal += 1
            run_args = copy.copy(args)
            run_args.offline_example = True
            run_args.example_youtube_id = item.platform_video_id
            run_args.source_url = f"https://www.youtube.com/watch?v={item.platform_video_id}"
            run_args.platform_video_id = item.platform_video_id
            run_args.cold_ingestion = False
            if item.text_json is not None:
                run_args.text_document = item.text_json
                run_args.offline_use_example_text_file = True
                run_args.audit_scenario_id = None
            else:
                run_args.offline_use_example_text_file = False
                run_args.audit_scenario_id = item.audit_scenario_id
            label = (
                f"exsuite_plan_{plan_i}_of_{len(suite)}__run_{run_ordinal}_of_{len(to_run)}__"
                f"vid_{item.platform_video_id}"
            )
            if not args.no_dp_wait_capacity and args.dp_wait_capacity_sec > 0:
                dp_url = os.environ.get("TF_BACKEND_DATAPROCESSOR_API_URL", "http://localhost:8002")
                dp_key = os.environ.get("TF_BACKEND_DATAPROCESSOR_API_KEY") or os.environ.get(
                    "DATAPROCESSOR_API_KEY"
                )
                _wait_dataprocessor_slot(
                    dataprocessor_url=dp_url,
                    api_key=dp_key,
                    max_wait_sec=int(args.dp_wait_capacity_sec),
                    poll_sec=float(args.dp_wait_poll_sec),
                    label=label,
                )
            code: int
            try:
                code = _run_one_e2e(
                    root,
                    run_args,
                    proc_set,
                    base_cfg,
                    triton_url,
                    suite_label=label,
                )
            except (FileNotFoundError, ValueError) as e:
                print(f"FAIL: {e}", file=sys.stderr)
                code = 2
            executed_codes.append(code)
            manifest_items.append(
                {
                    "index": plan_i,
                    "platform_video_id": item.platform_video_id,
                    "text_json": str(item.text_json) if item.text_json else None,
                    "audit_scenario_id": item.audit_scenario_id,
                    "skipped": False,
                    "skip_reason": None,
                    "exit_code": code,
                }
            )
            _between_suite_videos_cleanup(
                plan_i,
                code,
                item.platform_video_id,
                run_host_scrub=not bool(getattr(args, "no_e2e_host_scrub", False)),
            )

        overall = max(executed_codes) if executed_codes else 0
        manifest = {
            "kind": "example_suite",
            "example_suite_count": n_plan,
            "plan_total_builtin": len(plan_all),
            "skipped_due_to_result_store": len(skipped_plan_indices),
            "ran_count": len(to_run),
            "exit_codes_executed": executed_codes,
            "overall_exit": overall,
            "items": manifest_items,
        }
        (suite_root_dir / "suite_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"\nExample suite manifest (N={n_plan}): {suite_root_dir / 'suite_manifest.json'}",
            flush=True,
        )
        return overall

    try:
        return _run_one_e2e(root, args, proc_set, base_cfg, triton_url, suite_label=None)
    except (FileNotFoundError, ValueError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        if started_triton and not args.keep_triton_docker:
            _stop_triton_container()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
