import os
import subprocess
import shutil
import sys
import tempfile
import uuid
import hashlib
import time
import re
import json
from pathlib import Path
import numpy as np

# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Text colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

def _best_effort_torch_cuda_empty_cache_in_parent() -> None:
    """После тяжёлого Audio/Text subprocess — сбросить кэш alloc CUDA в процессе DataProcessor main."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc):
                ipc()
    except Exception:
        pass


def _is_tty() -> bool:
    """Проверяет, является ли stdout терминалом (для включения цветов)."""
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

def _colorize(text: str, color: str) -> str:
    """Добавляет цвет к тексту, если вывод в терминал."""
    if _is_tty():
        return f"{color}{text}{Colors.RESET}"
    return text

def _shorten_path(path: str, max_length: int = 60) -> str:
    """Сокращает путь, оставляя только важные части."""
    if not path:
        return path
    
    # Абсолютный путь -> относительный от workspace
    workspace = os.getcwd()
    if path.startswith(workspace):
        rel_path = os.path.relpath(path, workspace)
        if len(rel_path) <= max_length:
            return rel_path
        # Если все еще длинный, обрезаем начало
        parts = rel_path.split(os.sep)
        if len(parts) > 3:
            return os.sep.join(['...'] + parts[-3:])
        return rel_path
    
    # Если путь не в workspace, обрезаем начало
    parts = path.split(os.sep)
    if len(parts) > 4:
        return os.sep.join(['...'] + parts[-4:])
    return path

def _print_header(text: str):
    """Печатает заголовок секции."""
    print(_colorize(f"\n{'='*80}", Colors.CYAN))
    print(_colorize(f"  {text}", Colors.BOLD + Colors.CYAN))
    print(_colorize(f"{'='*80}", Colors.CYAN))

def _print_step(step: str, status: str = "OK", details: str = None):
    """Печатает шаг выполнения."""
    status_color = Colors.GREEN if status == "OK" else Colors.RED if status == "ERROR" else Colors.YELLOW
    status_symbol = "✓" if status == "OK" else "✗" if status == "ERROR" else "⚠"
    
    status_text = _colorize(f"[{status_symbol} {status}]", status_color)
    step_text = _colorize(step, Colors.BOLD)
    
    if details:
        details_text = _colorize(f"  {details}", Colors.DIM)
        print(f"  {status_text} {step_text}\n      {details_text}")
    else:
        print(f"  {status_text} {step_text}")

def _print_info(text: str, indent: int = 4):
    """Печатает информационное сообщение."""
    indent_str = " " * indent
    print(f"{indent_str}{_colorize('→', Colors.BLUE)} {_colorize(text, Colors.DIM)}")

def _format_log_line(line: str, processor: str = None) -> str | None:
    """Форматирует строку лога из subprocess."""
    line = line.strip()
    if not line:
        return None
    
    # Определяем тип сообщения по префиксам
    if "[Segmenter]" in line or "[Segmenter.run]" in line:
        processor = "Segmenter"
        if "built" in line.lower() or "starting" in line.lower():
            # Информационное сообщение
            if "built" in line.lower():
                # Извлекаем количество и путь
                match = re.search(r'built (\d+)', line)
                count = match.group(1) if match else "?"
                path_match = re.search(r'from (.+)$', line)
                path = _shorten_path(path_match.group(1) if path_match else "")
                return f"  {_colorize('→', Colors.BLUE)} {_colorize('Segmenter', Colors.CYAN)}: построено {_colorize(count, Colors.BOLD)} конфигов из {_colorize(path, Colors.DIM)}"
            elif "starting processing" in line.lower():
                path_match = re.search(r'processing (.+)$', line)
                path = _shorten_path(path_match.group(1) if path_match else "")
                return f"  {_colorize('→', Colors.BLUE)} {_colorize('Segmenter', Colors.CYAN)}: обработка {_colorize(path, Colors.DIM)}"
            else:
                return f"  {_colorize('→', Colors.BLUE)} {_colorize('Segmenter', Colors.CYAN)}: {_colorize(line.replace('[Segmenter]', '').replace('[Segmenter.run]', '').strip(), Colors.DIM)}"
        elif "saved" in line.lower() or "wrote" in line.lower():
            # Успешное сохранение
            path_match = re.search(r'-> (.+)$', line) or re.search(r'to (.+)$', line)
            if path_match:
                path = _shorten_path(path_match.group(1))
                return f"    {_colorize('✓', Colors.GREEN)} Сохранено: {_colorize(path, Colors.DIM)}"
            return f"    {_colorize('✓', Colors.GREEN)} {_colorize(line.replace('[Segmenter]', '').strip(), Colors.DIM)}"
        else:
            # Обычное сообщение
            return f"    {_colorize(line.replace('[Segmenter]', '').replace('[Segmenter.run]', '').strip(), Colors.DIM)}"
    
    elif "[TextProcessor" in line or "TextProcessor:" in line:
        processor = "TextProcessor"
        if "loaded document" in line.lower():
            path_match = re.search(r'from (.+)$', line)
            path = _shorten_path(path_match.group(1) if path_match else "")
            return f"  {_colorize('→', Colors.BLUE)} {_colorize('TextProcessor', Colors.MAGENTA)}: загружен документ из {_colorize(path, Colors.DIM)}"
        elif "starting" in line.lower() and "extractor" in line.lower():
            count_match = re.search(r'(\d+) extractor', line)
            count = count_match.group(1) if count_match else "?"
            return f"  {_colorize('→', Colors.BLUE)} {_colorize('TextProcessor', Colors.MAGENTA)}: запуск {_colorize(count, Colors.CYAN)} экстракторов"
        elif "running" in line.lower() and "extractor" in line.lower() or "embedder" in line.lower():
            # Format: "TextProcessor: [1/4] running LexicalStatsExtractor (device=cpu)"
            # Игнорируем строки с params (слишком длинные и дублируются)
            if "params=" in line.lower():
                return None  # Пропускаем строки с параметрами
            extractor_match = re.search(r'running (\w+(?:Extractor|Embedder))', line)
            progress_match = re.search(r'\[(\d+)/(\d+)\]', line)
            device_match = re.search(r'device=(\w+)', line)
            extractor = extractor_match.group(1) if extractor_match else "extractor"
            progress = progress_match.group(0) if progress_match else ""
            device = device_match.group(1) if device_match else "cpu"
            device_color = Colors.CYAN if device == "cuda" else Colors.DIM
            return f"    {_colorize('→', Colors.BLUE)} {_colorize(progress, Colors.CYAN)} {_colorize(extractor, Colors.YELLOW)} {_colorize(f'({device})', device_color)}"
        elif "completed" in line.lower() and ("extractor" in line.lower() or "embedder" in line.lower()):
            # Format: "TextProcessor: [1/4] LexicalStatsExtractor completed (ok) (0.123s) (16 features)"
            extractor_match = re.search(r'(\w+(?:Extractor|Embedder))', line)
            progress_match = re.search(r'\[(\d+)/(\d+)\]', line)
            status_match = re.search(r'\((ok|empty|error)\)', line)
            time_match = re.search(r'\(([\d.]+)s\)', line)
            features_match = re.search(r'\((\d+) features?\)', line)
            extractor = extractor_match.group(1) if extractor_match else "extractor"
            progress = progress_match.group(0) if progress_match else ""
            status = status_match.group(1) if status_match else "ok"
            time_str = time_match.group(1) if time_match else ""
            features_str = features_match.group(1) if features_match else None
            status_color = Colors.GREEN if status == "ok" else (Colors.YELLOW if status == "empty" else Colors.RED)
            status_icon = "✓" if status == "ok" else ("○" if status == "empty" else "✗")
            # Компактное форматирование: [progress] extractor time [features]
            time_part = f"{_colorize(time_str, Colors.DIM)}s" if time_str else ""
            features_part = f" {_colorize(f'({features_str} feat)', Colors.DIM)}" if features_str else ""
            return f"    {_colorize(status_icon, status_color)} {_colorize(progress, Colors.CYAN)} {_colorize(extractor, Colors.YELLOW)} {time_part}{features_part}"
        elif "failed" in line.lower() or "raised exception" in line.lower() or "traceback:" in line.lower():
            # Обработка ошибок с traceback
            if "traceback:" in line.lower():
                # Это строка с traceback - показываем как есть, но с цветом
                return f"    {_colorize(line.strip(), Colors.RED)}"
            extractor_match = re.search(r'(\w+Extractor|\w+Embedder)', line)
            progress_match = re.search(r'\[(\d+)/(\d+)\]', line)
            extractor = extractor_match.group(1) if extractor_match else "extractor"
            progress = progress_match.group(0) if progress_match else ""
            # Извлекаем детали ошибки
            error_match = re.search(r'error:\s*(.+?)(?:\n|$)', line, re.IGNORECASE)
            error_msg = error_match.group(1).strip() if error_match else "unknown error"
            if len(error_msg) > 100:
                error_msg = error_msg[:97] + "..."
            return f"    {_colorize('✗', Colors.RED)} {_colorize(progress, Colors.CYAN)} {_colorize(extractor, Colors.YELLOW)} {_colorize('failed', Colors.RED)}: {_colorize(error_msg, Colors.RED)}"
        elif "completed with status" in line.lower():
            status_match = re.search(r'status=(\w+)', line)
            status = status_match.group(1) if status_match else "unknown"
            status_color = Colors.GREEN if status == "ok" else Colors.RED
            return f"  {_colorize('✓', Colors.GREEN)} {_colorize('TextProcessor', Colors.MAGENTA)}: завершено со статусом {_colorize(status.upper(), status_color)}"
        elif "wrote npz" in line.lower():
            path_match = re.search(r'to (.+)$', line)
            path = _shorten_path(path_match.group(1) if path_match else "")
            return f"    {_colorize('✓', Colors.GREEN)} NPZ сохранен: {_colorize(path, Colors.DIM)}"
        else:
            return f"    {_colorize(line.replace('[TextProcessor', '').replace('TextProcessor:', '').strip(), Colors.DIM)}"
    
    elif "src.core.renderer" in line or "src.extractors" in line:
        if "Render-context saved" in line or "HTML render saved" in line or "HTML report generated" in line:
            path_match = re.search(r'to (.+)$', line) or re.search(r': (.+)$', line)
            if path_match:
                path = _shorten_path(path_match.group(1))
                # Пытаемся извлечь имя экстрактора из пути или строки
                extractor_match = re.search(r'extractors\.(\w+)\.', line) or re.search(r'/(\w+)_report\.html', path) or re.search(r'/(\w+)/render', path)
                extractor = extractor_match.group(1) if extractor_match else "renderer"
                # Красивое имя экстрактора (замена подчеркиваний на пробелы)
                extractor_display = extractor.replace("_", " ").replace("extractor", "").replace("embedder", "").strip()
                if not extractor_display:
                    extractor_display = extractor
                return f"    {_colorize('✓', Colors.GREEN)} {_colorize(extractor_display, Colors.YELLOW)}: отчет сохранен → {_colorize(path, Colors.DIM)}"
        return None  # Пропускаем технические строки renderer
    
    elif "[extract_audio]" in line or "[process_video_union]" in line:
        if "saved" in line.lower() or "already exists" in line.lower():
            path_match = re.search(r'-> (.+)$', line) or re.search(r': (.+)$', line)
            if path_match:
                path = _shorten_path(path_match.group(1))
                return f"    {_colorize('✓', Colors.GREEN)} {_colorize(path, Colors.DIM)}"
        return f"    {_colorize(line, Colors.DIM)}"

    # VisualProcessor / другие CLI с единым форматом логов:
    # "2026-02-12 23:40:07,261 INFO: VisualProcessor | main | core_provider core_clip start"
    m = re.match(
        r"^(?P<ts>\d{4}-\d{2}-\d{2} [0-9:,]+)\s+(?P<level>[A-Z]+):\s+(?P<msg>.*)$",
        line,
    )
    if m:
        ts = m.group("ts")
        level = m.group("level")
        msg = m.group("msg")

        # Выделяем namespace VisualProcessor / Segmenter / AudioProcessor и т.п.
        parts = [p.strip() for p in msg.split("|")]
        namespace = None
        scope = None
        core_msg = msg

        if len(parts) >= 2:
            namespace = parts[0]            # например, "VisualProcessor"
            scope = parts[1]                # например, "main"
            core_msg = " | ".join(parts[2:]) if len(parts) > 2 else ""

        # Менее информативные части убираем: оставляем единый шаблон
        #   TS [LEVEL] [NAMESPACE/SCOPE] message
        ns_scope = None
        if namespace and scope:
            ns_scope = f"{namespace}/{scope}"
        elif namespace:
            ns_scope = namespace

        level_str = f"[{level}]"
        if ns_scope:
            return f"    {ts} {level_str} {_colorize(f'[{ns_scope}]', Colors.CYAN)} {core_msg}"
        # fallback: без namespace
        return f"    {ts} {level_str} {core_msg}"
    
    # Общие паттерны
    if "error" in line.lower() or "failed" in line.lower() or "exception" in line.lower():
        return f"    {_colorize('✗', Colors.RED)} {_colorize(line, Colors.RED)}"
    elif "warning" in line.lower() or "warn" in line.lower():
        return f"    {_colorize('⚠', Colors.YELLOW)} {_colorize(line, Colors.YELLOW)}"
    
    # По умолчанию - просто возвращаем с отступом
    return f"    {_colorize(line, Colors.DIM)}"

def _run_subprocess_with_formatted_output(cmd: list, processor_name: str = None, check: bool = False, env: dict = None) -> subprocess.CompletedProcess:
    """Запускает subprocess и форматирует его вывод в реальном времени."""
    import re
    
    # Объединяем текущее окружение с переданным
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    
    # Запускаем процесс с перехватом вывода
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=process_env
    )
    
    # Читаем вывод построчно и форматируем
    output_lines = []
    last_formatted = None  # Для фильтрации дубликатов
    for line in process.stdout:
        formatted = _format_log_line(line, processor_name)
        if formatted and formatted != last_formatted:  # Пропускаем дубликаты
            print(formatted)
            last_formatted = formatted
        output_lines.append(line)
    
    # Ждем завершения процесса
    returncode = process.wait()
    
    # Создаем CompletedProcess для совместимости
    result = subprocess.CompletedProcess(
        cmd,
        returncode,
        stdout=''.join(output_lines),
        stderr=''
    )
    
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd, result.stdout, result.stderr)
    
    return result


def _dp_strip_ansi_for_error_text(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s or "")


def _line_is_error_candidate(ln: str) -> bool:
    """Строка похожа на сообщение об ошибке (не рутинный INFO)."""
    s = _dp_strip_ansi_for_error_text(ln)
    if re.search(r"\[(?:ERROR|CRITICAL)\]", s, re.I):
        return True
    if "Traceback (most recent call last)" in s:
        return True
    if re.search(
        r"\b(?:CalledProcessError|FileNotFoundError|RuntimeError|ValueError|KeyError|"
        r"OSError|ImportError|ModuleNotFoundError|AssertionError|KeyboardInterrupt)\b",
        s,
    ):
        return True
    if (
        re.search(r"\b(?:Error|Exception)\s*:\s*\S", s)
        and not re.search(r"\[INFO\]", s, re.I)
    ):
        return True
    if re.search(r"^\s*(?:raise |During handling of the)", s):
        return True
    if re.search(r"\b(?:FAIL|FATAL|fatal error)\b", s, re.I) and not re.search(
        r"\[INFO\]", s, re.I
    ):
        return True
    return False


def _line_is_benign_tail_noise(ln: str) -> bool:
    """Успешные/шумные INFO в конце лога, которые не объясняют ненулевой exit."""
    s = _dp_strip_ansi_for_error_text(ln)
    if not re.search(r"\[INFO\]", s, re.I):
        return False
    if re.search(
        r"(?:saved to|HTML render|render saved|Cosine metrics|Writing |Wrote |"
        r"completed successfully|MainProcessor initialized|Initializing MainProcessor|"
        r"Initializing .*?extractors|extractors\.\.\.)",
        s,
        re.I,
    ):
        return True
    return False


def _subprocess_output_error_detail(
    completed: subprocess.CompletedProcess,
    *,
    tail_non_empty_lines: int = 14,
    max_chars: int = 1400,
) -> str:
    """
    Сообщение для run_state/API при ненулевом коде: exit + фрагмент stdout.
    Предпочитаем строки с ERROR/Traceback/исключениями; отбрасываем хвост из «успешных» INFO.
    """
    rc = int(completed.returncode or 0)
    if rc == 0:
        return ""
    raw = (completed.stdout or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return f"exit={rc} (no subprocess output captured)"
    lines = [_dp_strip_ansi_for_error_text(ln.strip()) for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return f"exit={rc}"

    cand_idx = [i for i, ln in enumerate(lines) if _line_is_error_candidate(ln)]
    if cand_idx:
        i = cand_idx[-1]
        lo = max(0, i - 3)
        hi = min(len(lines), i + tail_non_empty_lines)
        snip = lines[lo:hi]
    else:
        window = lines[-max(80, tail_non_empty_lines * 5) :]
        meaningful = [ln for ln in window if not _line_is_benign_tail_noise(ln)]
        snip = (
            meaningful[-tail_non_empty_lines:]
            if meaningful
            else window[-tail_non_empty_lines:]
        )

    msg = " · ".join(snip)
    msg = re.sub(r"\s+", " ", msg).strip()
    if len(msg) > max_chars:
        msg = msg[: max_chars - 3] + "…"
    return f"exit={rc}: {msg}"

_path = os.path.dirname(__file__)

def _get_data_venv_python() -> str:
    """
    Возвращает путь к Python из DataProcessor/.data_venv.
    Если venv не найден, возвращает sys.executable (fallback) с предупреждением.
    """
    venv_python = os.path.join(_path, ".data_venv", "bin", "python")
    if os.path.exists(venv_python):
        return venv_python
    # Fallback: используем текущий интерпретатор, но выводим предупреждение
    import warnings
    warnings.warn(
        f"DataProcessor/.data_venv не найден. Используется текущий интерпретатор: {sys.executable}. "
        "Рекомендуется создать виртуальное окружение: python3 -m venv DataProcessor/.data_venv",
        UserWarning
    )
    return sys.executable

def _get_processor_venv_python(processor_name: str, venv_name: str) -> str:
    """
    Возвращает путь к Python из venv процессора.
    
    Args:
        processor_name: Имя процессора (для сообщений об ошибках)
        venv_name: Имя venv (например, ".tp_venv", ".ap_venv", ".vp_venv")
    
    Returns:
        Путь к Python из venv или fallback на .data_venv или sys.executable
    """
    venv_python = os.path.join(_path, processor_name, venv_name, "bin", "python")
    if os.path.exists(venv_python):
        return venv_python
    
    # Fallback 1: пробуем .data_venv
    data_venv_python = _get_data_venv_python()
    if data_venv_python != sys.executable:
        import warnings
        warnings.warn(
            f"{processor_name}/{venv_name} не найден. Используется DataProcessor/.data_venv. "
            f"Рекомендуется создать: python3 -m venv {processor_name}/{venv_name}",
            UserWarning
        )
        return data_venv_python
    
    # Fallback 2: sys.executable
    import warnings
    warnings.warn(
        f"{processor_name}/{venv_name} не найден. Используется текущий интерпретатор: {sys.executable}",
        UserWarning
    )
    return sys.executable

def _get_audio_venv_python() -> str:
    """Возвращает Python из AudioProcessor/.ap_venv или fallback на .data_venv."""
    return _get_processor_venv_python("AudioProcessor", ".ap_venv")

def _get_text_venv_python() -> str:
    """Возвращает Python из TextProcessor/.tp_venv или fallback на .data_venv."""
    return _get_processor_venv_python("TextProcessor", ".tp_venv")

def _get_visual_venv_python() -> str:
    """Возвращает Python из VisualProcessor/.vp_venv или fallback на .data_venv."""
    return _get_processor_venv_python("VisualProcessor", ".vp_venv")

def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found in PATH: {name}")

def _probe_video_duration_sec(video_path: str) -> float:
    """
    Prod behavior: fail-fast early for too-long videos (before Segmenter).
    Uses ffprobe (required).
    """
    _require_executable("ffprobe")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        os.path.abspath(video_path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr.strip()}")
    try:
        return float((p.stdout or "").strip())
    except Exception as e:
        raise RuntimeError(f"ffprobe returned invalid duration: {p.stdout!r}") from e

if __name__ == "__main__":
    import argparse
    import yaml

    # Глобальная переменная для cleanup временных файлов
    _cleanup_files = []

    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video-path', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default=f"{_path}/Segmenter/data", help='Base output directory for Segmenter')
    parser.add_argument('--chunk-size', type=int, default=64, help='Batch size for storing frames (union frames)')

    parser.add_argument('--global-config', type=str, default=None, help='Path to global config YAML (unified config for all processors). If provided, overrides individual processor settings.')
    parser.add_argument('--visual-cfg-path', type=str, default=f"{_path}/VisualProcessor/config.yaml", help='Path to VisualProcessor/config.yaml')
    parser.add_argument('--profile-path', type=str, default=None, help='Optional analysis profile YAML (required/optional components)')
    parser.add_argument('--dag-path', type=str, default=f"{_path}/docs/reference/component_graph.yaml", help='Path to component_graph.yaml (PR-6)')
    parser.add_argument('--dag-stage', type=str, default="baseline", help='DAG stage: baseline|v1|v2 (PR-6)')
    parser.add_argument('--platform-id', type=str, default="youtube")
    parser.add_argument('--video-id', type=str, default=None)
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--sampling-policy-version', type=str, default="v1")
    parser.add_argument('--dataprocessor-version', type=str, default="unknown")
    parser.add_argument('--analysis-fps', type=float, default=None, help="Analysis fps for Segmenter metadata (default: use source_fps)")
    parser.add_argument('--analysis-width', type=int, default=None, help="Resize width for analysis timeline (optional)")
    parser.add_argument('--analysis-height', type=int, default=None, help="Resize height for analysis timeline (optional)")
    # Segmenter sampling knobs (Audit v3): ASR windows
    parser.add_argument("--segmenter-asr-sampling-profile", type=str, default="semantic", choices=["semantic", "proxy"], help="Segmenter: ASR windows profile for families.asr")
    parser.add_argument("--segmenter-asr-window-sec", type=float, default=None, help="Segmenter: override ASR window_sec (seconds)")
    parser.add_argument("--segmenter-asr-stride-sec", type=float, default=None, help="Segmenter: override ASR stride_sec (seconds)")
    parser.add_argument("--segmenter-asr-max-windows", type=int, default=None, help="Segmenter: optional cap for ASR windows")

    # Output directory arguments (aliases for convenience)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('--rs-base', type=str, default=f"{_path}/VisualProcessor/result_store", help='Base result_store for VisualProcessor (per-run will be inside)')
    output_group.add_argument('--output-dir', type=str, default=None, dest='rs_base', help='Alias for --rs-base: output directory for results')
    output_group.add_argument('--result-dir', type=str, default=None, dest='rs_base', help='Alias for --rs-base: result directory for artifacts')
    parser.add_argument('--run-audio', action='store_true', help='Also run AudioProcessor Tier-0 extractors (clap/tempo/loudness) into the same per-run result_store')
    parser.add_argument('--audio-device', type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument('--audio-extractors', type=str, default="clap,tempo,loudness", help='Comma-separated audio extractor keys for AudioProcessor (clap,tempo,loudness)')
    # AudioProcessor feature flags passthrough (subset; audit v3 requires token-only ASR->Text path)
    parser.add_argument("--asr-enable-token-sequences", action="store_true", help="AudioProcessor/asr: enable token_ids_by_segment (needed for privacy-safe Text autogen)")
    parser.add_argument("--asr-enable-token-counts", action="store_true", help="AudioProcessor/asr: enable token_counts")
    parser.add_argument("--asr-enable-token-total", action="store_true", help="AudioProcessor/asr: enable token_total")
    parser.add_argument("--asr-enable-token-density", action="store_true", help="AudioProcessor/asr: enable token_density_per_sec")
    parser.add_argument("--asr-enable-speech-rate", action="store_true", help="AudioProcessor/asr: enable speech_rate_wpm")
    parser.add_argument("--asr-enable-lang-distribution", action="store_true", help="AudioProcessor/asr: enable lang_distribution")
    parser.add_argument("--asr-enable-segments-with-speech", action="store_true", help="AudioProcessor/asr: enable segments_with_speech")
    parser.add_argument("--asr-enable-avg-segment-duration", action="store_true", help="AudioProcessor/asr: enable avg_segment_duration_sec")
    parser.add_argument("--asr-enable-token-variance", action="store_true", help="AudioProcessor/asr: enable token_variance")
    parser.add_argument("--asr-save-segment-text", action="store_true", help="AudioProcessor/asr: persist raw segment text (debug-only)")
    # Scheduler-controlled knobs for AudioProcessor (L2/L3)
    parser.add_argument('--audio-segment-parallelism', type=int, default=None, help='AudioProcessor: concurrent segment workers (if supported)')
    parser.add_argument('--audio-max-inflight', type=int, default=None, help='AudioProcessor: max in-flight segment tasks (safety cap)')
    parser.add_argument('--audio-clap-batch-size', type=int, default=None, help='AudioProcessor: CLAP micro-batch size (may increase VRAM)')
    parser.add_argument('--run-text', action='store_true', help='Also run TextProcessor into the same per-run result_store')
    text_input_group = parser.add_mutually_exclusive_group()
    text_input_group.add_argument('--text-input-json', type=str, default=None, help='Path to TextProcessor VideoDocument JSON (single document)')
    text_input_group.add_argument('--text-input-dir', type=str, default=None, help='Directory with JSON documents (batch mode)')
    text_input_group.add_argument('--text-input-json-list', type=str, default=None, help='Comma-separated list of JSON paths (batch mode)')
    parser.add_argument('--no-run-visual', action='store_false', dest='run_visual', default=True, help='Disable VisualProcessor (can also be disabled via --global-config)')
    parser.add_argument('--text-enable-embeddings', action='store_true', help='Enable GPU-heavy text embedders (optional)')
    args = parser.parse_args()

    # Prod guardrail: reject videos longer than 20 minutes before any heavy work.
    dur = _probe_video_duration_sec(args.video_path)
    if dur > 20.0 * 60.0:
        raise RuntimeError(f"Video too long for baseline (>{20} min): duration_sec={dur}")

    root_path = os.path.abspath(_path)
    video_id = args.video_id
    if not video_id:
        video_id = os.path.splitext(os.path.basename(args.video_path))[0]
    run_id = args.run_id or uuid.uuid4().hex[:12]

    # ----------------------------
    # PR-5: state-files/state-manager (Level2+Level3)
    # We store state under: <runs_root>/state/<platform>/<video>/<run>/...
    # runs_root is derived from rs-base: e.g. rs-base=_runs/result_store -> runs_root=_runs
    # ----------------------------
    runs_root = os.path.dirname(os.path.abspath(args.rs_base))
    try:
        from storage.fs import FileSystemStorage
        from storage.paths import KeyLayout
        from state.managers import RunStateManager, ProcessorStateManager
        from state.enums import Status
    except Exception:
        FileSystemStorage = None  # type: ignore
        KeyLayout = None  # type: ignore
        RunStateManager = None  # type: ignore
        ProcessorStateManager = None  # type: ignore
        Status = None  # type: ignore

    state_storage = FileSystemStorage(runs_root) if FileSystemStorage else None
    state_layout = KeyLayout(prefix="") if KeyLayout else None
    run_state_mgr = None
    proc_mgrs = {}

    # Stable run config hash (shared across Segmenter/Visual/Audio/Text for idempotency)
    def _sha256_text(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    # Global config parser (если указан --global-config)
    global_config_parser = None
    if args.global_config:
        try:
            sys.path.insert(0, _path)  # Добавляем путь для импорта configs.config_parser
            from configs.config_parser import GlobalConfigParser
            global_config_parser = GlobalConfigParser(args.global_config)
            
            # Валидация конфига
            validation_errors = global_config_parser.validate()
            if validation_errors:
                raise RuntimeError(f"Global config validation failed:\n" + "\n".join(f"  - {e}" for e in validation_errors))
            
            # Переопределяем настройки из глобального конфига
            global_settings = global_config_parser.get_global_settings()
            if global_settings.get("platform_id"):
                args.platform_id = global_settings["platform_id"]
            if global_settings.get("sampling_policy_version"):
                args.sampling_policy_version = global_settings["sampling_policy_version"]
            if global_settings.get("dataprocessor_version"):
                args.dataprocessor_version = global_settings["dataprocessor_version"]
            
            # Audio processor settings
            if global_config_parser.is_processor_enabled("audio"):
                args.run_audio = True
            elif global_config_parser.get_processor_config("audio") is not None:
                # Если конфиг явно указывает enabled: false
                args.run_audio = False
            
            # Text processor settings
            if global_config_parser.is_processor_enabled("text"):
                args.run_text = True
            elif global_config_parser.get_processor_config("text") is not None:
                args.run_text = False
            
            # Visual processor settings
            if global_config_parser.is_processor_enabled("visual"):
                args.run_visual = True
            elif global_config_parser.get_processor_config("visual") is not None:
                # Если конфиг явно указывает enabled: false
                args.run_visual = False
            
            # Visual processor inline config (приоритет над cfg_path)
            visual_inline_config = global_config_parser.get_visual_inline_config()
            if visual_inline_config:
                # Сохраняем inline конфиг для использования в VisualProcessor
                # (будет использован вместо загрузки из файла)
                pass  # Обработка inline конфига будет в секции запуска VisualProcessor
            
        except Exception as e:
            raise RuntimeError(f"Failed to load global config from {args.global_config}: {e}") from e

    # Optional profile (PR-4): can override processor enablement and visual cfg path.
    # Profile имеет приоритет над глобальным конфигом (для обратной совместимости)
    profile = None
    if args.profile_path:
        with open(args.profile_path, "r", encoding="utf-8") as f:
            profile = yaml.safe_load(f) or {}

    # Определяем конфиг VisualProcessor: inline из глобального конфига или из файла
    visual_inline_config = None
    if global_config_parser:
        visual_inline_config = global_config_parser.get_visual_inline_config()
    
    visual_cfg_path = args.visual_cfg_path
    visual_cfg_temp_file = None  # Для временного файла из inline конфига
    
    # Если visual отключён и нет inline конфига — создаём минимальный конфиг для Segmenter (audio-only)
    if not args.run_visual and not visual_inline_config and (not os.path.exists(visual_cfg_path) or not os.path.isfile(visual_cfg_path)):
        fd, visual_cfg_temp_file = tempfile.mkstemp(suffix=".yaml", prefix="visual_cfg_minimal_", dir=tempfile.gettempdir())
        os.close(fd)
        with open(visual_cfg_temp_file, "w", encoding="utf-8") as f:
            yaml.safe_dump({"core_providers": {}, "modules": {}}, f, sort_keys=False, allow_unicode=True)
        visual_cfg_path = visual_cfg_temp_file
        _cleanup_files.append(visual_cfg_temp_file)
    
    # Если есть inline конфиг из глобального конфига, создаем временный файл для Segmenter
    if visual_inline_config:
        # Segmenter ожидает формат с core_providers и modules на верхнем уровне
        segmenter_visual_cfg = {
            "core_providers": visual_inline_config.get("core_providers", {}),
            "modules": visual_inline_config.get("modules", {}),
        }
        # Добавляем конфигурации компонентов (для sampling настроек)
        for key, value in visual_inline_config.items():
            if key not in ("core_providers", "modules", "global"):
                segmenter_visual_cfg[key] = value
        
        # Создаем временный файл
        fd, visual_cfg_temp_file = tempfile.mkstemp(suffix=".yaml", prefix="visual_cfg_", dir=tempfile.gettempdir())
        os.close(fd)
        with open(visual_cfg_temp_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(segmenter_visual_cfg, f, sort_keys=False, allow_unicode=True)
        visual_cfg_path = visual_cfg_temp_file
        _cleanup_files.append(visual_cfg_temp_file)  # Добавляем в список для cleanup
    
    if isinstance(profile, dict):
        vis = profile.get("visual") or {}
        if isinstance(vis, dict) and vis.get("cfg_path"):
            # Если profile указывает на файл, используем его (приоритет над inline конфигом)
            if visual_cfg_temp_file:
                try:
                    os.remove(visual_cfg_temp_file)
                except Exception:
                    pass
                visual_cfg_temp_file = None
            visual_cfg_path = str(vis.get("cfg_path"))
            # Important: if profile overrides visual cfg path, VisualProcessor runtime must use the file-based cfg,
            # not the inline_config from global_config.
            visual_inline_config = None

        procs = profile.get("processors") or {}
        if isinstance(procs, dict):
            audio_cfg = procs.get("audio") or {}
            if isinstance(audio_cfg, dict) and audio_cfg.get("enabled") is True:
                args.run_audio = True
            if isinstance(audio_cfg, dict) and audio_cfg.get("enabled") is False:
                args.run_audio = False
            text_cfg = procs.get("text") or {}
            if isinstance(text_cfg, dict) and text_cfg.get("enabled") is True:
                args.run_text = True
            if isinstance(text_cfg, dict) and text_cfg.get("enabled") is False:
                args.run_text = False
            visual_cfg = procs.get("visual") or {}
            if isinstance(visual_cfg, dict) and visual_cfg.get("enabled") is True:
                args.run_visual = True
            if isinstance(visual_cfg, dict) and visual_cfg.get("enabled") is False:
                args.run_visual = False

    # Optional override: orchestration.processors in global YAML (after profile).
    # Lets unified configs / E2E runs force a subset even when profile enables all processors.
    if global_config_parser:
        orch = (global_config_parser.config or {}).get("orchestration") or {}
        op = orch.get("processors") if isinstance(orch, dict) else None
        if isinstance(op, dict):
            if op.get("audio") is True:
                args.run_audio = True
            elif op.get("audio") is False:
                args.run_audio = False
            if op.get("text") is True:
                args.run_text = True
            elif op.get("text") is False:
                args.run_text = False
            if op.get("visual") is True:
                args.run_visual = True
            elif op.get("visual") is False:
                args.run_visual = False

    # Загружаем конфиг VisualProcessor: приоритет inline конфигу из глобального конфига
    # Если visual отключён — не загружаем файл (для audio-only профилей)
    if not args.run_visual:
        vp_cfg_for_hash = {}
    elif visual_inline_config:
        vp_cfg_for_hash = visual_inline_config
    else:
        with open(visual_cfg_path, "r", encoding="utf-8") as f:
            vp_cfg_for_hash = yaml.safe_load(f) or {}

    # PR-6: load DAG and compute Visual execution order (subset of enabled components)
    exec_order: list[str] = []
    try:
        with open(args.dag_path, "r", encoding="utf-8") as f:
            dag_yaml = yaml.safe_load(f) or {}
        from dag.component_graph import ComponentGraph

        g = ComponentGraph.from_yaml_dict(dag_yaml, stage=str(args.dag_stage))
        enabled_visual: set[str] = set()
        enabled_visual.update([k for k, v in (vp_cfg_for_hash.get("core_providers") or {}).items() if v])
        enabled_visual.update([k for k, v in (vp_cfg_for_hash.get("modules") or {}).items() if v])

        # Build execution order only for nodes known to the DAG.
        # Unknown enabled components are allowed in MVP but will be executed after DAG-ordered ones.
        known_enabled = [n for n in enabled_visual if n in g.by_name]
        exec_order = [n for n in g.topo_order(known_enabled) if n in enabled_visual]
    except Exception:
        exec_order = []
    cfg_for_hash = {
        "chunk_size": int(args.chunk_size),
        "sampling_policy_version": str(args.sampling_policy_version),
        "dataprocessor_version": str(args.dataprocessor_version),
        "analysis_fps": args.analysis_fps,
        "analysis_width": args.analysis_width,
        "analysis_height": args.analysis_height,
        "visual_cfg": vp_cfg_for_hash,
        "profile": profile,
        "dag_stage": str(args.dag_stage),
        "dag_path": str(args.dag_path),
        "run_audio": bool(args.run_audio),
        "run_text": bool(args.run_text),
    }
    
    # Добавляем настройки из глобального конфига или CLI аргументов
    if global_config_parser:
        # Используем весь глобальный конфиг для hash (для идемпотентности)
        cfg_for_hash["global_config"] = global_config_parser.config
    else:
        # Fallback на CLI аргументы (обратная совместимость)
        cfg_for_hash.update({
            "audio_device": str(args.audio_device),
            "audio_extractors": str(args.audio_extractors),
            "audio_segment_parallelism": (int(args.audio_segment_parallelism) if args.audio_segment_parallelism is not None else None),
            "audio_max_inflight": (int(args.audio_max_inflight) if args.audio_max_inflight is not None else None),
            "audio_clap_batch_size": (int(args.audio_clap_batch_size) if args.audio_clap_batch_size is not None else None),
            "text_input_json": os.path.abspath(args.text_input_json) if args.text_input_json else None,
            "text_enable_embeddings": bool(args.text_enable_embeddings),
        })
    config_hash = _sha256_text(yaml.safe_dump(cfg_for_hash, sort_keys=True, allow_unicode=True))[:16]

    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Unified per-run result_store directory (single source of truth for all processors)
    run_rs_path = os.path.join(os.path.abspath(args.rs_base), args.platform_id, video_id, run_id)
    os.makedirs(run_rs_path, exist_ok=True)
    if state_storage and state_layout and RunStateManager and ProcessorStateManager and Status:
        run_meta = {
            "platform_id": args.platform_id,
            "video_id": video_id,
            "run_id": run_id,
            "config_hash": config_hash,
            "sampling_policy_version": args.sampling_policy_version,
            "dataprocessor_version": str(args.dataprocessor_version),
            "created_at": created_at,
        }
        run_state_mgr = RunStateManager(
            storage=state_storage,
            layout=state_layout,
            platform_id=args.platform_id,
            video_id=video_id,
            run_id=run_id,
            run_meta=run_meta,
        )
        run_state_mgr.init(["segmenter", "audio", "text", "visual"])
        for p in ("segmenter", "audio", "text", "visual"):
            proc_mgrs[p] = ProcessorStateManager(
                storage=state_storage,
                layout=state_layout,
                platform_id=args.platform_id,
                video_id=video_id,
                run_id=run_id,
                processor_name=p,
                run_meta=run_meta,
            )
            # Materialize initial waiting state-files (Level-3).
            proc_mgrs[p].flush()
            run_state_mgr.merge_processor_state(p, proc_mgrs[p].state)

    # 1) Segmenter (union-sampled frames_dir)
    if proc_mgrs.get("segmenter") and run_state_mgr and Status:
        proc_mgrs["segmenter"].set_status(Status.running, started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        run_state_mgr.merge_processor_state("segmenter", proc_mgrs["segmenter"].state)
    seg_cmd = [
        _get_data_venv_python(),
        f"{_path}/Segmenter/segmenter.py",
        "--video-path", args.video_path,
        "--output", args.output,
        "--chunk-size", str(args.chunk_size),
        "--visual-cfg-path", visual_cfg_path,
        "--platform-id", args.platform_id,
        f"--video-id={video_id}",
        "--run-id", run_id,
        "--sampling-policy-version", args.sampling_policy_version,
        "--config-hash", config_hash,
        "--dataprocessor-version", str(args.dataprocessor_version),
    ]
    if args.analysis_fps is not None:
        seg_cmd.extend(["--analysis-fps", str(args.analysis_fps)])
    if args.analysis_width is not None:
        seg_cmd.extend(["--analysis-width", str(args.analysis_width)])
    if args.analysis_height is not None:
        seg_cmd.extend(["--analysis-height", str(args.analysis_height)])
    # Forward ASR sampling knobs to Segmenter
    if args.segmenter_asr_sampling_profile:
        seg_cmd.extend(["--asr-sampling-profile", str(args.segmenter_asr_sampling_profile)])
    if args.segmenter_asr_window_sec is not None:
        seg_cmd.extend(["--asr-window-sec", str(float(args.segmenter_asr_window_sec))])
    if args.segmenter_asr_stride_sec is not None:
        seg_cmd.extend(["--asr-stride-sec", str(float(args.segmenter_asr_stride_sec))])
    if args.segmenter_asr_max_windows is not None:
        seg_cmd.extend(["--asr-max-windows", str(int(args.segmenter_asr_max_windows))])
    _print_header("Segmenter")
    _print_step("Запуск Segmenter", "OK", _shorten_path(args.video_path))
    t0 = time.time()
    r = _run_subprocess_with_formatted_output(seg_cmd, processor_name="Segmenter", check=False)
    seg_duration_ms = int((time.time() - t0) * 1000)
    if r.returncode == 0:
        _print_step("Segmenter завершен", "OK", f"время: {seg_duration_ms}ms")
        if proc_mgrs.get("segmenter") and run_state_mgr and Status:
            proc_mgrs["segmenter"].set_status(
                Status.success,
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                duration_ms=seg_duration_ms,
            )
            run_state_mgr.merge_processor_state("segmenter", proc_mgrs["segmenter"].state)
    elif r.returncode == 10:
        # Segmenter-level skip (e.g., video cannot be opened/decoded).
        if proc_mgrs.get("segmenter") and run_state_mgr and Status:
            proc_mgrs["segmenter"].set_status(
                Status.skipped,
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                duration_ms=seg_duration_ms,
                error="segmenter_skipped",
                error_code="video_unreadable",
            )
            run_state_mgr.merge_processor_state("segmenter", proc_mgrs["segmenter"].state)
        # Do not proceed to audio/text/visual if Segmenter did not produce frames_dir/audio.
        raise SystemExit(0)
    else:
        if proc_mgrs.get("segmenter") and run_state_mgr and Status:
            proc_mgrs["segmenter"].set_status(
                Status.error,
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                duration_ms=seg_duration_ms,
                error=f"exit={r.returncode}",
                error_code="non_zero_exit",
            )
            run_state_mgr.merge_processor_state("segmenter", proc_mgrs["segmenter"].state)
        raise RuntimeError(f"Segmenter failed (exit={r.returncode})")

    frames_dir = os.path.join(args.output, video_id, "video")

    # Create/merge manifest.json early so Audio/Text/Visual can upsert without racing on first write.
    # (VisualProcessor and Audio/Text CLIs will load+merge if it already exists.)
    try:
        vp_root = Path(__file__).resolve().parent / "VisualProcessor"
        if str(vp_root) not in sys.path:
            sys.path.insert(0, str(vp_root))
        from utils.manifest import RunManifest  # type: ignore

        manifest_path = os.path.join(run_rs_path, "manifest.json")
        manifest = RunManifest(
            path=manifest_path,
            run_meta={
                "platform_id": args.platform_id,
                "video_id": video_id,
                "run_id": run_id,
                "status": "running",
                "config_hash": config_hash,
                "sampling_policy_version": args.sampling_policy_version,
                "dataprocessor_version": str(args.dataprocessor_version),
                "created_at": created_at,
                "frames_dir": os.path.join(os.path.abspath(args.output), video_id, "video"),
                "root_path": os.path.abspath(_path),
            },
        )
        manifest.flush()
    except Exception:
        # Best-effort: processors still create/merge manifests themselves.
        pass

    # 1.5) AudioProcessor (optional): write per-run NPZ artifacts into the same result_store
    if args.run_audio:
        audio_required = False
        if global_config_parser:
            audio_required = global_config_parser.is_processor_required("audio")
        elif isinstance(profile, dict):
            audio_required = bool(((profile.get("processors") or {}).get("audio") or {}).get("required") is True)
        
        # Устанавливаем DP_MODELS_ROOT если не установлен
        dp_models_root = os.environ.get("DP_MODELS_ROOT")
        if not dp_models_root:
            # По умолчанию используем bundled_models в репозитории
            default_models_root = os.path.join(_path, "dp_models", "bundled_models")
            if os.path.exists(default_models_root):
                dp_models_root = default_models_root
                os.environ["DP_MODELS_ROOT"] = dp_models_root
        
        audio_cmd = [
            _get_audio_venv_python(),
            f"{_path}/AudioProcessor/run_cli.py",
            "--video-path", args.video_path,
            "--frames-dir", os.path.join(os.path.abspath(args.output), video_id),
            "--rs-base", os.path.abspath(args.rs_base),
            "--run-rs-path", run_rs_path,
            "--platform-id", args.platform_id,
            f"--video-id={video_id}",
            "--run-id", run_id,
            "--sampling-policy-version", args.sampling_policy_version,
            "--config-hash", config_hash,
            "--dataprocessor-version", str(args.dataprocessor_version),
        ]
        
        # Используем аргументы из глобального конфига, если он указан
        if global_config_parser:
            audio_cli_args = global_config_parser.get_audio_cli_args()
            audio_cmd.extend(audio_cli_args)
        else:
            # Fallback на CLI аргументы (обратная совместимость)
            audio_cmd.extend(["--device", args.audio_device])
            audio_cmd.extend(["--extractors", args.audio_extractors])
            # Forward ASR flags if present
            if args.asr_enable_token_sequences:
                audio_cmd.append("--asr-enable-token-sequences")
            if args.asr_enable_token_counts:
                audio_cmd.append("--asr-enable-token-counts")
            if args.asr_enable_token_total:
                audio_cmd.append("--asr-enable-token-total")
            if args.asr_enable_token_density:
                audio_cmd.append("--asr-enable-token-density")
            if args.asr_enable_speech_rate:
                audio_cmd.append("--asr-enable-speech-rate")
            if args.asr_enable_lang_distribution:
                audio_cmd.append("--asr-enable-lang-distribution")
            if args.asr_enable_segments_with_speech:
                audio_cmd.append("--asr-enable-segments-with-speech")
            if args.asr_enable_avg_segment_duration:
                audio_cmd.append("--asr-enable-avg-segment-duration")
            if args.asr_enable_token_variance:
                audio_cmd.append("--asr-enable-token-variance")
            if args.asr_save_segment_text:
                audio_cmd.append("--asr-save-segment-text")
            if args.audio_segment_parallelism is not None:
                audio_cmd.extend(["--segment-parallelism", str(int(args.audio_segment_parallelism))])
            if args.audio_max_inflight is not None:
                audio_cmd.extend(["--max-inflight", str(int(args.audio_max_inflight))])
            if args.audio_clap_batch_size is not None:
                audio_cmd.extend(["--clap-batch-size", str(int(args.audio_clap_batch_size))])
        if proc_mgrs.get("audio") and run_state_mgr and Status:
            proc_mgrs["audio"].set_status(Status.running, started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            run_state_mgr.merge_processor_state("audio", proc_mgrs["audio"].state)
        _print_header("AudioProcessor")
        _print_step("Запуск AudioProcessor", "OK")
        t0 = time.time()
        # Передаем DP_MODELS_ROOT и HF cache в окружение subprocess (emotion_diarization, speaker_diarization)
        audio_env = {}
        if dp_models_root:
            audio_env["DP_MODELS_ROOT"] = dp_models_root
            # HF cache: bundled hf_cache/hub или default ~/.cache/huggingface (для WavLM и др.)
            # Используем realpath для symlink'ов (bundled_models/hf_cache/hub -> ~/.cache/huggingface/hub)
            # чтобы transformers/huggingface_hub корректно находили модели в offline режиме
            hf_cache_dir = os.path.join(dp_models_root, "hf_cache")
            hf_hub = os.path.join(hf_cache_dir, "hub")
            hf_hub_resolved = os.path.realpath(hf_hub) if os.path.exists(hf_hub) else hf_hub
            default_hf = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
            default_hub = os.path.join(default_hf, "hub")

            def _hf_has_models(path):
                if not path or not os.path.isdir(path):
                    return False
                try:
                    return any(p.startswith("models--") for p in os.listdir(path))
                except OSError:
                    return False

            if _hf_has_models(hf_hub_resolved):
                audio_env["HF_HOME"] = os.path.dirname(hf_hub_resolved)
                audio_env["HF_HUB_CACHE"] = hf_hub_resolved
                audio_env["TRANSFORMERS_CACHE"] = hf_hub_resolved
            elif _hf_has_models(default_hub):
                audio_env["HF_HOME"] = default_hf
                audio_env["HF_HUB_CACHE"] = default_hub
                audio_env["TRANSFORMERS_CACHE"] = default_hub
        # Wall clock anchor for AudioProcessor subprocess (run_cli logs manifest.run_meta breakdown).
        audio_env["DP_AUDIO_WALL_T0"] = str(t0)
        r = _run_subprocess_with_formatted_output(audio_cmd, processor_name="AudioProcessor", check=False, env=audio_env)
        audio_duration_ms = int((time.time() - t0) * 1000)
        if r.returncode == 0:
            _print_step("AudioProcessor завершен", "OK", f"время: {audio_duration_ms}ms")
        else:
            _print_step("AudioProcessor завершен", "ERROR", f"exit code: {r.returncode}")
        if proc_mgrs.get("audio") and run_state_mgr and Status:
            st = Status.success if r.returncode == 0 else Status.error
            proc_mgrs["audio"].set_status(
                st,
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                duration_ms=audio_duration_ms,
                error=None if r.returncode == 0 else _subprocess_output_error_detail(r),
                error_code=None if r.returncode == 0 else "non_zero_exit",
            )
            run_state_mgr.merge_processor_state("audio", proc_mgrs["audio"].state)
        if audio_required and r.returncode != 0:
            raise RuntimeError(f"AudioProcessor failed for required=true (exit={r.returncode})")
        _best_effort_torch_cuda_empty_cache_in_parent()

    if args.run_text:
        text_required = False
        if global_config_parser:
            text_required = global_config_parser.is_processor_required("text")
        elif isinstance(profile, dict):
            text_required = bool(((profile.get("processors") or {}).get("text") or {}).get("required") is True)

        # Определяем input: CLI → profile.processors.text.input_json → global-config
        text_input_json = args.text_input_json
        text_input_dir = args.text_input_dir
        text_input_json_list = args.text_input_json_list

        if isinstance(profile, dict) and not (
            text_input_json or text_input_dir or text_input_json_list
        ):
            ptxt = (profile.get("processors") or {}).get("text") or {}
            if isinstance(ptxt, dict) and ptxt.get("input_json"):
                text_input_json = str(ptxt["input_json"])

        if global_config_parser:
            text_cfg = global_config_parser.get_processor_config("text")
            if text_cfg and text_cfg.get("input_json") and not (text_input_json or text_input_dir or text_input_json_list):
                # Используем из конфига только если не указаны CLI аргументы
                text_input_json = text_cfg["input_json"]

        def _autogen_text_input_from_asr(*, run_rs_path: str, platform_id: str, video_id: str) -> str | None:
            """
            Best-effort: build a minimal VideoDocument JSON for TextProcessor from AudioProcessor ASR artifact.
            Privacy-first behavior (Audit v3):
            - Prefer token IDs (shared_tokenizer_v1) if present: no raw text is stored in the JSON.
            - Do NOT store raw per-segment transcript text in the JSON (token-only contract).
            """
            try:
                npz_path = os.path.join(run_rs_path, "asr_extractor", "asr_extractor_features.npz")
                if not os.path.exists(npz_path):
                    return None

                import numpy as np  # local import keeps DataProcessor lightweight

                npz = np.load(npz_path, allow_pickle=True)
                # Most AudioProcessor components store scalars/arrays as top-level NPZ keys.
                # Some may additionally store an object "payload" dict; support both.
                payload = None
                if "payload" in npz.files:
                    payload = npz["payload"]
                    if isinstance(payload, np.ndarray) and payload.dtype == object:
                        payload = payload.item() if payload.size == 1 else None
                payload = payload if isinstance(payload, dict) else {}

                def _get_any(key: str):
                    if key in payload:
                        return payload.get(key)
                    if key in npz.files:
                        v = npz[key]
                        # convert 0-d object arrays
                        if isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
                            try:
                                return v.item()
                            except Exception:
                                return v
                        # convert numeric arrays to python list for JSON
                        if isinstance(v, np.ndarray):
                            try:
                                return v.tolist()
                            except Exception:
                                return v
                        return v
                    return None

                # Prefer audio duration from Segmenter contract (source-of-truth): frames_dir/audio/segments.json
                audio_duration_sec = None
                try:
                    manifest_path = os.path.join(run_rs_path, "manifest.json")
                    if os.path.exists(manifest_path):
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            man = json.load(f) or {}
                        frames_dir = ((man.get("run") or {}).get("frames_dir"))
                        if frames_dir:
                            # frames_dir points to .../<video_id>/video; audio lives next to it: .../<video_id>/audio
                            seg_path = os.path.join(str(Path(str(frames_dir)).resolve().parent), "audio", "segments.json")
                            if os.path.exists(seg_path):
                                with open(seg_path, "r", encoding="utf-8") as f:
                                    segp = json.load(f) or {}
                                audio_duration_sec = segp.get("audio_duration_sec")
                except Exception:
                    audio_duration_sec = None

                seg_st = _get_any("segment_start_sec")
                seg_en = _get_any("segment_end_sec")
                if not (isinstance(seg_st, list) and isinstance(seg_en, list)) or (len(seg_st) != len(seg_en)):
                    return None

                # Preferred: token IDs (privacy-safe, no raw text in JSON)
                token_ids_by_segment = _get_any("token_ids_by_segment")
                token_ids_clean: list[list[int]] = []
                full_token_ids: list[int] = []
                if isinstance(token_ids_by_segment, list) and token_ids_by_segment:
                    for tok_arr in token_ids_by_segment:
                        if isinstance(tok_arr, np.ndarray):
                            ids = [int(x) for x in tok_arr.reshape(-1).tolist()]
                        elif isinstance(tok_arr, (list, tuple)):
                            ids = [int(x) for x in tok_arr]
                        else:
                            continue
                        token_ids_clean.append(ids)
                        full_token_ids.extend(ids)

                # Best-effort metadata (optional, token-only ASR payload)
                lang_code_by_segment = _get_any("lang_code_by_segment")
                lang_conf_by_segment = _get_any("lang_conf_by_segment")
                segment_quality_by_segment = _get_any("segment_quality_by_segment")
                asr_text_contract_version = _get_any("asr_text_contract_version")

                asr_payload = None
                if token_ids_clean and isinstance(seg_st, list) and isinstance(seg_en, list) and len(token_ids_clean) == len(seg_st) == len(seg_en):
                    asr_payload = {
                        "schema_version": "asr_payload_v2",
                        "tokenizer_spec": "shared_tokenizer_v1",
                        "asr_text_contract_version": asr_text_contract_version,
                        "token_ids_by_segment": token_ids_clean,
                        "segment_start_sec": [float(x) for x in seg_st],
                        "segment_end_sec": [float(x) for x in seg_en],
                        "segment_center_sec": [float(x) for x in (_get_any("segment_center_sec") or [])] if isinstance(_get_any("segment_center_sec"), list) else None,
                        "lang_code_by_segment": (lang_code_by_segment if isinstance(lang_code_by_segment, list) else None),
                        "lang_conf_by_segment": (lang_conf_by_segment if isinstance(lang_conf_by_segment, list) else None),
                        "segment_quality_by_segment": (segment_quality_by_segment if isinstance(segment_quality_by_segment, list) else None),
                    }

                doc = {
                    "schema_version": "video_document_v1",
                    "platform_id": platform_id,
                    "video_id": video_id,
                    "title": "",
                    "description": "",
                    "transcripts": {},
                    "transcripts_token_ids": ({"whisper": full_token_ids} if full_token_ids else {}),
                    # Required by some TextProcessor extractors (e.g. ASRTextProxyExtractor).
                    "audio_duration_sec": (float(audio_duration_sec) if audio_duration_sec is not None else None),
                    "asr": asr_payload,
                    "comments": [],
                }

                out_dir = os.path.join(run_rs_path, "_tmp")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, "text_input_autogen.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(doc, f, ensure_ascii=False, indent=2)
                return out_path
            except Exception:
                return None

        # If no explicit text input provided, try auto-generate from ASR output.
        if not (text_input_json or text_input_dir or text_input_json_list):
            autogen = _autogen_text_input_from_asr(run_rs_path=run_rs_path, platform_id=args.platform_id, video_id=video_id)
            if autogen:
                text_input_json = autogen

        # Проверяем, что указан хотя бы один источник
        if not (text_input_json or text_input_dir or text_input_json_list):
            raise ValueError(
                "TextProcessor requires input (specify --text-input-json, --text-input-dir, or --text-input-json-list, "
                "or set in --global-config). If you want DataProcessor to auto-generate input from ASR, "
                "run AudioProcessor with --asr-enable-token-sequences (preferred, privacy-safe) or --asr-save-segment-text (debug-only)."
            )
        
        text_cmd = [
            _get_text_venv_python(),
            f"{_path}/TextProcessor/run_cli.py",
            "--rs-base", os.path.abspath(args.rs_base),
            "--run-rs-path", run_rs_path,
            "--platform-id", args.platform_id,
            f"--video-id={video_id}",
            "--run-id", run_id,
            "--sampling-policy-version", args.sampling_policy_version,
            "--config-hash", config_hash,
            "--dataprocessor-version", str(args.dataprocessor_version),
        ]
        
        # Добавляем флаг для input
        if text_input_json:
            text_cmd.extend(["--input-json", os.path.abspath(text_input_json)])
        elif text_input_dir:
            text_cmd.extend(["--input-dir", os.path.abspath(text_input_dir)])
        elif text_input_json_list:
            text_cmd.extend(["--input-json-list", text_input_json_list])
        
        # Используем аргументы из глобального конфига, если он указан
        if global_config_parser:
            text_cli_args = global_config_parser.get_text_cli_args()
            text_cmd.extend(text_cli_args)
        else:
            # Fallback на CLI аргументы
            if args.text_enable_embeddings:
                text_cmd.append("--enable-embeddings")
        if proc_mgrs.get("text") and run_state_mgr and Status:
            proc_mgrs["text"].set_status(Status.running, started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            run_state_mgr.merge_processor_state("text", proc_mgrs["text"].state)
        _print_header("TextProcessor")
        input_desc = None
        if text_input_json:
            input_desc = _shorten_path(text_input_json)
        elif text_input_dir:
            input_desc = f"dir: {_shorten_path(text_input_dir)}"
        elif text_input_json_list:
            count = len(text_input_json_list.split(","))
            input_desc = f"list: {count} documents"
        _print_step("Запуск TextProcessor", "OK", input_desc)
        text_t0 = time.time()
        r = _run_subprocess_with_formatted_output(
            text_cmd,
            processor_name="TextProcessor",
            check=False,
            env={"DP_TEXT_WALL_T0": str(text_t0)},
        )
        text_duration_ms = int((time.time() - text_t0) * 1000)
        if r.returncode == 0:
            _print_step("TextProcessor завершен", "OK", f"время: {text_duration_ms}ms")
        else:
            _print_step("TextProcessor завершен", "ERROR", f"exit code: {r.returncode}")
        if proc_mgrs.get("text") and run_state_mgr and Status:
            st = Status.success if r.returncode == 0 else Status.error
            proc_mgrs["text"].set_status(
                st,
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                duration_ms=text_duration_ms,
                error=None if r.returncode == 0 else _subprocess_output_error_detail(r),
                error_code=None if r.returncode == 0 else "non_zero_exit",
            )
            run_state_mgr.merge_processor_state("text", proc_mgrs["text"].state)
        if text_required and r.returncode != 0:
            raise RuntimeError(f"TextProcessor failed for required=true (exit={r.returncode})")
        _best_effort_torch_cuda_empty_cache_in_parent()

    # 2) VisualProcessor: generate a temp cfg overriding global paths/ids
    if args.run_visual:
        visual_required = False
        if global_config_parser:
            visual_required = global_config_parser.is_processor_required("visual")
        elif isinstance(profile, dict):
            visual_required = bool(((profile.get("processors") or {}).get("visual") or {}).get("required") is True)
        
        vp_cfg = dict(vp_cfg_for_hash or {})
        vp_cfg = vp_cfg or {}
        vp_cfg["global"] = vp_cfg.get("global") or {}
        vp_cfg["global"].update(
            {
                "root_path": root_path,
                "frames_dir": frames_dir,
                # VisualProcessor expects rs_path; we pass the per-run directory to avoid re-deriving.
                "rs_path": run_rs_path,
                "rs_path_is_run_dir": True,
                "platform_id": args.platform_id,
                "video_id": video_id,
                "run_id": run_id,
                "config_hash": config_hash,
                "sampling_policy_version": args.sampling_policy_version,
                "dataprocessor_version": str(args.dataprocessor_version),
            }
        )
        # PR-8: pass resolved model mapping into VisualProcessor runtime cfg (and manifest.run).
        # MVP source-of-truth is profile YAML; later this comes from DB.
        if isinstance(profile, dict):
            rmm = profile.get("resolved_model_mapping")
            if isinstance(rmm, dict) and rmm:
                vp_cfg["resolved_model_mapping"] = rmm
        # PR-4: pass requirements map to VisualProcessor (enables required/optional enforcement).
        if isinstance(profile, dict):
            vis = profile.get("visual") or {}
            if isinstance(vis, dict):
                req = vis.get("requirements")
                if isinstance(req, dict) and req:
                    vp_cfg["requirements"] = req

        # PR-6: pass execution order into VisualProcessor (optional)
        if exec_order:
            vp_cfg["execution_order"] = exec_order

        fd, tmp_cfg_path = tempfile.mkstemp(prefix="vp_runtime_", suffix=".yaml")
        os.close(fd)
        try:
            with open(tmp_cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(vp_cfg, f, sort_keys=False, allow_unicode=True)

            vp_cmd = [
                _get_visual_venv_python(),
                f"{_path}/VisualProcessor/main.py",
                "--cfg-path",
                tmp_cfg_path,
            ]
            if proc_mgrs.get("visual") and run_state_mgr and Status:
                proc_mgrs["visual"].set_status(Status.running, started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                run_state_mgr.merge_processor_state("visual", proc_mgrs["visual"].state)
            _print_header("VisualProcessor")
            _print_step("Запуск VisualProcessor", "OK")
            t0 = time.time()
            try:
                r = _run_subprocess_with_formatted_output(vp_cmd, processor_name="VisualProcessor", check=False)
                visual_duration_ms = int((time.time() - t0) * 1000)
                if r.returncode == 0:
                    _print_step("VisualProcessor завершен", "OK", f"время: {visual_duration_ms}ms")
                else:
                    _print_step("VisualProcessor завершен", "ERROR", f"exit code: {r.returncode}")
                if proc_mgrs.get("visual") and run_state_mgr and Status:
                    st = Status.success if r.returncode == 0 else Status.error
                    proc_mgrs["visual"].set_status(
                        st,
                        finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        duration_ms=visual_duration_ms,
                        error=None if r.returncode == 0 else f"exit={r.returncode}",
                        error_code=None if r.returncode == 0 else "non_zero_exit",
                    )
                    run_state_mgr.merge_processor_state("visual", proc_mgrs["visual"].state)
                if r.returncode != 0:
                    if visual_required:
                        raise RuntimeError(f"VisualProcessor failed for required=true (exit={r.returncode})")
                    # Если не required, просто логируем ошибку и продолжаем
            except Exception as e:
                if proc_mgrs.get("visual") and run_state_mgr and Status:
                    proc_mgrs["visual"].set_status(
                        Status.error,
                        finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        duration_ms=int((time.time() - t0) * 1000),
                        error=str(e),
                        error_code="exception",
                    )
                    run_state_mgr.merge_processor_state("visual", proc_mgrs["visual"].state)
                if visual_required:
                    raise

            # After VisualProcessor finished, sync component statuses from manifest into processor state.
            try:
                manifest_path = os.path.join(os.path.abspath(args.rs_base), args.platform_id, video_id, run_id, "manifest.json")
                if os.path.exists(manifest_path) and proc_mgrs.get("visual") and run_state_mgr and Status:
                    import json as _json
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        m = _json.load(f) or {}
                    comps = m.get("components") or []
                    if isinstance(comps, list):
                        for c in comps:
                            if not isinstance(c, dict):
                                continue
                            # Visual в manifest: module + core (не audio/text).
                            if str(c.get("kind") or "").lower() not in ("module", "core"):
                                continue
                            name = c.get("name")
                            st = c.get("status")
                            if not isinstance(name, str) or not name:
                                continue
                            if st == "ok":
                                sst = Status.success
                            elif st == "empty":
                                sst = Status.empty
                            elif st == "error":
                                sst = Status.error
                            elif st == "running":
                                sst = Status.running
                            elif st == "skipped":
                                sst = Status.skipped
                            else:
                                sst = Status.error
                            proc_mgrs["visual"].upsert_component(
                                component_name=name,
                                status=sst,
                                artifacts=c.get("artifacts") if isinstance(c.get("artifacts"), list) else None,
                                error=c.get("error"),
                                error_code=c.get("error_code"),
                                notes=c.get("notes"),
                                device_used=c.get("device_used"),
                                started_at=c.get("started_at"),
                                finished_at=c.get("finished_at"),
                                duration_ms=c.get("duration_ms"),
                            )
                    run_state_mgr.merge_processor_state("visual", proc_mgrs["visual"].state)
            except Exception:
                # best-effort: do not break successful run if state sync fails
                if proc_mgrs.get("visual") and run_state_mgr and Status:
                    run_state_mgr.merge_processor_state("visual", proc_mgrs["visual"].state)
        finally:
            # Cleanup: удаляем временные файлы конфигов VisualProcessor
            if 'tmp_cfg_path' in locals():
                if tmp_cfg_path and os.path.exists(tmp_cfg_path):
                    try:
                        os.remove(tmp_cfg_path)
                    except Exception:
                        pass
    
    # Global cleanup для всех временных файлов
    # Best-effort: finalize manifest run status.
    try:
        vp_root = Path(__file__).resolve().parent / "VisualProcessor"
        if str(vp_root) not in sys.path:
            sys.path.insert(0, str(vp_root))
        from utils.manifest import RunManifest  # type: ignore

        manifest_path = os.path.join(run_rs_path, "manifest.json")
        RunManifest(path=manifest_path, run_meta={"status": "success"}).flush()
    except Exception:
        pass

    # Best-effort: write a compact manifest summary into _reports/.
    try:
        summary_script = os.path.join(_path, "scripts", "summarize_run_manifest.py")
        if os.path.exists(summary_script):
            _run_subprocess_with_formatted_output(
                [sys.executable, summary_script, "--manifest", os.path.join(run_rs_path, "manifest.json")],
                processor_name="ManifestSummary",
                check=False,
            )
    except Exception:
        pass

    for cleanup_file in _cleanup_files:
        if cleanup_file and os.path.exists(cleanup_file):
            try:
                os.remove(cleanup_file)
            except Exception:
                pass