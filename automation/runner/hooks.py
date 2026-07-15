"""PreToolUse-хук для Bash (общий для Первого и Второго агента): два барьера.

1) Изменение подов в обход pod_control — запрещено (deny).
2) Потенциально разрушительные/дорогие команды — требуют вердикта Второго агента (supervisor.answer())
   (агент работает с bypassPermissions, поэтому это единственная защита от rm -rf, скачивания
   гигабайтов на volume, git push и т.п. — вспомни инцидент с 85 ГБ HF-снапшота). Вердикты логируются
   в state/hook_decisions.log.
"""
from __future__ import annotations
import datetime as dt
import re

import config
import hub


def _log_decision(command: str, reason: str, verdict: str) -> None:
    try:
        with open(config.HOOK_DECISIONS_LOG, "a", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now().isoformat()} | {reason} | {verdict} | {command[:200]}\n")
    except Exception:
        pass

# --- (1) мутации RunPod мимо pod_control ---
_PODS = re.compile(r"rest\.runpod\.io/v1/pods", re.I)
_MUTATING = re.compile(r"(/start|/stop|-X\s*DELETE|--request\s*DELETE|-X\s*POST|--request\s*POST|-X\s*PATCH)", re.I)

# --- (2) опасные команды: (regex, человекочитаемая причина) ---
_DANGER = [
    (re.compile(r"\brm\s+(-[a-z]*r[a-z]*\s+|-[a-z]*f[a-z]*\s+).*(/workspace|/\s|~|\*)", re.I), "рекурсивное удаление файлов"),
    (re.compile(r"\brm\s+-[rf]{1,2}\b", re.I), "rm -rf"),
    (re.compile(r"\bmkfs\b|\bdd\s+if=|\bof=/dev/|\bshred\b|\bfdisk\b|\bwipefs\b", re.I), "операция с диском/устройством"),
    (re.compile(r"\bchmod\s+-R\s+777\b|\bchown\s+-R\b\s+.*\s+/", re.I), "рекурсивная смена прав/владельца от корня"),
    (re.compile(r"snapshot_download\((?![^)]*allow_patterns)", re.I), "HuggingFace snapshot_download БЕЗ allow_patterns (может забить volume)"),
    (re.compile(r"huggingface-cli\s+download\b(?!.*(--include|--allow-patterns))", re.I), "huggingface-cli download без --include"),
    (re.compile(r"\bgit\s+reset\s+--hard\b|\bgit\s+clean\s+-[a-z]*f", re.I), "уничтожение локальных изменений git"),
    (re.compile(r"curl[^\n|]*\|\s*(sudo\s+)?(ba)?sh|wget[^\n|]*\|\s*(sudo\s+)?(ba)?sh", re.I), "скачать-и-выполнить из интернета"),
    (re.compile(r":\(\)\s*\{\s*:\|:&\s*\};", re.I), "fork-бомба"),
    (re.compile(r"\bpip\s+uninstall\b|\bpip\s+install\b[^\n]*\b(torch|cuda)\b", re.I), "переустановка критичных пакетов окружения"),
]


_HEAVY = re.compile(r"\bssh\b|run_[a-z_]*\.py|\bpytest\b|\bffmpeg\b|snapshot_download|huggingface-cli\s+download|main\.py", re.I)


async def guard_bash(input_data, tool_use_id, context):
    if input_data.get("tool_name") != "Bash":
        return {}
    command = (input_data.get("tool_input") or {}).get("command", "")

    # (0) ЛИМИТ: на рабочем пороге блокируем НОВЫЕ тяжёлые прогоны — принудительное сворачивание.
    try:
        import limits, settings
        pct = limits.max_used_pct()
        if pct is not None and pct >= float(settings.get("limit_pct_stop")) and _HEAVY.search(command):
            return _deny(f"🧯 Лимит Claude {pct:.0f}% ≥ {settings.get('limit_pct_stop')}%. НОВЫЕ прогоны "
                         f"заблокированы. Только СВОРАЧИВАНИЕ: обнови REPORT/last_session/прогресс (Write/Edit), "
                         f"затем pod_control stop_all и заверши. Тяжёлый bash разблокируется после сброса лимита.")
    except Exception:
        pass

    # (1) hard deny
    if _PODS.search(command) and _MUTATING.search(command):
        return _deny("Прямое изменение подов запрещено. Используй pod_control (подтверждение в VK).")

    # (2) danger → проверяет Второй агент (быстрый авто-ответчик), а не владелец (полная автономия).
    for rx, reason in _DANGER:
        if rx.search(command):
            import supervisor
            verdict = await supervisor.answer(
                f"Первый агент хочет выполнить потенциально опасную bash-команду ({reason}):\n{command[:600]}\n"
                f"Оцени риск для проекта/данных/денег. Ответь строго 'РАЗРЕШАЮ' если безопасно в контексте "
                f"валидации компонентов, или 'ЗАПРЕЩАЮ: <причина и безопасная альтернатива>'.", "")
            up = verdict.upper()
            allowed = "РАЗРЕШ" in up and "ЗАПРЕЩ" not in up and "NEEDS_OWNER" not in up
            _log_decision(command, reason, "ALLOW" if allowed else "DENY")
            if allowed:
                return {}  # allow
            return _deny(f"Супервайзер не разрешил ({reason}): {verdict[:300]}")
    return {}


def _deny(reason: str):
    return {"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }}
