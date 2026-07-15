#!/usr/bin/env python3
"""Ночной watchdog для контроля Opus в режиме /sas.
Запускается каждые 20 минут, проверяет состояние и исправляет проблемы.

Запуск: python nightwatch.py
"""
import json
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import config
import limits
import podmanager

def ts():
    """ISO-time для логирования."""
    return datetime.now().isoformat(timespec='seconds')

def log(msg: str):
    """Логирование с timestamp."""
    print(f"[{ts()}] [WATCH] {msg}", flush=True)

def check_opus_active() -> dict | None:
    """Вернуть информацию об активном Opus если он работает, иначе None."""
    try:
        agents = json.loads(config.AGENTS_FILE.read_text())
        # Ищем component-runner (Opus) или Opus
        opus_agents = [
            v for v in agents.values()
            if v.get('role') in ('Opus', 'component-runner')
        ]
        if opus_agents:
            return opus_agents[0]  # есть живой Opus
        return None
    except Exception as e:
        log(f"❌ Ошибка при чтении agents.json: {e}")
        return None

def check_pod_status() -> dict:
    """Вернуть статус подов: {pod_id: status_str}."""
    try:
        result = subprocess.run(
            ["python", str(config.RUNNER_DIR / "podmanager.py"), "list"],
            capture_output=True, text=True, timeout=30, cwd=str(config.RUNNER_DIR)
        )
        if result.returncode == 0 and result.stdout.strip():
            pods = {}
            # Парсим текстовый вывод summary_text()
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # пропускаем первую строку "🖥️ Машин: N"
                if '•' in line and '[' in line:
                    # Формат: • pod_id [provider/kind/policy] status ssh=...
                    parts = line.split(' ')
                    pod_id = parts[1]  # второе слово после •
                    # Ищем статус в скобке
                    for i, part in enumerate(parts):
                        if '[' in part:
                            # Статус идёт после ] в следующих словах
                            status_idx = i + 1
                            if status_idx < len(parts):
                                status = parts[status_idx]
                                pods[pod_id] = status
                                break
            return pods if pods else {}
        else:
            if result.stderr:
                log(f"❌ podmanager list ошибка: {result.stderr}")
            return {}
    except Exception as e:
        log(f"❌ Ошибка при проверке подов: {e}")
        return {}

def check_limits() -> dict:
    """Вернуть текущие лимиты."""
    try:
        snap = config.LIMITS_SNAPSHOT.read_text()
        return json.loads(snap)
    except Exception as e:
        log(f"⚠️  Не удалось прочитать лимиты: {e}")
        return {}

def is_opus_stuck(timeout_sec: int = 300) -> bool:
    """Проверить, не залип ли Opus (нет активности за timeout_sec)."""
    try:
        opus = check_opus_active()
        if not opus:
            return False  # если его вообще нет, это не "stuck"

        last_seen = opus.get('last_seen', 0)
        now = time.time()
        inactive_sec = now - last_seen

        if inactive_sec > timeout_sec:
            log(f"⚠️  Opus залип: не пинговался {inactive_sec:.0f}с (лимит: {timeout_sec}с)")
            return True
        return False
    except Exception as e:
        log(f"❌ Ошибка при проверке timeout: {e}")
        return False

def handle_stuck_opus():
    """Убить и перезапустить залипший Opus."""
    log("🔴 Попытка восстановления Opus...")
    try:
        # Find running agent processes and kill them
        result = subprocess.run(
            ["pgrep", "-f", "agent_runner.py"],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                log(f"  Убиваю процесс {pid}...")
                subprocess.run(["kill", "-9", pid], timeout=10)

        # Stop the pod if it's running
        try:
            pods = check_pod_status()
            for pod_id, status in pods.items():
                if 'running' in status.lower():
                    log(f"  Останавливаю под {pod_id}...")
                    subprocess.run(
                        ["python", str(config.RUNNER_DIR / "podmanager.py"), "stop", pod_id],
                        timeout=30
                    )
        except Exception as e:
            log(f"  ⚠️  Не удалось остановить под: {e}")

        # Clean up agents registry
        config.AGENTS_FILE.write_text("{}")

        time.sleep(2)

        # Restart agent_runner
        log("  Перезапускаю agent_runner.py в фоне...")
        subprocess.Popen(
            ["python", str(config.RUNNER_DIR / "agent_runner.py")],
            cwd=str(config.RUNNER_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        log("✅ Opus перезапущен")
        return True
    except Exception as e:
        log(f"❌ Ошибка при восстановлении: {e}")
        return False

def handle_exited_pods():
    """Если 2+ пода EXITED, удалить 1 и создать новый."""
    try:
        pods = check_pod_status()
        exited = [pid for pid, st in pods.items() if st.lower() == 'exited']

        if len(exited) >= 2:
            log(f"⚠️  Обнаружено {len(exited)} упавших подов: {exited}")
            to_delete = exited[0]
            log(f"  Удаляю под {to_delete}...")

            result = subprocess.run(
                ["python", str(config.RUNNER_DIR / "podmanager.py"), "terminate", to_delete],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                log(f"  ✅ Под {to_delete} удален")
                time.sleep(2)

                log("  Создаю новый под...")
                result = subprocess.run(
                    ["python", str(config.RUNNER_DIR / "podmanager.py"), "launch"],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0:
                    log(f"  ✅ Новый под создан")
                    return True
                else:
                    log(f"  ❌ Ошибка при создании пода: {result.stderr}")
            else:
                log(f"  ❌ Ошибка при удалении пода: {result.stderr}")

        return False
    except Exception as e:
        log(f"❌ Ошибка при обработке упавших подов: {e}")
        return False

def report():
    """Сформировать отчёт о текущем состоянии."""
    log("=" * 60)

    # Opus статус
    opus = check_opus_active()
    if opus:
        log(f"✅ Opus активен (модель: {opus.get('model', '?')}, last_seen: {time.time() - opus.get('last_seen', 0):.0f}s ago)")
    else:
        log("❌ Opus НЕ активен")

    # Лимиты
    limits_data = check_limits()
    if limits_data:
        log(f"📊 Лимиты: 5h={limits_data.get('5h', '?')}% | 1d={limits_data.get('1d', '?')}% | credits={limits_data.get('credits', '?')}%")
    else:
        log("⚠️  Лимиты не обновлены")

    # Поды
    pods = check_pod_status()
    if pods:
        for pid, st in pods.items():
            icon = "✅" if "running" in st.lower() else "❌"
            log(f"  {icon} {pid}: {st}")
    else:
        log("⚠️  Информация о подах недоступна")

    # Проверим процесс agent_runner
    result = subprocess.run(
        ["pgrep", "-f", "agent_runner.py"],
        capture_output=True, text=True
    )
    if result.stdout.strip():
        log(f"✅ agent_runner процесс запущен (PID: {result.stdout.strip().split()[0]})")
    else:
        log("❌ agent_runner процесс НЕ запущен")

    log("=" * 60)

def save_status_file():
    """Сохранить текущий статус в JSON файл для быстрой проверки."""
    try:
        status = {
            "timestamp": ts(),
            "opus_active": bool(check_opus_active()),
            "limits": check_limits(),
            "pods": check_pod_status(),
            "process_running": bool(subprocess.run(
                ["pgrep", "-f", "agent_runner.py"],
                capture_output=True
            ).stdout.strip())
        }
        status_file = config.RUNNER_DIR / ".nightwatch_status.json"
        status_file.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    except Exception as e:
        log(f"⚠️  Не удалось сохранить статус файл: {e}")

def main():
    """Главный цикл watchdog."""
    try:
        log("🌙 Ночной watchdog запущен (интервал: 20 минут)")

        while True:
            try:
                log("\n📍 Проверка статуса...")

                # 1. Базовый отчёт
                report()

                # 2. Проверить залипание Opus
                if is_opus_stuck(timeout_sec=300):
                    handle_stuck_opus()

                # 3. Проверить упавшие поды
                handle_exited_pods()

                # 4. Сохранить статус для быстрой проверки
                save_status_file()

                # 5. Финальный статус
                log("✅ Проверка завершена\n")

            except Exception as e:
                log(f"❌ Ошибка в цикле: {e}")
                import traceback
                traceback.print_exc()

            # Ждём 20 минут до следующей проверки
            log("⏳ Ожидание 20 минут до следующей проверки...")
            time.sleep(20 * 60)

    except KeyboardInterrupt:
        log("⏹️  Watchdog остановлен")
        sys.exit(0)
    except Exception as e:
        log(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
