"""
Управление прогрессом и отчеты о прогрессе.
Использует простые текстовые выводы вместо прогресс-бара для равномерного отображения.
"""
import sys
import os
import json
import logging
import time
from typing import Optional, Any

from .cli_utils import utc_iso_now

logger = logging.getLogger(__name__)

# ANSI color codes для терминала
class Colors:
    """ANSI escape codes для цветного вывода."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Основные цвета
    BLUE = '\033[34m'      # AudioProcessor
    CYAN = '\033[36m'      # Проценты [XX%]
    YELLOW = '\033[33m'    # Названия компонентов/extractors
    GREEN = '\033[32m'     # Время
    GRAY = '\033[90m'      # Разделители и дополнительный текст
    
    # Проверка поддержки цветов
    @staticmethod
    def supports_color() -> bool:
        """Проверяет, поддерживает ли терминал цвета."""
        # Принудительное отключение через FORCE_NO_COLOR (высший приоритет)
        if os.getenv('FORCE_NO_COLOR') in ('1', 'true', 'yes'):
            return False
        # Принудительное включение цветов через FORCE_COLOR
        if os.getenv('FORCE_COLOR') in ('1', 'true', 'yes'):
            return True
        # Если stderr это терминал - используем цвета
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            return True
        # По умолчанию используем цвета (большинство современных терминалов поддерживают ANSI)
        # Игнорируем NO_COLOR и TERM=dumb, так как они могут быть установлены неправильно
        # Пользователь может явно отключить через FORCE_NO_COLOR если нужно
        return True

# Отслеживание последнего вывода для предотвращения дублирования
_last_stage_id: Optional[str] = None
_last_progress_pct: int = -1
_last_update_time: float = 0.0
_min_update_interval: float = 0.5  # Минимальный интервал между обновлениями (секунды)


def emit_progress(
    *,
    platform_id: str,
    video_id: str,
    run_id: str,
    component: str,
    stage_id: str,
    stage_name: str,
    progress_pct: int,
    extractor: Optional[str] = None,
    elapsed_sec: Optional[float] = None,
    total_elapsed_sec: Optional[float] = None,
) -> None:
    """
    Отправляет событие прогресса с простым текстовым выводом.
    Выводит информацию своевременно и равномерно.
    
    Args:
        platform_id: ID платформы
        video_id: ID видео
        run_id: ID запуска
        component: Компонент
        stage_id: ID этапа
        stage_name: Название этапа
        progress_pct: Процент прогресса (0-100)
        extractor: Опциональное имя extractor'а
        elapsed_sec: Время выполнения текущей стадии (секунды)
        total_elapsed_sec: Общее время выполнения (секунды)
    """
    global _last_stage_id, _last_progress_pct, _last_update_time
    
    progress_pct = max(0, min(100, progress_pct))
    current_time = time.time()
    
    # Определяем, нужно ли выводить сообщение
    should_emit = False
    
    # Всегда выводим при смене этапа
    if stage_id != _last_stage_id:
        should_emit = True
        _last_stage_id = stage_id
    # Выводим при изменении прогресса (>= 3%) или через интервал времени
    elif abs(progress_pct - _last_progress_pct) >= 3:
        should_emit = True
    elif current_time - _last_update_time >= _min_update_interval:
        # Выводим если прошло достаточно времени (для долгих операций)
        should_emit = True
    
    if not should_emit:
        return
    
    _last_update_time = current_time
    _last_progress_pct = progress_pct
    
    # Проверяем поддержку цветов
    use_colors = Colors.supports_color()
    
    # Строим сообщение для отображения с цветами
    if use_colors:
        # Цветной вывод
        audio_processor_prefix = f"{Colors.BLUE}{Colors.BOLD}AudioProcessor{Colors.RESET} {Colors.GRAY}|{Colors.RESET}"
        progress_pct_str = f"{Colors.CYAN}[{progress_pct:3d}%]{Colors.RESET}"
        
        if extractor:
            extractor_name = f"{Colors.YELLOW}{extractor}{Colors.RESET}"
            stage_name_part = f"{Colors.GRAY}:{Colors.RESET} {stage_name}"
            display_msg = f"{progress_pct_str} {extractor_name}{stage_name_part}"
        else:
            display_msg = f"{progress_pct_str} {stage_name}"
        
        # Информация о времени
        time_info = ""
        if elapsed_sec is not None:
            time_str = f"{Colors.GREEN}{elapsed_sec:.2f}s{Colors.RESET}"
            time_info = f" {Colors.GRAY}({Colors.RESET}{time_str}"
            if total_elapsed_sec is not None:
                total_time_str = f"{Colors.GREEN}{total_elapsed_sec:.2f}s{Colors.RESET}"
                time_info += f"{Colors.GRAY}, total: {Colors.RESET}{total_time_str}"
            time_info += f"{Colors.GRAY}){Colors.RESET}"
        
        # Выводим в stderr для видимости, но не мешаем stdout
        print(f"{audio_processor_prefix} {display_msg}{time_info}", file=sys.stderr, flush=True)
    else:
        # Обычный вывод без цветов
        if extractor:
            display_msg = f"[{progress_pct:3d}%] {extractor}: {stage_name}"
        else:
            display_msg = f"[{progress_pct:3d}%] {stage_name}"
        
        # Добавляем информацию о времени
        time_info = ""
        if elapsed_sec is not None:
            time_info = f" ({elapsed_sec:.2f}s"
            if total_elapsed_sec is not None:
                time_info += f", total: {total_elapsed_sec:.2f}s"
            time_info += ")"
        
        # Выводим в stderr для видимости, но не мешаем stdout
        print(f"AudioProcessor | {display_msg}{time_info}", file=sys.stderr, flush=True)
    
    # Также выводим JSON для машинной обработки (опционально, можно отключить)
    # event = {
    #     "platform_id": platform_id,
    #     "video_id": video_id,
    #     "run_id": run_id,
    #     "component": component,
    #     "stage_id": stage_id,
    #     "stage_name": stage_name,
    #     "progress_pct": progress_pct,
    #     "ts": utc_iso_now(),
    # }
    # if extractor is not None:
    #     event["extractor"] = extractor
    # try:
    #     print(json.dumps(event, ensure_ascii=False), flush=True)
    # except Exception:
    #     pass


def close_progress_bar() -> None:
    """Завершает вывод прогресса."""
    global _last_stage_id, _last_progress_pct, _last_update_time
    # Проверяем поддержку цветов
    use_colors = Colors.supports_color()
    
    # Выводим финальное сообщение
    if use_colors:
        audio_processor_prefix = f"{Colors.BLUE}{Colors.BOLD}AudioProcessor{Colors.RESET} {Colors.GRAY}|{Colors.RESET}"
        progress_pct_str = f"{Colors.CYAN}[100%]{Colors.RESET}"
        complete_text = f"{Colors.GREEN}Complete{Colors.RESET}"
        print(f"{audio_processor_prefix} {progress_pct_str} {complete_text}", file=sys.stderr, flush=True)
    else:
        print("AudioProcessor | [100%] Complete", file=sys.stderr, flush=True)
    
    # Сбрасываем состояние
    _last_stage_id = None
    _last_progress_pct = -1
    _last_update_time = 0.0

