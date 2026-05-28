"""
Processor Service - интеграция с main.py

Этот сервис отвечает за запуск обработки через main.py DataProcessor.

Для MVP используется ThreadPoolExecutor с ограничением параллелизма.
В будущем (Этап 2) будет заменён на Redis Queue + Worker процессы.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1381, 1457-1481)
"""

import subprocess
import asyncio
import os
import sys
import yaml
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional
from pathlib import Path

from api.schemas.requests import ProcessRequest
from api.config import config
from api.utils.logging import get_logger

logger = get_logger(__name__)

# Worker читает эти поля для Prometheus: см. api/services/worker.py (processing_time, failure_rate).
MAIN_PY_PIPELINE_PROCESSOR = "pipeline"
MAIN_PY_PIPELINE_COMPONENT = "main_py"


def _with_main_py_metric_labels(result: Dict[str, Any]) -> Dict[str, Any]:
    """Добавить processor/component для гистограммы времени и счётчика ошибок (не затирать, если уже заданы)."""
    out = dict(result)
    out.setdefault("processor", MAIN_PY_PIPELINE_PROCESSOR)
    out.setdefault("component", MAIN_PY_PIPELINE_COMPONENT)
    return out


def _get_dataprocessor_root() -> str:
    """Вернуть абсолютный путь к корню пакета DataProcessor."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", ".."))


class ProcessorService:
    """
    Сервис для запуска обработки через main.py.
    
    Для MVP использует ThreadPoolExecutor.
    В будущем будет заменён на Redis Queue + Worker.
    """
    
    def __init__(self, executor: Optional[ThreadPoolExecutor] = None):
        """
        Инициализация ProcessorService.
        
        Args:
            executor: ThreadPoolExecutor для запуска задач (опционально)
        """
        self.executor = executor or ThreadPoolExecutor(max_workers=config.max_concurrent_runs)
    
    def _convert_to_cli_args(self, request: ProcessRequest) -> list[str]:
        """
        Конвертировать ProcessRequest в аргументы командной строки для main.py.
        
        Args:
            request: Запрос на обработку
            
        Returns:
            Список аргументов командной строки
            
        Ссылка: DataProcessor/main.py (строки 408-432) для формата аргументов
        """
        args = [
            "--video-path", request.video_path,
            "--platform-id", request.platform_id,
        ]
        
        # Обязательные параметры (--video-id=… чтобы id вроде -Q6fnPIybEI не ломал argparse)
        if request.video_id:
            args.append(f"--video-id={request.video_id}")
        if request.run_id:
            args.append(f"--run-id={request.run_id}")
        
        # Опциональные параметры из ProcessRequest
        if request.visual_cfg_path:
            args.extend(["--visual-cfg-path", request.visual_cfg_path])
        if request.dag_path:
            args.extend(["--dag-path", request.dag_path])
        if request.dag_stage:
            args.extend(["--dag-stage", request.dag_stage])
        if request.output:
            args.extend(["--output", request.output])
        if request.rs_base:
            args.extend(["--rs-base", request.rs_base])
        if request.sampling_policy_version:
            args.extend(["--sampling-policy-version", request.sampling_policy_version])
        if request.dataprocessor_version:
            args.extend(["--dataprocessor-version", request.dataprocessor_version])
        if request.analysis_fps is not None:
            args.extend(["--analysis-fps", str(request.analysis_fps)])
        if request.analysis_width is not None:
            args.extend(["--analysis-width", str(request.analysis_width)])
        if request.analysis_height is not None:
            args.extend(["--analysis-height", str(request.analysis_height)])
        if request.chunk_size is not None:
            args.extend(["--chunk-size", str(request.chunk_size)])
        if request.run_audio is not None:
            if request.run_audio:
                args.append("--run-audio")
        if request.run_text is not None:
            # run_text не имеет отдельного флага, управляется через profile_config
            pass
        if request.global_config_path:
            args.extend(["--global-config", str(request.global_config_path)])

        # profile_config нужно сохранить в файл и передать через --profile-path
        # Это будет сделано в _run_main_py_sync перед запуском subprocess
        
        return args
    
    def _save_profile_config(
        self,
        profile_config: Dict[str, Any],
        run_id: str
    ) -> str:
        """
        Сохранить profile_config в временный YAML файл.
        
        Args:
            profile_config: Конфигурация профиля
            run_id: UUID run'а для создания уникального пути
            
        Returns:
            Путь к сохранённому файлу
            
        Ссылка: backend/app/services/dataprocessor.py (строки 64-67, 319-321)
        """
        # Создать директорию для профилей
        # Создать директорию profiles_cache в корне проекта
        dataprocessor_root = _get_dataprocessor_root()
        profiles_cache_dir = os.path.join(dataprocessor_root, "_profiles_cache", run_id)
        os.makedirs(profiles_cache_dir, exist_ok=True)
        
        # Сохранить профиль в YAML файл
        profile_path = os.path.join(profiles_cache_dir, "profile.yaml")
        
        try:
            with open(profile_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(profile_config, f, sort_keys=False, allow_unicode=True)
            
            logger.debug(f"Saved profile config to {profile_path}")
            return profile_path
            
        except Exception as e:
            logger.error(f"Failed to save profile config: {e}")
            raise
    
    async def _run_main_py_async(self, request: ProcessRequest) -> Dict[str, Any]:
        """
        Асинхронный запуск main.py через subprocess с изоляцией.
        
        Использует asyncio.create_subprocess_exec для лучшего контроля над процессом.
        
        Args:
            request: Запрос на обработку
            
        Returns:
            Результат выполнения (словарь с информацией о результате)
            
        Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 744-778)
        """
        # Определяем путь к main.py относительно текущего файла
        # api/services/processor.py -> DataProcessor/main.py
        dataprocessor_root = _get_dataprocessor_root()
        main_py_path = Path(dataprocessor_root) / "main.py"
        
        # Сохранить profile_config в временный файл
        profile_path = None
        try:
            profile_path = self._save_profile_config(request.profile_config, request.run_id)
        except Exception as e:
            logger.error(f"Failed to save profile config for run_id={request.run_id}: {e}")
            return _with_main_py_metric_labels({
                "success": False,
                "run_id": request.run_id,
                "error": f"Failed to save profile config: {str(e)}",
                "error_type": "profile_config_error",
            })
        
        # Добавить --profile-path к аргументам
        args = self._convert_to_cli_args(request)
        if profile_path:
            args.extend(["--profile-path", profile_path])
        
        # Определить Python интерпретатор
        python_cmd = os.environ.get("PYTHON", sys.executable)
        if not python_cmd:
            python_cmd = "python3"
        
        # Подготовить команду
        cmd = [python_cmd, str(main_py_path)] + args
        
        try:
            logger.info(
                "Starting processing",
                run_id=request.run_id,
                video_id=request.video_id,
                platform_id=request.platform_id
            )
            logger.debug(
                "Command for processing",
                run_id=request.run_id,
                command=" ".join(cmd)
            )
            
            # Запуск subprocess с изоляцией
            # Используем asyncio.create_subprocess_exec для лучшего контроля
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=dataprocessor_root,
                # Установить переменные окружения для изоляции
                env=os.environ.copy()
            )
            
            # Опциональный мониторинг памяти (если psutil доступен)
            monitor_task = None
            if config.subprocess_memory_limit_mb:
                monitor_task = asyncio.create_task(
                    self._monitor_subprocess_memory(process, request.run_id)
                )
            
            # Ожидание завершения процесса и чтение stdout/stderr
            stdout_bytes, stderr_bytes = await process.communicate()
            
            # Отменить мониторинг если он был запущен
            if monitor_task:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Декодировать вывод
            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

            # Метрики по компонентам из manifest (тайминги из VisualProcessor/Audio/…)
            try:
                from api.utils.result_store_path import resolve_run_result_store_path
                from api.services.manifest_metrics import record_from_manifest

                mpath = os.path.join(
                    resolve_run_result_store_path(
                        platform_id=request.platform_id,
                        video_id=request.video_id,
                        run_id=request.run_id,
                        rs_base=request.rs_base,
                    ),
                    "manifest.json",
                )
                n = record_from_manifest(mpath)
                if n:
                    logger.debug(
                        "Manifest component metrics recorded",
                        run_id=request.run_id,
                        manifest=mpath,
                        components=n,
                    )
            except Exception as e:
                logger.debug("Manifest component metrics skipped: %s", e)
            
            # Логирование stdout/stderr
            if stdout:
                logger.debug(f"stdout for run_id={request.run_id}:\n{stdout}")
            if stderr:
                logger.warning(f"stderr for run_id={request.run_id}:\n{stderr}")
            
            # Обработка exit code
            returncode = process.returncode
            
            if returncode == 0:
                logger.info(f"Processing completed successfully for run_id={request.run_id}")
                return _with_main_py_metric_labels({
                    "success": True,
                    "run_id": request.run_id,
                    "exit_code": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                })
            else:
                # Определить тип ошибки по exit code
                error_type = "unknown"
                if returncode == 1:
                    error_type = "general_error"
                elif returncode == 2:
                    error_type = "misuse_of_shell_commands"
                elif returncode > 128:
                    error_type = "signal_terminated"
                elif returncode == -9:  # SIGKILL
                    error_type = "killed_by_memory_limit"
                
                logger.error(
                    f"Processing failed for run_id={request.run_id}, "
                    f"exit_code={returncode}, error_type={error_type}"
                )
                
                # Извлечь основную ошибку из stderr если возможно
                error_message = f"Process exited with code {returncode}"
                if stderr:
                    # Попробовать найти последнюю строку с ошибкой
                    stderr_lines = stderr.strip().split("\n")
                    if stderr_lines:
                        last_line = stderr_lines[-1]
                        if "error" in last_line.lower() or "exception" in last_line.lower():
                            error_message = last_line
                
                return _with_main_py_metric_labels({
                    "success": False,
                    "run_id": request.run_id,
                    "exit_code": returncode,
                    "error_type": error_type,
                    "stdout": stdout,
                    "stderr": stderr,
                    "error": error_message,
                })
                
        except FileNotFoundError as e:
            logger.error(f"Python interpreter or main.py not found: {e}")
            return _with_main_py_metric_labels({
                "success": False,
                "run_id": request.run_id,
                "error": f"Python interpreter or main.py not found: {str(e)}",
                "error_type": "file_not_found",
            })
        except Exception as e:
            logger.exception(f"Error running main.py for run_id={request.run_id}: {e}")
            return _with_main_py_metric_labels({
                "success": False,
                "run_id": request.run_id,
                "error": str(e),
                "error_type": "exception",
            })
    
    async def _monitor_subprocess_memory(
        self,
        process: asyncio.subprocess.Process,
        run_id: str
    ) -> None:
        """
        Мониторинг использования памяти subprocess.
        
        Если превышен лимит, процесс будет убит.
        
        Args:
            process: Subprocess процесс
            run_id: UUID run'а
            
        Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2394-2416)
        """
        try:
            import psutil
        except ImportError:
            logger.debug("psutil not available, skipping memory monitoring")
            return
        
        memory_limit_mb = config.subprocess_memory_limit_mb
        if not memory_limit_mb:
            return
        
        check_interval = 10  # Проверять каждые 10 секунд
        
        while True:
            try:
                # Проверить что процесс еще работает
                if process.returncode is not None:
                    break
                
                # Получить информацию о процессе
                proc = psutil.Process(process.pid)
                memory_info = proc.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # Логировать использование памяти (info для важных событий)
                if memory_mb > memory_limit_mb * 0.8:  # Предупреждение при 80% использования
                    logger.warning(f"Run {run_id} memory usage: {memory_mb:.0f}MB / {memory_limit_mb}MB (80%+ threshold)")
                else:
                    logger.info(f"Run {run_id} memory usage: {memory_mb:.0f}MB / {memory_limit_mb}MB")
                
                # Обновить метрику использования памяти
                try:
                    from api.services.metrics import memory_usage
                    memory_usage.labels(run_id=run_id).set(int(memory_mb * 1024 * 1024))  # Конвертировать в байты
                except Exception as e:
                    logger.debug(f"Failed to update memory_usage metric: {e}")
                
                # Если превышен лимит → убить процесс
                if memory_mb > memory_limit_mb:
                    logger.warning(
                        f"Run {run_id} exceeded memory limit ({memory_mb:.0f}MB > {memory_limit_mb}MB), "
                        f"killing process"
                    )
                    try:
                        process.kill()
                    except ProcessLookupError:
                        # Процесс уже завершился
                        pass
                    break
                
                await asyncio.sleep(check_interval)
                
            except psutil.NoSuchProcess:
                # Процесс завершился
                break
            except Exception as e:
                logger.warning(f"Error monitoring memory for run {run_id}: {e}")
                await asyncio.sleep(check_interval)
    
    def _run_main_py_sync(self, request: ProcessRequest) -> Dict[str, Any]:
        """
        Синхронный запуск main.py через subprocess (legacy метод для совместимости).
        
        Этот метод используется для ThreadPoolExecutor и вызывает async версию.
        
        Args:
            request: Запрос на обработку
            
        Returns:
            Результат выполнения (словарь с информацией о результате)
            
        Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1474-1477)
        """
        # Запустить async версию в новом event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._run_main_py_async(request))
        except FileNotFoundError as e:
            logger.error(f"Python interpreter or main.py not found: {e}")
            return _with_main_py_metric_labels({
                "success": False,
                "run_id": request.run_id,
                "error": f"Python interpreter or main.py not found: {str(e)}",
                "error_type": "file_not_found",
            })
        except Exception as e:
            logger.exception(f"Error running main.py for run_id={request.run_id}: {e}")
            return _with_main_py_metric_labels({
                "success": False,
                "run_id": request.run_id,
                "error": str(e),
                "error_type": "exception",
            })
        finally:
            loop.close()

        stdout = result.get("stdout") or ""
        stderr = result.get("stderr") or ""
        if stdout:
            logger.debug(f"stdout for run_id={request.run_id}:\n{stdout}")
        if stderr:
            logger.warning(f"stderr for run_id={request.run_id}:\n{stderr}")

        if not result.get("success", False):
            out = dict(result)
            out.setdefault("error_type", "processing_failed")
            return _with_main_py_metric_labels(out)

        exit_code = result.get("exit_code", 0)
        if exit_code == 0:
            logger.info(f"Processing completed successfully for run_id={request.run_id}")
            return _with_main_py_metric_labels({
                "success": True,
                "run_id": request.run_id,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
            })

        error_type = "unknown"
        if exit_code == 1:
            error_type = "general_error"
        elif exit_code == 2:
            error_type = "misuse_of_shell_commands"
        elif exit_code > 128:
            error_type = "signal_terminated"

        logger.error(
            f"Processing failed for run_id={request.run_id}, "
            f"exit_code={exit_code}, error_type={error_type}"
        )

        error_message = f"Process exited with code {exit_code}"
        if stderr:
            stderr_lines = stderr.strip().split("\n")
            if stderr_lines:
                last_line = stderr_lines[-1]
                if "error" in last_line.lower() or "exception" in last_line.lower():
                    error_message = last_line

        return _with_main_py_metric_labels({
            "success": False,
            "run_id": request.run_id,
            "exit_code": exit_code,
            "error_type": error_type,
            "stdout": stdout,
            "stderr": stderr,
            "error": error_message,
        })
    
    async def run_processing(self, request: ProcessRequest) -> Dict[str, Any]:
        """
        Асинхронный запуск обработки.
        
        Использует async subprocess для лучшей изоляции и контроля.
        
        Args:
            request: Запрос на обработку
            
        Returns:
            Результат выполнения
        """
        # Использовать async версию напрямую для лучшей изоляции
        return await self._run_main_py_async(request)

