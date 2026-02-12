"""
Мониторинг ресурсов (CPU, GPU, память) во время выполнения.
"""
import os
import threading
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """
    Класс для мониторинга использования ресурсов (CPU RSS, GPU VRAM).
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sampler_thread: Optional[threading.Thread] = None
        self._max_rss_mb: Optional[float] = None
        self._max_gpu_used_mb: Optional[float] = None
    
    def _maybe_import_psutil(self):
        """Импортирует psutil если доступен."""
        try:
            import psutil  # type: ignore
            return psutil
        except Exception:
            return None
    
    def _rss_mb(self) -> Optional[float]:
        """Получает RSS память в MB."""
        psutil = self._maybe_import_psutil()
        if psutil is None:
            return None
        try:
            p = psutil.Process(os.getpid())
            return float(p.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            return None
    
    def _gpu_used_mb0(self) -> Optional[float]:
        """
        Получает используемую GPU память в MB (best-effort).
        Использует NVML если установлен, иначе torch memory_allocated.
        """
        # Проверяем доступность CUDA
        try:
            import torch
            if not torch.cuda.is_available():
                return None
        except Exception:
            return None
        
        # Пробуем NVML (более точный)
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            used = float(info.used) / (1024.0 * 1024.0)
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            return used
        except Exception:
            # Fallback на torch
            try:
                import torch
                return float(torch.cuda.memory_allocated(0)) / (1024.0 * 1024.0)
            except Exception:
                return None
    
    def _sampler_loop(self):
        """Основной цикл сэмплера ресурсов."""
        # Сэмплируем относительно часто чтобы поймать пики; best-effort only.
        while not self._stop_event.wait(0.2):
            r = self._rss_mb()
            g = self._gpu_used_mb0()
            with self._lock:
                if r is not None:
                    self._max_rss_mb = r if self._max_rss_mb is None else max(self._max_rss_mb, r)
                if g is not None:
                    self._max_gpu_used_mb = g if self._max_gpu_used_mb is None else max(self._max_gpu_used_mb, g)
    
    def start(self):
        """Запускает мониторинг ресурсов в отдельном потоке."""
        if self._sampler_thread is not None:
            logger.warning("ResourceMonitor already started")
            return
        
        self._stop_event.clear()
        self._sampler_thread = threading.Thread(
            target=self._sampler_loop,
            name="audio_runtime_sampler",
            daemon=True
        )
        self._sampler_thread.start()
        logger.debug("ResourceMonitor started")
    
    def stop(self, timeout: float = 1.0):
        """
        Останавливает мониторинг ресурсов.
        
        Args:
            timeout: Таймаут для ожидания завершения потока (секунды)
        """
        if self._sampler_thread is None:
            return
        
        self._stop_event.set()
        try:
            if self._sampler_thread.is_alive():
                self._sampler_thread.join(timeout=timeout)
        except Exception:
            pass
        
        self._sampler_thread = None
        logger.debug("ResourceMonitor stopped")
    
    def get_metrics(self) -> Dict[str, Optional[float]]:
        """
        Возвращает текущие метрики ресурсов.
        
        Returns:
            Словарь с метриками:
            - cpu_rss_peak_mb: пиковая RSS память CPU (MB)
            - gpu_vram_peak_mb: пиковая GPU VRAM (MB)
        """
        with self._lock:
            return {
                "cpu_rss_peak_mb": self._max_rss_mb,
                "gpu_vram_peak_mb": self._max_gpu_used_mb,
            }

