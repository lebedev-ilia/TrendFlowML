#!/usr/bin/env python3
"""Load testing script для Fetcher.

Скрипт для тестирования производительности Fetcher на целевой нагрузке
(например, 10k видео/день).

Использование:
    python scripts/load_test.py --target 10000 --duration 86400
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Добавляем путь к корню проекта
sys.path.insert(0, "/media/ilya/Новый том/TrendFlowML/Fetcher")

from fetcher.db import session_scope
from fetcher.metrics import (
    fetcher_cache_hits_total,
    fetcher_cache_miss_total,
    fetcher_videos_downloaded_total,
    fetcher_videos_failed_total,
)
from fetcher.models import Run, VideoSource
from fetcher.orchestrator import fetch_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LoadTestRunner:
    """Класс для выполнения load-тестов Fetcher."""

    def __init__(
        self,
        target_requests: int,
        duration_seconds: int,
        video_urls: Optional[List[str]] = None,
    ):
        """Инициализация load test runner.

        Args:
            target_requests: Целевое количество запросов (например, 10000)
            duration_seconds: Длительность теста в секундах (например, 86400 для дня)
            video_urls: Список URL видео для тестирования (если None, генерируются)
        """
        self.target_requests = target_requests
        self.duration_seconds = duration_seconds
        self.video_urls = video_urls or self._generate_test_urls()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.results: Dict[str, any] = defaultdict(list)

    def _generate_test_urls(self) -> List[str]:
        """Генерировать список тестовых URL видео.

        Для тестирования используем популярные YouTube видео.
        """
        # Примеры популярных YouTube видео для тестирования
        test_videos = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Astley - Never Gonna Give You Up
            "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # Me at the zoo
            "https://www.youtube.com/watch?v=9bZkp7q19f0",  # PSY - GANGNAM STYLE
        ]

        # Дублируем список для достижения target_requests
        urls = []
        while len(urls) < self.target_requests:
            urls.extend(test_videos)

        return urls[: self.target_requests]

    def create_run(self, url: str) -> str:
        """Создать run для тестирования.

        Args:
            url: URL видео

        Returns:
            UUID run'а
        """
        import uuid

        run_id = uuid.uuid4()
        with session_scope() as db:
            run = Run(
                id=run_id,
                source_type="youtube",
                source_url=url,
                status="PENDING",
            )
            db.add(run)

            video_source = VideoSource(
                run_id=run_id,
                platform="youtube",
                url=url,
            )
            db.add(video_source)
            db.commit()

        return str(run_id)

    def run_load_test(self) -> Dict[str, any]:
        """Запустить load test.

        Returns:
            Словарь с результатами теста
        """
        logger.info(
            f"Starting load test: {self.target_requests} requests over {self.duration_seconds}s"
        )

        self.start_time = time.time()
        self.end_time = self.start_time + self.duration_seconds

        # Вычисляем rate (запросов в секунду)
        requests_per_second = self.target_requests / self.duration_seconds
        interval = 1.0 / requests_per_second if requests_per_second > 0 else 1.0

        logger.info(f"Target rate: {requests_per_second:.2f} requests/second")
        logger.info(f"Interval between requests: {interval:.3f} seconds")

        completed = 0
        failed = 0
        run_ids: List[str] = []

        # Запускаем тест
        current_time = time.time()
        request_index = 0

        while current_time < self.end_time and request_index < len(self.video_urls):
            # Создаём run
            url = self.video_urls[request_index]
            run_id = self.create_run(url)
            run_ids.append(run_id)

            # Запускаем orchestrator
            request_start = time.time()
            try:
                fetch_video(run_id)
                completed += 1
                self.results["success"].append(time.time() - request_start)
            except Exception as e:
                failed += 1
                self.results["failed"].append(time.time() - request_start)
                logger.error(f"Request failed for run_id={run_id}: {e}")

            request_index += 1

            # Ждём до следующего запроса
            elapsed = time.time() - current_time
            if elapsed < interval:
                time.sleep(interval - elapsed)

            current_time = time.time()

        # Ждём завершения всех задач (с таймаутом)
        logger.info("Waiting for tasks to complete...")
        max_wait_time = 3600  # 1 час
        wait_start = time.time()

        while time.time() - wait_start < max_wait_time:
            with session_scope() as db:
                completed_runs = (
                    db.query(Run)
                    .filter(Run.id.in_([uuid.UUID(rid) for rid in run_ids]))
                    .filter(Run.status.in_(["COMPLETED", "FAILED"]))
                    .count()
                )

                if completed_runs >= len(run_ids):
                    break

            time.sleep(5)

        # Собираем финальные результаты
        return self._collect_results(run_ids, completed, failed)

    def _collect_results(self, run_ids: List[str], completed: int, failed: int) -> Dict[str, any]:
        """Собрать результаты теста.

        Args:
            run_ids: Список UUID run'ов
            completed: Количество успешных запросов
            failed: Количество неуспешных запросов

        Returns:
            Словарь с результатами
        """
        import uuid

        from fetcher.db import session_scope

        with session_scope() as db:
            runs = (
                db.query(Run)
                .filter(Run.id.in_([uuid.UUID(rid) for rid in run_ids]))
                .all()
            )

            status_counts = defaultdict(int)
            for run in runs:
                status_counts[run.status] += 1

        # Вычисляем метрики
        total_time = time.time() - self.start_time if self.start_time else 0
        throughput = len(run_ids) / total_time if total_time > 0 else 0

        success_times = self.results["success"]
        failed_times = self.results["failed"]

        avg_success_time = sum(success_times) / len(success_times) if success_times else 0
        avg_failed_time = sum(failed_times) / len(failed_times) if failed_times else 0

        results = {
            "test_config": {
                "target_requests": self.target_requests,
                "duration_seconds": self.duration_seconds,
                "actual_duration": total_time,
            },
            "requests": {
                "total": len(run_ids),
                "completed": completed,
                "failed": failed,
                "throughput_per_second": throughput,
            },
            "run_statuses": dict(status_counts),
            "latency": {
                "avg_success_seconds": avg_success_time,
                "avg_failed_seconds": avg_failed_time,
                "min_success_seconds": min(success_times) if success_times else 0,
                "max_success_seconds": max(success_times) if success_times else 0,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        return results

    def print_results(self, results: Dict[str, any]) -> None:
        """Вывести результаты теста.

        Args:
            results: Результаты теста
        """
        print("\n" + "=" * 80)
        print("LOAD TEST RESULTS")
        print("=" * 80)
        print(f"\nTest Configuration:")
        print(f"  Target requests: {results['test_config']['target_requests']}")
        print(f"  Duration: {results['test_config']['duration_seconds']}s")
        print(f"  Actual duration: {results['test_config']['actual_duration']:.2f}s")

        print(f"\nRequests:")
        print(f"  Total: {results['requests']['total']}")
        print(f"  Completed: {results['requests']['completed']}")
        print(f"  Failed: {results['requests']['failed']}")
        print(f"  Throughput: {results['requests']['throughput_per_second']:.2f} req/s")

        print(f"\nRun Statuses:")
        for status, count in results["run_statuses"].items():
            print(f"  {status}: {count}")

        print(f"\nLatency:")
        print(f"  Avg success: {results['latency']['avg_success_seconds']:.3f}s")
        print(f"  Avg failed: {results['latency']['avg_failed_seconds']:.3f}s")
        print(f"  Min success: {results['latency']['min_success_seconds']:.3f}s")
        print(f"  Max success: {results['latency']['max_success_seconds']:.3f}s")

        print(f"\nTimestamp: {results['timestamp']}")
        print("=" * 80 + "\n")


def main():
    """Главная функция для запуска load test."""
    parser = argparse.ArgumentParser(description="Load test для Fetcher")
    parser.add_argument(
        "--target",
        type=int,
        default=10000,
        help="Целевое количество запросов (по умолчанию 10000)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=86400,
        help="Длительность теста в секундах (по умолчанию 86400 = 1 день)",
    )
    parser.add_argument(
        "--urls",
        type=str,
        nargs="+",
        help="Список URL видео для тестирования (опционально)",
    )

    args = parser.parse_args()

    runner = LoadTestRunner(
        target_requests=args.target,
        duration_seconds=args.duration,
        video_urls=args.urls,
    )

    try:
        results = runner.run_load_test()
        runner.print_results(results)
    except KeyboardInterrupt:
        logger.info("Load test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Load test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

