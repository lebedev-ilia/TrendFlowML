#!/usr/bin/env python3
"""
Алгоритм для агрегации результатов бенчмарков из 4 попыток.
Вычисляет средние значения для числовых метрик и определяет выбросы.

Поддерживает два типа результатов:
1. Обычные результаты (out_component) - одиночные запуски компонента
2. Параллельные результаты (out_component_parallel) - многопоточные запуски

Использование:
    # Для обычных бенчмарков
    python aggregate_benchmark_results.py benchmarks/out_component
    
    # Для параллельных бенчмарков
    python aggregate_benchmark_results.py benchmarks/out_component_parallel
    
    # Автоматическое определение (по умолчанию использует out_component)
    python aggregate_benchmark_results.py

Формат вывода соответствует FINAL_BENCH_TABLE.md с дополнительной колонкой "Threads"
для параллельных бенчмарков.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import statistics


def detect_outliers(values: List[float], method: str = "iqr") -> List[float]:
    """
    Определяет выбросы в списке значений.
    
    Args:
        values: Список числовых значений
        method: Метод определения выбросов ("iqr" или "zscore")
    
    Returns:
        Список значений-выбросов
    """
    if len(values) < 3:
        return []
    
    if method == "iqr":
        sorted_vals = sorted(values)
        # Вычисляем квантили правильно
        n = len(sorted_vals)
        if n % 2 == 0:
            q1 = statistics.median(sorted_vals[:n//2])
            q3 = statistics.median(sorted_vals[n//2:])
        else:
            q1 = statistics.median(sorted_vals[:n//2])
            q3 = statistics.median(sorted_vals[n//2+1:])
        
        iqr = q3 - q1
        
        if iqr == 0:
            # Если IQR = 0, все значения одинаковые - выбросов нет
            return []
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [v for v in values if v < lower_bound or v > upper_bound]
    elif method == "zscore":
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        if std == 0:
            return []
        # Используем более строгий порог для малых выборок
        threshold = 2.5 if len(values) >= 3 else 2.0
        outliers = [v for v in values if abs((v - mean) / std) > threshold]
    else:
        return []
    
    return outliers


def is_parallel_benchmark(data: Dict[str, Any]) -> bool:
    """Определяет, является ли результат параллельным бенчмарком."""
    return 'num_threads' in data and 'threads' in data and 'statistics' in data


def extract_models_from_parallel_result(data: Dict[str, Any], json_path: Path, component: str = "core_clip") -> Tuple[str, str, str]:
    """Извлекает информацию о моделях из параллельного результата."""
    # Определяем путь к NPZ файлу в зависимости от компонента
    if component == "core_clip":
        npz_subpath = "core_clip" / "embeddings.npz"
        default_model_1 = "clip_image_224"
        default_model_2 = "clip_text"
        default_preprocess = "preprocess_clip_image_224"
    elif component == "core_depth_midas":
        npz_subpath = "core_depth_midas" / "depth.npz"
        default_model_1 = "midas_384"
        default_model_2 = "N/A"  # MiDaS не имеет text модели
        default_preprocess = "midas_384"
    else:
        return default_model_1, default_model_2, default_preprocess
    
    # Пытаемся извлечь из первого успешного потока
    threads = data.get('threads', [])
    for thread in threads:
        if thread.get('success', False):
            # Пытаемся найти models_used в NPZ файле
            run_id = thread.get('run_id', '')
            if run_id:
                # Ищем NPZ файл относительно директории с results.json
                result_dir = json_path.parent
                npz_path = result_dir / 'result_store' / run_id / npz_subpath
                
                if npz_path.exists():
                    try:
                        import numpy as np
                        npz_data = np.load(npz_path, allow_pickle=True)
                        meta = npz_data.get('meta')
                        if meta is not None:
                            if hasattr(meta, 'item'):
                                meta = meta.item()
                            if isinstance(meta, dict):
                                models_used = meta.get('models_used', [])
                                if models_used:
                                    triton_model_1 = models_used[0].get('model_name', 'N/A') if models_used else 'N/A'
                                    # Для core_depth_midas нет второй модели (text), для core_clip есть
                                    if component == "core_depth_midas":
                                        triton_model_2 = "N/A"  # MiDaS не имеет text модели
                                    else:
                                        triton_model_2 = models_used[1].get('model_name', 'N/A') if len(models_used) > 1 else 'N/A'
                                    
                                    # Определяем preprocess preset в зависимости от компонента
                                    if component == "core_depth_midas":
                                        # Для MiDaS preset определяется из model_name (midas_256, midas_384, midas_512)
                                        model_name_lower = triton_model_1.lower()
                                        if "256" in model_name_lower:
                                            triton_preprocess = "midas_256"
                                        elif "512" in model_name_lower:
                                            triton_preprocess = "midas_512"
                                        else:
                                            triton_preprocess = "midas_384"  # default
                                    else:
                                        # Для core_clip ищем preprocess в models_used
                                        triton_preprocess = default_preprocess
                                        for model in models_used:
                                            model_name = model.get('model_name', '').lower()
                                            if 'preprocess' in model_name or 'clip_image' in model_name:
                                                triton_preprocess = model.get('model_name', triton_preprocess)
                                                break
                                    
                                    return triton_model_1, triton_model_2, triton_preprocess
                    except Exception:
                        pass
    
    # Fallback: используем дефолтные значения
    return default_model_1, default_model_2, default_preprocess


def extract_metrics_from_json(json_path: Path) -> Dict[str, Any]:
    """Извлекает метрики из JSON файла результата бенчмарка."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    is_parallel = is_parallel_benchmark(data)
    
    if is_parallel:
        # Параллельный бенчмарк
        component_name = data.get('component', 'core_clip')
        triton_model_1, triton_model_2, triton_preprocess = extract_models_from_parallel_result(data, json_path, component=component_name)
        triton_batch = data.get('batch_size', 1)
        num_threads = data.get('num_threads', 1)
        
        # Ресурсы
        resources = data.get('resources', {})
        before_triton = resources.get('before_triton', {})
        after_triton = resources.get('after_triton', {})
        after_execution = resources.get('after_execution', {})
        peaks = resources.get('peaks', {})
        
        # Вычисляем дельты
        # Для Triton дельты остаются теми же (измеряются до и после запуска Triton)
        triton_delta_ram = after_triton.get('cpu_mem_used_mb', 0) - before_triton.get('cpu_mem_used_mb', 0)
        triton_delta_vram = after_triton.get('gpu_mem_used_mb', 0) - before_triton.get('gpu_mem_used_mb', 0)
        
        # Для параллельных бенчмарков используем ПИКОВЫЕ значения во время выполнения,
        # а не значения после завершения, чтобы правильно отразить использование памяти
        # при параллельном выполнении нескольких потоков
        peak_ram_mb = peaks.get('ram_used_peak_mb', 0)
        peak_vram_mb = peaks.get('vram_used_peak_mb', 0)
        
        # Component delta = пиковое использование во время выполнения - использование после Triton
        component_delta_vram = peak_vram_mb - after_triton.get('gpu_mem_used_mb', 0)
        component_delta_ram = peak_ram_mb - after_triton.get('cpu_mem_used_mb', 0)
        
        # Summary delta = пиковое использование во время выполнения - использование до Triton
        summary_delta_ram = peak_ram_mb - before_triton.get('cpu_mem_used_mb', 0)
        summary_delta_vram = peak_vram_mb - before_triton.get('gpu_mem_used_mb', 0)
        
        # Статистика из параллельного выполнения
        statistics_data = data.get('statistics', {})
        avg_duration_sec = statistics_data.get('avg_duration_sec', 0.0)
        total_wall_time_sec = data.get('total_wall_time_sec', 0.0)
        throughput = statistics_data.get('throughput_runs_per_sec', 0.0)
        
        # Усредняем inference время из всех потоков
        threads = data.get('threads', [])
        component_name = data.get('component', 'core_clip')
        
        if component_name == "core_depth_midas":
            # Для core_depth_midas нет image/text inference, только depth inference
            depth_inference_times = []
            for thread in threads:
                if thread.get('success', False):
                    timing = thread.get('component_timing', {})
                    # Используем depth_inference или total как fallback
                    if timing and 'depth_inference' in timing:
                        depth_inference_times.append(timing['depth_inference'])
                    elif timing and 'total' in timing:
                        depth_inference_times.append(timing['total'])
                    else:
                        # Если timing не доступен, используем duration_sec потока
                        depth_inference_times.append(thread.get('duration_sec', 0.0))
            
            image_inference_sec = statistics.mean(depth_inference_times) if depth_inference_times else 0.0
            text_inference_sec = 0.0  # Нет text inference для MiDaS
        else:
            # Для core_clip
            image_inference_times = []
            text_inference_times = []
            for thread in threads:
                if thread.get('success', False):
                    timing = thread.get('component_timing', {})
                    if 'image_inference' in timing:
                        image_inference_times.append(timing['image_inference'])
                    if 'text_inference' in timing:
                        text_inference_times.append(timing['text_inference'])
            
            image_inference_sec = statistics.mean(image_inference_times) if image_inference_times else 0.0
            text_inference_sec = statistics.mean(text_inference_times) if text_inference_times else 0.0
        
        # Пиковый CPU timestamp
        peak_timestamps = resources.get('peak_timestamps', {})
        cpu_peak_elapsed_sec = peak_timestamps.get('cpu_util_peak_elapsed_sec', None)
        if cpu_peak_elapsed_sec:
            try:
                cpu_peak_elapsed_sec = float(cpu_peak_elapsed_sec)
            except (ValueError, TypeError):
                cpu_peak_elapsed_sec = None
        
        metrics = {
            'triton_model_1': triton_model_1,
            'triton_model_2': triton_model_2,
            'triton_preprocess': triton_preprocess,
            'triton_batch': triton_batch,
            'num_threads': num_threads,
            'frames_cnt': data.get('frames_count', 0),
            'runs': 4,  # Всегда 4 для агрегации
            'duration_sec': total_wall_time_sec,  # Используем wall time для параллельных
            'avg_duration_per_thread_sec': avg_duration_sec,
            'image_inference_sec': image_inference_sec,
            'text_inference_sec': text_inference_sec,
            'throughput_runs_per_sec': throughput,
            'peak_cpu_pct': peaks.get('cpu_util_peak_pct', 0.0),
            'peak_cpu_occurred_sec': cpu_peak_elapsed_sec,
            'peak_gpu_pct': peaks.get('gpu_util_peak_pct', 0.0),
            'triton_delta_ram_mb': triton_delta_ram,
            'triton_delta_vram_mb': triton_delta_vram,
            'component_delta_vram_mb': component_delta_vram,
            'component_delta_ram_mb': component_delta_ram,
            'summary_delta_ram_mb': summary_delta_ram,
            'summary_delta_vram_mb': summary_delta_vram,
            'is_parallel': True,
        }
    else:
        # Обычный бенчмарк
        component_name = data.get('component', 'core_clip')
        models_used = data.get('models_used', [])
        
        # Если models_used нет в JSON, пытаемся извлечь из NPZ файла
        if not models_used:
            try:
                import numpy as np
                if component_name == "core_clip":
                    npz_path = json_path.parent / "result_store" / "core_clip" / "embeddings.npz"
                elif component_name == "core_depth_midas":
                    npz_path = json_path.parent / "result_store" / "core_depth_midas" / "depth.npz"
                else:
                    npz_path = None
                
                if npz_path and npz_path.exists():
                    npz_data = np.load(npz_path, allow_pickle=True)
                    meta = npz_data.get('meta')
                    if meta is not None:
                        if hasattr(meta, 'item'):
                            meta = meta.item()
                        if isinstance(meta, dict):
                            models_used = meta.get('models_used', [])
            except Exception:
                pass
        
        triton_model_1 = models_used[0].get('model_name', 'N/A') if models_used else 'N/A'
        # Для core_depth_midas нет второй модели (text), для core_clip есть
        if component_name == "core_depth_midas":
            triton_model_2 = "N/A"  # MiDaS не имеет text модели
        else:
            triton_model_2 = models_used[1].get('model_name', 'N/A') if len(models_used) > 1 else 'N/A'
        
        # Triton Preprocess - определяем в зависимости от компонента
        if component_name == "core_depth_midas":
            # Для MiDaS preset определяется из model_name
            model_name_lower = triton_model_1.lower()
            if "256" in model_name_lower:
                triton_preprocess = "midas_256"
            elif "512" in model_name_lower:
                triton_preprocess = "midas_512"
            else:
                triton_preprocess = "midas_384"  # default
        else:
            # Для core_clip ищем в models_used
            triton_preprocess = 'preprocess_clip_image_224'  # Дефолтное значение
            for model in models_used:
                model_name = model.get('model_name', '').lower()
                if 'preprocess' in model_name or 'clip_image' in model_name:
                    triton_preprocess = model.get('model_name', triton_preprocess)
                    break
        
        triton_batch = data.get('batch_size', 1)
        
        # Ресурсы
        resources = data.get('resources', {})
        before_triton = resources.get('before_triton', {})
        after_triton = resources.get('after_triton', {})
        after_component = resources.get('after_component', {})
        peaks = resources.get('peaks', {})
        
        # Вычисляем дельты
        triton_delta_ram = after_triton.get('cpu_mem_used_mb', 0) - before_triton.get('cpu_mem_used_mb', 0)
        triton_delta_vram = after_triton.get('gpu_mem_used_mb', 0) - before_triton.get('gpu_mem_used_mb', 0)
        component_delta_vram = after_component.get('gpu_mem_used_mb', 0) - after_triton.get('gpu_mem_used_mb', 0)
        component_delta_ram = after_component.get('cpu_mem_used_mb', 0) - after_triton.get('cpu_mem_used_mb', 0)
        summary_delta_ram = after_component.get('cpu_mem_used_mb', 0) - before_triton.get('cpu_mem_used_mb', 0)
        summary_delta_vram = after_component.get('gpu_mem_used_mb', 0) - before_triton.get('gpu_mem_used_mb', 0)
        
        # Пиковый CPU timestamp
        peak_timestamps = resources.get('peak_timestamps', {})
        cpu_peak_elapsed_sec = peak_timestamps.get('cpu_util_peak_elapsed_sec', None)
        if cpu_peak_elapsed_sec:
            try:
                cpu_peak_elapsed_sec = float(cpu_peak_elapsed_sec)
            except (ValueError, TypeError):
                cpu_peak_elapsed_sec = None
        
        # Извлекаем timing информацию в зависимости от компонента
        component_timing = data.get('component_timing', {})
        if component_name == "core_depth_midas":
            # Для core_depth_midas нет image/text inference, только depth inference
            # Если component_timing пустой, используем общее время выполнения
            if component_timing:
                image_inference_sec = component_timing.get('depth_inference', component_timing.get('total', data.get('component_duration_sec', 0.0)))
            else:
                # Если timing не доступен, используем общее время выполнения компонента
                image_inference_sec = data.get('component_duration_sec', 0.0)
            text_inference_sec = 0.0  # Нет text inference для MiDaS
        else:
            # Для core_clip
            image_inference_sec = component_timing.get('image_inference', 0.0)
            text_inference_sec = component_timing.get('text_inference', 0.0)
        
        metrics = {
            'triton_model_1': triton_model_1,
            'triton_model_2': triton_model_2,
            'triton_preprocess': triton_preprocess,
            'triton_batch': triton_batch,
            'num_threads': 1,  # Одиночный запуск
            'frames_cnt': data.get('frames_count', 0),
            'runs': 4,  # Всегда 4 для агрегации
            'duration_sec': data.get('component_duration_sec', 0.0),
            'avg_duration_per_thread_sec': data.get('component_duration_sec', 0.0),
            'image_inference_sec': image_inference_sec,
            'text_inference_sec': text_inference_sec,
            'throughput_runs_per_sec': 0.0,  # Не применимо для одиночных запусков
            'peak_cpu_pct': peaks.get('cpu_util_peak_pct', 0.0),
            'peak_cpu_occurred_sec': cpu_peak_elapsed_sec,
            'peak_gpu_pct': peaks.get('gpu_util_peak_pct', 0.0),
            'triton_delta_ram_mb': triton_delta_ram,
            'triton_delta_vram_mb': triton_delta_vram,
            'component_delta_vram_mb': component_delta_vram,
            'component_delta_ram_mb': component_delta_ram,
            'summary_delta_ram_mb': summary_delta_ram,
            'summary_delta_vram_mb': summary_delta_vram,
            'is_parallel': False,
        }
    
    return metrics


def aggregate_results(results_dir: Path) -> Dict[str, Any]:
    """Агрегирует результаты из 4 попыток."""
    # Находим все директории с результатами
    result_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    
    if len(result_dirs) < 4:
        # Добавляем детальную информацию для отладки
        error_msg = f"Необходимо минимум 4 попытки, найдено: {len(result_dirs)}\n"
        error_msg += f"Директория результатов: {results_dir}\n"
        if result_dirs:
            error_msg += f"Найденные директории ({len(result_dirs)}):\n"
            for d in result_dirs:
                error_msg += f"  - {d.name}\n"
                # Проверяем содержимое каждой директории
                files = list(d.iterdir())
                error_msg += f"    Файлы в директории: {[f.name for f in files]}\n"
        else:
            error_msg += "Директории не найдены. Проверьте путь к результатам.\n"
            # Проверяем, что находится в results_dir
            all_items = list(results_dir.iterdir())
            if all_items:
                error_msg += f"Найденные элементы в {results_dir}:\n"
                for item in all_items:
                    error_msg += f"  - {item.name} ({'директория' if item.is_dir() else 'файл'})\n"
            else:
                error_msg += f"Директория {results_dir} пуста.\n"
        raise ValueError(error_msg)
    
    # Берем последние 4 попытки
    latest_4_dirs = result_dirs[-4:]
    
    # Извлекаем метрики из каждой попытки
    all_metrics = []
    missing_files = []
    for result_dir in latest_4_dirs:
        json_path = result_dir / 'results.json'
        if json_path.exists():
            try:
                metrics = extract_metrics_from_json(json_path)
                metrics['run_dir'] = result_dir.name
                all_metrics.append(metrics)
            except Exception as e:
                print(f"[warning] Ошибка при чтении {json_path}: {e}", file=sys.stderr)
                missing_files.append((result_dir.name, f"Ошибка чтения: {e}"))
        else:
            missing_files.append((result_dir.name, "Файл results.json не найден"))
            # Показываем, что есть в директории
            files = list(result_dir.iterdir())
            print(f"[warning] В директории {result_dir.name} нет results.json. Найденные файлы: {[f.name for f in files]}", file=sys.stderr)
    
    if len(all_metrics) != 4:
        error_msg = f"Не удалось найти results.json во всех 4 попытках. Найдено: {len(all_metrics)}\n"
        error_msg += f"Директория результатов: {results_dir}\n"
        error_msg += f"Проверенные директории ({len(latest_4_dirs)}):\n"
        for d in latest_4_dirs:
            error_msg += f"  - {d.name}\n"
        if missing_files:
            error_msg += "\nПроблемы:\n"
            for dir_name, reason in missing_files:
                error_msg += f"  - {dir_name}: {reason}\n"
        raise ValueError(error_msg)
    
    # Определяем числовые поля для агрегации
    numeric_fields = [
        'duration_sec',
        'avg_duration_per_thread_sec',
        'image_inference_sec',
        'text_inference_sec',
        'throughput_runs_per_sec',
        'peak_cpu_pct',
        'peak_cpu_occurred_sec',
        'peak_gpu_pct',
        'triton_delta_ram_mb',
        'triton_delta_vram_mb',
        'component_delta_vram_mb',
        'component_delta_ram_mb',
        'summary_delta_ram_mb',
        'summary_delta_vram_mb',
    ]
    
    # Агрегируем числовые поля
    aggregated = {
        'runs_count': 4,
        'run_directories': [m['run_dir'] for m in all_metrics],
    }
    
    # Берем общие поля из первого результата
    for field in ['triton_model_1', 'triton_model_2', 'triton_preprocess', 'triton_batch', 'frames_cnt', 'num_threads', 'is_parallel']:
        aggregated[field] = all_metrics[0].get(field)
    
    # Вычисляем средние и определяем выбросы
    for field in numeric_fields:
        values = [m[field] for m in all_metrics if m[field] is not None]
        
        if not values:
            aggregated[field] = {
                'mean': None,
                'values': [m[field] for m in all_metrics],
                'outliers': []
            }
            continue
        
        # Конвертируем в float для вычислений
        float_values = [float(v) for v in values if v is not None]
        
        if len(float_values) == 0:
            aggregated[field] = {
                'mean': None,
                'values': [m[field] for m in all_metrics],
                'outliers': []
            }
            continue
        
        mean_value = statistics.mean(float_values)
        outliers = detect_outliers(float_values)
        
        aggregated[field] = {
            'mean': round(mean_value, 3),
            'values': [round(v, 3) for v in float_values],
            'outliers': [round(v, 3) for v in outliers]
        }
    
    return aggregated


def extract_video_name(results_dir: Path) -> str:
    """Извлекает имя видео из результатов бенчмарка."""
    # Ищем в последнем JSON файле
    result_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    if result_dirs:
        latest_dir = result_dirs[-1]
        json_path = latest_dir / 'results.json'
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                video_path = data.get('video_path', '')
                if video_path:
                    return os.path.basename(video_path)
            except Exception:
                pass
    return 'N/A'


def format_for_table(aggregated: Dict[str, Any], video_name: str = 'N/A') -> Dict[str, Any]:
    """Форматирует агрегированные результаты для вставки в таблицу."""
    is_parallel = aggregated.get('is_parallel', False)
    num_threads = aggregated.get('num_threads', 1)
    
    formatted = {
        'Model': f"{aggregated.get('triton_model_1', 'N/A')} vunknown",
        'Video': video_name,
        'Triton model 1': aggregated.get('triton_model_1', 'N/A'),
        'Triton model 2': aggregated.get('triton_model_2', 'N/A'),
        'Triton Preprocess': aggregated.get('triton_preprocess', 'N/A'),
        'Triton Batch': aggregated.get('triton_batch', 'N/A'),
        'Threads': num_threads if is_parallel else 1,
        'Frames cnt': aggregated.get('frames_cnt', 'N/A'),
        'Runs': aggregated.get('runs_count', 4),
    }
    
    # Форматируем числовые поля
    numeric_fields_mapping = {
        'duration_sec': 'Duration (s)' if is_parallel else 'Duration (s)',
        'avg_duration_per_thread_sec': 'Avg Duration per Thread (s)',
        'image_inference_sec': 'Image Inf (s)',
        'text_inference_sec': 'Text Inf (s)',
        'throughput_runs_per_sec': 'Throughput (runs/s)',
        'peak_cpu_pct': 'Peak CPU %',
        'peak_cpu_occurred_sec': 'Peak CPU occurred (s)',
        'peak_gpu_pct': 'Peak GPU %',
        'triton_delta_ram_mb': 'Triton Delta RAM (MB)',
        'triton_delta_vram_mb': 'Triton Delta VRAM (MB)',
        'component_delta_vram_mb': 'Component Delta VRAM (MB)',
        'component_delta_ram_mb': 'Component Delta RAM (MB)',
        'summary_delta_ram_mb': 'Summary Delta RAM',
        'summary_delta_vram_mb': 'Summary Delta VRAM',
    }
    
    for field, display_name in numeric_fields_mapping.items():
        field_data = aggregated.get(field, {})
        if not isinstance(field_data, dict):
            # Если поле не было агрегировано (например, для параллельных бенчмарков некоторые поля могут отсутствовать)
            continue
        
        mean_val = field_data.get('mean')
        outliers = field_data.get('outliers', [])
        
        if mean_val is None:
            formatted[display_name] = None
        elif outliers:
            formatted[display_name] = {
                'mean': mean_val,
                'outliers': outliers,
                'display': f"{mean_val} ({', '.join(map(str, outliers))})"
            }
        else:
            formatted[display_name] = {
                'mean': mean_val,
                'outliers': [],
                'display': str(mean_val)
            }
    
    return formatted


def main():
    """Главная функция."""
    name = None
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
        if len(sys.argv) > 2:
            name = sys.argv[2]
    else:
        # По умолчанию используем out_component, но проверяем также out_component_parallel
        script_dir = Path(__file__).parent
        results_dir = script_dir / 'out_component'
        
        # Если out_component не существует, пробуем out_component_parallel
        if not results_dir.exists():
            parallel_dir = script_dir / 'out_component_parallel'
            if parallel_dir.exists():
                results_dir = parallel_dir
                print(f"[info] Используется директория параллельных бенчмарков: {results_dir}", file=sys.stderr)
    
    if not results_dir.exists():
        print(f"Ошибка: директория {results_dir} не существует", file=sys.stderr)
        print(f"[info] Попробуйте указать путь к директории с результатами:", file=sys.stderr)
        print(f"  python {sys.argv[0]} <path_to_results_dir>", file=sys.stderr)
        sys.exit(1)
    
    try:
        aggregated = aggregate_results(results_dir)
        video_name = extract_video_name(results_dir)
        formatted = format_for_table(aggregated, video_name)
        
        # Определяем тип бенчмарка
        is_parallel = aggregated.get('is_parallel', False)
        benchmark_type = "параллельный" if is_parallel else "одиночный"
        num_threads = aggregated.get('num_threads', 1)
        
        print(f"[info] Тип бенчмарка: {benchmark_type}", file=sys.stderr)
        if is_parallel:
            print(f"[info] Количество потоков: {num_threads}", file=sys.stderr)
        
        # Также выводим упрощенную версию для быстрой вставки
        simplified = {}
        for key, value in formatted.items():
            if isinstance(value, dict) and 'display' in value:
                simplified[key] = value['display']
            elif isinstance(value, dict):
                simplified[key] = value.get('mean', value)
            else:
                simplified[key] = value
        
        # Генерируем имя файла
        if name is None:
            # Используем имя директории результатов как имя файла
            name = results_dir.name
            if name in ['out_component', 'out_component_parallel']:
                # Если это стандартная директория, пытаемся определить имя из первой попытки
                result_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
                if result_dirs:
                    # Извлекаем имя из первой директории (обычно содержит информацию о компоненте)
                    first_dir = result_dirs[0]
                    json_path = first_dir / 'results.json'
                    if json_path.exists():
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            component = data.get('component', 'unknown')
                            name = f"{component}_{results_dir.name}"
                        except Exception:
                            name = f"benchmark_{results_dir.name}"
                else:
                    name = f"benchmark_{results_dir.name}"
        
        summary_file = Path(__file__).parent / 'summary' / f"res_{name}.json"
        summary_file.parent.mkdir(exist_ok=True)
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(simplified, f, ensure_ascii=False, indent=2)
        print(f"[info] Результаты сохранены в: {summary_file}", file=sys.stderr)
        
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
