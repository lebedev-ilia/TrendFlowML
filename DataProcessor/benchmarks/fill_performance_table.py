#!/usr/bin/env python3
"""
Скрипт для заполнения таблицы производительности в README.md
на основе JSON файлов из папки summary.
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def parse_filename(filename: str) -> Optional[Tuple[str, int, int]]:
    """
    Парсит имя файла формата: res_core_optical_flow_raft_{256|384|512}_{batch}_{frames}.json
    Возвращает: (model_version, batch, frames) или None
    """
    pattern = r'res_core_optical_flow_raft_(\d+)_(\d+)_(\d+)\.json'
    match = re.match(pattern, filename)
    if not match:
        return None
    
    size = int(match.group(1))
    batch = int(match.group(2))
    frames = int(match.group(3))
    
    # Преобразуем размер в формат x256, x384, x512
    model_version = f"x{size}"
    
    return (model_version, batch, frames)


def load_json_data(json_path: Path) -> Dict:
    """Загружает данные из JSON файла."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_metrics(data: Dict) -> Dict[str, str]:
    """Извлекает нужные метрики из JSON данных."""
    return {
        'duration': data.get('Duration (s)', ''),
        'peak_cpu': data.get('Peak CPU %', ''),
        'peak_gpu': data.get('Peak GPU %', ''),
        'component_vram': data.get('Component Delta VRAM (MB)', ''),
        'component_ram': data.get('Component Delta RAM (MB)', ''),
    }


def read_table_from_readme(readme_path: Path) -> List[str]:
    """Читает README.md и возвращает все строки."""
    with open(readme_path, 'r', encoding='utf-8') as f:
        return f.readlines()


def find_table_section(lines: List[str]) -> Tuple[int, int, bool]:
    """Находит начало и конец секции таблицы Performance.
    Возвращает: (table_start, end_idx, has_header)
    has_header = True если заголовок найден, False если нужно добавить.
    """
    section_start = None
    table_start = None
    end_idx = None
    has_header = False
    
    # Находим секцию Performance
    for i, line in enumerate(lines):
        if '## Perfomance' in line or '## Performance' in line:
            section_start = i
        elif section_start is not None and '| Model Version' in line:
            table_start = i
            has_header = True
        elif section_start is not None and table_start is None and line.strip().startswith('|') and ('x256' in line or 'x384' in line or 'x512' in line):
            # Нашли первую строку данных, но заголовка нет
            table_start = i
            has_header = False
        elif section_start is not None and line.startswith('##') and i > section_start:
            end_idx = i
            break
    
    if section_start is None:
        raise ValueError("Не найдена секция Performance в README.md")
    
    if table_start is None:
        # Ищем первую строку с данными после секции
        for i in range(section_start + 1, len(lines)):
            line = lines[i]
            if line.strip().startswith('|') and ('x256' in line or 'x384' in line or 'x512' in line):
                table_start = i
                has_header = False
                break
    
    if table_start is None:
        raise ValueError("Не найдена таблица в секции Performance")
    
    if end_idx is None:
        end_idx = len(lines)
    
    return table_start, end_idx, has_header


def parse_table_row(line: str) -> Optional[Dict[str, str]]:
    """Парсит строку таблицы и возвращает данные или None."""
    if not line.strip().startswith('|'):
        return None
    
    parts = [p.strip() for p in line.split('|')]
    if len(parts) < 4:
        return None
    
    # Пропускаем разделитель таблицы
    if parts[1] == '------' or parts[1] == '':
        return None
    
    return {
        'model_version': parts[1] if len(parts) > 1 else '',
        'batch': parts[2] if len(parts) > 2 else '',
        'frames': parts[3] if len(parts) > 3 else '',
        'duration': parts[4] if len(parts) > 4 else '',
        'peak_cpu': parts[5] if len(parts) > 5 else '',
        'peak_gpu': parts[6] if len(parts) > 6 else '',
        'triton_ram': parts[7] if len(parts) > 7 else '',
        'triton_vram': parts[8] if len(parts) > 8 else '',
        'component_vram': parts[9] if len(parts) > 9 else '',
        'component_ram': parts[10] if len(parts) > 10 else '',
        'summary_ram': parts[11] if len(parts) > 11 else '',
        'summary_vram': parts[12] if len(parts) > 12 else '',
    }


def format_table_row(row_data: Dict[str, str]) -> str:
    """Форматирует строку таблицы."""
    return (f"| {row_data['model_version']} | {row_data['batch']} | "
            f"{row_data['frames']} | {row_data['duration']} | "
            f"{row_data['peak_cpu']} | {row_data['peak_gpu']} | "
            f"{row_data['triton_ram']} | {row_data['triton_vram']} | "
            f"{row_data['component_vram']} | {row_data['component_ram']} | "
            f"{row_data['summary_ram']} | {row_data['summary_vram']} |")


def update_table(
    lines: List[str],
    table_start: int,
    table_end: int,
    has_header: bool,
    data_map: Dict[Tuple[str, int, int], Dict[str, str]]
) -> List[str]:
    """Обновляет таблицу данными из data_map."""
    if has_header:
        new_lines = lines[:table_start + 2]  # Заголовок и разделитель
        data_start = table_start + 2
    else:
        # Добавляем заголовок и разделитель
        new_lines = lines[:table_start]
        new_lines.append("| Model Version | Triton Batch | Frames cnt | Duration (s) | Peak CPU % | Peak GPU % | Triton Delta RAM (MB) | Triton Delta VRAM (MB) | Component Delta VRAM (MB) | Component Delta RAM (MB) | Summary Delta RAM | Summary Delta VRAM |\n")
        new_lines.append("|------|------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|\n")
        data_start = table_start
    
    # Парсим существующие строки таблицы
    existing_rows = []
    for i in range(data_start, table_end):
        line = lines[i]
        if line.strip().startswith('|'):
            row_data = parse_table_row(line)
            if row_data:
                existing_rows.append(row_data)
    
    # Создаем словарь существующих строк для быстрого поиска
    # Если встречаются дубликаты, объединяем данные (приоритет у заполненных значений)
    existing_map = {}
    for row in existing_rows:
        # Извлекаем ключ, проверяя что поля не пустые
        model_version = row['model_version'].strip()
        batch = row['batch'].strip()
        frames = row['frames'].strip()
        
        # Пропускаем строки с пустыми ключами
        if not model_version or not batch or not frames:
            continue
            
        # Преобразуем batch и frames в int для ключа
        try:
            batch_int = int(batch)
            frames_int = int(frames)
        except ValueError:
            continue
            
        key = (model_version, batch_int, frames_int)
        
        if key not in existing_map:
            existing_map[key] = row.copy()
        else:
            # Объединяем данные: берем заполненные значения из обеих строк
            existing_row = existing_map[key]
            for field in ['duration', 'peak_cpu', 'peak_gpu', 'triton_ram', 
                         'triton_vram', 'component_vram', 'component_ram', 
                         'summary_ram', 'summary_vram']:
                if not existing_row[field].strip() and row[field].strip():
                    existing_row[field] = row[field]
                elif existing_row[field].strip() and not row[field].strip():
                    pass  # Оставляем существующее значение
                elif existing_row[field].strip() != row[field].strip():
                    # Если оба заполнены и разные, берем из новой строки (приоритет новым данным)
                    existing_row[field] = row[field]
    
    # Обновляем существующие строки и добавляем новые
    # Используем только уникальные ключи (удаляем дубликаты)
    all_keys = set(existing_map.keys()) | set(data_map.keys())
    
    # Сортируем ключи: сначала по model_version, потом по batch, потом по frames
    sorted_keys = sorted(all_keys, key=lambda x: (
        x[0],  # model_version
        x[1],  # batch (уже int)
        x[2]   # frames (уже int)
    ))
    
    # Группируем по model_version для разделения пустыми строками
    current_model = None
    processed_keys = set()  # Отслеживаем обработанные ключи для предотвращения дубликатов
    
    for key in sorted_keys:
        # Пропускаем уже обработанные ключи (защита от дубликатов)
        if key in processed_keys:
            continue
        processed_keys.add(key)
        
        model_version, batch, frames = key
        
        # Добавляем пустую строку при смене model_version
        if current_model is not None and current_model != model_version:
            new_lines.append('\n')
        current_model = model_version
        
        # Получаем или создаем строку
        if key in existing_map:
            row = existing_map[key].copy()
            # Убеждаемся, что batch и frames - строки для форматирования
            row['batch'] = str(batch)
            row['frames'] = str(frames)
        else:
            # Создаем новую строку
            row = {
                'model_version': model_version,
                'batch': str(batch),
                'frames': str(frames),
                'duration': '',
                'peak_cpu': '',
                'peak_gpu': '',
                'triton_ram': '',
                'triton_vram': '',
                'component_vram': '',
                'component_ram': '',
                'summary_ram': '',
                'summary_vram': '',
            }
        
        # Обновляем данные из data_map (приоритет данным из JSON)
        if key in data_map:
            metrics = data_map[key]
            row['duration'] = metrics['duration']
            row['peak_cpu'] = metrics['peak_cpu']
            row['peak_gpu'] = metrics['peak_gpu']
            row['component_vram'] = metrics['component_vram']
            row['component_ram'] = metrics['component_ram']
        
        new_lines.append(format_table_row(row) + '\n')
    
    # Добавляем оставшиеся строки после таблицы
    new_lines.extend(lines[table_end:])
    
    return new_lines


def main():
    # Пути
    script_dir = Path(__file__).parent
    summary_dir = script_dir / 'summary'
    readme_path = Path(__file__).parent.parent / 'VisualProcessor' / 'core' / 'model_process' / 'core_optical_flow' / 'README.md'
    
    if not summary_dir.exists():
        print(f"Ошибка: папка {summary_dir} не найдена")
        return
    
    if not readme_path.exists():
        print(f"Ошибка: файл {readme_path} не найден")
        return
    
    # Загружаем все JSON файлы
    data_map = {}
    json_files = list(summary_dir.glob('res_core_optical_flow_raft_*.json'))
    
    print(f"Найдено {len(json_files)} JSON файлов")
    
    for json_file in json_files:
        parsed = parse_filename(json_file.name)
        if parsed is None:
            print(f"Пропущен файл с неверным форматом: {json_file.name}")
            continue
        
        model_version, batch, frames = parsed
        
        try:
            data = load_json_data(json_file)
            metrics = extract_metrics(data)
            data_map[(model_version, batch, frames)] = metrics
            print(f"Обработан: {json_file.name} -> {model_version}, batch={batch}, frames={frames}")
        except Exception as e:
            print(f"Ошибка при обработке {json_file.name}: {e}")
            continue
    
    if not data_map:
        print("Не найдено данных для заполнения таблицы")
        return
    
    # Читаем README
    lines = read_table_from_readme(readme_path)
    
    # Находим секцию таблицы
    try:
        table_start, table_end, has_header = find_table_section(lines)
    except ValueError as e:
        print(f"Ошибка: {e}")
        return
    
    # Обновляем таблицу
    new_lines = update_table(lines, table_start, table_end, has_header, data_map)
    
    # Сохраняем обновленный README
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"\nТаблица успешно обновлена в {readme_path}")
    print(f"Заполнено {len(data_map)} записей")


if __name__ == '__main__':
    main()

