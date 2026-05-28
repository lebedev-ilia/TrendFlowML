"""
Dependency resolution для AudioProcessor extractors и feature flags.

Обеспечивает:
- Автоматическое определение порядка выполнения extractors на основе зависимостей
- Валидацию feature flags внутри extractors
- Предупреждения и ошибки при отсутствии зависимостей
"""

import logging
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


# ============================================================================
# Dependency Graph: Extractors
# ============================================================================

# Граф зависимостей между extractors (опциональные зависимости через shared_features)
# Формат: {extractor_key: [list of dependency keys]}
# Примечание: speech_analysis имеет условные зависимости (только если включены feature flags),
# поэтому не добавляется в автоматическое добавление зависимостей
# Примечание: key, band_energy, spectral_entropy НЕ добавляются сюда, так как они могут использовать
# существующие результаты других extractors, но не требуют их запуска (опциональные зависимости)
EXTRACTOR_DEPENDENCIES: Dict[str, List[str]] = {
    # "key": ["chroma"],  # Убрано: key_extractor использует существующий chroma, но не требует его запуска
    # "band_energy": ["spectral"],  # Убрано: band_energy_extractor использует существующий spectral, но не требует его запуска
    # "spectral_entropy": ["spectral"],  # Убрано: spectral_entropy_extractor использует существующий spectral, но не требует его запуска
    # speech_analysis НЕ добавляется сюда, так как зависимости условные (зависят от feature flags)
    # Зависимости проверяются во время выполнения через feature flags (enable_asr_metrics, enable_diarization_metrics)
}

# Обязательные зависимости (required): компонент бесполезен без зависимости → ошибка если зависимость не включена
# Опциональные зависимости (optional): компонент может работать без зависимости, но с ухудшенным функционалом → предупреждение
# Формат: {extractor_key: [list of required dependency keys]}
# Примечание: speech_analysis НЕ добавляется сюда, так как зависимости условные (зависят от feature flags)
REQUIRED_EXTRACTOR_DEPENDENCIES: Dict[str, List[str]] = {
    # speech_analysis имеет условные зависимости (только если enable_asr_metrics=True или enable_diarization_metrics=True)
    # Зависимости проверяются во время выполнения через feature flags, а не через dependency resolver
}

# Опциональные зависимости (optional): компонент может работать без зависимости, но с ухудшенным функционалом → предупреждение
# Формат: {extractor_key: [list of optional dependency keys]}
OPTIONAL_EXTRACTOR_DEPENDENCIES: Dict[str, List[str]] = {
    "key": ["chroma"],  # key_extractor может использовать chroma из chroma_extractor
    "band_energy": ["spectral"],  # band_energy_extractor может использовать STFT из spectral_extractor
    "spectral_entropy": ["spectral"],  # spectral_entropy_extractor может использовать STFT/Mel из spectral_extractor
    "voice_quality": ["pitch"],  # voice_quality_extractor может использовать pitch из pitch_extractor
}

# Обратная совместимость: используем старую переменную для обратной совместимости
EXTRACTOR_REQUIRED_DEPENDENCIES = REQUIRED_EXTRACTOR_DEPENDENCIES

# Обратный граф: какие extractors зависят от данного
EXTRACTOR_DEPENDENTS: Dict[str, List[str]] = defaultdict(list)
for extractor, deps in EXTRACTOR_DEPENDENCIES.items():
    for dep in deps:
        EXTRACTOR_DEPENDENTS[dep].append(extractor)


# ============================================================================
# Dependency Graph: Feature Flags внутри extractors
# ============================================================================

# Зависимости между feature flags внутри extractors
# Формат: {extractor_key: {feature_flag: [list of required feature flags]}}
FEATURE_FLAG_DEPENDENCIES: Dict[str, Dict[str, List[str]]] = {
    "key": {
        # key_top_k требует enable_detailed_scores (для key_scores) или вычисляет локально
        # Но если enable_top_k=True, лучше иметь enable_detailed_scores для точности
        "enable_top_k": ["enable_detailed_scores"],  # Warning, not error
        # key_transitions требует enable_time_series
        "enable_key_changes": ["enable_time_series"],  # Error if missing
        # Метрики стабильности требуют enable_time_series
        "enable_stability_metrics": ["enable_time_series"],  # Error if missing
    },
    "asr": {
        # token_total зависит от token_counts (сумма)
        "enable_token_total": ["enable_token_counts"],  # Error if missing
        # token_density_per_sec зависит от token_total
        "enable_token_density": ["enable_token_total"],  # Error if missing
        # speech_rate_wpm зависит от token_total
        "enable_speech_rate": ["enable_token_total"],  # Error if missing
        # token_variance зависит от token_counts
        "enable_token_variance": ["enable_token_counts"],  # Error if missing
        # segments_with_speech зависит от token_sequences
        "enable_segments_with_speech": ["enable_token_sequences"],  # Error if missing
    },
    "mel": {
        # stats_vector зависит от enable_statistics
        "enable_stats_vector": ["enable_statistics"],  # Error if missing
    },
    "voice_quality": {
        # Некоторые метрики требуют комбинации флагов
        # Но это обрабатывается внутри extractor'а, не здесь
    },
}


# ============================================================================
# Dependency Resolution: Extractors
# ============================================================================

def resolve_extractor_dependencies(
    requested_extractors: List[str],
    available_extractors: Optional[List[str]] = None,
    auto_add_dependencies: bool = True,
    strict_mode: bool = False,
    enabled_extractors: Optional[List[str]] = None,  # Список включенных extractors из конфига (для проверки перед auto-add)
    speech_enable_asr_metrics: bool = False,  # Feature flags для speech_analysis (для условных зависимостей)
    speech_enable_diarization_metrics: bool = False,
    speech_enable_pitch_metrics: bool = False,
    speech_pitch_enabled: bool = False,  # pitch_enabled флаг для speech_analysis (нужен для pitch зависимости)
) -> Tuple[List[str], List[str], List[str]]:
    """
    Разрешает зависимости между extractors и возвращает упорядоченный список.

    Args:
        requested_extractors: Список запрошенных extractors (ключи)
        available_extractors: Список всех доступных extractors (если None, используется только EXTRACTOR_DEPENDENCIES)
        auto_add_dependencies: Автоматически добавлять недостающие зависимости
        strict_mode: Если True, выдавать ошибки вместо предупреждений при отсутствии зависимостей

    Returns:
        Tuple[ordered_extractors, warnings, errors]:
        - ordered_extractors: Упорядоченный список extractors (топологическая сортировка)
        - warnings: Список предупреждений
        - errors: Список ошибок
    """
    warnings: List[str] = []
    errors: List[str] = []

    # Нормализация: убираем дубликаты и приводим к lowercase
    requested = list(dict.fromkeys(k.lower().strip() for k in requested_extractors))
    
    # Определяем доступные extractors
    if available_extractors is not None:
        # Используем переданный список всех доступных extractors
        available = set(k.lower().strip() for k in available_extractors)
    else:
        # Fallback: используем только extractors с зависимостями (для обратной совместимости)
        available = set(EXTRACTOR_DEPENDENCIES.keys()) | set(EXTRACTOR_DEPENDENTS.keys())

    # Проверка существования extractors
    unknown = [k for k in requested if k not in available]
    if unknown:
        errors.append(f"Unknown extractors: {unknown}. Available: {sorted(available)}")
        return [], warnings, errors

    # Условные зависимости для speech_analysis на основе feature flags
    # Создаем динамический граф зависимостей с учетом feature flags
    dynamic_dependencies = dict(EXTRACTOR_DEPENDENCIES)
    if "speech_analysis" in requested:
        speech_deps = []
        if speech_enable_asr_metrics:
            speech_deps.append("asr")
        if speech_enable_diarization_metrics:
            speech_deps.append("speaker_diarization")
        # pitch зависимость требует ОБА условия: pitch_enabled=True И enable_pitch_metrics=True
        if speech_pitch_enabled and speech_enable_pitch_metrics:
            speech_deps.append("pitch")
        if speech_deps:
            dynamic_dependencies["speech_analysis"] = speech_deps
    
    # Добавление зависимостей (если auto_add_dependencies=True)
    resolved = set(requested)
    if auto_add_dependencies:
        enabled_set = set(enabled_extractors) if enabled_extractors else None
        for extractor in requested:
            deps = dynamic_dependencies.get(extractor, [])
            for dep in deps:
                if dep not in resolved:
                    # Проверяем, включен ли dependency в конфиге (если enabled_extractors предоставлен)
                    if enabled_set is not None and dep not in enabled_set:
                        # Dependency выключен в конфиге - не добавляем автоматически
                        if strict_mode:
                            errors.append(
                                f"Extractor '{extractor}' requires dependency '{dep}', but '{dep}' is disabled in config. "
                                f"Enable '{dep}' in config or disable '{extractor}'."
                            )
                        else:
                            warnings.append(
                                f"Extractor '{extractor}' requires dependency '{dep}', but '{dep}' is disabled in config. "
                                f"Enable '{dep}' in config or disable '{extractor}'."
                            )
                    else:
                        # Dependency включен в конфиге или enabled_extractors не предоставлен - добавляем
                        resolved.add(dep)
                        warnings.append(
                            f"Extractor '{extractor}' depends on '{dep}'. "
                            f"Automatically added '{dep}' to the execution list."
                        )

    # Проверка наличия зависимостей (если strict_mode=True или auto_add_dependencies=False)
    if not auto_add_dependencies or strict_mode:
        for extractor in requested:
            # Обязательные зависимости
            required_deps = REQUIRED_EXTRACTOR_DEPENDENCIES.get(extractor, [])
            missing_required = [d for d in required_deps if d not in resolved]
            if missing_required:
                msg = f"Extractor '{extractor}' requires dependencies: {missing_required}. "
                if strict_mode:
                    errors.append(msg + "Component is useless without these dependencies.")
                else:
                    errors.append(msg + "Component is useless without these dependencies. Enable them or remove '{extractor}' from --extractors.")
            
            # Опциональные зависимости
            optional_deps = OPTIONAL_EXTRACTOR_DEPENDENCIES.get(extractor, [])
            missing_optional = [d for d in optional_deps if d not in resolved]
            if missing_optional:
                msg = f"Extractor '{extractor}' has optional dependencies: {missing_optional}. "
                if strict_mode:
                    warnings.append(msg + "Missing dependencies will cause suboptimal performance.")
                else:
                    warnings.append(msg + "Consider adding them to --extractors for optimization.")

    # Добавляем опциональные зависимости в dynamic_dependencies для правильного порядка выполнения
    # Это гарантирует, что даже опциональные зависимости будут выполняться в правильном порядке
    # Важно: делаем это ДО топологической сортировки, чтобы зависимости учитывались
    for extractor in sorted(resolved):  # Сортируем для детерминированного порядка
        optional_deps = OPTIONAL_EXTRACTOR_DEPENDENCIES.get(extractor, [])
        if optional_deps:
            # Добавляем опциональные зависимости, если они присутствуют в resolved
            existing_optional = [dep for dep in optional_deps if dep in resolved]
            if existing_optional:
                if extractor not in dynamic_dependencies:
                    dynamic_dependencies[extractor] = []
                # Добавляем только те опциональные зависимости, которые уже есть в списке
                for dep in existing_optional:
                    if dep not in dynamic_dependencies[extractor]:
                        dynamic_dependencies[extractor].append(dep)
                        logger.debug(f"Dependency resolver: Added optional dependency '{dep}' -> '{extractor}' for ordering")

    # Топологическая сортировка (Kahn's algorithm)
    # Используем dynamic_dependencies для правильного порядка выполнения
    ordered = _topological_sort_extractors(list(resolved), dynamic_dependencies)

    if not ordered:
        errors.append("Circular dependency detected in extractor dependencies!")
        return [], warnings, errors

    return ordered, warnings, errors


def _topological_sort_extractors(
    extractors: List[str], dependencies: Dict[str, List[str]]
) -> List[str]:
    """
    Топологическая сортировка extractors на основе зависимостей.

    Returns:
        Упорядоченный список extractors (или пустой список при циклических зависимостях)
    """
    # Строим граф: {node: [list of nodes that depend on this node]}
    # И инвертируем зависимости: если A зависит от B, то B должен быть перед A
    graph: Dict[str, Set[str]] = defaultdict(set)
    in_degree: Dict[str, int] = defaultdict(int)

    # Инициализация in_degree для всех extractors
    for ext in extractors:
        in_degree[ext] = 0

    # Построение графа (инвертированные зависимости)
    for ext in extractors:
        deps = dependencies.get(ext, [])
        for dep in deps:
            if dep in extractors:  # Только если dependency тоже в списке
                graph[dep].add(ext)  # dep должен быть перед ext
                in_degree[ext] += 1

    # Kahn's algorithm
    queue = deque([ext for ext in extractors if in_degree[ext] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Проверка на циклы
    if len(result) != len(extractors):
        return []

    return result


# ============================================================================
# Dependency Resolution: Feature Flags
# ============================================================================

def validate_feature_flags(
    extractor_key: str,
    enabled_flags: Set[str],
    strict_mode: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Валидирует feature flags для extractor'а на основе зависимостей.

    Args:
        extractor_key: Ключ extractor'а (например, "key", "asr")
        enabled_flags: Множество включенных feature flags
        strict_mode: Если True, выдавать ошибки вместо предупреждений

    Returns:
        Tuple[warnings, errors]: Списки предупреждений и ошибок
    """
    warnings: List[str] = []
    errors: List[str] = []

    deps = FEATURE_FLAG_DEPENDENCIES.get(extractor_key, {})
    if not deps:
        return warnings, errors

    for flag, required_flags in deps.items():
        if flag not in enabled_flags:
            continue  # Флаг не включен, пропускаем

        missing = [rf for rf in required_flags if rf not in enabled_flags]
        if missing:
            msg = (
                f"Extractor '{extractor_key}': feature flag '{flag}' requires: {missing}. "
                f"Enable them for proper functionality."
            )
            if strict_mode:
                errors.append(msg)
            else:
                warnings.append(msg)

    return warnings, errors


def get_feature_flag_dependencies(extractor_key: str) -> Dict[str, List[str]]:
    """
    Возвращает зависимости feature flags для extractor'а.

    Args:
        extractor_key: Ключ extractor'а

    Returns:
        Словарь {feature_flag: [list of required flags]}
    """
    return FEATURE_FLAG_DEPENDENCIES.get(extractor_key, {})


# ============================================================================
# Utility Functions
# ============================================================================

def get_extractor_dependencies(extractor_key: str) -> List[str]:
    """
    Возвращает список зависимостей extractor'а.

    Args:
        extractor_key: Ключ extractor'а

    Returns:
        Список ключей зависимых extractors
    """
    return EXTRACTOR_DEPENDENCIES.get(extractor_key, [])


def get_extractor_dependents(extractor_key: str) -> List[str]:
    """
    Возвращает список extractors, которые зависят от данного.

    Args:
        extractor_key: Ключ extractor'а

    Returns:
        Список ключей зависимых extractors
    """
    return EXTRACTOR_DEPENDENTS.get(extractor_key, [])


def print_dependency_graph():
    """Выводит граф зависимостей для отладки."""
    print("=== Extractor Dependencies ===")
    for ext, deps in sorted(EXTRACTOR_DEPENDENCIES.items()):
        if deps:
            print(f"  {ext} → {', '.join(deps)}")

    print("\n=== Feature Flag Dependencies ===")
    for ext, flags in sorted(FEATURE_FLAG_DEPENDENCIES.items()):
        if flags:
            print(f"  {ext}:")
            for flag, reqs in sorted(flags.items()):
                print(f"    {flag} → {', '.join(reqs)}")


if __name__ == "__main__":
    # Тестирование
    print_dependency_graph()
    print("\n=== Test: resolve_extractor_dependencies ===")
    ordered, warnings, errors = resolve_extractor_dependencies(
        ["key", "band_energy", "spectral_entropy"], auto_add_dependencies=True
    )
    print(f"Ordered: {ordered}")
    print(f"Warnings: {warnings}")
    print(f"Errors: {errors}")

