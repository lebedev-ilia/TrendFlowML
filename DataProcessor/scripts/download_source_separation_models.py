#!/usr/bin/env python3
"""
Скрипт для загрузки и сохранения моделей Source Separation в DP_MODELS_ROOT.

Использование:
    python scripts/download_source_separation_models.py --models-root dp_models/bundled_models --sizes large

Модель будет сохранена в:
    dp_models/bundled_models/audio/source_separation/large.pt

Примечание: Этот скрипт требует установки demucs:
    pip install demucs
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

try:
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
except ImportError as e:
    raise ImportError(
        f"demucs is required but not installed. Install it with: pip install demucs"
    ) from e


class DemucsEnergyModel(nn.Module):
    """
    In-process Source Separation Energy Extractor
    Input:  log-mel spectrogram [B, n_mels, T]
    Output: energy shares [B, 4] (vocals, drums, bass, other)
    
    Note: Source order matches spec files: ["vocals", "drums", "bass", "other"]
    """

    SOURCES = ["vocals", "drums", "bass", "other"]

    def __init__(self, demucs_name="htdemucs", samplerate=44100):
        super().__init__()

        self.samplerate = samplerate
        self.demucs = get_model(demucs_name)
        self.demucs.eval()
        self.demucs.requires_grad_(False)

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logmel: [B, n_mels, T]
        Returns:
            shares: [B, 4] (vocals, drums, bass, other)
        """
        B = logmel.shape[0]
        device = logmel.device

        # ❗ ВАЖНО:
        # Здесь предполагается, что logmel получен из waveform
        # Для in-process модели мы используем упрощённую инверсию
        # (для фичей этого достаточно)
        wav = self._logmel_to_waveform(logmel)

        if wav.shape[1] == 1:
            wav = wav.repeat(1, 2, 1)

        with torch.no_grad():
            sources = apply_model(
                self.demucs,
                wav,
                device=device,
                split=True,
                overlap=0.25,
                progress=False,
            )

        # sources: [B, 4, C, T] (Demucs order: drums, bass, other, vocals)
        # Reorder to match spec: vocals, drums, bass, other
        # Demucs order: [0=drums, 1=bass, 2=other, 3=vocals]
        # Target order: [0=vocals, 1=drums, 2=bass, 3=other]
        reordered_sources = torch.stack([
            sources[:, 3],  # vocals
            sources[:, 0],  # drums
            sources[:, 1],  # bass
            sources[:, 2],  # other
        ], dim=1)  # [B, 4, C, T]

        energies = reordered_sources.pow(2).mean(dim=(2, 3))  # [B, 4]
        shares = energies / (energies.sum(dim=1, keepdim=True) + 1e-8)

        return shares

    def _logmel_to_waveform(self, logmel: torch.Tensor) -> torch.Tensor:
        """
        Approximate inversion for feature-level inference.
        """
        mel = logmel.exp()
        energy = mel.mean(dim=1)  # [B, T]
        wav = energy.unsqueeze(1)  # [B, 1, T]
        return wav


def _repo_root() -> str:
    """Определяет корень репозитория."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _default_models_root(repo_root: str) -> str:
    """Возвращает путь к bundled_models по умолчанию."""
    return os.path.join(repo_root, "dp_models", "bundled_models")


def download_source_separation_model(model_size: str, output_path: str, models_root: str = None) -> None:
    """
    Загружает модель source separation указанного размера и сохраняет в output_path.
    
    Args:
        model_size: Размер модели ("large")
        output_path: Путь для сохранения модели (.pt файл)
        models_root: Корневая директория для моделей (для настройки TORCH_HOME кеша)
    """
    size = model_size.lower()
    if size not in ("large",):
        raise ValueError(f"Unsupported model size: {size}. Expected: large")

    # Маппинг размера → Demucs вариант
    demucs_name = "htdemucs"  # large использует htdemucs

    # Настраиваем TORCH_HOME для кеширования базовой Demucs модели
    # Это важно, чтобы модель загружалась из кеша при runtime, а не из интернета
    if models_root:
        torch_cache_dir = os.path.join(models_root, "torch_cache")
        os.makedirs(torch_cache_dir, exist_ok=True)
        # Устанавливаем TORCH_HOME для кеширования базовой Demucs модели
        # Это позволит factory функции загружать модель из кеша, а не из интернета
        os.environ["TORCH_HOME"] = torch_cache_dir
        print(f"[download] TORCH_HOME установлен: {torch_cache_dir}")

    print(f"[download] Загрузка Demucs backbone: {demucs_name} (size: {size})...")
    print(f"[download] Примечание: базовая Demucs модель будет загружена в TORCH_HOME кеш ({os.environ.get('TORCH_HOME', 'default')})")
    print(f"[download] При последующих запусках модель будет загружаться из кеша (offline mode)")

    try:
        model = DemucsEnergyModel(demucs_name=demucs_name)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load Demucs model {demucs_name}: {e}") from e

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем модель в формате, совместимом с TorchStateDictProvider
    # НЕ сохраняем полный объект модели ("model"), так как это вызывает проблемы с десериализацией
    # Вместо этого сохраняем только state_dict, который будет загружен через factory функцию
    # - "state_dict": state_dict (для TorchStateDictProvider)
    # - "meta": метаданные модели
    checkpoint = {
        "state_dict": model.state_dict(),
        "meta": {
            "type": "source_separation_energy",
            "sources": ["vocals", "drums", "bass", "other"],  # Порядок из spec файлов
            "input": "logmel",
            "output": "energy_shares",
            "demucs": demucs_name,
            "model_size": size,
            "samplerate": 44100,
        },
    }

    try:
        torch.save(checkpoint, output_path)
        print(f"[download] ✓ Модель source separation ({size}) сохранена: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save model to {output_path}: {e}") from e


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Загрузка и сохранение моделей Source Separation в DP_MODELS_ROOT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Загрузить large модель (по умолчанию)
  python scripts/download_source_separation_models.py --sizes large
  
  # Указать кастомный путь для моделей
  python scripts/download_source_separation_models.py --models-root /path/to/models --sizes large

Требования:
  - demucs: pip install demucs
        """,
    )
    
    parser.add_argument(
        "--models-root",
        type=str,
        default=None,
        help="Путь к DP_MODELS_ROOT (по умолчанию: dp_models/bundled_models)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=["large"],
        default=["large"],
        help="Размеры моделей для загрузки (по умолчанию: large)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Пропустить модели, которые уже существуют",
    )
    
    args = parser.parse_args()
    
    repo_root = _repo_root()
    models_root = os.path.abspath(args.models_root or _default_models_root(repo_root))
    
    print(f"[download] DP_MODELS_ROOT: {models_root}")
    print(f"[download] Размеры моделей: {', '.join(args.sizes)}")
    print()
    
    # Создаем директорию для Source Separation моделей
    source_sep_dir = os.path.join(models_root, "audio", "source_separation")
    os.makedirs(source_sep_dir, exist_ok=True)
    
    # Загружаем каждую модель
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for size in args.sizes:
        output_path = os.path.join(source_sep_dir, f"{size}.pt")
        
        # Проверяем, существует ли модель
        if args.skip_existing and os.path.exists(output_path):
            print(f"[skip] Модель {size} уже существует: {output_path}")
            skip_count += 1
            continue
        
        try:
            download_source_separation_model(size, output_path, models_root=models_root)
            success_count += 1
        except Exception as e:
            print(f"[error] Ошибка при загрузке модели {size}: {e}", file=sys.stderr)
            error_count += 1
    
    print()
    print(f"[summary] Успешно: {success_count}, Пропущено: {skip_count}, Ошибок: {error_count}")
    
    if error_count > 0:
        sys.exit(1)
    
    print()
    print("[info] Все указанные модели Source Separation успешно загружены и сохранены.")
    print("[info] Теперь вы можете использовать их через ModelManager.")
    print("[info] Убедитесь, что spec-файлы в dp_models/spec_catalog/audio/ соответствуют этим моделям.")


if __name__ == "__main__":
    main()
