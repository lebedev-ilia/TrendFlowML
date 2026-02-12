#!/usr/bin/env python3
"""
Скрипт для исправления сохраненной модели source separation.

Удаляет полный объект модели из checkpoint, оставляя только state_dict.
Это исправляет проблему "Can't get attribute 'DemucsEnergyModel'" при загрузке.
"""

import argparse
import os
import sys
from pathlib import Path

import torch


def fix_checkpoint(checkpoint_path: str, output_path: str = None) -> None:
    """
    Исправляет checkpoint, удаляя полный объект модели и оставляя только state_dict.
    
    Args:
        checkpoint_path: Путь к существующему checkpoint файлу
        output_path: Путь для сохранения исправленного checkpoint (если None, перезаписывает исходный)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"[fix] Загрузка checkpoint: {checkpoint_path}")
    
    # Пробуем загрузить checkpoint
    # Если он содержит полный объект модели, который не может быть десериализован,
    # мы не сможем его загрузить обычным способом
    # В этом случае нужно использовать другой подход
    
    try:
        # Пробуем загрузить обычным способом
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        if isinstance(ckpt, dict):
            # Если checkpoint содержит полный объект модели, удаляем его
            if "model" in ckpt:
                print("[fix] Обнаружен полный объект модели, удаляем его...")
                # Сохраняем state_dict и meta, удаляем model
                new_ckpt = {
                    "state_dict": ckpt.get("state_dict"),
                    "meta": ckpt.get("meta", {}),
                }
                # Удаляем все ключи, кроме state_dict и meta
                for key in list(ckpt.keys()):
                    if key not in ("state_dict", "meta"):
                        if key != "model":
                            # Сохраняем другие ключи (если есть)
                            new_ckpt[key] = ckpt[key]
                
                ckpt = new_ckpt
                print("[fix] Полный объект модели удален из checkpoint")
            else:
                print("[fix] Checkpoint уже не содержит полный объект модели")
        else:
            # Если checkpoint - это не dict, возможно это просто state_dict
            print("[fix] Checkpoint не является словарем, возможно это state_dict")
            ckpt = {"state_dict": ckpt, "meta": {}}
        
        # Сохраняем исправленный checkpoint
        output = output_path or checkpoint_path
        print(f"[fix] Сохранение исправленного checkpoint: {output}")
        torch.save(ckpt, output)
        print(f"[fix] ✓ Checkpoint исправлен и сохранен: {output}")
        
    except Exception as e:
        error_str = str(e).lower()
        if "can't get attribute" in error_str or "cannot get attribute" in error_str:
            print(f"[fix] Ошибка десериализации класса: {e}")
            print("[fix] Попытка загрузить только state_dict через альтернативный метод...")
            
            # Пробуем использовать pickle напрямую для извлечения state_dict
            try:
                import pickle
                import io
                
                # Читаем файл как байты
                with open(checkpoint_path, "rb") as f:
                    data = f.read()
                
                # Пробуем использовать Unpickler с custom find_class для пропуска проблемных объектов
                class SafeUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        # Если это проблемный класс, возвращаем заглушку
                        if name == "DemucsEnergyModel" or "DemucsEnergyModel" in str(module):
                            # Возвращаем None или пропускаем
                            return None
                        return super().find_class(module, name)
                
                # Пробуем загрузить через SafeUnpickler
                try:
                    unpickler = SafeUnpickler(io.BytesIO(data))
                    ckpt = unpickler.load()
                    
                    if isinstance(ckpt, dict) and "state_dict" in ckpt:
                        # Успешно загрузили state_dict
                        new_ckpt = {
                            "state_dict": ckpt["state_dict"],
                            "meta": ckpt.get("meta", {}),
                        }
                        output = output_path or checkpoint_path
                        torch.save(new_ckpt, output)
                        print(f"[fix] ✓ Checkpoint исправлен через SafeUnpickler: {output}")
                        return
                except Exception as e2:
                    print(f"[fix] SafeUnpickler также не сработал: {e2}")
            
            except Exception as e2:
                print(f"[fix] Альтернативный метод не сработал: {e2}")
            
            print("[fix] ✗ Не удалось исправить checkpoint автоматически")
            print("[fix] Рекомендация: пересохраните модель через download_source_separation_models.py")
            raise RuntimeError(f"Failed to fix checkpoint: {e}") from e
        else:
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Исправление сохраненной модели source separation (удаление полного объекта модели)"
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Путь к checkpoint файлу для исправления",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Путь для сохранения исправленного checkpoint (по умолчанию: перезаписывает исходный)",
    )
    
    args = parser.parse_args()
    
    try:
        fix_checkpoint(args.checkpoint_path, args.output)
        print("\n[fix] ✓ Готово! Модель исправлена и готова к использованию.")
    except Exception as e:
        print(f"\n[fix] ✗ Ошибка: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

