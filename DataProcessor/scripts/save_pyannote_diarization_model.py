#!/usr/bin/env python3
"""
Скрипт для сохранения pyannote.audio speaker diarization модели локально.

Использование:
    python scripts/save_pyannote_diarization_model.py \
        --huggingface-token "hf_xxx" \
        --output-dir "dp_models/bundled_models/audio/pyannote_speaker_diarization"
"""

import argparse
import os
import sys
from pathlib import Path
import json

import torch
import omegaconf

# Разрешаем omegaconf для torch.load(weights_only=True)
torch.serialization.add_safe_globals([
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
])

def main():
    parser = argparse.ArgumentParser(description="Сохранить pyannote.audio speaker diarization модель локально")
    parser.add_argument(
        "--huggingface-token",
        type=str,
        required=True,
        help="HuggingFace token для доступа к pyannote.audio моделям (или установите HUGGINGFACE_TOKEN env var)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dp_models/bundled_models/audio/pyannote_speaker_diarization",
        help="Директория для сохранения модели (относительно корня репозитория)"
    )
    
    args = parser.parse_args()
    
    # Определяем корень репозитория
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / args.output_dir
    
    # Создаем директорию
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Используем токен из аргумента или env var
    hf_token = args.huggingface_token or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("Ошибка: huggingface-token не указан. Укажите --huggingface-token или установите HUGGINGFACE_TOKEN env var")
        sys.exit(1)
    
    print(f"Сохранение pyannote.audio speaker diarization модели в '{output_dir}'...")
    print(f"⚠️  Внимание: модель большая, загрузка может занять время...")
    
    try:
        from pyannote.audio import Pipeline  # type: ignore
        import torch  # type: ignore
    except ImportError as e:
        print(f"Ошибка: требуемые библиотеки не установлены. Установите: pip install pyannote.audio torch")
        print(f"Детали: {e}")
        sys.exit(1)
    
    try:
        # Устанавливаем токен через huggingface_hub (если доступен)
        try:
            import huggingface_hub  # type: ignore
            huggingface_hub.login(token=hf_token, add_to_git_credential=False)
            print("✓ HuggingFace token установлен через huggingface_hub")
        except ImportError:
            # Если huggingface_hub не установлен, используем переменную окружения
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGINGFACE_TOKEN"] = hf_token
            print("✓ HuggingFace token установлен через переменные окружения")
        except Exception as e:
            print(f"⚠️  Предупреждение: не удалось установить токен через huggingface_hub: {e}")
            # Продолжаем с переменными окружения
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGINGFACE_TOKEN"] = hf_token
        
        # Загружаем pipeline
        print(f"Загрузка модели 'pyannote/speaker-diarization'...")
        print(f"Используется HuggingFace token: {'установлен' if hf_token else 'НЕ УСТАНОВЛЕН'}")
        
        # Пробуем разные варианты параметров для совместимости с разными версиями
        pipeline = None
        load_error = None
        
        # Вариант 1: token (новый API)
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                token=hf_token
            )
            if pipeline is not None:
                print("✓ Модель загружена с параметром 'token'")
        except Exception as e1:
            load_error = e1
            # Вариант 2: use_auth_token (старый API)
            try:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=hf_token
                )
                if pipeline is not None:
                    print("✓ Модель загружена с параметром 'use_auth_token'")
            except Exception as e2:
                load_error = e2
                # Вариант 3: без явного токена (используется из окружения)
                try:
                    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
                    if pipeline is not None:
                        print("✓ Модель загружена через переменные окружения")
                except Exception as e3:
                    load_error = e3
        
        if pipeline is None:
            error_msg = str(load_error) if load_error else "Неизвестная ошибка"
            print(f"\n❌ Не удалось загрузить модель.")
            print(f"   Ошибка: {error_msg}")
            
            if "gated" in error_msg.lower() or "private" in error_msg.lower() or "accept" in error_msg.lower():
                print("\n   ⚠️  Модель требует принятия условий использования.")
                print("   1. Посетите https://hf.co/pyannote/speaker-diarization")
                print("   2. Примите условия использования (нажмите 'Agree and access repository')")
                print("   3. Затем повторите попытку.")
            elif "token" in error_msg.lower() or "auth" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
                print("\n   ⚠️  Проблема с аутентификацией.")
                print("   Убедитесь, что:")
                print("   1. HuggingFace token корректен и активен")
                print("   2. Token имеет права на доступ к модели")
                print("   3. Посетите https://hf.co/settings/tokens для создания/проверки token")
            else:
                print("\n   Проверьте:")
                print("   1. Интернет-соединение")
                print("   2. Корректность HuggingFace token")
                print("   3. Принятие условий использования модели")
            
            raise RuntimeError(f"Pipeline не загружен: {error_msg}")
        
        # Сохраняем pipeline в указанную директорию
        print(f"Сохранение модели в '{output_dir}'...")
        pipeline.to(torch.device("cpu"))  # Сохраняем на CPU для совместимости
        pipeline.save(output_dir)
        
        # Сохраняем метаданные
        metadata = {
            "model_name": "pyannote/speaker-diarization",
            "model_type": "speaker_diarization",
            "framework": "pyannote.audio",
            "saved_at": str(Path(output_dir).resolve()),
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Модель успешно сохранена в: {output_dir}")
        print(f"\nМодель готова к использованию через ModelManager!")
        print(f"\nSpec файл должен быть создан:")
        print(f"  DataProcessor/dp_models/spec_catalog/audio/pyannote_speaker_diarization.yaml")
        print(f"\nПуть в spec: \"audio/pyannote_speaker_diarization\"")
        
    except Exception as e:
        print(f"Ошибка при сохранении модели: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

