"""
Тест pyannote speaker diarization: по умолчанию аудио из короткого example-видео.

Рекомендуемый интерпретатор (как вы просили):
  src/extractors/emotion_diarization_extractor/.emotion_diarization_venv/bin/python3 ex.py

Если импорт pyannote падает, доустановите в этот venv (один раз):
  .../.emotion_diarization_venv/bin/python3 -m pip install pyannote.audio torch-audiomentations

Токен Hugging Face нужен только если локальный бандл ещё не скачан (модель gated):
  export HF_TOKEN=...

Веса кладутся в DataProcessor/dp_models/bundled_models/audio/pyannote_speaker_diarization
(репозиторий pyannote/speaker-diarization-community-1 через huggingface_hub.snapshot_download),
далее Pipeline грузится только из этой папки, без использования HF-кэша для этой модели.

Загрузка аудио: soundfile (waveform в RAM), без torchcodec. Видео → WAV: ffmpeg (-ar 16000 -ac 1).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import soundfile as sf
import torch

# Не используем встроенный декодер pyannote (waveform через soundfile) — длинный traceback torchcodec только мешает.
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", module="pyannote.audio.utils.reproducibility")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# .../TrendFlowML/DataProcessor/AudioProcessor
_AUDIO_PROC = Path(__file__).resolve().parent
_REPO_ROOT = _AUDIO_PROC.parent.parent
_EMOTION_VENV_PY = (
    _AUDIO_PROC
    / "src"
    / "extractors"
    / "emotion_diarization_extractor"
    / ".emotion_diarization_venv"
    / "bin"
    / "python3"
)

DEFAULT_VIDEO = _REPO_ROOT / "example" / "example_videos" / "-Q6fnPIybEI.mp4"
MODEL_ID = "pyannote/speaker-diarization-community-1"
# Согласовано со spec_catalog audio/pyannote_speaker_diarization → local_artifacts path
DEFAULT_BUNDLE_DIR = (
    _REPO_ROOT
    / "DataProcessor"
    / "dp_models"
    / "bundled_models"
    / "audio"
    / "pyannote_speaker_diarization"
)

_AUDIO_EXT = {".wav", ".flac", ".ogg"}


def _bundle_looks_ready(path: Path) -> bool:
    """HF snapshot / pipeline.save для pyannote кладут config.yaml в корень бандла."""
    return path.is_dir() and (path / "config.yaml").is_file()


def ensure_pyannote_bundle(local_dir: Path, repo_id: str, token: str | None) -> None:
    """Если бандла нет — скачиваем репозиторий целиком в local_dir (копии файлов, не symlink в кэш)."""
    local_dir = local_dir.resolve()
    if _bundle_looks_ready(local_dir):
        return
    if not token:
        print(
            f"Нет готового бандла в {local_dir} (нужен config.yaml). "
            "Укажите HF_TOKEN или HUGGINGFACE_TOKEN для первой загрузки с Hugging Face.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as e:
        print(f"Нужен huggingface_hub: {e}", file=sys.stderr)
        sys.exit(1)
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Скачивание {repo_id!r} в {local_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=token,
        local_dir_use_symlinks=False,
    )
    if not _bundle_looks_ready(local_dir):
        print(
            f"После загрузки в {local_dir} не найден config.yaml — проверьте токен и доступ к модели.",
            file=sys.stderr,
        )
        sys.exit(1)


def load_pyannote_from_bundle(local_dir: Path):
    """Загрузка только из локальной директории (local_files_only — без докачки из Hub)."""
    local_dir = local_dir.resolve()
    try:
        return Pipeline.from_pretrained(str(local_dir), local_files_only=True)
    except TypeError:
        return Pipeline.from_pretrained(str(local_dir))


def load_waveform(audio_path: str):
    """(channels, time) float32 + sample_rate для pyannote."""
    w, sr = sf.read(audio_path, dtype="float32")
    if w.ndim == 1:
        w = w[None, :]
    else:
        w = w.T
    return torch.from_numpy(w.copy()), int(sr)


def media_path_to_wav(
    path: str,
    wav_out: str | None,
    sample_rate: int,
    mono: bool,
) -> tuple[str, bool]:
    """
    Возвращает (путь к wav, temp_файл_нужно_удалить).
    Для wav/flac/ogg — исходный путь; для остального — ffmpeg во временный или указанный файл.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Нет файла: {p}")

    if p.suffix.lower() in _AUDIO_EXT:
        return str(p), False

    if wav_out:
        out = Path(wav_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        target = str(out)
        _ffmpeg_to_wav(str(p), target, sample_rate, mono)
        return target, False

    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="diar_extract_")
    os.close(fd)
    _ffmpeg_to_wav(str(p), tmp, sample_rate, mono)
    return tmp, True


def _ffmpeg_to_wav(src_video: str, dst_wav: str, sample_rate: int, mono: bool) -> None:
    ac = "1" if mono else "2"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        src_video,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        ac,
        dst_wav,
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("Нужен ffmpeg в PATH для извлечения аудио из видео.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg завершился с ошибкой (код {e.returncode}).") from e


def main() -> None:
    parser = argparse.ArgumentParser(description="Тест pyannote diarization на wav или видео (ffmpeg).")
    parser.add_argument(
        "media",
        nargs="?",
        default=str(DEFAULT_VIDEO),
        help=f"Видео или wav (по умолчанию: {DEFAULT_VIDEO})",
    )
    parser.add_argument(
        "--wav-out",
        default=None,
        help="Куда сохранить извлечённый wav (иначе временный файл, удалится после прогона).",
    )
    parser.add_argument(
        "--ffmpeg-sr",
        type=int,
        default=16000,
        help="Частота дискретизации wav после ffmpeg (по умолчанию 16000).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_BUNDLE_DIR),
        help="Локальная папка с весами pyannote (по умолчанию dp_models/.../pyannote_speaker_diarization).",
    )
    args = parser.parse_args()

    bundle_dir = Path(args.model_dir).expanduser()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    ensure_pyannote_bundle(bundle_dir, MODEL_ID, token)

    wav_path: str | None = None
    remove_wav = False
    try:
        wav_path, remove_wav = media_path_to_wav(
            args.media, args.wav_out, args.ffmpeg_sr, mono=True
        )
        print(f"media={args.media!r}")
        print(f"wav for pyannote={wav_path!r} (temp={'yes' if remove_wav else 'no'})")

        pipeline = load_pyannote_from_bundle(bundle_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        print(f"device={device}, model_dir={bundle_dir!r}, repo={MODEL_ID}")

        waveform, sample_rate = load_waveform(wav_path)
        waveform = waveform.to(device)

        with ProgressHook() as hook:
            output = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)

        for turn, speaker in output.speaker_diarization:
            print(f"start={turn.start:.2f}s stop={turn.end:.2f}s speaker_{speaker}")
    finally:
        if remove_wav and wav_path and os.path.isfile(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass


if __name__ == "__main__":
    if not _EMOTION_VENV_PY.is_file():
        print(
            f"Подсказка: ожидаемый python этого проекта: {_EMOTION_VENV_PY}",
            file=sys.stderr,
        )
    main()
