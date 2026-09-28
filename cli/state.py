from enum import Enum
from pathlib import Path

from model import NNModel


class ModelType(str, Enum):
    ENCODEC = "encodec"
    RAVE = "rave"


AUDIO_EXTENSIONS = {".wav", ".aif", ".aiff", ".mp3"}


def create_model(
    model_type: ModelType,
    rave_path: Path | None = None,
    sample_rate: int = 48_000,
) -> NNModel:
    if model_type == ModelType.ENCODEC:
        if rave_path is not None:
            raise ValueError("--rave-path can only be used with --model rave.")
        from encodec_lib import EncodecNNModel, raw_encodec_model

        return EncodecNNModel(raw_encodec_model(sample_rate))

    if rave_path is None:
        raise ValueError("--model rave requires --rave-path.")
    if not rave_path.is_dir():
        raise FileNotFoundError(f"RAVE run directory not found: {rave_path}")

    try:
        from rave_lib import RAVEModel
    except ModuleNotFoundError as exc:
        if exc.name == "rave":
            raise ModuleNotFoundError(
                "RAVE dependencies are missing. Install nbform with the [rave] extra."
            ) from exc
        raise

    return RAVEModel(rave_path)


def list_audio_files(path: str | Path, allow_directory: bool = True) -> list[Path]:
    audio_path = Path(path)
    if audio_path.is_file():
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio file: {audio_path}")
        return [audio_path]

    if not allow_directory:
        raise FileNotFoundError(f"Input must be an audio file: {audio_path}")
    if not audio_path.is_dir():
        raise FileNotFoundError(f"Input file or directory not found: {audio_path}")

    files = sorted(
        (
            child
            for child in audio_path.iterdir()
            if child.is_file() and child.suffix.lower() in AUDIO_EXTENSIONS
        ),
        key=lambda child: child.name.lower(),
    )
    if not files:
        raise FileNotFoundError(f"No supported audio files found in '{audio_path}'.")
    return files