import os
import os.path
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Self

import sounddevice as sd
import torch
import torchaudio
from numpy._typing import NDArray

from library.encodec import EncodecNNModel, raw_encodec_model
from library.model import LayerInfo, NNModel

AUDIO_EXTENSIONS = {
    ".wav",
    ".aif",
    ".aiff",
    ".mp3",
}


class ModelType(str, Enum):
    RAVE = "RAVE"
    ENCODEC = "Encodec"


class Command(str, Enum):
    BEND = "bend"
    ANALYZE = "analyze"


@dataclass
class CommonArgs:
    command: Command
    rave_path: Path | None
    model_type: ModelType | None
    output: Path
    sample_rate: int


@dataclass
class BendArgs(CommonArgs):
    input: Path


@dataclass
class AnalyzeArgs(CommonArgs):
    input: Path
    save_depth: int | None = None
    seed: int = 0
    seconds: float | None = None


Args = BendArgs | AnalyzeArgs


# TODO: replace tuple[torch.Tensor, int] with AudioTensor
@dataclass(frozen=True)
class AudioTensor:
    audio: torch.Tensor
    sr: int

    @classmethod
    def from_tuple(cls, tup: tuple[torch.Tensor, int]) -> Self:
        return cls(audio=tup[0], sr=tup[1])

    def output_stream(self) -> sd.OutputStream:
        return sd.OutputStream(
            samplerate=self.sr, channels=self.num_channels(), dtype=self.dtype()
        )

    def num_channels(self) -> int:
        return self.audio.shape[0]

    def num_samps(self) -> int:
        return self.audio.shape[1]

    def dtype(self) -> Any:
        return self.audio.numpy().dtype

    def chunks(self, chunk_size: int = 512) -> list[NDArray]:
        transposed = self.audio.numpy().T.astype(self.dtype(), copy=False)
        return [
            transposed[i : i + chunk_size].copy(order="C")
            for i in range(0, len(transposed), chunk_size)
        ]

    def readable_length(self) -> str:
        num_seconds = round(self.num_samps() / self.sr)
        seconds = num_seconds % 60
        minutes = num_seconds // 60
        return f"{minutes:02}:{seconds:02}"


@dataclass(frozen=True)
class File:
    path: Path
    wav: tuple[torch.Tensor, int]


class AppState:
    def __init__(self, args: Args) -> None:
        self.files: list[File] = []
        self.model: NNModel
        self.rave_path: Path | None

        self.args = args
        self.__load_model()
        self.__handle_output_dir()
        self.__load_files()

    def __load_model(self) -> None:
        if self.args.model_type == ModelType.RAVE:
            assert self.args.rave_path != None
            try:
                from library.rave import RAVEModel
            except ModuleNotFoundError as exc:
                if exc.name == "rave":
                    raise ModuleNotFoundError(
                        "RAVE dependencies are missing. Install nbform with the [rave] extra."
                    ) from exc
                raise
            self.model = RAVEModel(self.args.rave_path)
            self.backup_model = RAVEModel(self.args.rave_path)
            self.rave_path = self.args.rave_path
        else:
            raw_model = raw_encodec_model(self.args.sample_rate)
            self.model = EncodecNNModel(raw_model)
            self.backup_model = EncodecNNModel(raw_encodec_model(self.args.sample_rate))

    def __handle_output_dir(self) -> None:
        self.output = self.args.output
        command_name = self.args.command.value
        self.output = self.output / command_name

        if not self.output.exists():
            self.output.mkdir(parents=True, exist_ok=True)

    def __load_files(self) -> None:
        path = self.args.input

        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        if path.is_file():
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                raise ValueError(f"Not a supported audio file: {path}")

            self.files.append(File(wav=torchaudio.load(path), path=path))

        if path.is_dir():
            self.paths = []
            for f in os.listdir(path):
                file_path = Path(os.path.join(path, f))
                if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS:
                    self.files.append(
                        File(wav=torchaudio.load(file_path), path=file_path)
                    )

            if len(self.files) == 0:
                raise FileNotFoundError(
                    f"Path does not contain any valid audio files: {path}"
                )


def create_model(
    model_type: ModelType,
    rave_path: Path | None = None,
    sample_rate: int = 48_000,
) -> NNModel:
    if model_type == ModelType.ENCODEC:
        if rave_path is not None:
            raise ValueError("--rave-path can only be used with --type RAVE.")
        return EncodecNNModel(raw_encodec_model(sample_rate))

    if rave_path is None:
        raise ValueError("--type RAVE requires --rave-path.")
    if not rave_path.is_dir():
        raise FileNotFoundError(f"RAVE run directory not found: {rave_path}")
    try:
        from library.rave import RAVEModel
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
