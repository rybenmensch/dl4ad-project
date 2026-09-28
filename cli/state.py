import os
import os.path
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Self

import torch
import torchaudio
from numpy._typing import NDArray
import sounddevice as sd

from library.encodec import EncodecNNModel
from library.model import (
    LayerInfo,
    NNModel,
    get_all_layers,
    get_shape_preserving_layers,
    get_weighted_layers,
)
from library.rave import RAVEModel

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
    GENERATE = "generate"
    ANALYZE = "analyze"
    EXPORT = "export"


@dataclass
class CommonArgs:
    command: Command
    rave_path: Path | None
    model_type: ModelType | None
    output: Path


@dataclass
class GenerateArgs(CommonArgs):
    input: Path


@dataclass
class AnalyzeArgs(CommonArgs):
    input: Path
    save_depth: int | None = None
    seed: int = 0
    seconds: float | None = None


@dataclass
class ExportArgs(CommonArgs):
    pass


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


Args = GenerateArgs | AnalyzeArgs | ExportArgs


class AppState:
    def __init__(self, args: Args) -> None:
        self.shape_preserving_layers: list[LayerInfo] = []
        self.weighted_layers: list[LayerInfo] = []
        self.all_layers: list[LayerInfo] = []
        self.files: list[File] = []
        self.model: NNModel

        self.args = args
        self.__load_model()
        self.__handle_output_dir()
        self.__load_files()
        self.update_layer_lists()

    def update_layer_lists(self) -> None:
        self.shape_preserving_layers = get_shape_preserving_layers(self.model)
        self.weighted_layers = get_weighted_layers(self.model)
        self.all_layers = get_all_layers(self.model)

    def __load_model(self) -> None:
        if self.args.model_type == ModelType.RAVE:
            assert self.args.rave_path != None
            self.model = RAVEModel(self.args.rave_path)
            self.backup_model = RAVEModel(self.args.rave_path)
        else:
            self.model = EncodecNNModel()
            self.backup_model = EncodecNNModel()

    def __handle_output_dir(self) -> None:
        self.output = self.args.output
        command_name = self.args.command.value
        self.output = self.output / command_name

        if not self.output.exists():
            self.output.mkdir(parents=True, exist_ok=True)

    def __load_files(self) -> None:
        if isinstance(self.args, ExportArgs):
            self.input_list = None
            return

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
