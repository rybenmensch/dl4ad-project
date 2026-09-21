import os
from pathlib import Path
from typing import Any

import auraloss
import gin
import torch
import torchaudio

# TODO: clean up the mess


def print_all_attrs(obj: Any) -> None:
    for k in obj.__dict__:
        print(k)


def hasattr_from_attr_string(obj: Any, string: str) -> Any:
    attrs: list[str] = string.split(".")
    for attr in attrs:
        try:
            obj = getattr(obj, attr)
        except Exception:
            return False
    return True


def getattr_from_attr_string(obj: Any, string: str) -> Any:
    attrs: list[str] = string.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def setattr_from_attr_string(obj: object, string: str, value: Any) -> None:
    attrs: list[str] = string.split(".")
    if len(attrs) == 0:
        print(f"Invalid attr string f{string}")
        exit()

    last = attrs.pop()
    for attr in attrs:
        obj = getattr(obj, attr)
    setattr(obj, last, value)


def convert_audio(wav: tuple[torch.Tensor, int], target_sr: int, target_chans: int):
    audio, sr = wav
    assert audio.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_chans == 1:
        audio = audio.mean(0, keepdim=True)
    elif target_chans == 2:
        *shape, _, length = audio.shape
        audio = audio.expand(*shape, target_chans, length)
    elif audio.shape[0] == 1:
        audio = audio.expand(target_chans, -1)
    audio = torchaudio.transforms.Resample(sr, target_sr)(audio)
    return audio


def get_in_channels(model) -> int:
    if hasattr(model, "n_channels"):
        return model.n_channels
    return get_in_channels_from_state_dict(model.state_dict())


def get_in_channels_from_state_dict(state_dict: dict) -> int:
    """
    Find the input channels of the encoder's first conv layer from state_dict
    to detect n_channels. We need this because at this point, we do not have a
    RAVE model yet, only a state_dict!
    """
    in_channels = None

    # first conv layer of state_dict might have a different 'path' depending on
    # model version
    for key in [
        "encoder.encoder.net.0.weight_v",
        "encoder.net.0.weight_v",
        "encoder.encoder.net.0.weight",
    ]:
        if key in state_dict:
            in_channels = state_dict[key].shape[1]
            break

    # actual amount of input channels is input_channels // number of pqmf bands
    if in_channels is not None:
        try:
            n_band = gin.query_parameter("%N_BAND")
        except Exception:
            # assume 16 bands as in the paper if the parameter isn't found
            n_band = 16
        return in_channels // n_band
    else:
        return 1


class JITModel:
    # useful for JIT-compiled .ts models
    def __init__(self, path: Path | str) -> None:
        check_path(path)
        self.model = torch.jit.load(path)
        self.model.eval()

        self.state_dict = self.model.state_dict()

        if "sampling_rate" in self.state_dict:
            self.sr: int = self.state_dict["sampling_rate"].item()
        else:
            self.sr: int = 44100

        self.input_channels: int = self.state_dict["encode_params"][0].item()

    def get_model_keys(self):
        return self.state_dict.keys()

    def print_model_keys(self) -> None:
        for k in self.get_model_keys():
            print(k)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # TODO(low priority): go digging in git history for that old
        # process_audio function or re-implement anew with `convert_audio`
        # return process_audio(self.model, waveform)
        raise NotImplementedError

    def get_state_dict(self):
        return self.model.state_dict()

    def set_state_dict(self, state_dict) -> None:
        self.model.load_state_dict(state_dict)


def check_path(p: str | Path) -> Path:
    """Check whether a path exists and returns it if it does."""
    path: Path = Path(p)
    if not path.exists():
        raise Exception(f"Path {path} doesn't exist!")
    return path


def inout_paths(file_path: Path, in_path: Path, out_path: Path) -> tuple[Path, Path]:
    dir_path, filename_ext = os.path.split(file_path)
    filename, ext = os.path.splitext(filename_ext)

    # use no prefix for audio_source_path
    # use './' for current folder (unsafe anyway)
    if dir_path == "":
        dir_path = in_path

    input_path: Path = check_path(dir_path) / filename_ext
    input_path = check_path(input_path)

    output_name = f"{filename}_reconstructed{ext}"
    output_path = check_path(out_path) / output_name

    return input_path, output_path


# LOSS FUNCTIONS


def mean_absolute_error(x1: torch.Tensor, x2: torch.Tensor) -> float:
    return torch.mean(torch.abs(x1 - x2)).item()


def mrstft(x1: torch.Tensor, x2: torch.Tensor) -> float:
    m = auraloss.freq.MultiResolutionSTFTLoss()
    return m(x1.unsqueeze(0), x2.unsqueeze(0)).item()
