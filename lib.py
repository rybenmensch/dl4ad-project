import os
from typing import Tuple
import torch
from pathlib import Path
import gin
import auraloss


def get_in_channels(model) -> int:
    if hasattr(model, "n_channels"):
        return model.n_channels
    return get_state_dict_in_channels(model.state_dict())


def get_state_dict_in_channels(state_dict: dict) -> int:
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


def process_audio(model, waveform: torch.Tensor) -> torch.Tensor:
    model_channels = get_in_channels(model)
    audio_channels = waveform.shape[0]

    if model_channels == 2:
        if audio_channels == 1:
            waveform = torch.cat([waveform, waveform], dim=0)
            audio_channels = 2

        if audio_channels == 2:
            input_tensor = waveform.unsqueeze(0)
            with torch.no_grad():
                torch.manual_seed(0)
                output_tensor = model(input_tensor)
                output_waveform = output_tensor.squeeze(0)
                return output_waveform
        else:
            processed_channels = []
            for i in range(0, audio_channels, 2):
                if i + 1 < audio_channels:
                    pair = waveform[i : i + 2, ...]
                else:
                    pair = torch.cat(
                        [waveform[i : i + 1, ...], waveform[i : i + 1, ...]], dim=0
                    )
                input_tensor = pair.unsqueeze(0)
                with torch.no_grad():

                    torch.manual_seed(0)
                    output_tensor = model(input_tensor)
                    output_waveform = output_tensor.squeeze(0)
                    if i + 1 >= audio_channels:
                        output_waveform = output_waveform[0:1, ...]
                    processed_channels.append(output_waveform)
            return torch.cat(processed_channels, dim=0)

    else:
        processed_channels = []
        for i in range(audio_channels):
            # (channels, timesteps) -> (1, timesteps)
            channel = waveform[i : i + 1, ...]

            # (1, timesteps) -> (batch, 1, timesteps)
            input_tensor = channel.unsqueeze(0)

            with torch.no_grad():
                # (batch, num_chans, timesteps)
                torch.manual_seed(0)
                output_tensor = model(input_tensor)
                # (batch, num_chans, timesteps) -> (num_chans, timesteps)
                output_waveform = output_tensor.squeeze(0)
                # (num_chans, timesteps) -> (1, timesteps)
                output_waveform = output_waveform[0:1, ...]
                processed_channels.append(output_waveform)

        # [(1, timesteps), ..., (1, timesteps)] -> (num_audio_channels, timesteps)
        output_tensor = torch.cat(processed_channels, dim=0)

        return output_tensor


class Model:
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

    def process_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        return process_audio(self.model, waveform)

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


def inout_paths(file_path: Path, in_path: Path, out_path: Path) -> Tuple[Path, Path]:
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
