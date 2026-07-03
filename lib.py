import os
from typing import Tuple
import torch
from pathlib import Path

def check_path(p: str|Path) -> Path:
    path: Path = Path(p)
    if not path.exists():
        raise Exception(f"Path {path} doesn't exist!")
    return path

def process_audio(model, waveform: torch.Tensor) -> torch.Tensor:
    num_audio_channels = waveform.shape[0]

    processed_channels = []
    for i in range(num_audio_channels):
        # (channels, timesteps) -> (1, timesteps)
        channel = waveform[i:i+1, ...] 
        
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
    def __init__(self, path: Path|str) -> None:
        check_path(path)
        self.model = torch.jit.load(path);
        self.model.eval()

        self.state_dict = self.model.state_dict()
        # print(self.state_dict['decode_params'])
        # print(self.state_dict['forward_params'])

        if 'sampling_rate' in self.state_dict:
            self.sr: int = self.state_dict['sampling_rate'].item()
        else:
            self.sr: int = 44100

        self.input_channels: int = self.state_dict['encode_params'][0].item()

    def get_model_keys(self):
        return self.state_dict.keys()

    def print_model_keys(self) -> None:
        for k in self.get_model_keys():
            print(k)

    def process_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        return process_audio(self.model, waveform)

        # num_audio_channels = waveform.shape[0]
        #
        # processed_channels = []
        # for i in range(num_audio_channels):
        #     # (channels, timesteps) -> (1, timesteps)
        #     channel = waveform[i:i+1, ...] 
        #
        #     # (1, timesteps) -> (batch, 1, timesteps)
        #     input_tensor = channel.unsqueeze(0)
        #
        #     with torch.no_grad():
        #         # (batch, num_chans, timesteps)
        #         torch.manual_seed(0)
        #         output_tensor = self.model(input_tensor)
        #         # (batch, num_chans, timesteps) -> (num_chans, timesteps)
        #         output_waveform = output_tensor.squeeze(0)
        #         # (num_chans, timesteps) -> (1, timesteps)
        #         output_waveform = output_waveform[0:1, ...]
        #         processed_channels.append(output_waveform)
        #
        # # [(1, timesteps), ..., (1, timesteps)] -> (num_audio_channels, timesteps)
        # output_tensor = torch.cat(processed_channels, dim=0)
        #
        # return output_tensor

    def get_state_dict(self):
        return self.model.state_dict()

    def set_state_dict(self, state_dict) -> None:
        self.model.load_state_dict(state_dict)

def inout_paths(file_path: Path, in_path: Path, out_path: Path) -> Tuple[Path, Path]:
    dir_path, filename_ext = os.path.split(file_path)
    filename, ext = os.path.splitext(filename_ext)

    # use no prefix for audio_source_path
    # use './' for current folder (unsafe anyway)
    if dir_path == '':
        dir_path = in_path

    input_path: Path = check_path(dir_path) / filename_ext
    input_path = check_path(input_path)

    output_name = f"{filename}_reconstructed{ext}"
    output_path = check_path(out_path) / output_name

    return input_path, output_path
