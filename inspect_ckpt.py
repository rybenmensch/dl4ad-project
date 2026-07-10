# from lib import *
# import torch
# import torchaudio
# from pathlib import Path
# import numpy as np
# import torchinfo
# import cached_conv as cc
# import gin
# import nn_tilde
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# from absl import flags, app
# from typing import Union, Optional
# import rave

import torch
import gin
import rave
import rave.core
from scripts.export import VariationalScriptedRAVE
from pathlib import Path
import torchaudio
from lib import *


run_path = "models/checkpoint/"
config_file = rave.core.search_for_config(run_path)
gin.parse_config_file(config_file)

checkpoint_path = rave.core.search_for_run(run_path)
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Find the input channels of the encoder's first conv layer from state_dict to detect n_channels
in_channels = None
for key in ["encoder.encoder.net.0.weight_v", "encoder.net.0.weight_v", "encoder.encoder.net.0.weight"]:
    if key in checkpoint["state_dict"]:
        in_channels = checkpoint["state_dict"][key].shape[1]
        break

if in_channels is not None:
    try:
        n_band = gin.query_parameter('%N_BAND')
    except Exception:
        n_band = 16
    n_channels = in_channels // n_band
else:
    n_channels = 1

model = rave.RAVE(n_channels=n_channels)
model.load_state_dict(checkpoint["state_dict"], strict=False)
model.eval()
# # exit()


# EXPERIMENT

original_encoder_net = model.encoder.encoder.net
last_layer = model.encoder.encoder.net[-1]
if hasattr(last_layer, "module"):
    conv_layer = last_layer.module
else:
    conv_layer = last_layer

encoder_output_channels = conv_layer.out_channels
torch.manual_seed(0)

class CustomEncoderWrapper(torch.nn.Module):
    def __init__(self, original_net, channels):
        super().__init__()
        self.original_net = original_net

        self.custom_layer = torch.nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1
        )

        with torch.no_grad():
            self.custom_layer.weight.copy_(torch.eye(channels).unsqueeze(-1))
            self.custom_layer.bias.zero_()

    def forward(self, x):
        features = self.original_net(x)
        return self.custom_layer(features)

model.encoder.encoder.net = CustomEncoderWrapper(
    original_encoder_net,
    encoder_output_channels
)

print(model.encoder.encoder.net)

# AUDIO
source_path: Path = check_path("audio/source")
reconstructed_path: Path = check_path("audio/reconstructed")

# file_name: str = "tof.wav"
file_name: str = "GLM.wav"

input_path, output_path = inout_paths(Path(file_name), source_path, reconstructed_path)
waveform, sr = torchaudio.load(input_path)

output_tensor = process_audio(model, waveform)
torchaudio.save(output_path, output_tensor, sr)

