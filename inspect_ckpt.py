import torch
from pathlib import Path
import torchaudio
from lib import *

model = rave_from_checkpoint("models/checkpoint/")

# for k in checkpoint["state_dict"].keys():
#     print(k)

# for k in model.get_():
#     print(k)

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

