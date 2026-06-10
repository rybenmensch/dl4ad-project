import torch
import torchaudio
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from lib import *
import auraloss

# MODEL
model = Model("models/percussion.ts")
# assume mono model
if(model.input_channels != 1):
    raise Exception("Stereo models not supported yet!")

model_clean = Model("models/percussion.ts")

# AUDIO
source_path: Path = check_path("audio/source")
reconstructed_path: Path = check_path("audio/reconstructed")

# file_name: str = "tof.wav"
file_name: str = "GLM.wav"

input_path, output_path = inout_paths(Path(file_name), source_path, reconstructed_path)
waveform, sr = torchaudio.load(input_path)

output_clean = model_clean.process_audio(waveform)

# MANIPULATION
state_dict = model.get_state_dict()
x = np.linspace(1, 10, num = 64, endpoint = False)

means = []
errors = []
losses = []
mrstft = auraloss.freq.MultiResolutionSTFTLoss()

path = "decoder.synth.branches.1.weight"

for f in x:
    state_dict = model_clean.get_state_dict()
    weight = state_dict[path]

    weight = weight * f

    state_dict[path] = weight
    model.set_state_dict(state_dict)
    output_tensor = model.process_audio(waveform)

    absdiff = torch.abs(output_tensor - output_clean)
    means.append(torch.mean(absdiff).item())
    errors.append(torch.std(absdiff).item())
    losses.append(mrstft(output_tensor.unsqueeze(0), output_clean.unsqueeze(0)))

fig, ax = plt.subplots(1, 2)

ax[0].errorbar(
    x,
    y=means, 
    yerr=errors, 
    fmt='o-',        # 'o' for circle markers, '-' for a connecting line
    ecolor='red',    # Color of the error bars
    elinewidth=1.5,  # Width of the error bar lines
    capsize=5,       # Width of the horizontal caps on the error bars
    color='blue',    # Color of the data points and line
)
ax[0].set_title("mean absolute difference")
ax[1].scatter(x, losses)
ax[1].set_title("losses")
plt.tight_layout()
plt.show()

# # SAVE OUTPUT
# torchaudio.save(output_path, output_tensor, sr)

