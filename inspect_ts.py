import torch
import torchaudio
from pathlib import Path
import numpy as np
from lib import *
import torchinfo
from plotting import plot_comparison

# MODEL
model = Model("models/percussion.ts")
# assume mono model
if(model.input_channels != 1):
    raise Exception("Stereo models not supported yet!")

model_clean = Model("models/percussion.ts")

m: torch.nn.Module = model.model

# for name, param in m.named_parameters():
#     print(name, '->', param.shape)

# torchinfo.summary(m, input_data=torch.ones((1, 1, 44100), dtype=torch.float32))
# exit()

# AUDIO
source_path: Path = check_path("audio/source")
reconstructed_path: Path = check_path("audio/reconstructed")

# file_name: str = "tof.wav"
file_name: str = "GLM.wav"

input_path, output_path = inout_paths(Path(file_name), source_path, reconstructed_path)
print(input_path)
#input_path = str(input_path)
waveform, sr = torchaudio.load(input_path)

output_clean = model_clean.process_audio(waveform)

# MANIPULATION
state_dict = model.get_state_dict()

model.print_model_keys()
#exit()

x = np.linspace(-2, 2, num = 10, endpoint = False)

x = [0, 2, 4]

for i in x:
    state_dict = model_clean.get_state_dict()

    wave_weights = state_dict["decoder.synth.branches.0.weight"]
    state_dict["decoder.synth.branches.0.weight"] = torch.zeros_like(wave_weights)

    name = lambda s: f"decoder.synth.branches.2.net.{s}.bias"

    saved = state_dict[name(i)]
    for j in x:
        bias = state_dict[name(j)]
        state_dict[name(j)] = torch.zeros_like(bias)

    state_dict[name(i)] = saved

    model.set_state_dict(state_dict)
    output_tensor = model.process_audio(waveform)
    
    
    output_tensor *= 10
    
    plot_comparison(
        output_clean,
        output_tensor,
        sr=model.sr,
        title=f"Degradation {i}",
        save_path=f"plots/plot_{i}.png",
        show=False,
    )
    

    name, ext = os.path.splitext(output_path)
    out_path = f"{name}_{i}{ext}"
    torchaudio.save(out_path, output_tensor, sr)

# # SAVE OUTPUT
# torchaudio.save(output_path, output_tensor, sr)

exit()

# model inspection stuff

all_weights = list(filter(lambda x: "weight" in x, model.get_model_keys()))
for w in all_weights:
    print(f"{w}    {state_dict[w].shape}")

model.print_model_keys()

exit()

keys = model.get_model_keys()
keys = list(filter(lambda x: "encoder" in x, keys))

bruh = {}
for key in keys:
    start = len("encoder.net.")
    ln = key[start:].find('.')
    idx = int(key[start:start+ln])
    if idx not in bruh:
        bruh[idx] = []
    bruh[idx].append(key[start+ln+1:])

for key in bruh.keys():
    path = f"encoder.net.{key}"
    print(f"{path}:")
    val = bruh[key]
    for v in val:
        entry = state_dict[f"{path}.{v}"]
        shape = list(entry.shape)
        print(f"    {v}:    {entry.shape}")

