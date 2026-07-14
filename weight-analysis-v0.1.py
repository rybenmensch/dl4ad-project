import os
import torch
import torchaudio
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import auraloss
from copy import deepcopy
from lib import *
import re

# 1. SETUP PATHS & AUDIO
# Point this to the directory containing your gin config and the v3 .ckpt file
run_path = "./models/run/"
source_path = check_path("./audio/source/")
reconstructed_path = check_path("./audio/")

file_name = "GLM.wav"
input_path, output_path = inout_paths(Path(file_name), source_path, reconstructed_path)
waveform, sr = torchaudio.load(input_path)

# 2. LOAD MODELS (.ckpt)
print("Loading baseline and manipulation models...")
model_clean = rave_from_checkpoint(run_path)
model_manip = rave_from_checkpoint(run_path)

# Extract clean baseline output
# Note: process_audio from lib.py accepts raw model + waveform
output_clean = process_audio(model_clean, waveform)

# 3. IDENTIFY DECODER LAYERS AT DIFFERENT DEPTHS
state_dict = model_clean.state_dict()


# Function to extract numbers from the key for true sequential sorting
def extract_numbers(key):
    # Finds all numbers in the key string and converts them to integers
    return [int(text) if text.isdigit() else text for text in re.split(r"(\d+)", key)]


decoder_keys = [
    k for k in state_dict.keys() if "decoder" in k and ("weight" in k or "bias" in k)
]

# Sort keys naturally (e.g., 6 comes before 10, which comes before 22)
decoder_keys.sort(key=extract_numbers)
print(decoder_keys)

# Now selecting 3 representative depths will be perfectly sequential
if len(decoder_keys) >= 3:
    target_keys = [
        decoder_keys[
            int(len(decoder_keys) * 0.1)
        ],  # Truly Early (e.g., lower block numbers)
        decoder_keys[int(len(decoder_keys) * 0.5)],  # Truly Mid
        decoder_keys[
            int(len(decoder_keys) * 0.9)
        ],  # Truly Late (e.g., higher block numbers)
    ]
else:
    target_keys = decoder_keys[:3]

print(f"\nTargeting the following decoder layers for manipulation:")
for idx, key in enumerate(target_keys):
    print(f"  Depth {idx + 1}: {key} (Shape: {list(state_dict[key].shape)})")

# 4. EXPERIMENT SETUP
scaling_factors = np.linspace(0.5, 10.0, num=10, endpoint=True)
# scaling_factors = np.logspace(-2, 2, num=10, base=10)
mrstft = auraloss.freq.MultiResolutionSTFTLoss()

# Prepare plotting
fig, axes = plt.subplots(len(target_keys), 2, figsize=(12, 4 * len(target_keys)))
if len(target_keys) == 1:
    axes = np.expand_dims(axes, axis=0)

## 5. PERTURBATION LOOP
# for depth_idx, param_key in enumerate(target_keys):
#    print(f"\nEvaluating Layer Depth {depth_idx + 1}: {param_key}")
#
#    means = []
#    errors = []
#    losses = []
#
#    for f in scaling_factors:
#        # Always reset to a fresh copy of the clean state dict
#        fresh_state = deepcopy(model_clean.state_dict())
#        std_dev = fresh_state[param_key].std().item()
#
#        # Apply scaling perturbation
#        perturbed_param = fresh_state[param_key] * f
#        fresh_state[param_key] = perturbed_param
#
#        # Load into the manipulation model
#        model_manip.load_state_dict(fresh_state)
#
#        # Process audio
#        output_tensor = process_audio(model_manip, waveform)
#
#        # Calculate distance metrics
#        absdiff = torch.abs(output_tensor - output_clean)
#        means.append(torch.mean(absdiff).item())
#        errors.append(torch.std(absdiff).item())
#
#        # Multi-Resolution STFT Loss
#        loss_val = mrstft(output_tensor.unsqueeze(0), output_clean.unsqueeze(0)).item()
#        losses.append(loss_val)
#
#        # Save audio for key factor boundaries (e.g., min, max) to avoid overflowing storage
#        if f in [scaling_factors[0], scaling_factors[-1]]:
#            f_str = f"{f:.2f}".replace('.', '_')
#            layer_short_name = param_key.replace('.', '_')[-40:] # truncate if path is too long
#
#            out_name = reconstructed_path / f"{Path(file_name).stem}_depth{depth_idx+1}_factor{f_str}.wav"
#            torchaudio.save(out_name, output_tensor, sr)
#            print(f"  -> Saved modified audio: {out_name.name}")
#
#    # Plot metrics for this specific depth
#    # Subplot 1: Mean Absolute Difference
#    axes[depth_idx, 0].errorbar(
#        scaling_factors, y=means, yerr=errors, fmt='o-',
#        ecolor='red', elinewidth=1.5, capsize=4, color='blue'
#    )
#    axes[depth_idx, 0].set_title(f"Depth {depth_idx+1} MA Diff\n({param_key[-35:]})", fontsize=9)
#    axes[depth_idx, 0].set_xlabel("Scaling Factor")
#
#    # Subplot 2: MRSTFT Spectral Loss
#    axes[depth_idx, 1].scatter(scaling_factors, losses, color='purple')
#    axes[depth_idx, 1].set_title(f"Depth {depth_idx+1} MRSTFT Loss", fontsize=9)
#    axes[depth_idx, 1].set_xlabel("Scaling Factor")

from torch.nn.utils import remove_weight_norm

# 5. PERTURBATION LOOP
for depth_idx, param_key in enumerate(target_keys):
    print(f"\nEvaluating Layer Depth {depth_idx + 1}: {param_key}")

    # Extract the module path (e.g., "decoder.net.18.aligned.branches.0.net.1")
    # We strip off '.weight_g' or '.weight_v'
    submodule_path = ".".join(param_key.split(".")[:-1])

    means, errors, losses = [], [], []

    for f in scaling_factors:
        # 1. Start with a fresh model copy to isolate tests
        model_manip = rave_from_checkpoint(str(run_path))

        # 2. Navigate to the exact sub-layer module dynamically
        module = model_manip
        for part in submodule_path.split("."):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        # 3. Remove weight normalization hook to expose the raw 'weight'
        try:
            remove_weight_norm(module)
        except ValueError:
            # If it's already removed or a different hook type
            pass

        # 4. Now 'module.weight' is fully exposed and un-normalized!
        with torch.no_grad():
            module.weight.copy_(module.weight * f)
            print(module.weight)

        # 5. Process and evaluate
        output_tensor = process_audio(model_manip, waveform)

        absdiff = torch.abs(output_tensor - output_clean)
        means.append(torch.mean(absdiff).item())
        errors.append(torch.std(absdiff).item())
        losses.append(
            mrstft(output_tensor.unsqueeze(0), output_clean.unsqueeze(0)).item()
        )

        # Save audio files
        if f in [scaling_factors[0], scaling_factors[-1]]:
            f_str = f"{f:.2f}".replace(".", "_")
            out_name = (
                reconstructed_path
                / f"{Path(file_name).stem}_depth{depth_idx+1}_drastic_factor{f_str}.wav"
            )
            torchaudio.save(out_name, output_tensor, sr)

        # Plot metrics for this specific depth
        # Subplot 1: Mean Absolute Difference
        axes[depth_idx, 0].errorbar(
            scaling_factors,
            y=means,
            yerr=errors,
            fmt="o-",
            ecolor="red",
            elinewidth=1.5,
            capsize=4,
            color="blue",
        )
        axes[depth_idx, 0].set_title(
            f"Depth {depth_idx+1} MA Diff\n({param_key[-35:]})", fontsize=9
        )
        axes[depth_idx, 0].set_xlabel("Scaling Factor")

        # Subplot 2: MRSTFT Spectral Loss
        axes[depth_idx, 1].scatter(scaling_factors, losses, color="purple")
        axes[depth_idx, 1].set_title(f"Depth {depth_idx+1} MRSTFT Loss", fontsize=9)
        axes[depth_idx, 1].set_xlabel("Scaling Factor")


plt.tight_layout()
plt.show()
