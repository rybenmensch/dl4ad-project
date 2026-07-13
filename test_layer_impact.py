import torch
import torchaudio
from pathlib import Path
from lib import *
from rave_lib import *
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import numpy as np

import warnings

# Suppress the lightning_fabric pkg_resources warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm` is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*return_complex.*argument is now deprecated.*")

model = rave_from_checkpoint("models/satyr")
# model = rave_from_checkpoint("models/checkpoint/")

source_path: Path = check_path("audio/source")
reconstructed_path: Path = check_path("audio/reconstructed")

file_name: str = "GLM.wav"

input_path, output_path = inout_paths(Path(file_name), source_path, reconstructed_path)
base_source, sr = torchaudio.load(input_path)
base_reconstruction = process_audio(model, base_source)

# model.encoder.encoder.net = SequentialWithSkip(model.encoder.encoder.net, [20])
# with torch.no_grad():
#     mod_reconstruction = process_audio(model, base_source)

# model.decoder.net = ManipulatedSequential(model.decoder.net, repeats={14: 3})
# with torch.no_grad():
#     mod_reconstruction = process_audio(model, base_source)

# torchaudio.save(output_path, mod_reconstruction, sr)

class Mode(Enum):
    skip = 1
    repeat = 2

class Stats:
    def __init__(self, baseline: torch.Tensor, reconstruction: torch.Tensor, mode: Mode):
        self.baseline = baseline
        self.reconstruction: torch.Tensor = reconstruction
        self.mae: float = 0
        self.mrstft: float = 0
        self.mode = mode

        self.calc_mae()
        self.calc_mrstft()

    def __str__(self) -> str:
        return str(self.mae)

    def calc_mae(self):
        self.mae = mean_absolute_error(self.baseline, self.reconstruction)

    def calc_mrstft(self):
        self.mrstft = mrstft(self.baseline, self.reconstruction)

@dataclass
class Layer:
    model: rave.RAVE
    net: cached_conv.convs.CachedSequential
    net_path: str
    index: int
    name: str
    stats: List[Stats]
    skip_recon: Optional[torch.Tensor] = None
    repeat_recon: Optional[torch.Tensor] = None

# collect layers from both encoder and decoder
shape_preserving_layers: List[Layer] = []
for net, net_path in get_nets(model):
    layers = get_shape_preserving_layers(net)
    for layer in layers:
        shape_preserving_layers.append(Layer(
            model,
            net,
            net_path,
            layer["index"],
            layer["name"],
            []
        ))

# the following function is less-than-optimal only to be used for current task!
# should be structured differently if we want to do layer skipping for actually
# producing sounds! also, things are hardcoded and idiotic but I don't have
# time to deal with it now :)

def process_audio_with_modified_layer(layer: Layer, layer_factory) -> torch.Tensor:
    model = layer.model
    original_net = layer.net

    # set to new net
    # this is rather dumb. should do something with getattr and setattr or
    # something, but also borderline unbearable to do it that way. so kludge it
    # is for now
    # skip_net = SequentialWithSkip(original_net, skips=[layer.index])
    # repeat_net = SequentialWithRepeat(original_net, repeats={layer.index: num_repeats})
    new_net = layer_factory(original_net, layer.index)
    if layer.net_path == get_encoder_net(model)[1]:
        model = set_encoder_net(model, new_net)
    elif layer.net_path == get_decoder_net(model)[1]:
        model = set_decoder_net(model, new_net)

    with torch.no_grad():
        mod_reconstruction = process_audio(model, base_source)

    # set to old net again
    # honk honk
    if layer.net_path == get_encoder_net(model)[1]:
        model = set_encoder_net(model, original_net)
    elif layer.net_path == get_decoder_net(model)[1]:
        model = set_decoder_net(model, original_net)

    return mod_reconstruction


def process_audio_with_skipped_layer(layer: Layer) -> torch.Tensor:
    return process_audio_with_modified_layer(layer, lambda n, i:
                                             ManipulatedSequential(n, skips=[i]))


def process_audio_with_repeated_layer(layer: Layer) -> torch.Tensor:
    num_repeats = 2
    return process_audio_with_modified_layer(layer, lambda n, i:
                                             ManipulatedSequential(n, repeats={i: num_repeats}))


for l in shape_preserving_layers:
    l.skip_recon = process_audio_with_skipped_layer(l)
    l.repeat_recon = process_audio_with_repeated_layer(l)
    # l.skip_recon = torch.zeros_like(base_reconstruction)
    # l.repeat_recon = torch.zeros_like(base_reconstruction)

    l.stats.append(Stats(base_reconstruction, l.skip_recon, Mode.skip))
    l.stats.append(Stats(base_reconstruction, l.repeat_recon, Mode.repeat))



# thx gemini
for L in [mean_absolute_error, mrstft]:
    # Determine which property name to look at based on the function
    stat_attr = "mae" if L.__name__ == "mean_absolute_error" else "mrstft"

    # Flatten out layers and stats into single row entries and apply the near-zero filter
    flattened_rows = []
    for layer in shape_preserving_layers:
        for stat in layer.stats:
            diff = getattr(stat, stat_attr)
            
            # One-liner to filter out values super close to 0 using numpy
            if not np.isclose(diff, 0.0, atol=1e-5):
                flattened_rows.append({
                    "path": f"{layer.net_path}[{layer.index}]",
                    "type": layer.name,
                    "operation": stat.mode.name,
                    "change": diff
                })

    # Sort every row independently from least impact to most impact
    flattened_rows.sort(key=lambda x: x["change"])
    flattened_rows.reverse()

    print("\n=======================================================================")
    print("\n--- RESULTS: Layer Impact (Sorted from Least Impact to Most Impact) ---")
    print(" | ".join([
        f"{'Layer path':<24}",
        f"{'Type':<10}",
        f"{'Operation':<7}",
        f"{f'Reconstruction Change ({stat_attr.upper()})':<30}"
    ]))
    print("-" * 72)
    for row in flattened_rows:
        print(" | ".join([
            f"{row['path']:<24}",
            f"{row['type']:<10}",
            f"{row['operation']:<7}",
            f"{row['change']:.6f}"
        ]))

