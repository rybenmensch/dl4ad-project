import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torchaudio

from lib import *
from plotting import plot_comparison
from rave_lib import *

# Suppress the lightning_fabric pkg_resources warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*weight_norm` is deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*return_complex.*argument is now deprecated.*",
)

model = rave_from_checkpoint("models/satyr")
# model = rave_from_checkpoint("models/checkpoint/")

source_path: Path = check_path("audio/source")
reconstructed_root: Path = check_path("audio/reconstructed/satyr/")

file_name: str = "GLM.wav"

base_source, sr = torchaudio.load("audio/source/GLM.wav")
base_recon = process_audio(model, base_source)


torchaudio.save(reconstructed_root / "base_reconstruction.wav", base_recon, sr)

# model.encoder.encoder.net = SequentialWithSkip(model.encoder.encoder.net, [20])
# with torch.no_grad():
#     mod_recon = process_audio(model, base_source)

# model.decoder.net = ManipulatedSequential(model.decoder.net, repeats={14: 3})
# with torch.no_grad():
#     mod_recon = process_audio(model, base_source)
# torchaudio.save(output_path, mod_recon, sr)


class Mode(Enum):
    skip = 1
    repeat = 2


class Stats:
    def __init__(
        self, baseline: torch.Tensor, reconstruction: torch.Tensor, mode: Mode
    ):
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
        shape_preserving_layers.append(
            Layer(model, net, net_path, layer["index"], layer["name"], [])
        )

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
        mod_recon = process_audio(model, base_source)

    # set to old net again
    # honk honk
    if layer.net_path == get_encoder_net(model)[1]:
        model = set_encoder_net(model, original_net)
    elif layer.net_path == get_decoder_net(model)[1]:
        model = set_decoder_net(model, original_net)

    return mod_recon


def process_audio_with_skipped_layer(layer: Layer) -> torch.Tensor:
    return process_audio_with_modified_layer(
        layer, lambda n, i: ManipulatedSequential(n, skips=[i])
    )


def process_audio_with_repeated_layer(layer: Layer) -> torch.Tensor:
    num_repeats = 2
    return process_audio_with_modified_layer(
        layer, lambda n, i: ManipulatedSequential(n, repeats={i: num_repeats})
    )


for l in shape_preserving_layers:

    def bruh(op, audiofile):
        net_path = f"{l.net_path}_{l.index}"
        net_path = "_".join(net_path.split("."))
        net_path = f"{op}_{net_path}"
        fn_a = str(reconstructed_root / (net_path + ".wav"))
        fn_p = str(reconstructed_root / (net_path + ".png"))

        torchaudio.save(fn_a, audiofile, sr)
        plot_comparison(base_recon, audiofile, sr, save_path=fn_p, show=False)

    def norm(x):
        return x / torch.max(x)

    skip_recon = process_audio_with_skipped_layer(l)
    repeat_recon = process_audio_with_repeated_layer(l)

    skip_recon = norm(skip_recon)
    repeat_recon = norm(repeat_recon)

    bruh("skip", skip_recon)
    bruh("repeat", repeat_recon)
    # plot_comparison(base_recon, skip_recon, sr, save_path=")


exit()

for l in shape_preserving_layers:
    l.skip_recon = process_audio_with_skipped_layer(l)
    l.repeat_recon = process_audio_with_repeated_layer(l)
    # l.skip_recon = torch.zeros_like(base_reconstruction)
    # l.repeat_recon = torch.zeros_like(base_reconstruction)

    l.stats.append(Stats(base_recon, l.skip_recon, Mode.skip))
    l.stats.append(Stats(base_recon, l.repeat_recon, Mode.repeat))


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
                flattened_rows.append(
                    {
                        "path": f"{layer.net_path}[{layer.index}]",
                        "type": layer.name,
                        "operation": stat.mode.name,
                        "change": diff,
                    }
                )

    # Sort every row independently from most impact to last impact
    flattened_rows.sort(key=lambda x: x["change"])
    flattened_rows.reverse()

    from table2md import MarkdownTable

    data = []

    print("\n--- RESULTS: Layer Impact (Sorted from Least Impact to Most Impact) ---\n")

    for row in flattened_rows:
        data.append(
            {
                "Layer path": row["path"],
                "Type": row["type"],
                "Operation": row["operation"],
                f"Reconstruction Change ({stat_attr.upper()})": round(row["change"], 4),
            }
        )

    MarkdownTable.from_dicts(data).print()
