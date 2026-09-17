import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torchaudio

from encodec_lib import EncodecNNModel
from lib import *
from model import Net, NetTypeEnum, get_shape_preserving_layers_from_net
from modules import *
from plotting import plot_comparison
from rave_lib import RAVEModel
from torch_lib import get_layer_name

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

source_path: Path = check_path("audio/source")
reconstructed_root: Path = check_path("audio/reconstructed")
file_name: str = "GLM.wav"
base_wav = torchaudio.load("audio/source/GLM.wav")
base_source, base_sr = base_wav

model = RAVEModel("models/satyr/")
base_recon = model(base_wav)

processed = model(base_wav)
torchaudio.save(reconstructed_root / "lmao.wav", processed, model.get_sample_rate())

torchaudio.save(
    reconstructed_root / "base_reconstruction.wav", base_recon, model.get_sample_rate()
)

# - weight:
#     - shuffling
#     - multiplication
#     - roll
#     - oscillate
#     - invert
#     - ablate
# - skip layers
# - repeat layers

exit()


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
    model: RAVEModel
    net: Net
    net_path: str
    index: int
    name: str
    stats: list[Stats]
    skip_recon: torch.Tensor | None = None
    repeat_recon: torch.Tensor | None = None


# # collect layers from both encoder and decoder
# shape_preserving_layers: List[Layer] = []
# for net, net_path in model.get_nets_and_paths():
#     layers = get_shape_preserving_layers(net)
#     for layer in layers:
#         shape_preserving_layers.append(
#             Layer(model, net, net_path, layer["index"], layer["name"], [])
#         )


def process_audio_with_modified_layer(layer: Layer, make_net) -> torch.Tensor:
    model = layer.model
    # TODO: implement an actual copy here!
    original_net = layer.net

    net_type: NetTypeEnum = model.from_net_path.get_net_type(layer.net_path)
    new_net = make_net(original_net, layer.index)
    model.set_net(net_type, new_net)

    with torch.no_grad():
        mod_recon = process_audio(model.model, base_source)

    model.set_net(net_type, original_net)

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


for layer in shape_preserving_layers:

    def bruh(op, audiofile):
        net_path = f"{layer.net_path}_{layer.index}"
        net_path = "_".join(net_path.split("."))
        net_path = f"{op}_{net_path}"
        fn_a = str(reconstructed_root / (net_path + ".wav"))
        fn_p = str(reconstructed_root / (net_path + ".png"))

        torchaudio.save(fn_a, audiofile, sr)
        plot_comparison(base_recon, audiofile, sr, save_path=fn_p, show=False)

    def norm(x):
        return x / torch.max(x)

    skip_recon = process_audio_with_skipped_layer(layer)
    repeat_recon = process_audio_with_repeated_layer(layer)

    skip_recon = norm(skip_recon)
    repeat_recon = norm(repeat_recon)

    bruh("skip", skip_recon)
    bruh("repeat", repeat_recon)
    # plot_comparison(base_recon, skip_recon, sr, save_path=")


exit()

for layer in shape_preserving_layers:
    layer.skip_recon = process_audio_with_skipped_layer(layer)
    layer.repeat_recon = process_audio_with_repeated_layer(layer)
    # l.skip_recon = torch.zeros_like(base_reconstruction)
    # l.repeat_recon = torch.zeros_like(base_reconstruction)

    layer.stats.append(Stats(base_recon, layer.skip_recon, Mode.skip))
    layer.stats.append(Stats(base_recon, layer.repeat_recon, Mode.repeat))


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
