from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

from encodec_lib import (
    EncodecNNModel,
    HFEncodecNNModel,
    process_audio,
    raw_encodec_model,
    raw_hf_encodec_model,
)
from lib import check_path
from model import NetTypeEnum
from plotting import plot_comparison
from torch_lib import get_shape_preserving_layers


def make_skipped_modulelist(original_net, skip_index: int) -> nn.ModuleList:
    """Baut eine neue ModuleList ohne den Layer an skip_index."""
    return nn.ModuleList([l for i, l in enumerate(original_net) if i != skip_index])


def make_repeated_modulelist(
    original_net, repeat_index: int, times: int = 2
) -> nn.ModuleList:
    """Baut eine neue ModuleList, in der der Layer an repeat_index
    wiederholt wird."""
    new_layers = []
    for i, l in enumerate(original_net):
        if i == repeat_index:
            for _ in range(times):
                new_layers.append(l)
        else:
            new_layers.append(l)
    return nn.ModuleList(new_layers)


def norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.max(torch.abs(x))


# MODELL LADEN

raw_model = raw_hf_encodec_model()
model = HFEncodecNNModel(raw_model)
print(model.get_sample_rate())
model.reset()

raw_model = raw_encodec_model(24_000)
model = EncodecNNModel(raw_model)
print(model.get_sample_rate())
model.reset()

# raw_model = encodec_model_24khz()
# model = EncodecNNModel(raw_model)
# model.reset()
# print(model.model.sample_rate)
#
# model = EncodecNNModel()
# model.reset()
# print(model.model.sample_rate)

exit()

# raw_model = encodec_from_hf("facebook/encodec_24khz")
# model = HFEncodecNNModel(raw_model)

# skip and repeat sweep
shape_preserving_layers = []
for net, net_path in model.get_nets_and_paths():
    layers = get_shape_preserving_layers(net)
    for layer in layers:
        shape_preserving_layers.append(
            {"net_path": net_path, "index": layer["index"], "name": layer["name"]}
        )


index = 0
net_type = NetTypeEnum.Encoder
original_net = model.get_net(net_type)

model.set_net(NetTypeEnum.Decoder, None)

# print("==========================================================")
# print(model.model)
new_net = make_repeated_modulelist(original_net, index, times=2)
new_net = None
model.set_net(net_type, new_net)

print("==========================================================")
print(model.model)
# recon = process_audio(model.model, base_source)
# model.set_net(net_type, original_net)

exit()

source_path: Path = check_path("audio/source")
reconstructed_root: Path = check_path("audio/reconstructed")

base_source, sr = torchaudio.load("audio/source/GLM.wav")

# Stereo auf Mono
if base_source.shape[0] > 1:
    base_source = base_source[0:1, :]

# baseline reconstruction
base_recon = process_audio(model.model, base_source)
# torchaudio.save(
#     str(reconstructed_root / "base_reconstruction_encodec.wav"), base_recon, sr
# )
# print("Baseline gespeichert.")


def process_with_modification(net_path: str, index: int, mode: str) -> torch.Tensor:
    net_type = (
        NetTypeEnum.Encoder
        if net_path == model.get_net_path(NetTypeEnum.Encoder)
        else NetTypeEnum.Decoder
    )
    original_net = model.get_net(net_type)

    if mode == "skip":
        new_net = make_skipped_modulelist(original_net, index)
    else:
        new_net = make_repeated_modulelist(original_net, index, times=2)

    model.set_net(net_type, new_net)
    recon = process_audio(model.model, base_source)
    model.set_net(net_type, original_net)
    return recon


for l in shape_preserving_layers:
    net_path, index, name = l["net_path"], l["index"], l["name"]

    skip_recon = norm(process_with_modification(net_path, index, "skip"))
    # repeat_recon = norm(process_with_modification(net_path, index, "repeat"))
    #
    # tag = "_".join(f"{net_path}_{index}".split("."))
    #
    # for op, audio in [("skip", skip_recon), ("repeat", repeat_recon)]:
    #     out_name = f"{op}_{tag}"
    #     fn_a = str(reconstructed_root / f"{out_name}.wav")
    #     fn_p = str(reconstructed_root / f"{out_name}.png")
    #
    #     torchaudio.save(fn_a, audio, sr)
    #     plot_comparison(
    #         base_recon,
    #         audio,
    #         sr,
    #         title=f"{op}: {net_path}[{index}] ({name})",
    #         save_path=fn_p,
    #         show=False,
    #     )
