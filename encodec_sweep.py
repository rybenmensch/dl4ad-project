import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

from lib import *
from plotting import plot_comparison
from encodec import EncodecNNModel, encodec_from_pretrained, get_shape_preserving_layers
from model import NetTypeEnum

warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*weight_norm` is deprecated.*"
)


def encodec_process_audio(model, waveform: torch.Tensor, bandwidth: float | None = None) -> torch.Tensor:
    """
    Adapter fuer HF EncodecModel: forward() gibt ein EncodecOutput-Objekt
    zurueck (mit .audio_values), keinen direkten Tensor wie RAVE.

    bandwidth: Ziel-Bitrate in kbps. None -> hoechste verfuegbare Qualitaet
    """
    if bandwidth is None:
        bandwidth = max(model.config.target_bandwidths)

    input_tensor = waveform.unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor, bandwidth=bandwidth)
    return output.audio_values.squeeze(0)


def make_skipped_modulelist(original_net, skip_index: int) -> nn.ModuleList:
    """Baut eine neue ModuleList ohne den Layer an skip_index."""
    return nn.ModuleList([l for i, l in enumerate(original_net) if i != skip_index])


def make_repeated_modulelist(original_net, repeat_index: int, times: int = 2) -> nn.ModuleList:
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

raw_model = encodec_from_pretrained("facebook/encodec_24khz")
model = EncodecNNModel(raw_model)


source_path: Path = check_path("audio/source")
reconstructed_root: Path = check_path("audio/reconstructed")

base_source, sr = torchaudio.load("audio/source/GLM.wav")

# Stereo auf Mono
if base_source.shape[0] > 1:
    base_source = base_source[0:1, :]

# baseline reconstruction
base_recon = encodec_process_audio(model.model, base_source)
torchaudio.save(str(reconstructed_root / "base_reconstruction_encodec.wav"), base_recon, sr)
print("Baseline gespeichert.")

# skip and repeat sweep
shape_preserving_layers = []
for net, net_path in model.get_nets_and_paths():
    layers = get_shape_preserving_layers(net)
    for layer in layers:
        shape_preserving_layers.append(
            {"net_path": net_path, "index": layer["index"], "name": layer["name"]}
        )



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
    recon = encodec_process_audio(model.model, base_source)
    model.set_net(net_type, original_net)
    return recon


for l in shape_preserving_layers:
    net_path, index, name = l["net_path"], l["index"], l["name"]

    skip_recon = norm(process_with_modification(net_path, index, "skip"))
    repeat_recon = norm(process_with_modification(net_path, index, "repeat"))

    tag = "_".join(f"{net_path}_{index}".split("."))

    for op, audio in [("skip", skip_recon), ("repeat", repeat_recon)]:
        out_name = f"{op}_{tag}"
        fn_a = str(reconstructed_root / f"{out_name}.wav")
        fn_p = str(reconstructed_root / f"{out_name}.png")

        torchaudio.save(fn_a, audio, sr)
        plot_comparison(
            base_recon, audio, sr,
            title=f"{op}: {net_path}[{index}] ({name})",
            save_path=fn_p, show=False,
        )

