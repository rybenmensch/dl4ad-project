import torch
import torchaudio
from pathlib import Path
from lib import *
from rave_lib import *
from dataclasses import dataclass
from typing import Any, List

# model = rave_from_checkpoint("models/satyr")
model = rave_from_checkpoint("models/checkpoint/")

source_path: Path = check_path("audio/source")
reconstructed_path: Path = check_path("audio/reconstructed")

file_name: str = "GLM.wav"

input_path, _ = inout_paths(Path(file_name), source_path, reconstructed_path)
base_source, sr = torchaudio.load(input_path)
base_reconstruction = process_audio(model, base_source)

@dataclass
class Layer:
    model: rave.RAVE
    net: cached_conv.convs.CachedSequential
    net_path: str
    index: int
    name: str

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
            layer["name"]
        ))

# the following functions are less-than-optimal only to be used for current
# task! should be structured differently if we want to do layer skipping for
# actually producing sounds! also, things are hardcoded and idiotic but I don't
# have time to deal with it now :)

def process_audio_with_skipped_layer(layer: Layer) -> torch.Tensor:
    model = layer.model
    original_net = layer.net

    # set to new net
    # this is rather dumb. should do something with getattr and setattr or
    # something, but also borderline unbearable to do it that way. so kludge it
    # is for now
    skip_net = SequentialWithSkip(original_net, skips=[layer.index])
    if layer.net_path == get_encoder_net(model)[1]:
        model = set_encoder_net(model, skip_net)
    elif layer.net_path == get_decoder_net(model)[1]:
        model = set_decoder_net(model, skip_net)

    with torch.no_grad():
        mod_reconstruction = process_audio(model, base_source)

    # set to old net again
    # honk honk
    if layer.net_path == get_encoder_net(model)[1]:
        model = set_encoder_net(model, original_net)
    elif layer.net_path == get_decoder_net(model)[1]:
        model = set_decoder_net(model, original_net)

    return mod_reconstruction

def process_audio_with_repeated_layer(layer: Layer) -> torch.Tensor:
    model = layer.model
    original_net = layer.net
    num_repeats = 2

    # set to new net
    # this is rather dumb. should do something with getattr and setattr or
    # something, but also borderline unbearable to do it that way. so kludge it
    # is for now
    # skip_net = SequentialWithSkip(original_net, skips=[layer.index])
    repeat_net = SequentialWithRepeat(original_net, repeats={layer.index: num_repeats})
    if layer.net_path == get_encoder_net(model)[1]:
        model = set_encoder_net(model, repeat_net)
    elif layer.net_path == get_decoder_net(model)[1]:
        model = set_decoder_net(model, repeat_net)

    with torch.no_grad():
        mod_reconstruction = process_audio(model, base_source)

    # set to old net again
    # honk honk
    if layer.net_path == get_encoder_net(model)[1]:
        model = set_encoder_net(model, original_net)
    elif layer.net_path == get_decoder_net(model)[1]:
        model = set_decoder_net(model, original_net)

    return mod_reconstruction

process_audio_with_skipped_layer(shape_preserving_layers[0])
process_audio_with_skipped_layer(shape_preserving_layers[1])

exit()
# WHAT ABOUT CHANGING LAYER ORDER?

def measure_impact(layers, loss_function):
    results  = []

    for skip_idx, layer_name in layers:
        # save original decoder
        original_net = model.decoder.net
        model.decoder.net = SequentialWithSkip(original_net, skips=[skip_idx])

        with torch.no_grad():
            try:
                mod_reconstruction = process_audio(model, base_source)
                diff = loss_function(base_reconstruction, mod_reconstruction)
                results.append((skip_idx, layer_name, diff))
                # print(f"Skipping Layer {skip_idx:<2} ({layer_name:<30}): Mean Diff = {diff:.6f}")

            except Exception as e:
                print(f"Layer nr {skip_idx} of type {layer_name} raised {e}")
        model.decoder.net = original_net

    return results

for L in [mean_absolute_error, mrstft]:
    impact_results = measure_impact(layers, L)
    impact_results.sort(key=lambda x: x[2])

    print("\n=======================================================================")
    print("\n--- RESULTS: Layer Impact (Sorted from Least Impact to Most Impact) ---")
    print(f"{'Index':<6} | {'Layer Type':<30} | {f'Reconstruction Change ({L.__name__})':<30}")
    print("-" * 72)
    for idx, name, diff in impact_results:
        print(f"{idx:<6} | {name:<30} | {diff:.6f}")
