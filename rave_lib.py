import cached_conv
import torch
import gin
import rave
from lib import *
import torch.nn as nn
from typing import Any

# In here, just stuff to interface with RAVE models and components!


def rave_from_checkpoint(run_path: Path | str) -> rave.RAVE:
    """
    Create a full RAVE model from the path to a run.
    input:
        run_path: Path | str
    output:
        RAVE model
    """

    config_file = rave.core.search_for_config(run_path)
    gin.parse_config_file(config_file)

    checkpoint_path = rave.core.search_for_run(run_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint["state_dict"]
    n_channels = get_state_dict_in_channels(state_dict)

    model = rave.RAVE(n_channels=n_channels)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def get_encoder_net(model: rave.RAVE) -> Tuple[cached_conv.convs.CachedSequential, str]:
    """
    Returns the encoder net and the path to it.
    Exists because the topology can change between RAVE updates, in which case
    we will update the getter here.
    """
    encoder = model.encoder
    encoder_str = "encoder"
    if hasattr(encoder, "encoder"):
        encoder = encoder.encoder
        encoder_str += ".encoder"
    return (encoder.net, encoder_str + ".net")


def set_encoder_net(model: rave.RAVE, net: Any) -> rave.RAVE:
    """
    Returns the model with updated encoder_net.
    Exists because the topology can change between RAVE updates, in which case
    we will update the getter here.
    Also exists because of a kludge and will maybe possibly be removed
    """
    if hasattr(model.encoder, "encoder"):
        model.encoder.encoder.net = net
    else:
        model.encoder.net = net
    return model


def get_decoder_net(model: rave.RAVE) -> Tuple[cached_conv.convs.CachedSequential, str]:
    """
    Returns the decoder net and the path to it.
    Exists because the topology can change between RAVE updates, in which case
    we will update the getter here.
    """
    decoder = model.decoder
    decoder_str = "decoder"
    if hasattr(decoder, "decoder"):
        decoder = decoder.decoder
        decoder_str += ".decoder"
    return (decoder.net, decoder_str + ".net")


def set_decoder_net(model: rave.RAVE, net: Any) -> rave.RAVE:
    """
    Returns the model with updated decoder_net.
    Exists because the topology can change between RAVE updates, in which case
    we will update the getter here.
    Also exists because of a kludge and will maybe possibly be removed
    """
    if hasattr(model.decoder, "decoder"):
        model.decoder.decoder.net = net
    else:
        model.decoder.net = net
    return model


def get_nets(model: rave.RAVE) -> Tuple[
        Tuple[cached_conv.convs.CachedSequential, str],
        Tuple[cached_conv.convs.CachedSequential, str]]:
    """
    Returns the encoder and the decoder nets and the paths to them.
    """
    return (get_encoder_net(model), get_decoder_net(model))


def get_last_encoder_layer(model: rave.RAVE) -> cached_conv.convs.Conv1d:
    return model.encoder.encoder.net[-1]


def get_encoder_output_channels(model: rave.RAVE) -> int:
    return model.encoder.encoder.net[-1].out_channels


def get_shape_preserving_layers(net: cached_conv.CachedSequential):
    """
    Returns information about every layer that preserves the input shape.
    Input:
        - Sequential net
    Output:
        - List of dicts with content {index, name}
    """
    results = []

    try:
        net[0]
    except:
        print("Model should be sequential!")
        exit()

    input_size = net[0].in_channels

    # batch=1, channels=input_size, time=64
    x = torch.zeros(1, input_size, 64)

    for idx, layer in enumerate(net):
        layer_name = type(layer).__name__
        try:
            with torch.no_grad():
                out = layer(x)
                if out.shape == x.shape:
                    # layer preserves shape
                    results.append({"index": idx, "name": layer_name})
                else:
                    # layer does not preserve shape
                    pass
                x = out
        except Exception as e:
            print(f"Layer nr {idx} of type {layer_name} raised {e}")

    return results


# CUSTOM MODULES


class SequentialWithSkip(nn.Module):
    def __init__(self, original_net, skips=None):
        super().__init__()
        self.original_net = original_net
        self.skips = set(skips) if skips else set()
    def forward(self, x):
        for i, layer in enumerate(self.original_net):
            if i in self.skips:
                continue
            x = layer(x)
        return x


class SequentialWithRepeat(nn.Module):
    def __init__(self, original_net, repeats=None):
        super().__init__()
        self.original_net = original_net
        self.repeats = repeats if repeats else {}
    def forward(self, x):
        for i, layer in enumerate(self.original_net):
            r = self.repeats.get(i, 1)
            for r in range(r):
                x  = layer(x)
            return x


class ManipulatedSequential(nn.Module):
    def __init__(self, original_net, skips=None, repeats=None):
        super().__init__()
        self.original_net = original_net
        self.skips = set(skips) if skips else set()
        self.repeats = repeats if repeats else {}
    def forward(self, x):
        for i, layer in enumerate(self.original_net):
            if i in self.skips:
                continue
            r = self.repeats.get(i, 1)
            for _ in range(r):
                x = layer(x)
        return x


# class CustomEncoderWrapper(torch.nn.Module):
#     def __init__(self, original_net, channels):
#         super().__init__()
#         self.original_net = original_net
#
#         self.custom_layer = torch.nn.Conv1d(
#             in_channels=channels,
#             out_channels=channels,
#             kernel_size=1
#         )
#
#         with torch.no_grad():
#             self.custom_layer.weight.copy_(torch.eye(channels).unsqueeze(-1))
#             self.custom_layer.bias.zero_()
#
#     def forward(self, x):
#         features = self.original_net(x)
#         return self.custom_layer(features)
#
# """
# # Usage
# encoder_output_channels = conv_layer.out_channels
# torch.manual_seed(0)
# model.encoder.encoder.net = CustomEncoderWrapper(
#     original_encoder_net,
#     encoder_output_channels
# )
#
# print(model.encoder.encoder.net)
# """
