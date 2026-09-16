from collections.abc import Iterator
from typing import Protocol, cast

import torch
from torch import nn

from lib import hasattr_from_attr_string


class IterableModule(Protocol):
    def __iter__(self) -> Iterator[nn.Module]: ...
    def __getitem__(self, idx: int) -> nn.Module: ...


def is_layer_iterable(net: nn.Module | IterableModule) -> bool:
    try:
        net_cast = cast(IterableModule, net)
        net_cast[0]
    except Exception:
        return False
    return True


# TODO:
# this should be the responsibility of the models tbh
# maybe make the models itself
def get_input_size(net: IterableModule) -> int:
    first_layer = net[0]

    # intentionally not writing this more cleanly
    # so that we can see the path here
    if hasattr(first_layer, "in_channels"):
        # used in RAVE
        return first_layer.in_channels
    elif hasattr_from_attr_string(first_layer, "conv.in_channels"):
        # used in HF Encodec
        return first_layer.conv.in_channels
    elif hasattr_from_attr_string(first_layer, "conv.conv.in_channels"):
        # used in Encodec
        return first_layer.conv.conv.in_channels
    else:
        print("Number of input channels unknown!")
        exit()


def get_shape_preserving_layers(net: IterableModule):
    """
    Returns information about every layer that preserves the input shape.
    Input:  nn.Module (needs to be iterable!)
    Output: List of dicts with content {index, name}
    """
    # TODO: make this recursive, so that we can check nets of sub-nets

    if not is_layer_iterable(net):
        print("Model should be sequential!")
        exit()

    results = []

    input_size: int = get_input_size(net)

    # batch=1, channels=input_size, time=64
    x = torch.zeros(1, input_size, 64)
    for idx, layer in enumerate(net):
        layer_name = type(layer).__name__
        try:
            with torch.no_grad():
                out = layer(x)
                if isinstance(out, tuple):
                    # EncodecLSTM returns tuple (output, (h_n, c_n))
                    out = out[0]
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
