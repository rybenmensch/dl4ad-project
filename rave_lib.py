from pathlib import Path
from typing import TypeAlias

import cached_conv
import gin
import rave
import torch
import torch.nn as nn

from lib import get_attr_from_attr_string, get_in_channels_from_state_dict
from model import Net, NetTypeEnum, NNModel

Conv1d: TypeAlias = cached_conv.convs.Conv1d | cached_conv.convs.CachedConv1d
CachedSequential: TypeAlias = cached_conv.convs.CachedSequential

# In here, just stuff to interface with RAVE models and components!


class RAVEModel(NNModel):
    def __init__(self, model):
        super(RAVEModel, self).__init__(model)

    def get_net_path(self, net_type: NetTypeEnum) -> str:
        """Returns the path of the net"""
        comp_name = net_type.value
        comp = getattr(self.model, comp_name)

        path_str = comp_name
        if hasattr(comp, comp_name):
            path_str += f".{comp_name}"
        return path_str + ".net"

    def get_net(self, net_type: NetTypeEnum) -> CachedSequential:
        """Returns the net."""
        net_path = self.get_net_path(net_type)
        return get_attr_from_attr_string(self.model, net_path)

    def set_net(self, net_type: NetTypeEnum, net: Net):
        """
        Returns the model with updated net.
        Exists because the topology can change between RAVE updates, in which case
        we will update the setter here.
        Also exists because of a kludge and will maybe possibly be removed
        """
        comp_name = net_type.value
        comp = getattr(self.model, comp_name)
        if hasattr(comp, comp_name):
            getattr(comp, comp_name).net = net
        else:
            comp.net = net


# actually specific to RAVE
def rave_from_checkpoint(run_path: Path | str) -> rave.RAVE:
    """Create a full RAVE model from the path to a run."""

    config_file = rave.core.search_for_config(run_path)
    gin.parse_config_file(config_file)

    checkpoint_path = rave.core.search_for_run(run_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint["state_dict"]
    n_channels = get_in_channels_from_state_dict(state_dict)

    model = rave.RAVE(n_channels=n_channels)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def get_shape_preserving_layers(net: nn.Module):
    """
    Returns information about every layer that preserves the input shape.
    Input:
        - CachedSequential net
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


# GRAVEYARD

# def get_last_encoder_layer(model: rave.RAVE) -> cached_conv.convs.Conv1d:
#     return model.encoder.encoder.net[-1]


# def get_encoder_output_channels(model: rave.RAVE) -> int:
#     return model.encoder.encoder.net[-1].out_channels
