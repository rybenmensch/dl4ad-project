from pathlib import Path
from typing import TypeAlias

import cached_conv
import gin
import rave
import torch
import torch.nn as nn

from lib import get_attr_from_attr_string, get_in_channels_from_state_dict
from model import Net, NNModel

Conv1d: TypeAlias = cached_conv.convs.Conv1d | cached_conv.convs.CachedConv1d
CachedSequential: TypeAlias = cached_conv.convs.CachedSequential

# In here, just stuff to interface with RAVE models and components!


class RAVEModel(NNModel):
    def __init__(self, model):
        super(RAVEModel, self).__init__(model)

    def get_encoder_net_path(self) -> str:
        """Returns the path of the encoder net"""
        encoder_str = "encoder"
        if hasattr(self.model.encoder, "encoder"):
            encoder_str += ".encoder"
        return encoder_str + ".net"

    def get_decoder_net_path(self) -> str:
        """Returns the path of the decoder net."""
        decoder_str = "decoder"
        if hasattr(self.model.decoder, "decoder"):
            decoder_str += ".decoder"
        return decoder_str + ".net"

    def get_encoder_net(self) -> CachedSequential:
        """Returns the encoder net."""
        net_path = self.get_encoder_net_path()
        return get_attr_from_attr_string(self.model, net_path)

    def get_decoder_net(self) -> CachedSequential:
        """Returns the decoder net."""
        net_path = self.get_decoder_net_path()
        return get_attr_from_attr_string(self.model, net_path)

    def set_encoder_net(self, net: Net):
        """
        Returns the model with updated encoder_net.
        Exists because the topology can change between RAVE updates, in which case
        we will update the setter here.
        Also exists because of a kludge and will maybe possibly be removed
        """
        if hasattr(self.model.encoder, "encoder"):
            self.model.encoder.encoder.net = net
        else:
            self.model.encoder.net = net

    def set_decoder_net(self, net: Net):
        """
        Returns the model with updated decoder_net.
        Exists because the topology can change between RAVE updates, in which case
        we will update the getter here.
        Also exists because of a kludge and will maybe possibly be removed
        """
        if hasattr(self.model.decoder, "decoder"):
            self.model.decoder.decoder.net = net
        else:
            self.model.decoder.net = net


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
