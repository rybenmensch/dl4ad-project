from pathlib import Path

import gin
import rave
import torch
import torch.nn as nn

from lib import get_in_channels_from_state_dict

# In here, just stuff to interface with RAVE models and components!

# still have to think about how to separate these things out??
# because most everything here does not just apply to RAVE
# but could be used for any model (I guess)
# but how to interface is kind of the question
# the cheapest way would just be to do something like
#
# when we want to use RAVE:
# from rave_lib import get_encoder_net_path, get_decoder_net_path
#
# when we want to use blabla:
# from blabla_lib import get_encoder_net_path, get_decoder_net_path
# but obviously this is very very restricted
#
# The approach might just have to be some kind of base class thing. Thin
# wrapper around the model, has the all the functions that are not specific to
# rave as normal member functions, and then get_encoder_net_path and
# get_decoder_net_path as virtual functions depending on the type of model.


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
