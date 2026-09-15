from pathlib import Path
from typing import List, TypeAlias, cast

import cached_conv
import gin
import rave
import torch
import torch.nn as nn

from lib import (
    get_in_channels_from_state_dict,
    getattr_from_attr_string,
    setattr_from_attr_string,
)
from model import Net, NetTypeEnum, NNModel

Conv1d: TypeAlias = cached_conv.convs.Conv1d | cached_conv.convs.CachedConv1d
LayerSequence: TypeAlias = cached_conv.convs.CachedSequential

# In here, just stuff to interface with RAVE models and components!


class RAVEModel(NNModel):
    def __init__(self, path_or_model: Path | str | rave.RAVE):
        self.path: Path | str | None
        model = None
        if isinstance(path_or_model, Path) or isinstance(path_or_model, str):
            self.path = path_or_model
            model = raw_rave_model(path_or_model)
        super(RAVEModel, self).__init__(model)
        self.model: rave.RAVE

    def reset(self):
        if self.path == None:
            print(
                "Cannot reset model that was initialized without a path to a checkpoint!"
            )
            exit()
        self.model = raw_rave_model(self.path)

    def get_sample_rate(self) -> int:
        return self.model.sr

    def get_first_layer(self, net_type: NetTypeEnum) -> Conv1d:
        return cast(Conv1d, self.get_net(net_type)[0])

    def get_net_path(self, net_type: NetTypeEnum) -> str:
        """Returns the path of the net."""
        net_type_name = net_type.value
        net = getattr(self.model, net_type_name)
        path_str = net_type_name  # 'encoder' or 'decoder'
        if hasattr(net, net_type_name):
            # name is nested, e.g. 'decoder.decoder'
            path_str += f".{net_type_name}"
        return path_str + ".net"


def raw_rave_model(run_path: Path | str) -> rave.RAVE:
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


# GRAVEYARD


# def get_weighted_layers(net: nn.Module):
#     pass


# def get_last_encoder_layer(model: rave.RAVE) -> cached_conv.convs.Conv1d:
#     return model.encoder.encoder.net[-1]


# def get_encoder_output_channels(model: rave.RAVE) -> int:
#     return model.encoder.encoder.net[-1].out_channels
