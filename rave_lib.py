from pathlib import Path
from typing import TypeAlias, cast

import cached_conv
import gin
import rave
import torch
from cached_conv.convs import CachedSequential
from rave import Residual
from torch import nn

from lib import get_in_channels_from_state_dict
from model import NetTypeEnum, NNModel
from torch_lib import get_layer_name

Conv1d: TypeAlias = (
    cached_conv.convs.Conv1d
    | cached_conv.convs.CachedConv1d
    | cached_conv.convs.ConvTranspose1d
)


class RAVEModel(NNModel):
    def __init__(self, path_or_model: Path | str | rave.RAVE):
        self.path: Path | str | None
        model = None
        if isinstance(path_or_model, (Path, str)):
            self.path = path_or_model
            model = raw_rave_model(path_or_model)
        super().__init__(model)
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

    def get_channels(self) -> int:
        return self.model.n_channels

    def get_first_layer(self, net_type: NetTypeEnum) -> Conv1d:
        return cast(Conv1d, self.get_net(net_type)[0])

    def get_layer_channels(self, layer: nn.Module) -> tuple[int, int] | None:
        """Returns `None` if layer accepts any input/output size."""
        if isinstance(layer, Conv1d):
            return (layer.in_channels, layer.out_channels)
        elif isinstance(layer, Residual):
            net = cast(CachedSequential, layer.aligned.branches[0].net)
            if (ch_in := self.get_layer_channels(net[1])) and (
                ch_out := self.get_layer_channels(net[3])
            ):
                return (ch_in[0], ch_out[1])
        elif isinstance(layer, nn.LeakyReLU):
            return None
        else:
            return super().get_layer_channels(layer)

    def get_net_path(self, net_type: NetTypeEnum) -> str:
        """Returns the path of the net specified by net_type."""
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
    # TODO: move get_in_channels_from_state_dict to this file
    # as it is specific to RAVE and then rework it to fit
    # the current state of the library
    n_channels = get_in_channels_from_state_dict(state_dict)

    model = rave.RAVE(n_channels=n_channels)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model
