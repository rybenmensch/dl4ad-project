from pathlib import Path
from typing import cast

import gin
import rave
import torch
from cached_conv.convs import CachedSequential, Conv1d, ConvTranspose1d
from rave import Residual
from torch import Tensor, nn
from torch.nn import LeakyReLU
from torch.nn.utils import remove_weight_norm

from lib import get_in_channels_from_state_dict
from model import NetTypeEnum, NNModel, WeightAndBias


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

    def layer_get_channels(self, layer: nn.Module) -> tuple[int, int] | None:
        """Returns `None` if layer accepts any input/output size."""
        if isinstance(layer, (Conv1d, ConvTranspose1d)):
            return (layer.in_channels, layer.out_channels)
        elif isinstance(layer, Residual):
            net = cast(CachedSequential, layer.aligned.branches[0].net)
            if (ch_in := self.layer_get_channels(net[1])) and (
                ch_out := self.layer_get_channels(net[3])
            ):
                return (ch_in[0], ch_out[1])
        elif isinstance(layer, LeakyReLU):
            return None
        else:
            return super().layer_get_channels(layer)

    def layer_has_subnet(self, layer: nn.Module) -> bool:
        return isinstance(layer, Residual)

    def layer_has_weights(self, layer: nn.Module) -> bool:
        return not isinstance(layer, LeakyReLU)

    def layer_get_weight_and_bias(self, layer: nn.Module) -> list[WeightAndBias]:
        try:
            remove_weight_norm(layer)
        except (ValueError, AttributeError):
            pass

        if isinstance(layer, (Conv1d, ConvTranspose1d)):
            bias = layer.bias if layer.bias != None else torch.empty((0, 0))
            return [WeightAndBias(weight=layer.weight, bias=bias)]
        elif isinstance(layer, Residual):
            net = cast(CachedSequential, layer.aligned.branches[0].net)
            all = []
            for l in net:
                all += self.layer_get_weight_and_bias(l)
            return all
        elif isinstance(layer, LeakyReLU):
            return [WeightAndBias(weight=torch.empty((0, 0)), bias=torch.empty((0, 0)))]
        else:
            return super().layer_get_weight_and_bias(layer)

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

    if isinstance(run_path, Path):
        run_path = run_path.as_posix()
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
