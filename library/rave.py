import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cached_conv as cc
import gin
import rave
import rave.core
import torch
from cached_conv.convs import CachedSequential, Conv1d, ConvTranspose1d
from rave import Residual
from scripts import export as rave_export
from torch import nn
from torch.nn import LeakyReLU
from torch.nn.utils import remove_weight_norm

from library.model import NetTypeEnum, NNModel, WeightAndBias
from library.torch_utils import unwrap_layer


@dataclass
class ExportOptions:
    path: Path
    fidelity: float = 0.99
    streaming: bool = True


class RAVEModel(NNModel):
    def __init__(self, path_or_model: Path | str | rave.RAVE):
        self.path: Path | str | None
        model = None
        if isinstance(path_or_model, (Path, str)):
            self.path = path_or_model
            model = raw_rave_model(path_or_model)
        super().__init__(model)
        self.model: rave.RAVE

    def reset(self) -> None:
        if self.path == None:
            print(
                "Cannot reset model that was initialized without a path to a checkpoint!"
            )
            exit()
        self.model = raw_rave_model(self.path)

    def export(self, options: ExportOptions) -> None:
        export_rave_model(self, options)

    def get_sample_rate(self) -> int:
        return self.model.sr

    def get_channels(self) -> int:
        return self.model.n_channels

    def layer_get_channels(self, layer: nn.Module) -> tuple[int, int] | None:
        """Returns `None` if layer accepts any input/output size."""
        layer = unwrap_layer(layer)
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
        layer = unwrap_layer(layer)
        return isinstance(layer, Residual)

    def layer_has_weights(self, layer: nn.Module) -> bool:
        layer = unwrap_layer(layer)
        return not isinstance(layer, LeakyReLU)

    def layer_get_weight_and_bias(self, layer: nn.Module) -> list[WeightAndBias]:
        layer = unwrap_layer(layer)
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


def rave_get_in_channels_from_state_dict(state_dict: dict) -> int:
    """
    Find the input channels of the encoder's first conv layer from state_dict
    to detect n_channels. We need this because at this point, we do not have a
    RAVE model yet, only a state_dict!
    """
    in_channels = None

    # first conv layer of state_dict might have a different 'path' depending on
    # model version
    for key in [
        "encoder.encoder.net.0.weight_v",
        "encoder.net.0.weight_v",
        "encoder.encoder.net.0.weight",
    ]:
        if key in state_dict:
            in_channels = state_dict[key].shape[1]
            break

    # actual amount of input channels is input_channels // number of pqmf bands
    if in_channels is not None:
        try:
            n_band = gin.query_parameter("%N_BAND")
        except ValueError:
            # assume 16 bands as in the paper if the parameter isn't found
            n_band = 16
        return in_channels // n_band
    else:
        return 1


def raw_rave_model(run_path: Path | str) -> rave.RAVE:
    """Create a full RAVE model from the path to a run."""

    if isinstance(run_path, Path):
        run_path = run_path.as_posix()
    config_file = rave.core.search_for_config(run_path)
    if config_file is None:
        raise FileNotFoundError(f"No config file found at path: {run_path}")
    gin.parse_config_file(config_file)

    checkpoint_path = rave.core.search_for_run(run_path)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint file found at path: {run_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    n_channels = rave_get_in_channels_from_state_dict(state_dict)

    cc.use_cached_conv(True)
    model = rave.RAVE(n_channels=n_channels)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def export_rave_model(model: NNModel, options: ExportOptions) -> None:
    cc.use_cached_conv(True)

    pretrained = model.model
    pretrained.eval()

    if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
        script_class = rave_export.VariationalScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.DiscreteEncoder):
        script_class = rave_export.DiscreteScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.WasserteinEncoder):
        script_class = rave_export.WasserteinScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.SphericalEncoder):
        script_class = rave_export.SphericalScriptedRAVE
    else:
        raise ValueError(
            f"Encoder type not supported for export: {type(pretrained.encoder)}"
        )

    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)

    scripted_rave = script_class(
        pretrained=pretrained,
        channels=model.get_channels(),
        prior=prior_scripted,
        fidelity=options.fidelity,
    )

    x = torch.zeros(1, pretrained.n_channels, 2**14)
    z = scripted_rave.encode(x)
    x = scripted_rave.decode(z)

    scripted_rave.export_to_ts(options.path)

    try:
        if pretrained.n_channels <= 2:
            # test stereo mode for VST export
            scripted_rave.set_stereo_mode(True)
            z_vst_input = torch.zeros(2, scripted_rave.full_latent_size, z.shape[-1])
            out = scripted_rave.decode(z_vst_input)
            assert out.shape[1] == 2, "model output is not stereo"
    except Exception as e:
        print("this model will not work with the RAVE VST. \n Caught error : %s" % e)
