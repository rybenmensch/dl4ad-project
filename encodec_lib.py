from typing import cast

import torch
from encodec.model import EncodecModel
from encodec.modules.seanet import SLSTM, SConv1d, SConvTranspose1d, SEANetResnetBlock
from torch import Tensor, nn
from torch.nn import ELU, Conv1d, ConvTranspose1d, Sequential
from torch.nn.utils import remove_weight_norm

from model import NetTypeEnum, NNModel, WeightAndBias


class EncodecNNModel(NNModel):
    def __init__(self, model: EncodecModel | None = None) -> None:
        if model == None:
            model = raw_encodec_model(48_000)
        super().__init__(model)
        self.model: EncodecModel

    def reset(self) -> None:
        self.model = raw_encodec_model(self.get_sample_rate())

    def get_sample_rate(self) -> int:
        return self.model.sample_rate

    def get_channels(self) -> int:
        return self.model.channels

    def layer_get_channels(self, layer: nn.Module) -> tuple[int, int] | None:
        if isinstance(layer, SConv1d):
            conv = cast(Conv1d, layer.conv.conv)
            return (conv.in_channels, conv.out_channels)
        elif isinstance(layer, SConvTranspose1d):
            conv = cast(ConvTranspose1d, layer.convtr.convtr)
            return (conv.in_channels, conv.out_channels)
        elif isinstance(layer, SEANetResnetBlock):
            net = cast(Sequential, layer.block)
            if (ch_in := self.layer_get_channels(net[1])) and (
                ch_out := self.layer_get_channels(net[3])
            ):
                return (ch_in[0], ch_out[1])
        elif isinstance(layer, SLSTM):
            lstm = layer.lstm
            return (lstm.input_size, lstm.input_size)
        elif isinstance(layer, ELU):
            return None
        else:
            return super().layer_get_channels(layer)

    def layer_has_subnet(self, layer: nn.Module) -> bool:
        return isinstance(layer, (SConv1d, SEANetResnetBlock))

    def layer_has_weights(self, layer: nn.Module) -> bool:
        return isinstance(layer, (SConv1d, SConvTranspose1d, SEANetResnetBlock))

    def layer_get_weight_and_bias(self, layer: nn.Module) -> list[WeightAndBias]:
        try:
            remove_weight_norm(layer)
        except (ValueError, AttributeError):
            pass

        if isinstance(layer, SConv1d):
            conv = cast(Conv1d, layer.conv.conv)
            bias = conv.bias if conv.bias != None else torch.empty((0, 0))
            return [WeightAndBias(weight=conv.weight, bias=bias)]
        elif isinstance(layer, SConvTranspose1d):
            conv = cast(ConvTranspose1d, layer.convtr.convtr)
            bias = conv.bias if conv.bias != None else torch.empty((0, 0))
            return [WeightAndBias(weight=conv.weight, bias=bias)]
        elif isinstance(layer, SEANetResnetBlock):
            net = cast(Sequential, layer.block)
            all = []
            for l in net:
                all += self.layer_get_weight_and_bias(l)
            return all
        elif isinstance(layer, (SLSTM, ELU)):
            return [WeightAndBias(weight=torch.empty((0, 0)), bias=torch.empty((0, 0)))]
        else:
            return super().layer_get_weight_and_bias(layer)

    def get_net_path(self, net_type: NetTypeEnum) -> str:
        """Returns the path of the net specified by net_type."""
        return net_type.value + ".model"


def raw_encodec_model(sample_rate: int = 48_000) -> EncodecModel:
    model: EncodecModel | None = None
    if sample_rate == 24_000:
        model = EncodecModel.encodec_model_24khz()
    elif sample_rate == 48_000:
        model = EncodecModel.encodec_model_48khz()
    else:
        print(f"Samplerate {sample_rate} not supported!")
        exit()
    model.eval()
    return model
