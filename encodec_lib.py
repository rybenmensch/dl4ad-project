from typing import cast

from encodec.model import EncodecModel
from encodec.modules.seanet import SLSTM, SConv1d, SConvTranspose1d, SEANetResnetBlock
from torch import nn
from torch.nn import LSTM, Conv1d, ConvTranspose1d, Sequential

from model import NetTypeEnum, NNModel


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

    def get_layer_channels(self, layer: nn.Module) -> tuple[int, int] | None:
        if isinstance(layer, SConv1d):
            conv = cast(Conv1d, layer.conv.conv)
            return (conv.in_channels, conv.out_channels)
        elif isinstance(layer, SConvTranspose1d):
            conv = cast(ConvTranspose1d, layer.convtr.convtr)
            return (conv.in_channels, conv.out_channels)
        elif isinstance(layer, SEANetResnetBlock):
            net = cast(Sequential, layer.block)
            if (ch_in := self.get_layer_channels(net[1])) and (
                ch_out := self.get_layer_channels(net[3])
            ):
                return (ch_in[0], ch_out[1])
        elif isinstance(layer, SLSTM):
            lstm = layer.lstm
            return (lstm.input_size, lstm.input_size)
        elif isinstance(layer, nn.ELU):
            return None
        else:
            return super().get_layer_channels(layer)

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
