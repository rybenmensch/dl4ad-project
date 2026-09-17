from typing import TypeAlias, cast

from encodec.model import EncodecModel
from encodec.modules.conv import SConv1d
from torch import nn

from model import NetTypeEnum, NNModel

Conv1d: TypeAlias = nn.Conv1d


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

    def get_first_layer(self, net_type: NetTypeEnum) -> Conv1d:
        first_layer = cast(SConv1d, self.get_net(net_type)[0])
        return cast(Conv1d, first_layer.conv.conv)

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
