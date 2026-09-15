from enum import Enum
from typing import TypeAlias, cast

import encodec
import torch
import torch.nn as nn
from encodec.model import EncodecModel
from encodec.modules.conv import SConv1d
from transformers import EncodecModel as HFEncodecModel
from transformers.models.encodec.modeling_encodec import EncodecConv1d

from lib import (
    getattr_from_attr_string,
    hasattr_from_attr_string,
    print_all_attrs,
    setattr_from_attr_string,
)
from model import Net, NetTypeEnum, NNModel
from torch_lib import is_layer_iterable


class EncodecNNModel(NNModel):
    def __init__(self, model: EncodecModel | None = None) -> None:
        if model == None:
            model = raw_encodec_model(48_000)
        super(EncodecNNModel, self).__init__(model)
        self.model: EncodecModel

    def reset(self) -> None:
        self.model = raw_encodec_model(self.get_sample_rate())

    def get_sample_rate(self) -> int:
        return self.model.sample_rate

    def get_first_layer(self, net_type: NetTypeEnum) -> nn.Conv1d:
        first_layer = cast(SConv1d, self.get_net(net_type)[0])
        return cast(nn.Conv1d, first_layer.conv.conv)

    def get_net_path(self, net_type: NetTypeEnum) -> str:
        """Returns the path of the net."""
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


class HFEncodecNNModel(NNModel):
    def __init__(self, model: HFEncodecModel | None = None) -> None:
        if model == None:
            model = raw_hf_encodec_model()
        super(HFEncodecNNModel, self).__init__(cast(EncodecModel, model))
        self.model: HFEncodecModel

    def reset(self) -> None:
        sr: int = self.model.config.sampling_rate
        self.model = raw_hf_encodec_model(self.get_sample_rate())

    def get_sample_rate(self) -> int:
        return self.model.config.sampling_rate

    def get_first_layer(self, net_type: NetTypeEnum) -> nn.Conv1d:
        first_layer = cast(EncodecConv1d, self.get_net(net_type)[0])
        return cast(nn.Conv1d, first_layer.conv)

    def get_net_path(self, net_type: NetTypeEnum) -> str:
        """Returns the path of the net."""
        return net_type.value + ".layers"


def raw_hf_encodec_model(sample_rate: int = 48_000) -> HFEncodecModel:
    model: HFEncodecModel | None = None
    sr: int
    if sample_rate == 24_000 or sample_rate == 48_000:
        sr = sample_rate // 1000
    else:
        print(f"Samplerate {sample_rate} not supported!")
        exit()
    model = HFEncodecModel.from_pretrained(f"facebook/encodec_{sr}khz")
    model.eval()
    return model


def process_audio(
    model, waveform: torch.Tensor, bandwidth: float | None = None
) -> torch.Tensor:
    """
    Adapter fuer HF EncodecModel: forward() gibt ein EncodecOutput-Objekt
    zurueck (mit .audio_values), keinen direkten Tensor wie RAVE.

    bandwidth: Ziel-Bitrate in kbps. None -> hoechste verfuegbare Qualitaet
    """
    if bandwidth is None:
        bandwidth = max(model.config.target_bandwidths)

    input_tensor = waveform.unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor, bandwidth=bandwidth)
    return output.audio_values.squeeze(0)
