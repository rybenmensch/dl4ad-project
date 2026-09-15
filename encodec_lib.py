from enum import Enum
from typing import TypeAlias, cast

import torch
import torch.nn as nn
from encodec.model import EncodecModel
from transformers import EncodecModel as HFEncodecModel

from lib import (
    getattr_from_attr_string,
    hasattr_from_attr_string,
    setattr_from_attr_string,
)
from model import Net, NetTypeEnum, NNModel
from torch_lib import is_layer_iterable

LayerSequence: TypeAlias = nn.ModuleList


class EncodecNNModel(NNModel):
    def __init__(self, model: EncodecModel):
        super(EncodecNNModel, self).__init__(model)
        self.subnet_name = "model"

    def get_net_path(self, net_type: NetTypeEnum) -> str:
        """Returns the path of the net."""
        return ".".join([net_type.value, self.subnet_name])

    def get_net(self, net_type: NetTypeEnum) -> LayerSequence:
        """Returns the net."""
        net_path = self.get_net_path(net_type)
        return getattr_from_attr_string(self.model, net_path)

    def set_net(self, net_type: NetTypeEnum, net: Net):
        """Update the net."""
        net_path = self.get_net_path(net_type)
        setattr_from_attr_string(self.model, net_path, net)


class HFEncodecNNModel(EncodecNNModel):
    def __init__(self, model: HFEncodecModel):
        super(HFEncodecNNModel, self).__init__(cast(EncodecModel, model))
        self.subnet_name = "layers"


def encodec_from_hf(
    model_name: str = "facebook/encodec_24khz",
) -> HFEncodecModel:
    model = HFEncodecModel.from_pretrained(model_name)
    model.eval()
    return model


def encodec_model_48khz() -> EncodecModel:
    model = EncodecModel.encodec_model_48khz()
    model.eval()
    return model


def encodec_model_24khz() -> EncodecModel:
    model = EncodecModel.encodec_model_24khz()
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
