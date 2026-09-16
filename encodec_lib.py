from typing import Any, TypeAlias, cast

import torch
import torch.nn as nn
from encodec.model import EncodecModel
from encodec.modules.conv import SConv1d
from rave import Tuple

from lib import convert_audio
from model import NetTypeEnum, NNModel

Conv1d: TypeAlias = nn.Conv1d


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

    def get_channels(self) -> int:
        return self.model.channels

    def get_first_layer(self, net_type: NetTypeEnum) -> Conv1d:
        first_layer = cast(SConv1d, self.get_net(net_type)[0])
        return cast(Conv1d, first_layer.conv.conv)

    def process_audio(self, audio_sr: Tuple[torch.Tensor, int]) -> torch.Tensor:
        audio = convert_audio(audio_sr, self.get_sample_rate(), self.get_channels())
        audio = audio.unsqueeze(0)
        return self.model(audio).squeeze(0)

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
