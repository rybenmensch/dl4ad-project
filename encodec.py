from typing import TypeAlias

import torch
import torch.nn as nn
from transformers import EncodecModel as HFEncodecModel

from lib import get_attr_from_attr_string
from model import Net, NetTypeEnum, NNModel

# EnCodec-Layer liegen als nn.ModuleList vor (nicht als CachedSequential wie bei RAVE)
LayerSequence: TypeAlias = nn.ModuleList



class EncodecNNModel(NNModel):
    def __init__(self, model):
        super(EncodecNNModel, self).__init__(model)

    def get_net_path(self, net_type: NetTypeEnum) -> str:
        """Returns the path of the net"""
        comp_name = net_type.value  # "encoder" oder "decoder"
        comp = getattr(self.model, comp_name)
        path_str = comp_name
        # Bei EnCodec (transformers) gibt es KEINE Verschachtelung wie bei
        # RAVE (kein encoder.encoder) - der Check bleibt trotzdem drin,
        # falls sich das in einer anderen transformers-Version aendert.
        if hasattr(comp, comp_name):
            path_str += f".{comp_name}"
        return path_str + ".layers"

    def get_net(self, net_type: NetTypeEnum) -> LayerSequence:
        """Returns the net."""
        net_path = self.get_net_path(net_type)
        return get_attr_from_attr_string(self.model, net_path)

    def set_net(self, net_type: NetTypeEnum, net: Net):
        """Setzt eine neue Layer-Liste (nn.ModuleList) als Encoder/Decoder ein."""
        comp_name = net_type.value
        comp = getattr(self.model, comp_name)
        if hasattr(comp, comp_name):
            getattr(comp, comp_name).layers = net
        else:
            comp.layers = net



def encodec_from_pretrained(model_name: str = "facebook/encodec_24khz") -> HFEncodecModel:
    """Laedt Architektur + Gewichte in einem Schritt von Hugging Face."""
    model = HFEncodecModel.from_pretrained(model_name)
    model.eval()
    return model


def get_shape_preserving_layers(net: nn.Module):
    """
    Returns information about every layer that preserves the input shape.
    Input:  nn.ModuleList (Encoder- oder Decoder-Layer)
    Output: List of dicts with content {index, name}
    """
    results = []
    try:
        net[0]
    except Exception:
        print("Model should be sequential!")
        exit()

    # EncodecConv1d ist ein Wrapper um nn.Conv1d - in_channels liegt daher
    # unter .conv, nicht direkt auf dem Layer selbst.
    first_layer = net[0]
    if hasattr(first_layer, "conv"):
        input_size = first_layer.conv.in_channels
    else:
        input_size = first_layer.in_channels

    x = torch.zeros(1, input_size, 64)
    for idx, layer in enumerate(net):
        layer_name = type(layer).__name__
        try:
            with torch.no_grad():
                out = layer(x)
                # EncodecLSTM gibt ein Tuple (output, (h_n, c_n)) zurueck
                if isinstance(out, tuple):
                    out = out[0]
                if out.shape == x.shape:
                    results.append({"index": idx, "name": layer_name})
                x = out
        except Exception as e:
            print(f"Layer nr {idx} of type {layer_name} raised {e}")
    return results
