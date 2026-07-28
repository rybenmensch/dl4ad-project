from enum import Enum

from model import (
    Model,
    Module,
    Net,
    get_decoder_net,
    get_decoder_net_path,
    get_encoder_net,
    get_encoder_net_path,
)


class NetType(Enum):
    Encoder = 0
    Decoder = 1


def get_net_path(model: Model, net_type: NetType) -> str:
    if net_type == NetType.Encoder:
        return get_encoder_net_path(model)
    elif net_type == NetType.Decoder:
        return get_decoder_net_path(model)


def get_net(model: Model, net_type: NetType) -> Net:
    if net_type == NetType.Encoder:
        return get_encoder_net(model)
    elif net_type == NetType.Decoder:
        return get_decoder_net(model)


# a bit ridiculous
def get_layer_from_index(model: Model, net_type: NetType, index: int) -> Module:
    net = get_net(model, net_type)
    return net[index]
