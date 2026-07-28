from from_net_type import NetType
from model import (
    Model,
    Module,
    Net,
    get_decoder_net,
    get_decoder_net_path,
    get_encoder_net,
    get_encoder_net_path,
)


def get_net_type(model: Model, net_path: str) -> NetType:
    if net_path == get_encoder_net_path(model):
        return NetType.Encoder
    elif net_path == get_decoder_net_path(model):
        return NetType.Decoder
    raise Exception(f"Unsupported NetType f{net_path}!")


def get_net(model: Model, net_path: str) -> Net:
    if net_path == get_encoder_net_path(model):
        return get_encoder_net(model)
    elif net_path == get_decoder_net_path(model):
        return get_decoder_net(model)
    raise Exception(f"Unsupported NetType f{net_path}!")


# a bit ridiculous
def get_layer_from_index(model: Model, net_path: str, index: int) -> Module:
    net: Net = get_net(model, net_path)
    return net[index]
