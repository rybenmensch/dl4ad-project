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


def get_net_type(model: Model, layer_path: str) -> NetType:
    if layer_path.startswith(get_encoder_net_path(model)):
        return NetType.Encoder
    elif layer_path.startswith(get_decoder_net_path(model)):
        return NetType.Decoder
    raise Exception("Unsupported net type!")


def get_net(model: Model, layer_path: str) -> Net:
    """Returns the net that the layer corresponding to the path belongs to."""
    net_type: NetType = get_net_type(model, layer_path)

    if net_type == NetType.Encoder:
        return get_encoder_net(model)
    elif net_type == NetType.Decoder:
        return get_decoder_net(model)


def get_net_path(model: Model, layer_path: str) -> str:
    """
    Returns the net path that the layer corresponding to the path belongs to.
    """
    net_type: NetType = get_net_type(model, layer_path)
    if net_type == NetType.Encoder:
        return get_encoder_net_path(model)
    elif net_type == NetType.Decoder:
        return get_decoder_net_path(model)


def get_layer_index(model: Model, layer_path: str) -> int:
    """Returns the index of the layer in the corresponding net."""
    net_path: str = get_net_path(model, layer_path)
    # + 1 to get rid of the dot in front of the index
    start = len(net_path) + 1
    return int(layer_path[start:])


def get_layer(model: Model, layer_path: str) -> Module:
    """Returns the layer corresponding to the path."""
    net: Net = get_net(model, layer_path)
    index: int = get_layer_index(model, layer_path)
    return net[index]
