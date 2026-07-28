from typing import Any, Tuple, TypeAlias

import cached_conv
import rave
from torch import nn

from lib import get_attr_from_attr_string

CachedSequential: TypeAlias = cached_conv.convs.CachedSequential


class RAVEWrapper:
    def __init__(self, model):
        self.model = model

    def get_encoder_net_path(self) -> str:
        """Returns the path of the encoder net"""
        encoder_str = "encoder"
        if hasattr(self.model.encoder, "encoder"):
            encoder_str += ".encoder"
        return encoder_str + ".net"

    def get_decoder_net_path(self) -> str:
        """Returns the path of the decoder net."""
        decoder_str = "decoder"
        if hasattr(self.model.decoder, "decoder"):
            decoder_str += ".decoder"
        return decoder_str + ".net"

    def get_encoder_net(self) -> CachedSequential:
        """Returns the encoder net."""
        net_path = self.get_encoder_net_path()
        return get_attr_from_attr_string(self.model, net_path)

    def get_decoder_net(self) -> CachedSequential:
        """Returns the decoder net."""
        net_path = self.get_decoder_net_path()
        return get_attr_from_attr_string(self.model, net_path)


class OtherWrapper:
    def __init__(self, model):
        self.model = model

    def get_encoder_net_path(self) -> str:
        raise NotImplemented

    def get_decoder_net_path(self) -> str:
        raise NotImplemented

    def get_encoder_net(self) -> CachedSequential:
        # would return whatever layer type it is
        raise NotImplemented

    def get_decoder_net(self) -> CachedSequential:
        raise NotImplemented


# TODO: add to the wrapper
def set_encoder_net(model: rave.RAVE, net: Any) -> rave.RAVE:
    """
    Returns the model with updated encoder_net.
    Exists because the topology can change between RAVE updates, in which case
    we will update the setter here.
    Also exists because of a kludge and will maybe possibly be removed
    """
    if hasattr(model.encoder, "encoder"):
        model.encoder.encoder.net = net
    else:
        model.encoder.net = net
    return model


# TODO: add to the wrapper
def set_decoder_net(model: rave.RAVE, net: Any) -> rave.RAVE:
    """
    Returns the model with updated decoder_net.
    Exists because the topology can change between RAVE updates, in which case
    we will update the getter here.
    Also exists because of a kludge and will maybe possibly be removed
    """
    if hasattr(model.decoder, "decoder"):
        model.decoder.decoder.net = net
    else:
        model.decoder.net = net
    return model


Model: TypeAlias = RAVEWrapper | OtherWrapper
Net: TypeAlias = CachedSequential
Module: TypeAlias = nn.Module | cached_conv.convs.Conv1d


def get_encoder_net_path(model: Model) -> str:
    return model.get_encoder_net_path()


def get_decoder_net_path(model: Model) -> str:
    return model.get_decoder_net_path()


def get_encoder_net(model: Model) -> CachedSequential:
    return model.get_encoder_net()


def get_decoder_net(model: Model) -> CachedSequential:
    return model.get_decoder_net()


def get_nets(
    model: Model,
) -> Tuple[
    CachedSequential,
    CachedSequential,
]:
    """Returns the encoder and decoder nets."""
    return (get_encoder_net(model), get_decoder_net(model))


def get_net_paths(model: Model) -> Tuple[str, str]:
    """Returns the encoder and decoder paths."""
    return (get_encoder_net_path(model), get_decoder_net_path(model))


def get_nets_and_paths(
    model: Model,
) -> Tuple[
    Tuple[CachedSequential, str],
    Tuple[CachedSequential, str],
]:
    """Returns the encoder and decoder nets and the paths to them."""
    return (
        (get_encoder_net(model), get_encoder_net_path(model)),
        (get_decoder_net(model), get_decoder_net_path(model)),
    )
