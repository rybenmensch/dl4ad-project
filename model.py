from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Tuple, TypeAlias

from torch import nn

# TODO: maybe make more specific
Net: TypeAlias = nn.Module
Module: TypeAlias = nn.Module


class VirtualException(BaseException):
    def __init__(self):
        BaseException(self)


class NNModel(metaclass=ABCMeta):
    """wrapper class for non-torchscript models"""

    def __init__(self, model):
        self.model = model
        self.from_net_type = NetType(self)
        self.from_net_path = NetPath(self)
        self.from_layer_path = LayerPath(self)

    @abstractmethod
    def get_encoder_net_path(self) -> str:
        pass

    @abstractmethod
    def get_decoder_net_path(self) -> str:
        pass

    @abstractmethod
    def get_encoder_net(self) -> Net:
        pass

    @abstractmethod
    def get_decoder_net(self) -> Net:
        pass

    @abstractmethod
    def set_decoder_net(self, net: Net):
        pass

    @abstractmethod
    def set_encoder_net(self, net: Net):
        pass

    def get_nets(self) -> Tuple[
        Net,
        Net,
    ]:
        """Returns the encoder and decoder nets."""
        return (self.get_encoder_net(), self.get_decoder_net())

    def get_net_paths(self) -> Tuple[str, str]:
        """Returns the encoder and decoder paths."""
        return (self.get_encoder_net_path(), self.get_decoder_net_path())

    def get_nets_and_paths(self) -> Tuple[
        Tuple[Net, str],
        Tuple[Net, str],
    ]:
        """Returns the encoder and decoder nets and the paths to them."""
        return (
            (self.get_encoder_net(), self.get_encoder_net_path()),
            (self.get_decoder_net(), self.get_decoder_net_path()),
        )


class NetTypeEnum(Enum):
    Encoder = 0
    Decoder = 1


class NetType:
    def __init__(self, model: NNModel):
        self.model = model

    def get_net_path(self, net_type: NetTypeEnum) -> str:
        if net_type == NetTypeEnum.Encoder:
            return self.model.get_encoder_net_path()
        elif net_type == NetTypeEnum.Decoder:
            return self.model.get_decoder_net_path()

    def get_net(self, net_type: NetTypeEnum) -> Net:
        if net_type == NetTypeEnum.Encoder:
            return self.model.get_encoder_net()
        elif net_type == NetTypeEnum.Decoder:
            return self.model.get_decoder_net()

    def get_layer_from_index(self, net_type: NetTypeEnum, index: int) -> Module:
        net = self.get_net(net_type)
        return net[index]


class NetPath:
    def __init__(self, model: NNModel):
        self.model = model

    def get_net_type(self, net_path: str) -> NetTypeEnum:
        if net_path == self.model.get_encoder_net_path():
            return NetTypeEnum.Encoder
        elif net_path == self.model.get_decoder_net_path():
            return NetTypeEnum.Decoder
        raise Exception(f"Unsupported NetType f{net_path}!")

    def get_net(self, net_path: str) -> Net:
        net_type: NetTypeEnum = self.get_net_type(net_path)
        if net_type == NetTypeEnum.Encoder:
            return self.model.get_encoder_net()
        elif net_type == NetTypeEnum.Decoder:
            return self.model.get_decoder_net()

    def get_layer(self, net_path: str, index: int) -> Module:
        net: Net = self.get_net(net_path)
        return net[index]


class LayerPath:
    def __init__(self, model: NNModel):
        self.model = model

    def get_net_type(self, layer_path: str) -> NetTypeEnum:
        if layer_path.startswith(self.model.get_encoder_net_path()):
            return NetTypeEnum.Encoder
        elif layer_path.startswith(self.model.get_decoder_net_path()):
            return NetTypeEnum.Decoder
        raise Exception("Unsupported net type!")

    def get_net(self, layer_path: str) -> Net:
        """Returns the net that the layer corresponding to the path belongs to."""
        net_type: NetTypeEnum = self.get_net_type(layer_path)
        if net_type == NetTypeEnum.Encoder:
            return self.model.get_encoder_net()
        elif net_type == NetTypeEnum.Decoder:
            return self.model.get_decoder_net()

    def get_net_path(self, layer_path: str) -> str:
        """
        Returns the net path that the layer corresponding to the path belongs to.
        """
        net_type: NetTypeEnum = self.get_net_type(layer_path)
        if net_type == NetTypeEnum.Encoder:
            return self.model.get_encoder_net_path()
        elif net_type == NetTypeEnum.Decoder:
            return self.model.get_decoder_net_path()

    def get_layer_index(self, layer_path: str) -> int:
        """Returns the index of the layer in the corresponding net."""
        net_path: str = self.get_net_path(layer_path)
        # + 1 to get rid of the dot in front of the index
        start = len(net_path) + 1
        return int(layer_path[start:])

    def get_layer(self, layer_path: str) -> Module:
        """Returns the layer corresponding to the path."""
        net: Net = self.get_net(layer_path)
        index: int = self.get_layer_index(layer_path)
        return net[index]
