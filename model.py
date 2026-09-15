import copy
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Tuple, TypeAlias

from torch import nn

from lib import getattr_from_attr_string, setattr_from_attr_string

# TODO: maybe make more specific
Net: TypeAlias = nn.Module
Module: TypeAlias = nn.Module
LayerSequence: TypeAlias = nn.Module


class NetTypeEnum(str, Enum):
    Encoder = "encoder"
    Decoder = "decoder"


class NNModel(metaclass=ABCMeta):
    """wrapper class for non-torchscript models"""

    def __init__(self, model):
        self.model = model
        self.from_net_type = NetType(self)
        self.from_net_path = NetPath(self)
        self.from_layer_path = LayerPath(self)
        self.from_layer = Layer(self)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        pass

    @abstractmethod
    def get_net_path(self, net_type: NetTypeEnum) -> str:
        """Implement this for looking up the actual path to the encoder or decoder."""
        pass

    def get_net(self, net_type: NetTypeEnum) -> LayerSequence:
        """Returns the net."""
        net_path = self.get_net_path(net_type)
        return getattr_from_attr_string(self.model, net_path)

    def set_net(self, net_type: NetTypeEnum, net: Net):
        """Update the net."""
        net_path = self.get_net_path(net_type)
        setattr_from_attr_string(self.model, net_path, net)

    def get_nets(self) -> Tuple[
        Net,
        Net,
    ]:
        """Returns the encoder and decoder nets."""
        return (self.get_net(NetTypeEnum.Encoder), self.get_net(NetTypeEnum.Decoder))

    def get_net_paths(self) -> Tuple[str, str]:
        """Returns the encoder and decoder paths."""
        return (
            self.get_net_path(NetTypeEnum.Encoder),
            self.get_net_path(NetTypeEnum.Decoder),
        )

    def get_nets_and_paths(self) -> Tuple[
        Tuple[Net, str],
        Tuple[Net, str],
    ]:
        """Returns the encoder and decoder nets and the paths to them."""
        return (
            (self.get_net(NetTypeEnum.Encoder), self.get_net_path(NetTypeEnum.Encoder)),
            (self.get_net(NetTypeEnum.Decoder), self.get_net_path(NetTypeEnum.Decoder)),
        )

    def get_nets_and_types(
        self,
    ) -> Tuple[Tuple[Net, NetTypeEnum], Tuple[Net, NetTypeEnum]]:

        return (
            (self.get_net(NetTypeEnum.Encoder), NetTypeEnum.Encoder),
            (self.get_net(NetTypeEnum.Encoder), NetTypeEnum.Encoder),
        )


class NetType:
    def __init__(self, model: NNModel):
        self.model = model

    def get_net_path(self, net_type: NetTypeEnum) -> str:
        return self.model.get_net_path(net_type)

    def get_net(self, net_type: NetTypeEnum) -> Net:
        return self.model.get_net(net_type)

    def get_layer_from_index(self, net_type: NetTypeEnum, index: int) -> Module:
        net = self.get_net(net_type)
        return net[index]


class NetPath:
    def __init__(self, model: NNModel):
        self.model = model

    def get_net_type(self, net_path: str) -> NetTypeEnum:
        for net_type in NetTypeEnum:
            if net_path == self.model.get_net_path(net_type):
                return net_type
        raise Exception(f"Unsupported NetPath {net_path}!")

    def get_net(self, net_path: str) -> Net:
        net_type = self.get_net_type(net_path)
        return self.model.get_net(net_type)

    def get_layer(self, net_path: str, index: int) -> Module:
        net: Net = self.get_net(net_path)
        return net[index]


class Layer:
    def __init__(self, model: NNModel):
        self.model = model

    def get_net_type(self, layer: Module) -> NetTypeEnum:
        net: Net = self.get_net(layer)
        for n, t in self.model.get_nets_and_types():
            # i have no clue how this equality check works, don't know if
            # that will still work if we have shuffled some things around
            if n == net:
                return t
        raise Exception("Layer not found!")

    def get_net(self, layer: Module) -> Net:
        for net in self.model.get_nets():
            for l in net:
                # i have no clue how this equality check works, don't know if
                # that will still work if we have shuffled some things around
                if l == layer:
                    return net
        raise Exception("Layer not found!")

    def get_net_path(self, layer: Module) -> str:
        net_type: NetTypeEnum = self.get_net_type(layer)
        return self.model.from_net_type.get_net_path(net_type)

    def get_layer_path(self, layer: Module) -> str:
        net_path, index = self.get_net_path_and_index(layer)
        return f"{net_path}.{index}"

    def get_net_path_and_index(self, layer: Module) -> Tuple[str, int]:
        net_path: str = self.get_net_path(layer)
        _, i = self.get_net_type_and_index(layer)
        return (net_path, i)

    def get_net_type_and_index(self, layer: Module) -> Tuple[NetTypeEnum, int]:
        net_type: NetTypeEnum = self.get_net_type(layer)
        for i, l in enumerate(self.model.get_net(net_type)):
            if l == layer:
                return (net_type, i)
        raise Exception("Layer not found!")


class LayerPath:
    def __init__(self, model: NNModel):
        self.model = model

    def get_net_type(self, layer_path: str) -> NetTypeEnum:
        for net_type in NetTypeEnum:
            if layer_path.startswith(self.model.get_net_path(net_type)):
                return net_type
        raise Exception(f"Unsupported LayerPath {layer_path}!")

    def get_net(self, layer_path: str) -> Net:
        """Returns the net that the layer corresponding to the path belongs to."""
        net_type = self.get_net_type(layer_path)
        return self.model.get_net(net_type)

    def get_net_path(self, layer_path: str) -> str:
        """
        Returns the net path that the layer corresponding to the path belongs to.
        """
        net_type = self.get_net_type(layer_path)
        return self.model.get_net_path(net_type)

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
