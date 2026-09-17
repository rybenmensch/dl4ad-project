from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias, runtime_checkable

import torch
from torch import nn

from lib import convert_audio, getattr_from_attr_string, setattr_from_attr_string
from torch_lib import IterableModule, get_layer_name

Net: TypeAlias = IterableModule
Module: TypeAlias = nn.Module


@runtime_checkable
class HasInChannels(Protocol):
    in_channels: int


class NetTypeEnum(str, Enum):
    Encoder = "encoder"
    Decoder = "decoder"


class NNModel(ABC):
    """wrapper class for non-torchscript models"""

    def __init__(self, model) -> None:
        self.model = model
        self.from_net_type = NetType(self)
        self.from_net_path = NetPath(self)
        self.from_layer_path = LayerPath(self)
        self.from_layer = Layer(self)

    def __str__(self) -> str:
        return str(self.model)

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Returns the sample rate the model is intended to run on."""

    # TODO: rename, possibly confusing because doesn't refer to model channels
    # but input channels for first conv layer
    def get_in_channels(self, net_type: NetTypeEnum) -> int:
        return self.get_first_layer(net_type).in_channels

    @abstractmethod
    def get_channels(self) -> int:
        pass

    @abstractmethod
    def get_first_layer(self, net_type: NetTypeEnum) -> HasInChannels:
        pass

    @abstractmethod
    def get_layer_channels(self, layer: Module) -> tuple[int, int] | None:
        raise TypeError(f"Layer has unhandled type {get_layer_name(layer)}!")

    def process_audio(self, wav: tuple[torch.Tensor, int]) -> torch.Tensor:
        torch.manual_seed(0)
        audio = convert_audio(wav, self.get_sample_rate(), self.get_channels())
        audio = audio.unsqueeze(0)
        with torch.no_grad():
            return self.model(audio).squeeze(0)

    @abstractmethod
    def get_net_path(self, net_type: NetTypeEnum) -> str:
        """Implement this for looking up the actual path to the encoder or decoder."""
        pass

    def get_net(self, net_type: NetTypeEnum) -> Net:
        """Returns the net."""
        net_path = self.get_net_path(net_type)
        return getattr_from_attr_string(self.model, net_path)

    def set_net(self, net_type: NetTypeEnum, net: Net) -> None:
        """Update the net."""
        net_path = self.get_net_path(net_type)
        setattr_from_attr_string(self.model, net_path, net)

    def get_nets(self) -> tuple[
        Net,
        Net,
    ]:
        """Returns the encoder and decoder nets."""
        return (self.get_net(NetTypeEnum.Encoder), self.get_net(NetTypeEnum.Decoder))

    def get_net_paths(self) -> tuple[str, str]:
        """Returns the encoder and decoder paths."""
        return (
            self.get_net_path(NetTypeEnum.Encoder),
            self.get_net_path(NetTypeEnum.Decoder),
        )

    def get_nets_and_paths(self) -> tuple[
        tuple[Net, str],
        tuple[Net, str],
    ]:
        """Returns the encoder and decoder nets and the paths to them."""
        return (
            (self.get_net(NetTypeEnum.Encoder), self.get_net_path(NetTypeEnum.Encoder)),
            (self.get_net(NetTypeEnum.Decoder), self.get_net_path(NetTypeEnum.Decoder)),
        )

    def get_nets_and_types(
        self,
    ) -> tuple[tuple[Net, NetTypeEnum], tuple[Net, NetTypeEnum]]:

        return (
            (self.get_net(NetTypeEnum.Encoder), NetTypeEnum.Encoder),
            (self.get_net(NetTypeEnum.Decoder), NetTypeEnum.Decoder),
        )


class NetType:
    def __init__(self, model: NNModel) -> None:
        self.model = model

    def get_net_path(self, net_type: NetTypeEnum) -> str:
        return self.model.get_net_path(net_type)

    def get_net(self, net_type: NetTypeEnum) -> Net:
        return self.model.get_net(net_type)

    def get_layer_from_index(self, net_type: NetTypeEnum, index: int) -> Module:
        net = self.get_net(net_type)
        return net[index]


class NetPath:
    def __init__(self, model: NNModel) -> None:
        self.model = model

    def get_net_type(self, net_path: str) -> NetTypeEnum:
        for net_type in NetTypeEnum:
            if net_path == self.model.get_net_path(net_type):
                return net_type
        raise ValueError(f"Unsupported NetPath {net_path}!")

    def get_net(self, net_path: str) -> Net:
        net_type = self.get_net_type(net_path)
        return self.model.get_net(net_type)

    def get_layer(self, net_path: str, index: int) -> Module:
        net: Net = self.get_net(net_path)
        return net[index]


class Layer:
    def __init__(self, model: NNModel) -> None:
        self.model = model

    def get_net_type(self, layer: Module) -> NetTypeEnum:
        net: Net = self.get_net(layer)
        for n, t in self.model.get_nets_and_types():
            # i have no clue how this equality check works, don't know if
            # that will still work if we have shuffled some things around
            if n == net:
                return t
        raise ValueError("Layer not found!")

    def get_net(self, layer: Module) -> Net:
        for net in self.model.get_nets():
            for l in net:
                # i have no clue how this equality check works, don't know if
                # that will still work if we have shuffled some things around
                if l == layer:
                    return net
        raise ValueError("Layer not found!")

    def get_net_path(self, layer: Module) -> str:
        net_type: NetTypeEnum = self.get_net_type(layer)
        return self.model.from_net_type.get_net_path(net_type)

    def get_layer_path(self, layer: Module) -> str:
        net_path, index = self.get_net_path_and_index(layer)
        return f"{net_path}.{index}"

    def get_net_path_and_index(self, layer: Module) -> tuple[str, int]:
        net_path: str = self.get_net_path(layer)
        _, i = self.get_net_type_and_index(layer)
        return (net_path, i)

    def get_net_type_and_index(self, layer: Module) -> tuple[NetTypeEnum, int]:
        net_type: NetTypeEnum = self.get_net_type(layer)
        for i, l in enumerate(self.model.get_net(net_type)):
            if l == layer:
                return (net_type, i)
        raise ValueError("Layer not found!")

    def get_layer_name(self, layer: Module) -> str:
        return get_layer_name(layer)


class LayerPath:
    def __init__(self, model: NNModel) -> None:
        self.model = model

    def get_net_type(self, layer_path: str) -> NetTypeEnum:
        for net_type in NetTypeEnum:
            if layer_path.startswith(self.model.get_net_path(net_type)):
                return net_type
        raise ValueError(f"Unsupported LayerPath {layer_path}!")

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


# TODO: think about what information is returned,
# path? net_type? the layer itself?
@dataclass(frozen=True)
class ShapePreservingLayer:
    index: int
    name: str
    net_type: NetTypeEnum
    layer_path: str


def get_shape_preserving_layers_from_net(
    model: NNModel, net_type: NetTypeEnum
) -> list[ShapePreservingLayer]:
    """
    Returns information about every layer that preserves the input shape.
    Input:  NNModel
    Output: List of ShapePreservingLayer dataclasses
    """

    net: Net = model.get_net(net_type)

    results = []
    input_size = model.get_in_channels(net_type)

    x = torch.zeros(1, input_size, 64)

    for idx, layer in enumerate(net):
        layer_name = model.from_layer.get_layer_name(layer)
        layer_path = model.from_layer.get_layer_path(layer)
        try:
            with torch.no_grad():
                out = layer(x)
                if out.shape == x.shape:
                    # layer DOES preserve shape
                    results.append(
                        ShapePreservingLayer(
                            index=idx,
                            name=layer_name,
                            net_type=net_type,
                            layer_path=layer_path,
                        )
                    )
                else:
                    # layer does NOT preserve shape
                    pass
                x = out
        except Exception as e:
            print(f"Layer nr {idx} of type {layer_name} raised {e}")

    return results


# TODO: should this be a method of the class?
def get_shape_preserving_layers(model: NNModel) -> list[ShapePreservingLayer]:
    results = []
    for _, net_type in model.get_nets_and_types():
        net_type: NetTypeEnum
        results += get_shape_preserving_layers_from_net(model, net_type)

    return results


def get_weighted_layers():
    pass
