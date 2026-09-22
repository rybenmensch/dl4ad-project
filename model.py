from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import TypeAlias

import torch
from torch import Tensor, nn

from lib import convert_audio, getattr_from_attr_string, setattr_from_attr_string
from torch_lib import IterableModule, get_layer_name

Net: TypeAlias = IterableModule
Module: TypeAlias = nn.Module


class NetTypeEnum(str, Enum):
    Encoder = "encoder"
    Decoder = "decoder"


@dataclass
class WeightAndBias:
    weight: Tensor
    bias: Tensor


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

    @abstractmethod
    def get_channels(self) -> int:
        pass

    @abstractmethod
    def layer_get_channels(self, layer: Module) -> tuple[int, int] | None:
        raise TypeError(f"Layer has unhandled type {get_layer_name(layer)}!")

    def __call__(self, wav: tuple[torch.Tensor, int]) -> torch.Tensor:
        torch.manual_seed(0)
        audio = convert_audio(wav, self.get_sample_rate(), self.get_channels())
        audio = audio.unsqueeze(0)
        with torch.no_grad():
            return self.model(audio).squeeze(0)

    # TODO: think if this is actually needed?
    @abstractmethod
    def layer_has_subnet(self, layer: nn.Module) -> bool:
        pass

    # TODO: think if this is actually needed?
    @abstractmethod
    def layer_has_weights(self, layer: nn.Module) -> bool:
        pass

    @abstractmethod
    def layer_get_weight_and_bias(self, layer: nn.Module) -> list[WeightAndBias]:
        raise TypeError(f"Layer has unhandled type {get_layer_name(layer)}!")

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
            if net == layer:
                # the layer is the net itself
                return net
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
        if index == -1:
            # the layer is the net itself
            return net_path
        return f"{net_path}.{index}"

    def get_net_path_and_index(self, layer: Module) -> tuple[str, int]:
        net_path: str = self.get_net_path(layer)
        _, i = self.get_net_type_and_index(layer)
        return (net_path, i)

    def get_net_type_and_index(self, layer: Module) -> tuple[NetTypeEnum, int]:
        net_type: NetTypeEnum = self.get_net_type(layer)
        net: Net = self.model.get_net(net_type)
        if net == layer:
            # the layer is the net itself
            return (net_type, -1)
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


@dataclass(frozen=True)
class LayerInfo:
    model: NNModel
    index: int
    name: str
    layer_path: str
    # TODO: instead of this type, it could have a more descriptive type
    # for the None connections. e.g. a type that tells us that the layer can be
    # put into any position, but that also tells us the inouts at the current position
    inout: tuple[int, int] | None
    net_type: NetTypeEnum

    def get_layer(self):
        return self.model.from_layer_path.get_layer(self.layer_path)

    @classmethod
    def from_layer(cls, model: NNModel, layer: Module) -> "LayerInfo":
        model.from_layer.get_layer_path(layer)
        net_type, index = model.from_layer.get_net_type_and_index(layer)
        return cls(
            model=model,
            index=index,
            name=get_layer_name(layer),
            layer_path=model.from_layer.get_layer_path(layer),
            inout=model.layer_get_channels(layer),
            net_type=net_type,
        )

    @classmethod
    def from_net_and_index(cls, model: NNModel, net: Net, idx: int) -> "LayerInfo":
        layer = net[idx]
        return cls(
            model=model,
            index=idx,
            name=get_layer_name(layer),
            layer_path=model.from_layer.get_layer_path(layer),
            inout=model.layer_get_channels(layer),
            net_type=model.from_layer.get_net_type(layer),
        )


def get_shape_preserving_layers_from_net(model: NNModel, net: Net) -> list[LayerInfo]:
    """
    Returns information about every layer that preserves the input shape.
    Input:  NNModel
    Output: List of LayerInfo dataclasses
    """

    results = []
    for l in net:
        inout = model.layer_get_channels(l)
        if inout == None or inout[0] == inout[1]:
            results.append(LayerInfo.from_layer(model, l))

    return results


# TODO: should this be a method of the class?
def get_shape_preserving_layers(model: NNModel) -> list[LayerInfo]:
    results = []
    for net in model.get_nets():
        results += get_shape_preserving_layers_from_net(model, net)
    return results


def info_replace_inout(infos: list[LayerInfo]) -> list[LayerInfo]:
    """
    If `inout` was previously `None`, replace `inout` with the output size of the previous
    and the input size of the next layer.
    """
    result = []
    for i in infos:
        in_chans: int | None = None
        out_chans: int | None = None

        if i.inout == None:
            index = i.index

            # check if we are about to commit an out-of-bounds read
            if i.index != 0:
                inout = infos[index - 1].inout
                in_chans = inout[0] if inout != None else None
            if i.index != len(infos) - 1:
                inout = infos[index + 1].inout
                out_chans = inout[1] if inout != None else None
        else:
            in_chans, out_chans = i.inout

        result.append(replace(i, inout=(in_chans, out_chans)))
    return result


@dataclass(frozen=True)
class SwapInfo:
    source: LayerInfo
    targets: list[LayerInfo]

    def __str__(self) -> str:
        s = self.source
        s_str = f"({s.index}) {s.name} {s.inout} <-> "
        space = " " * len(s_str)
        t_strs = [f"({t.index}) {t.name} {t.inout} " for t in self.targets]

        first = s_str + t_strs[0]
        following = [space + t for t in t_strs[1:]]
        return "\n".join([first, *following])


def get_swappable_layers_from_net(model: NNModel, net: Net) -> list[SwapInfo]:
    infos = [LayerInfo.from_layer(model, l) for l in net]
    infos_corrected = info_replace_inout(infos)

    results = []
    for source in infos:
        targets = []

        source_corr = infos_corrected[source.index]
        source_inout = source_corr.inout if source.inout == None else source.inout

        for target in infos:
            if source == target:
                continue
            if source.inout == target.inout:
                targets.append(target)
                continue

            target_corr = infos_corrected[target.index]
            target_inout = target_corr.inout if target.inout == None else target.inout

            if source_inout == target_inout:
                targets.append(target)

        if len(targets):
            test = SwapInfo(source=source, targets=targets)
            results.append(test)

    return results


def get_swappable_layers(model: NNModel) -> list[SwapInfo]:
    results = []
    for net in model.get_nets():
        results += get_swappable_layers_from_net(model, net)
    return results


@dataclass(frozen=True)
class Swap:
    source: LayerInfo
    target: LayerInfo

    @classmethod
    def from_info(cls, info: SwapInfo, index: int = 0) -> "Swap":
        return cls(source=info.source, target=info.targets[index])


def swap_layers(model: NNModel, swapList: list[SwapInfo], swap: Swap) -> list[SwapInfo]:
    """
    Rather unholyly, this both in-place modifies the model, along with
    returning a modified swaplist...
    """
    source = swap.source
    target = swap.target
    source_net = model.get_net(source.net_type)
    target_net = model.get_net(target.net_type)

    # first, do the actual swap
    source_tmp = source_net[source.index]
    source_net[source.index] = target_net[target.index]
    target_net[target.index] = source_tmp

    # basically just re-calculate swapList as swapping invalidates most of the list
    return get_swappable_layers(model)
