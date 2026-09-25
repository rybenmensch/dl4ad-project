from collections.abc import Iterator
from typing import cast

from torch import nn

# TODO:
# clean up lib and torch_lib


class IterableModule(nn.Module):
    def __iter__(self) -> Iterator[nn.Module]:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> nn.Module:
        raise NotImplementedError

    def __setitem__(self, idx: int, value: nn.Module) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


def is_layer_iterable(net: nn.Module | IterableModule) -> bool:
    try:
        net_cast = cast(IterableModule, net)
        net_cast[0]
        return True
    except (TypeError, IndexError):
        return False


def get_layer_name(layer: nn.Module) -> str:
    if type(layer).__repr__ is not nn.Module.__repr__:
        return repr(layer)
    return type(layer).__name__


def unwrap_layer(layer: nn.Module) -> nn.Module:
    while hasattr(layer, "layer"):
        assert isinstance(layer.layer, nn.Module)
        layer = layer.layer
    return layer
