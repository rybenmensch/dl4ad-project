import torch
from torch import Tensor
from torch.nn import Module

from library.model import LayerInfo
from library.torch_utils import get_layer_name


class WrapLayer(Module):
    def __init__(self, layer_info: LayerInfo, args: str) -> None:
        super().__init__()
        self.layer = layer_info.get_layer()
        self.__args = args

    def __repr__(self) -> str:
        layer_name = self.__class__.__name__
        if self.__args == "":
            args = ""
        else:
            args = ", " + self.__args
        return f"{layer_name}({get_layer_name(self.layer)}{args})"


class SkippingLayer(WrapLayer):
    """Layer that forwards its input unchanged."""

    def __init__(self, layer_info: LayerInfo) -> None:
        super().__init__(layer_info, "")

    def forward(self, x: Tensor) -> Tensor:
        return x


class RepeatingLayer(WrapLayer):
    """Layer that applies its processing multiple times."""

    def __init__(self, layer_info: LayerInfo, repeats: int = 1) -> None:
        super().__init__(layer_info, str(repeats))
        self.repeats = repeats

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.repeats):
            x = self.layer(x)
        return x


def make_w_b_string(w: float, b: float, default: int) -> str:
    if b == default:
        if w == default:
            return ""
        return f"{w=}"
    return f"{w=}, {b=}"


class MultiplierLayer(WrapLayer):
    """Layer that multiplies weights and biases."""

    def __init__(
        self, layer_info: LayerInfo, weight_mul: float = 1, bias_mul: float = 1
    ) -> None:
        super().__init__(layer_info, make_w_b_string(weight_mul, bias_mul, 1))

        self.weight_mul = weight_mul
        self.bias_mul = bias_mul

        with torch.no_grad():
            for w_b in layer_info.model.layer_get_weight_and_bias(self.layer):
                w_b.weight.mul_(weight_mul)
                w_b.bias.mul_(bias_mul)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class AdditionLayer(WrapLayer):
    """Layer that adds an offset to weights and biases."""

    def __init__(
        self, layer_info: LayerInfo, weight_add: float = 1, bias_add: float = 1
    ) -> None:
        super().__init__(layer_info, make_w_b_string(weight_add, bias_add, 0))

        self.weight_add = weight_add
        self.bias_add = bias_add

        with torch.no_grad():
            for w_b in layer_info.model.layer_get_weight_and_bias(self.layer):
                w_b.weight.add_(weight_add)
                w_b.bias.add_(bias_add)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)
