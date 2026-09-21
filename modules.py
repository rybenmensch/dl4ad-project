import torch
from torch import Tensor
from torch.nn import Module

from model import LayerInfo


class SkippingLayer(Module):
    """Layer that forwards its input unchanged."""

    def __init__(self, layer: LayerInfo) -> None:
        super().__init__()
        self.layer = layer.get_layer()

    def forward(self, x: Tensor) -> Tensor:
        return x


class RepeatingLayer(Module):
    """Layer that applies its processing multiple times."""

    def __init__(self, layer: LayerInfo, repeats: int = 1) -> None:
        super().__init__()
        self.layer = layer.get_layer()
        self.repeats = repeats

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.repeats):
            x = self.layer(x)
        return x


class MultiplierLayer(Module):
    """Layer that multiplies weights and biases."""

    def __init__(
        self, layer: LayerInfo, weight_mul: float = 1, bias_mul: float = 1
    ) -> None:
        super().__init__()
        self.layer = layer.get_layer()
        with torch.no_grad():
            for w_b in layer.model.layer_get_weight_and_bias(self.layer):
                w_b.weight.mul_(weight_mul)
                w_b.bias.mul_(bias_mul)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class AdditionLayer(Module):
    """Layer that adds an offset to weights and biases."""

    def __init__(
        self, layer: LayerInfo, weight_add: float = 1, bias_add: float = 1
    ) -> None:
        super().__init__()
        self.layer = layer.get_layer()
        with torch.no_grad():
            for w_b in layer.model.layer_get_weight_and_bias(self.layer):
                w_b.weight.add_(weight_add)
                w_b.bias.add_(bias_add)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)
