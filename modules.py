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
