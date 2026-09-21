from torch import Tensor
from torch.nn import Module

from model import LayerInfo


class SkippingLayer(Module):
    """Layer that forwards its input unchanged."""

    def __init__(self, layer: LayerInfo) -> None:
        super().__init__()
        self.layer = layer.model.from_layer_path.get_layer(layer.layer_path)

    def forward(self, x: Tensor) -> Tensor:
        return x


class RepeatingLayer(Module):
    """Layer that applies its processing multiple times."""

    def __init__(self, layer: LayerInfo, repeats: int = 1) -> None:
        super().__init__()
        self.layer = layer.model.from_layer_path.get_layer(layer.layer_path)
        self.repeats = repeats

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.repeats):
            x = self.layer(x)
        return x


class SequentialWithSkip(Module):
    def __init__(self, original_net, skips=None) -> None:
        super().__init__()
        self.original_net = original_net
        self.skips = set(skips) if skips else set()

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.original_net):
            if i in self.skips:
                continue
            x = layer(x)
        return x


class SequentialWithRepeat(Module):
    def __init__(self, original_net, repeats=None) -> None:
        super().__init__()
        self.original_net = original_net
        self.repeats = repeats or {}

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.original_net):
            r = self.repeats.get(i, 1)
            for _ in range(r):
                x = layer(x)
        # TODO: check if bug? return x was indented in one more
        return x


class ManipulatedSequential(Module):
    def __init__(self, original_net, skips=None, repeats=None) -> None:
        super().__init__()
        self.original_net = original_net
        self.skips = set(skips) if skips else set()
        self.repeats = repeats or {}

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.original_net):
            if i in self.skips:
                continue
            r = self.repeats.get(i, 1)
            for _ in range(r):
                x = layer(x)
        return x


# class CustomEncoderWrapper(torch.nn.Module):
#     def __init__(self, original_net, channels):
#         super().__init__()
#         self.original_net = original_net
#
#         self.custom_layer = torch.nn.Conv1d(
#             in_channels=channels,
#             out_channels=channels,
#             kernel_size=1
#         )
#
#         with torch.no_grad():
#             self.custom_layer.weight.copy_(torch.eye(channels).unsqueeze(-1))
#             self.custom_layer.bias.zero_()
#
#     def forward(self, x):
#         features = self.original_net(x)
#         return self.custom_layer(features)
#
# """
# # Usage
# encoder_output_channels = conv_layer.out_channels
# torch.manual_seed(0)
# model.encoder.encoder.net = CustomEncoderWrapper(
#     original_encoder_net,
#     encoder_output_channels
# )
#
# print(model.encoder.encoder.net)
# """
