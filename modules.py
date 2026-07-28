import torch.nn as nn


class SequentialWithSkip(nn.Module):
    def __init__(self, original_net, skips=None):
        super().__init__()
        self.original_net = original_net
        self.skips = set(skips) if skips else set()

    def forward(self, x):
        for i, layer in enumerate(self.original_net):
            if i in self.skips:
                continue
            x = layer(x)
        return x


class SequentialWithRepeat(nn.Module):
    def __init__(self, original_net, repeats=None):
        super().__init__()
        self.original_net = original_net
        self.repeats = repeats if repeats else {}

    def forward(self, x):
        for i, layer in enumerate(self.original_net):
            r = self.repeats.get(i, 1)
            for _ in range(r):
                x = layer(x)
            return x


class ManipulatedSequential(nn.Module):
    def __init__(self, original_net, skips=None, repeats=None):
        super().__init__()
        self.original_net = original_net
        self.skips = set(skips) if skips else set()
        self.repeats = repeats if repeats else {}

    def forward(self, x):
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
