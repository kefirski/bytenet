import torch.nn as nn

from .conv import *
from ..utils import LayerNorm


class ResNet(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilation=1, autoregressive=False):
        super(ResNet, self).__init__()

        self.residual_block = nn.Sequential(
            LayerNorm(input_size),
            nn.ReLU(),
            Conv1d(input_size, hidden_size, kernel_size=1, dilation=dilation),

            LayerNorm(hidden_size),
            nn.ReLU(),
            MaskedConv1d(hidden_size, hidden_size, kernel_size=kernel_size, dilation=dilation) if autoregressive else
            Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, dilation=dilation),

            LayerNorm(hidden_size),
            nn.ReLU(),
            Conv1d(hidden_size, input_size, kernel_size=1, dilation=dilation),
        )

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, input_size, seq_len]
        :return: An float tensor with shape of [batch_size, input_size, seq_len]
        """

        return input + self.residual_block(input)
