import torch as t
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single


class Conv1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding='SAME', dilation=1, groups=1, bias=True):
        padding = _single(self.same_padding(kernel_size, dilation)) if padding == 'SAME' else _single(int(padding))
        kernel_size = _single(kernel_size)
        dilation = _single(dilation)

        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, 1, padding, dilation,
            False, _single(0), groups, bias)

    def forward(self, input):
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    @staticmethod
    def same_padding(kernel_size, dilation):
        width = dilation * kernel_size - dilation + 1
        return width // 2


class MaskedConv1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding='SAME', dilation=1, bias=True):
        implied_kernel_size = kernel_size // 2 + 1
        padding = _single(self.same_padding(kernel_size, dilation)) if padding == 'SAME' else _single(int(padding))
        kernel_size = _single(kernel_size)
        dilation = _single(dilation)

        self.mask = t.ones(out_channels, in_channels, *kernel_size).byte()
        self.mask[:, :, :implied_kernel_size] = t.zeros(out_channels, in_channels, implied_kernel_size)

        super(MaskedConv1d, self).__init__(
            in_channels, out_channels, kernel_size, 1, padding, dilation,
            False, _single(0), 1, bias)

    def forward(self, input):
        return F.conv1d(input, self.masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    @property
    def masked_weight(self):
        self.weight.data.masked_fill_(self.mask, 0)
        return self.weight

    @staticmethod
    def same_padding(kernel_size, dilation):
        width = dilation * kernel_size - dilation + 1
        return width // 2

    def cuda(self, device=None):
        super(MaskedConv1d, self).cuda(device)
        self.mask = self.mask.cuda()
        return self

    def cpu(self):
        super(MaskedConv1d, self).cpu()
        self.mask = self.mask.cpu()
        return self
