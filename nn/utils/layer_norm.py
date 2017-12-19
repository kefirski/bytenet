import torch as t
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-8):
        super(LayerNorm, self).__init__()

        self.eps = eps

        self.sigma = nn.Parameter(t.ones(size))
        self.mu = nn.Parameter(t.zeros(size))

    def forward(self, z):

        z = z.transpose(1, 2)

        mu = t.mean(z, keepdim=True, dim=-1)
        sigma = t.std(z, keepdim=True, dim=-1)
        out = (z - mu) / (sigma + self.eps)
        out = out * self.sigma.expand_as(out) + self.mu.expand_as(out)

        return out.transpose(1, 2)
