import torch
import math
from rltools.common import utils
nn = torch.nn


class QuantileNetwork(nn.Module):
    def __init__(self, emb_dim, out_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.fc = nn.Linear(emb_dim, out_dim)
        self.emb = nn.Parameter(torch.arange(self.emb_dim), requires_grad=False)

    def forward(self, quantiles):
        emb = torch.tile(self.emb, quantiles.shape)
        x = emb * quantiles
        x = self.fc(torch.cos(math.pi * x))
        return torch.relu(x)


class Embedding(nn.Module):
    def __init__(self, *sizes, act=nn.ELU):
        super().__init__()
        self.emb = nn.Sequential(
            utils.build_mlp(*sizes, act=act),
            nn.LayerNorm(sizes[-1]),
            nn.Tanh()
        )

    def forward(self, x):
        return self.emb(x)
