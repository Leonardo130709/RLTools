import torch
import math
from ..common import utils
nn = torch.nn


class QuantileEmbedding(nn.Module):
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


class TanhLayerNormEmbedding(nn.Module):
    def __init__(self, *sizes, act=nn.ELU):
        super().__init__()
        self.emb = nn.Sequential(
            utils.build_mlp(*sizes, act=act),
            nn.LayerNorm(sizes[-1]),
            nn.Tanh()
        )

    def forward(self, x):
        return self.emb(x)


class TanhLayerNormMLP(nn.Module):
    def __init__(self, *sizes, act=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            TanhLayerNormEmbedding(sizes[0], sizes[1]),
            utils.build_mlp(*sizes[1:], act=act)
        )

    def forward(self, inp):
        return self.net(inp)


class ResNet(nn.Module):
    def __init__(self, hidden_dim, act=nn.ReLU):
        super().__init__()

        def make_block():
            return nn.Sequential(
                nn.LayerNorm(hidden_dim),
                act(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        self.net = nn.Sequential(*(make_block() for _ in range(2)))

    def forward(self, x):
        return x + self.net(x)


class ResidualTower(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim, act=nn.ReLU, num_blocks=2):
        super().__init__()
        self.net = nn.Sequential(
            TanhLayerNormEmbedding(in_features, hidden_dim),
            *(ResNet(hidden_dim, act) for _ in range(num_blocks)),
            TanhLayerNormEmbedding(hidden_dim, out_features)
        )

    def forward(self, x):
        return self.net(x)
