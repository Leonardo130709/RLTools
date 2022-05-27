import torch
from ..common import utils
from .common import QuantileEmbedding, TanhLayerNormMLP
nn = torch.nn
F = torch.nn.functional
td = torch.distributions


class Critic(nn.Module):
    def __init__(self, in_features: int, layers: tuple, heads=2):
        super().__init__()
        self.qs = nn.ModuleList([TanhLayerNormMLP(in_features, *layers, 1) for _ in range(heads)])

    def forward(self, obs, action):
        x = torch.cat([obs, action], -1)
        qs = [head(x) for head in self.qs]
        return torch.cat(qs, -1)


# TODO: add support for general UVFA critic, not only quantile
class DistributionalCritic(nn.Module):
    def __init__(self, in_features: int, layers: tuple, quantile_emb_dim: int = 64):
        super().__init__()
        self.qnet = QuantileEmbedding(quantile_emb_dim, in_features)
        self.mlp = TanhLayerNormMLP(in_features, *layers, 1)

    def forward(self, obs, action, tau) -> torch.Tensor:
        tau = self.qnet(tau)
        x = torch.cat([obs, action], -1)
        return self.mlp(x*tau)


class DistributionalValue(nn.Module):
    def __init__(self, in_features: int, layers: tuple, quantile_emb: int = 64):
        super().__init__()
        self.qnet = QuantileEmbedding(quantile_emb, in_features)
        self.mlp = TanhLayerNormMLP(in_features, *layers, 1)

    def forward(self, state, tau) -> torch.Tensor:
        tau = self.qnet(tau)
        return self.mlp(tau*state)
