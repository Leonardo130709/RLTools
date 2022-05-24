import torch
from ..common import utils
from .common import QuantileEmbedding
nn = torch.nn
F = torch.nn.functional
td = torch.distributions


class Critic(nn.Module):
    def __init__(self, in_features, layers, heads=2):
        super().__init__()
        self.qs = nn.ModuleList([utils.build_mlp(in_features, *layers, 1) for _ in range(heads)])

    def forward(self, obs, action):
        x = torch.cat([obs, action], -1)
        qs = [head(x) for head in self.qs]
        return torch.cat(qs, -1)


# TODO: add support for general UVFA critic, not only quantile
class DistributionalCritic(nn.Module):
    def __init__(self, in_features, layers=(32, 32), quantile_emb_dim=64):
        super().__init__()
        self.qnet = QuantileEmbedding(quantile_emb_dim, in_features)
        self.mlp = utils.build_mlp(in_features, *layers, 1)

    def forward(self, obs, action, tau):
        tau = self.qnet(tau)
        x = torch.cat([obs, action], -1)
        return self.mlp(x*tau)


class DistributionalValue(nn.Module):
    def __init__(self, in_features, layers, quantile_emb):
        super().__init__()
        self.qnet = QuantileEmbedding(quantile_emb, in_features)
        self.mlp = utils.build_mlp(in_features, *layers, 1)

    def forward(self, state, tau):
        tau = self.qnet(tau)
        return self.mlp(tau*state)
