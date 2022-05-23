import torch
from rltools.common import utils
from .common import QuantileNetwork
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


class DistributionalCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_number, layers=(32, 32)):
        super().__init__()
        dim = obs_dim+act_dim*act_number
        self.qnet = QuantileNetwork(64, dim)
        self.mlp = utils.build_mlp(dim, *layers, 1)

    def forward(self, obs, action, tau):
        tau = self.qnet(tau)
        x = torch.cat([obs, action], -1)
        return self.mlp(x*tau)
