import torch
from ..common import utils
from .common import TanhLayerNormMLP
nn = torch.nn
F = torch.nn.functional
td = torch.distributions


class CategoricalActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_number: int, layers: tuple = (32, 32), ordinal: bool = False):
        super().__init__()
        self.act_dim = act_dim
        self.act_num = act_number
        self.ordinal = ordinal
        self.mlp = TanhLayerNormMLP(obs_dim, *layers, act_dim*act_number)

    def forward(self, obs):
        logits = self.mlp(obs).reshape(*obs.shape[:-1], self.act_dim, self.act_num)
        if self.ordinal:
            logits = utils.ordinal_logits(logits, delta=1e-7)
        dist = td.OneHotCategorical(logits=logits)
        return td.Independent(dist, 1)


class ContinuousActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, layers: tuple, mean_scale: float = 1, init_std: float = 1.):
        super().__init__()
        self.mean_scale = mean_scale
        self.mlp = TanhLayerNormMLP(obs_dim, *layers, 2*act_dim)
        self.init_std = torch.log(torch.tensor(init_std).exp() - 1.)

    def forward(self, x) -> td.Distribution:
        x = self.mlp(x)
        mu, std = x.chunk(2, -1)
        mu = self.mean_scale * torch.tanh(mu / self.mean_scale)
        std = torch.maximum(std, torch.full_like(std, -18.))
        std = F.softplus(std + self.init_std) + 1e-4
        return self.get_dist(mu, std)

    @staticmethod
    def get_dist(mu, std) -> td.Distribution:
        dist = td.Normal(mu, std)
        dist = td.transformed_distribution.TransformedDistribution(dist, td.TanhTransform(cache_size=1))
        dist = td.Independent(dist, 1)
        return dist
