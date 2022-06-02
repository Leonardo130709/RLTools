import torch
from ..common import utils
from .common import MLP
nn = torch.nn
F = torch.nn.functional
td = torch.distributions


class CategoricalActor(nn.Module):
    """Categorical policy over actions."""
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 act_number: int,
                 layers: tuple = (32, 32),
                 ordinal: bool = False
                 ):
        super().__init__()
        self.act_dim = act_dim
        self.act_num = act_number
        self.ordinal = ordinal
        self.mlp = MLP(obs_dim, *layers, act_dim*act_number, layernormtanh='first')

    def forward(self, obs):
        logits = self.mlp(obs).reshape(*obs.shape[:-1], self.act_dim, self.act_num)
        if self.ordinal:
            logits = utils.ordinal_logits(logits, delta=1e-7)
        return td.OneHotCategorical(logits=logits)


class ContinuousActor(nn.Module):
    """TanhNormal policy over actions."""
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 layers: tuple,
                 mean_scale: float = 1,
                 init_std: float = 1.
                 ):
        super().__init__()
        self.mean_scale = mean_scale
        self.mlp = MLP(obs_dim, *layers, 2*act_dim, layernormtanh='first')
        self.init_std = torch.log(torch.tensor(init_std).exp() - 1.)

    def forward(self, x) -> td.Distribution:
        x = self.mlp(x)
        mean, std = x.chunk(2, -1)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std = torch.maximum(std, torch.full_like(std, -18.))
        std = F.softplus(std + self.init_std) + 1e-3
        return self.get_dist(mean, std)

    @staticmethod
    def get_dist(mean, std) -> td.Distribution:
        """Create corresponding distribution from the input parameters.
        May be useful to create copies with detached params."""
        dist = td.Normal(mean, std)
        dist = td.transformed_distribution.TransformedDistribution(
            dist,
            utils.TruncatedTanhTransform(cache_size=1)
        )
        return dist
