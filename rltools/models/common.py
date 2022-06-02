import math
from typing import Literal, Type, Callable, Any
import torch
nn = torch.nn
LayerNormTanhMode = Literal['first', 'last', 'both', 'none']
Activation = Type[nn.Module]  # make it more precise


class QuantileEmbedding(nn.Module):
    def __init__(self, emb_dim: int, out_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.fc = nn.Linear(emb_dim, out_dim)
        self.emb = nn.Parameter(torch.arange(self.emb_dim), requires_grad=False)

    def forward(self, quantiles):
        emb = torch.tile(self.emb, quantiles.shape)
        x = emb * quantiles
        x = self.fc(torch.cos(math.pi * x))
        return torch.relu(x)


class MLP(nn.Module):
    """Common Multi Layer Perceptron.
    With an option to place LayerNorm+Tanh on top of it and/or bottom.
    """
    def __init__(self,
                 *layers: int,
                 act: Activation = nn.ReLU,
                 layernormtanh: LayerNormTanhMode = 'none'
                 ):
        super().__init__()
        modules = []  # avoid unnecessary nested structures
        lnt_first = layernormtanh in ('first', 'both')
        lnt_last = layernormtanh in ('last', 'both')

        def _make_layernormtanh(in_features, out_features):
            return [
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features),
                nn.Tanh()
            ]

        if lnt_first:
            modules.extend(_make_layernormtanh(layers[0], layers[1]))

        for i in range(lnt_first, len(layers) - 1 - lnt_last):
            modules.extend([nn.Linear(layers[i], layers[i+1]), act()])

        if lnt_last:
            modules.extend(_make_layernormtanh(layers[-2], layers[-1]))
        else:
            modules = modules[:-1]  # Drop output activation which may not be required.

        self.net = nn.Sequential(*modules)

    def forward(self, tensor):
        return self.net(tensor)


class ResNet(nn.Module):
    def __init__(self, hidden_dim: int, act: Activation = nn.ReLU):
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
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_dim: int,
                 act: Activation = nn.ReLU,
                 num_blocks: int = 2
                 ):
        super().__init__()
        self.net = nn.Sequential(
            MLP(in_features, hidden_dim, act=act, layernormtanh='first'),
            *(ResNet(hidden_dim, act) for _ in range(num_blocks)),
            MLP(hidden_dim, out_features, act=act, layernormtanh='last')
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadWrapper(nn.Module):
    """Creates N copies of module and stacks them on forward call."""

    def __init__(self,
                 module_factory: Callable[[Any], nn.Module],
                 weight_init: Callable[[nn.Module], None] = None,
                 heads: int = 2,
                 dim: int = 0,
                 ):
        super().__init__()
        self.heads = nn.ModuleList([module_factory() for _ in range(heads)])
        self._dim = dim
        if weight_init:
            self.apply(weight_init)

    def forward(self, *args, **kwargs):
        """First dimension will be appended and populated with stacks."""
        outputs = []
        for module in self.heads:
            outputs.append(module(*args, **kwargs))

        return torch.stack(outputs, self._dim)
