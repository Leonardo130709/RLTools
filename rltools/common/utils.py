import copy
from itertools import chain
from typing import TypeVar, Union, Tuple, NoReturn, Type, List, Tuple
import torch
import dm_env
import numpy as np
from ..wrappers.base import Wrapper
nn = torch.nn
F = nn.functional
td = torch.distributions

Env = TypeVar('Env', bound=dm_env.Environment)
Module = TypeVar('Module', bound=nn.Module)


def build_mlp(*sizes: int, act: nn.Module = nn.ReLU) -> nn.Sequential:
    """Builds basic MLP."""
    mlp = []
    for i in range(1, len(sizes)):
        mlp.append(nn.Linear(sizes[i-1], sizes[i]))
        mlp.append(act())
    return nn.Sequential(*mlp[:-1])


def grads_sum(model: nn.Module) -> float:
    """Computes gradients norm of the module, mostly serves for debugging purpose."""
    norm = 0
    for p in model.parameters():
        if p.grad is not None:
            norm += p.grad.abs().sum().item()
    return np.sqrt(norm)


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, rho: float) -> NoReturn:
    """Soft update online network with its target counterpart."""
    for target_param, online_param in zip(target.parameters(), online.parameters()):
        target_param.data.copy_(rho * target_param.data + (1. - rho) * online_param.data)


class TruncatedTanhTransform(td.transforms.TanhTransform):
    """Prevents TanhTransform saturation by truncating it instead of caching."""
    _lim = .9997

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        y = torch.clamp(y, min=-self._lim, max=self._lim)
        return y.atanh()


def softplus(param: torch.Tensor) -> torch.Tensor:
    """Safe softplus."""
    param = torch.maximum(param, torch.full_like(param, -18.))
    return F.softplus(param) + 1e-8


def sigmoid(param: torch.Tensor, lower_lim: float = 0., upper_lim: float = 1000.) -> torch.Tensor:
    """Rescaled sigmoid."""
    return lower_lim + (upper_lim - lower_lim)*torch.sigmoid(param)


def make_param_group(*modules: Module):
    """Makes single param group from inputs."""
    return nn.ParameterList(chain(*map(nn.Module.parameters, modules)))


def make_targets(*modules: Module):
    """Foreach module create its target counterpart."""
    return map(lambda m: copy.deepcopy(m).requires_grad_(False), modules)


def retrace(resids: torch.Tensor, cs: torch.Tensor,
            discount: float, lambda_: float) -> torch.Tensor:
    """
    Retrace deltas from sequence of corresponding inputs.
    1606.02647
    """
    cs = torch.cat((cs[1:], torch.ones_like(cs[-1:])))
    cs *= lambda_
    resids, cs = map(lambda t: t.flip(0), (resids, cs))
    deltas = []
    last_val = torch.zeros_like(resids[0])
    for resid, c in zip(resids, cs):
        last_val = resod + last_val * discount * c
        deltas.append(last_val)
    return torch.stack(deltas).flip(0)


def ordinal_logits(logits: torch.Tensor, delta: float = 0.) -> torch.Tensor:
    """
    Compute logits for ordinal regression on factorized categorical distribution.
    However optimizations task with such logits could be cumbersome.
    """
    logits = torch.sigmoid(logits)
    logits = torch.clamp(logits, min=delta, max=1.-delta)
    lt = torch.log(logits)
    gt = torch.log(1.-logits)
    lt = torch.cumsum(lt, -1)
    gt = torch.cumsum(gt[..., 1:].flip(-1), -1).flip(-1)
    gt = F.pad(gt, [0, 1])
    return lt+gt


def dual_loss(loss: torch.Tensor, epsilon: Union[float, torch.Tensor],
              alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Constrained loss with lagrange multiplier. """
    scaled_loss = alpha.detach()*loss
    mult_loss = alpha*(epsilon - loss.detach())
    return scaled_loss, mult_loss


def sequence_discount(x: torch.Tensor, discount: float = 1.) -> torch.Tensor:
    """Simple sequence discount that matches input tensor dims."""
    discount = discount ** torch.arange(x.size(0), device=x.device)
    shape = (x.ndimension() - 1) * (1,)
    return discount.reshape(-1, *shape)


def chain_wrapper(env: Env, wrappers_with_configs: List[Tuple[Type[Wrapper], dict]]) -> Env:
    """Create env by sequentially wrapping it with input wrappers."""
    for wrapper, config in wrappers_with_configs:
        env = wrapper(env, **config)
    return env


def weight_init(module: nn.Module) -> NoReturn:
    """Should define common weight init strategy"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data, 1.4)
