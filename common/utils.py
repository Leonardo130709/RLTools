import copy
import torch
import numpy as np
from itertools import chain
nn = torch.nn
F = nn.functional
td = torch.distributions


def build_mlp(*sizes, act=nn.ELU):
    mlp = []
    for i in range(1, len(sizes)):
        mlp.append(nn.Linear(sizes[i-1], sizes[i]))
        mlp.append(act())
    return nn.Sequential(*mlp[:-1])


def grads_sum(model):
    s = 0
    for p in model.parameters():
        if p.grad is not None:
            s += p.grad.pow(2).sum().item()
    return np.sqrt(s)


@torch.no_grad()
def soft_update(target, online, rho):
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.data.copy_(rho * pt.data + (1. - rho) * po.data)


class TruncatedTanhTransform(td.transforms.TanhTransform):
    _lim = .9999997

    def _inverse(self, y):
        y = torch.clamp(y, min=-self._lim, max=self._lim)
        return y.atanh()


def softplus(param):
    param = torch.maximum(param, torch.full_like(param, -18.))
    return F.softplus(param) + 1e-8


def sigmoid(param, lower_lim=0., upper_lim=1000.):
    return lower_lim + (upper_lim - lower_lim)*torch.sigmoid(param)


def make_param_group(*modules):
    return nn.ParameterList(chain(*map(nn.Module.parameters, modules)))


def make_targets(*modules):
    return map(lambda m: copy.deepcopy(m).requires_grad_(False), modules)


def retrace(resids, cs, discount, disclam):
    cs = torch.cat((cs[1:], torch.ones_like(cs[-1:])))
    cs *= disclam
    resids, cs = map(lambda t: t.flip(0), (resids, cs))
    deltas = []
    last_val = torch.zeros_like(resids[0])
    for r, c in zip(resids, cs):
        last_val = r + last_val * discount * c
        deltas.append(last_val)
    return torch.stack(deltas).flip(0)


def ordinal_logits(logits):
    delta = 1e-7
    logits = torch.sigmoid(logits)
    logits = torch.clamp(logits, min=delta, max=1.-delta)
    lt = torch.log(logits)
    gt = torch.log(1.-logits)
    lt = torch.cumsum(lt, -1)
    gt = torch.cumsum(gt[..., 1:].flip(-1), -1).flip(-1)
    gt = F.pad(gt, [0, 1])
    return lt+gt


def dual_loss(loss, epsilon, alpha):
    """ Constrained loss with lagrange multiplier. """
    scaled_loss = alpha.detach()*loss
    mult_loss = alpha*(epsilon - loss.detach())
    return scaled_loss, mult_loss
