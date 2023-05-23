from collections.abc import MutableMapping
from typing import Any

import tree
import numpy as np
import dm_env.specs
import gymnasium as gym

from rltools.dmc_wrappers import base

GymTimeStep = tuple[base.Observation, float, bool, bool, dict[str, Any]]


def _convert_fn(sp: dm_env.specs.Array) -> gym.spaces.Space:
    """dmc->gym space."""
    if isinstance(sp, dm_env.specs.BoundedArray):
        return gym.spaces.Box(
            sp.minimum, sp.maximum, sp.shape, sp.dtype)
    if isinstance(sp, dm_env.specs.Array):
        return gym.spaces.Box(-np.inf, np.inf, sp.shape, sp.dtype)
    if isinstance(sp, dm_env.specs.DiscreteArray):
        return gym.spaces.Discrete(sp.num_values)
    raise NotImplementedError(sp)


def _inverse_convert_fn(sp: gym.spaces.Space) -> dm_env.specs.Array:
    """gym->dmc spec."""
    if isinstance(sp, gym.spaces.Box):
        if np.any(sp.low == -np.inf) and np.any(sp.high == np.inf):
            return dm_env.specs.Array(shape=sp.shape, dtype=sp.dtype)
        else:
            return dm_env.specs.BoundedArray(sp.shape, sp.dtype, sp.low, sp.high)
    if isinstance(sp, gym.spaces.Discrete):
        return dm_env.specs.DiscreteArray(sp.n)
    raise NotImplementedError(sp)


class DmcToGymnasium:
    """Unpacks dm_env.TimeStep to gymnasium-like tuple."""

    def __init__(self, env: dm_env.Environment) -> None:
        self.env = env
        rew_spec = env.reward_spec()
        if isinstance(rew_spec, dm_env.specs.BoundedArray):
            self.reward_range = (rew_spec.minimum, rew_spec.maximum)
        else:
            self.reward_range = (-float("inf"), float("inf"))
        self._action_spec = _convert_fn(env.action_spec())
        self._obs_spec = tree.map_structure(_convert_fn, env.observation_spec())

    def step(self, action: base.Action) -> GymTimeStep:
        timestep = self.env.step(action)
        obs = timestep.observation
        terminated = timestep.last()
        truncated = terminated and timestep.discount > 0
        reward = timestep.reward
        return obs, reward, terminated, truncated, {}

    def reset(self) -> base.Observation:
        return self.env.reset().observation, {}

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_spec

    @property
    def observation_space(self) -> MutableMapping[str, gym.spaces.Space]:
        return self._obs_spec


class GymnasiumToDmc:
    """Convert tuple to a proper dm_env.Timestep namedtuple."""

    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self._action_spec = _inverse_convert_fn(env.action_space)
        self._obs_spec = tree.map_structure(
            _inverse_convert_fn, env.observation_space)
        self._rew_spec = dm_env.specs.Array(shape=(), dtype=float)

    def reset(self) -> dm_env.TimeStep:
        obs, _ = self.env.reset()
        return dm_env.restart(obs)

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        obs, reward, terminated, truncated, _ = self.env.step(action)
        if terminated:
            return dm_env.termination(reward, obs)
        if truncated:
            return dm_env.truncation(reward, obs)
        return dm_env.transition(reward, obs, discount=1.)

    def action_spec(self) -> dm_env.specs.Array:
        return self._action_spec

    def observation_spec(self) -> base.ObservationSpec:
        return self._obs_spec

    def reward_spec(self) -> dm_env.specs.Array:
        return self._rew_spec
