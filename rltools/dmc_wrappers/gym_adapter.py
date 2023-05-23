from collections.abc import MutableMapping

import tree
import numpy as np
import dm_env.specs

from rltools.dmc_wrappers import base

import gym
if gym.__version__ >= "0.26":
    raise ImportError("gym<0.26 is required to use gym API.")


# TODO: update gymnasium API.
class DmcToGym:
    """ Unpacks dm_env.TimeStep to gym-like tuple. Cannot be wrapped further."""

    def __init__(self, env: dm_env.Environment) -> None:
        self.env = env

    def step(self, action: np.ndarray
             ) -> tuple[base.Observation, float, bool, float]:
        timestep = self.env.step(action)
        obs = timestep.observation
        done = timestep.last()
        reward = timestep.reward
        discount = timestep.discount
        return obs, reward, done, discount

    def reset(self) -> base.Observation:
        return self.env.reset().observation

    @property
    def action_space(self) -> gym.spaces.Box:
        spec = self.env.action_spec()
        return gym.spaces.Box(
            spec.minimum, spec.maximum, spec.shape, spec.dtype)

    @property
    def observation_space(self) -> MutableMapping[str, gym.spaces.Space]:
        spec = self.env.observation_spec()

        def convert_fn(sp):
            if isinstance(sp, dm_env.specs.BoundedArray):
                return gym.spaces.Box(
                    sp.minimum, sp.maximum, sp.shape, sp.dtype)
            if isinstance(sp, dm_env.specs.Array):
                return gym.spaces.Box(-np.inf, np.inf, sp.shape, sp.dtype)
            raise NotImplementedError

        return tree.map_structure(convert_fn, spec)


class GymToDmc:
    """Convert tuple to a proper dm_env.Timestep namedtuple."""

    def __init__(self, env: gym.Env) -> None:
        self.env = env

    def reset(self) -> dm_env.TimeStep:
        obs = self.env.reset()
        return dm_env.restart(obs)

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        obs, reward, done, _ = self.env.step(action)
        if done:
            return dm_env.termination(reward, obs)
        return dm_env.transition(reward, obs, discount=1.)

    def action_spec(self) -> dm_env.specs.BoundedArray:
        space = self.env.action_space
        if isinstance(space, gym.spaces.Box):
            return dm_env.specs.BoundedArray(
                minimum=space.low,
                maximum=space.high,
                shape=space.shape,
                dtype=space.dtype
            )
        raise NotImplementedError

    def observation_spec(self) -> base.ObservationSpec:
        space = self.env.observation_space

        def convert_fn(sp):
            if np.any(sp.low == -np.inf) and np.any(sp.high == np.inf):
                return dm_env.specs.Array(shape=sp.shape, dtype=sp.dtype)
            return dm_env.specs.BoundedArray(
                sp.shape, sp.dtype, sp.low, sp.high)

        return tree.map_structure(convert_fn, space)
