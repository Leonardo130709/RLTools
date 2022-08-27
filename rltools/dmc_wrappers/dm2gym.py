import gym
import numpy as np
from dm_control.rl.control import Environment

from dmc_wrappers.utils.nested import nested_fn


class GymWrapper:
    """ Unpacks dm_env.TimeStep to gym-like tuple. Cannot be wrapped further."""
    def __init__(self, env: Environment):
        self._env = env

    def step(self, action):
        timestep = self._env.step(action)
        obs = timestep.observation
        done = timestep.last()
        reward = timestep.reward
        return obs, reward, done

    def reset(self):
        return self._env.reset().observation

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)

    @property
    def observation_space(self):
        spec = self._env.observation_spec()
        return nested_fn(
            lambda sp: gym.spaces.Box(-np.inf, np.inf, sp.shape, sp.dtype), spec)
