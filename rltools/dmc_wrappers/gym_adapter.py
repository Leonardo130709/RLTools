import gym
import numpy as np
import dm_env.specs
from dm_env import TimeStep, StepType, Environment

from rltools.dmc_wrappers.utils.nested import nested_fn


# TODO: update gym API >= 0.26.
class DmcToGym:
    """ Unpacks dm_env.TimeStep to gym-like tuple. Cannot be wrapped further."""

    def __init__(self, env: Environment):
        self._env = env

    def step(self, action):
        timestep = self._env.step(action)
        obs = timestep.observation
        done = timestep.last()
        reward = timestep.reward
        discount = timestep.discount
        return obs, reward, done, discount

    def reset(self):
        return self._env.reset().observation

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(
            spec.minimum, spec.maximum, spec.shape, spec.dtype)

    @property
    def observation_space(self):
        spec = self._env.observation_spec()
        return nested_fn(
            lambda sp: gym.spaces.Box(
                -np.inf, np.inf, sp.shape, sp.dtype),
            spec
        )


class GymToDmc:
    def __init__(self, env):
        self._env = env

    def reset(self):
        obs = self._env.reset()
        return TimeStep(observation=obs,
                        reward=0.,
                        discount=1.,
                        step_type=StepType.FIRST)

    def step(self, action):
        obs, reward, done, _ = self._env.step(action)
        step_type = StepType.LAST if done else StepType.MID
        return TimeStep(observation=obs,
                        reward=reward,
                        discount=done,
                        step_type=step_type
                        )

    def action_spec(self):
        spec = self._env.action_space
        return dm_env.specs.BoundedArray(minimum=spec.low,
                                         maximum=spec.high,
                                         shape=spec.shape,
                                         dtype=spec.dtype)

    def observation_spec(self):
        spec = self._env.observation_spec

        def convert_spec(sp):
            if sp.low == -np.inf or sp.high == np.inf:
                return dm_env.specs.Array(shape=sp.shape, dtype=sp.dtype)
            return dm_env.specs.BoundedArray(
                sp.shape, sp.dtype, sp.low, sp.high)

        return nested_fn(convert_spec, spec)
