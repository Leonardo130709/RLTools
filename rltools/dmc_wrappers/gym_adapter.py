import gym
import numpy as np
import tree
from dm_env import specs
from dm_env import TimeStep, StepType, Environment


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
        
        def convert_fn(sp):
            if isinstance(sp, specs.Array):
                return gym.spaces.Box(-np.inf, np.inf, sp.shape, sp.dtype)
            elif isinstance(sp, specs.BoundedArray):
                return gym.spaces.Box(sp.minimum, sp.maximum, 
                                      sp.shape, sp.dtype)
            else:
                raise NotImplemented
        return tree.map_structure(convert_fn, spec)


class GymToDmc:
    """Convert tuple to proper dm_env.Timestep namedtuple."""

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
        return specs.BoundedArray(minimum=spec.low,
                                  maximum=spec.high,
                                  shape=spec.shape,
                                  dtype=spec.dtype)

    def observation_spec(self):
        space = self._env.observation_space

        def convert_fn(sp):
            if sp.low == -np.inf or sp.high == np.inf:
                return specs.Array(shape=sp.shape, dtype=sp.dtype)
            return specs.BoundedArray(
                sp.shape, sp.dtype, sp.low, sp.high)

        return tree.map_structure(convert_fn, space)
