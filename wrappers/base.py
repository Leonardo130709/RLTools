import abc
from dm_env import TimeStep, Environment


class Wrapper(abc.ABC):
    def __init__(self, env: Environment):
        self.env = env

    def observation(self, timestep):
        return timestep.observation

    def reward(self, timestep):
        return timestep.reward

    def done(self, timestep):
        return timestep.last()

    def step_type(self, timestep):
        return timestep.step_type

    def discount(self, timestep):
        return timestep.discount

    def step(self, action) -> TimeStep:
        timestep = self.env.step(action)
        return self._wrap_timestep(timestep)

    def reset(self) -> TimeStep:
        return self._wrap_timestep(self.env.reset())

    def _wrap_timestep(self, timestep) -> TimeStep:
        return TimeStep(
            step_type=self.step_type(timestep),
            reward=self.reward(timestep),
            discount=self.discount(timestep),
            observation=self.observation(timestep)
        )

    # TODO: explicit declaration
    def __getattr__(self, item):
        return getattr(self.env, item)

    @property
    def unwrapped(self) -> Environment:
        if hasattr(self.env, 'unwrapped'):
            return self.env.unwrapped
        else:
            return self.env
