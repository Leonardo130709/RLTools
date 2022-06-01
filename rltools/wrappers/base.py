from typing import Any
import dm_env


class Wrapper(dm_env.Environment):
    """This allows to modify attributes which agent observes and to pack it back."""
    def __init__(self, env: dm_env.Environment):
        self.env = env

    @staticmethod
    def observation(timestep: dm_env.TimeStep) -> Any:
        return timestep.observation

    @staticmethod
    def reward(timestep: dm_env.TimeStep) -> float:
        return timestep.reward

    @staticmethod
    def done(timestep: dm_env.TimeStep) -> bool:
        return timestep.last()

    @staticmethod
    def step_type(timestep: dm_env.TimeStep) -> dm_env.StepType:
        return timestep.step_type

    @staticmethod
    def discount(timestep: dm_env.TimeStep) -> float:
        return timestep.discount

    def step(self, action) -> dm_env.TimeStep:
        timestep = self.env.step(action)
        return self._wrap_timestep(timestep)

    def reset(self) -> dm_env.TimeStep:
        return self._wrap_timestep(self.env.reset())

    def _wrap_timestep(self, timestep) -> dm_env.TimeStep:
        # pylint: disable-next=protected-access
        return timestep._replace(
            step_type=self.step_type(timestep),
            reward=self.reward(timestep),
            discount=self.discount(timestep),
            observation=self.observation(timestep)
        )

    def action_spec(self) -> dm_env.specs.Array:
        return self.env.action_spec()

    def observation_spec(self) -> dm_env.specs.Array:
        return self.env.observation_spec()

    # TODO: explicit declaration
    def __getattr__(self, item):
        return getattr(self.env, item)

    @property
    def unwrapped(self) -> dm_env.Environment:
        if hasattr(self.env, 'unwrapped'):
            return self.env.unwrapped
        else:
            return self.env
