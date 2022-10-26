from typing import Any, Mapping, NamedTuple, Type

import dm_env
from dm_env import specs

OBS_SPECS = Mapping[str, specs.Array]


class EnvironmentSpecs(NamedTuple):
    observation_spec: OBS_SPECS
    action_spec: specs.BoundedArray
    reward_spec: specs.Array
    discount_spec: specs.BoundedArray


# pylint: disable=no-staticmethod-decorator
class Wrapper(dm_env.Environment):
    """This allows to modify methods of an already initialized environment."""
    
    def __init__(self, env: dm_env.Environment):
        self._env = env

    def observation(self, timestep: dm_env.TimeStep) -> Any:
        return timestep.observation

    def reward(self, timestep: dm_env.TimeStep) -> float:
        return timestep.reward

    def done(self, timestep: dm_env.TimeStep) -> bool:
        return timestep.last()

    def step_type(self, timestep: dm_env.TimeStep) -> dm_env.StepType:
        return timestep.step_type

    def discount(self, timestep: dm_env.TimeStep) -> float:
        return timestep.discount

    def step(self, action) -> dm_env.TimeStep:
        timestep = self._env.step(action)
        return self._wrap_timestep(timestep)

    def reset(self) -> dm_env.TimeStep:
        return self._wrap_timestep(self._env.reset())

    def _wrap_timestep(self, timestep) -> dm_env.TimeStep:
        # pylint: disable-next=protected-access
        return timestep._replace(
            step_type=self.step_type(timestep),
            reward=self.reward(timestep),
            discount=self.discount(timestep),
            observation=self.observation(timestep)
        )

    def action_spec(self) -> specs.Array:
        return self._env.action_spec()

    def observation_spec(self) -> OBS_SPECS:
        return self._env.observation_spec()

    def reward_spec(self) -> specs.Array:
        return self._env.reward_spec()

    def discount_spec(self) -> specs.Array:
        return self._env.discount_spec()

    @property
    def environment_specs(self):
        return EnvironmentSpecs(
            observation_spec=self.observation_spec(),
            action_spec=self.action_spec(),
            reward_spec=self.reward_spec(),
            discount_spec=self.discount_spec()
        )

    def __getattr__(self, item):
        return getattr(self._env, item)

    @property
    def unwrapped(self) -> dm_env.Environment:
        if hasattr(self._env, "unwrapped"):
            return self._env.unwrapped
        else:
            return self._env
