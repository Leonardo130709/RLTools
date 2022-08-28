from typing import Any, Mapping, TypedDict, Union, NamedTuple, Type

import dm_env.specs
from dm_control.rl.control import Environment

OBS_SPECS = Mapping[str, dm_env.specs.Array]


class CameraParams(TypedDict):
    width: float
    height: float
    camera_id: Union[int, str]


class EnvironmentSpecs(NamedTuple):
    observation_spec: OBS_SPECS
    action_spec: dm_env.specs.Array
    reward_spec: dm_env.specs.Array
    discount_spec: dm_env.specs.Array


# pylint: disable=no-staticmethod-decorator
class Wrapper:
    """This allows to modify methods of an already initialized environment."""
    def __init__(self, env: Union[Environment, Type["Wrapper"]]):
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

    def action_spec(self) -> dm_env.specs.Array:
        return self._env.action_spec()

    def observation_spec(self) -> OBS_SPECS:
        return self._env.observation_spec()

    def reward_spec(self):
        return self._env.reward_spec()

    def discount_spec(self):
        return self._env.discount_spec()

    @property
    def environment_specs(self):
        return EnvironmentSpecs(
            observation_spec=self.observation_spec(),
            action_spec=self.action_spec(),
            reward_spec=self.reward_spec(),
            discount_spec=self.discount_spec()
        )

    # TODO: explicit declaration
    def __getattr__(self, item):
        return getattr(self._env, item)

    @property
    def unwrapped(self) -> Environment:
        if hasattr(self._env, "unwrapped"):
            return self._env.unwrapped
        else:
            return self._env
