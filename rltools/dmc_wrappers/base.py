from typing import NamedTuple, Any
from collections.abc import MutableMapping

import numpy as np
import dm_env
from dm_env import specs

Action = np.ndarray
ActionSpec = specs.Array
Observation = MutableMapping[str, np.ndarray]
ObservationSpec = MutableMapping[str, specs.Array]


class EnvironmentSpecs(NamedTuple):
    observation_spec: ObservationSpec
    action_spec: ActionSpec
    reward_spec: specs.Array
    discount_spec: specs.BoundedArray


# pylint: disable=no-staticmethod-decorator
class Wrapper:
    """This allows to alternate methods of a dm_env.Environment instance."""

    def __init__(self, env: dm_env.Environment):
        self.env = env

    def _wrap_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        """Modify dm_env.TimeStep fields.

        All the methods are no-op by default.
        Child class supposed to implement any of them
            and make corresponding changes to specs methods.
        However, one may decide to avoid calling this
            and directly reimplement step/reset of dm_env.Environment.
        """
        # pylint: disable-next=protected-access
        return timestep._replace(
            step_type=self._step_type_fn(timestep),
            reward=self._reward_fn(timestep),
            discount=self._discount_fn(timestep),
            observation=self._observation_fn(timestep)
        )

    def _observation_fn(self, timestep: dm_env.TimeStep) -> Observation:
        """Modify observation."""
        return timestep.observation

    def _reward_fn(self, timestep: dm_env.TimeStep) -> float:
        """Modify reward."""
        return timestep.reward

    def _step_type_fn(self, timestep: dm_env.TimeStep) -> dm_env.StepType:
        """Modify dm_env.StepType."""
        return timestep.step_type

    def _discount_fn(self, timestep: dm_env.TimeStep) -> float:
        """Modify discount."""
        return timestep.discount

    def step(self, action: Action) -> dm_env.TimeStep:
        timestep = self.env.step(action)
        return self._wrap_timestep(timestep)

    def reset(self) -> dm_env.TimeStep:
        return self._wrap_timestep(self.env.reset())

    def action_spec(self) -> specs.Array:
        return self.env.action_spec()

    def observation_spec(self) -> ObservationSpec:
        return self.env.observation_spec()

    def reward_spec(self) -> specs.Array:
        return self.env.reward_spec()

    def discount_spec(self) -> specs.BoundedArray:
        return self.env.discount_spec()

    @property
    def environment_specs(self) -> EnvironmentSpecs:
        """Return tuple of all specs."""
        return EnvironmentSpecs(
            observation_spec=self.observation_spec(),
            action_spec=self.action_spec(),
            reward_spec=self.reward_spec(),
            discount_spec=self.discount_spec()
        )

    @property
    def unwrapped(self) -> dm_env.Environment:
        """Strip down wrappers and return original env."""
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        return self.env

    def __del__(self) -> None:
        self.env.close()

    def __getattr__(self, item: str) -> Any:
        return getattr(self.env, item)


Wrapper.reset.__doc__ = dm_env.Environment.reset.__doc__
Wrapper.step.__doc__ = dm_env.Environment.step.__doc__
Wrapper.action_spec.__doc__ = dm_env.Environment.action_spec.__doc__
Wrapper.observation_spec.__doc__ = dm_env.Environment.observation_spec.__doc__
Wrapper.reward_spec.__doc__ = dm_env.Environment.reward_spec.__doc__
Wrapper.discount_spec.__doc__ = dm_env.Environment.discount_spec.__doc__
