import dm_env

from rltools.dmc_wrappers import base


class TimeLimit(base.Wrapper):
    """Truncate episode after specified number of steps."""

    def __init__(self, env: dm_env.Environment, time_limit: int) -> None:
        super().__init__(env)
        self.time_limit = time_limit
        self._steps = 0

    def reset(self) -> dm_env.TimeStep:
        self._steps = 0
        return super().reset()

    def step(self, action: base.Action) -> dm_env.TimeStep:
        self._steps += 1
        return super().step(action)

    def _step_type_fn(self, timestep: dm_env.TimeStep) -> dm_env.StepType:
        if self._steps >= self.time_limit:
            return dm_env.StepType.LAST
        return timestep.step_type
