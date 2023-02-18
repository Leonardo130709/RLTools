import dm_env.specs

from rltools.dmc_wrappers import base


class AutoReset(base.Wrapper):
    """Normalize action to [-1, 1] range."""

    def step(self, action: base.Action) -> dm_env.TimeStep:
        ts = super().step(action)
        if ts.last():
            new_ts = self._env.reset()
            return ts._replace(observation=new_ts.observation)
        return ts
