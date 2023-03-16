import dm_env

from rltools.dmc_wrappers import base


class AutoReset(base.Wrapper):
    """Automatically reset episode if it terminates.

    Intended for use with vectorization wrappers.
    """

    def reset(self) -> dm_env.TimeStep:
        return self.env.reset()

    def step(self, action: base.Action) -> dm_env.TimeStep:
        ts = super().step(action)
        if ts.last():
            new_ts = self.env.reset()
            return ts._replace(observation=new_ts.observation)
        return ts
