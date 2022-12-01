import numpy as np
import dm_env.specs

from rltools.dmc_wrappers import base


class ActionRescale(base.Wrapper):
    """Normalize action to [-1, 1] range."""

    def __init__(self, env: dm_env.Environment) -> None:
        super().__init__(env)
        spec = env.action_spec()
        assert isinstance(spec, dm_env.specs.BoundedArray)
        low = spec.minimum
        high = spec.maximum
        assert np.isfinite(low).all() and np.isfinite(high).all()
        self._diff = (high - low) / 2.
        self._spec = spec.replace(
            minimum=np.full_like(low, -1),
            maximum=np.full_like(high, 1)
        )
        self._low = low

    def step(self, action: base.Action) -> dm_env.TimeStep:
        action = (action + 1.) * self._diff + self._low
        return self._env.step(action)

    def action_spec(self) -> dm_env.specs.BoundedArray:
        return self._spec
