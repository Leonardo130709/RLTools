import numpy as np
import dm_env.specs
from dm_env.specs import BoundedArray

from rltools.dmc_wrappers.base import Wrapper


class ActionRescale(Wrapper):
    """Normalize action to [-1, 1] range."""
    
    def __init__(self, env):
        super().__init__(env)
        spec = env.action_spec()
        assert isinstance(spec, BoundedArray)
        low = spec.minimum
        high = spec.maximum
        self._diff = (high - low) / 2.
        self._spec = spec.replace(
            minimum=np.full_like(low, -1),
            maximum=np.full_like(high, 1)
        )
        self._low = low

    def step(self, action):
        action = (action + 1.) * self._diff + self._low
        return self._env.step(action)

    def action_spec(self) -> dm_env.specs.Array:
        return self._spec
