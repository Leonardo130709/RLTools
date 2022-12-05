import numpy as np
import dm_env.specs

from rltools.dmc_wrappers import base


class DiscreteActionWrapper(base.Wrapper):
    """Discretizing action space."""

    def __init__(self, env, bins: int):
        super().__init__(env)
        act_spec = env.action_spec()
        assert isinstance(act_spec, dm_env.specs.BoundedArray)
        shape = act_spec.shape + (bins,)
        self._action_spec = dm_env.specs.BoundedArray(
            shape=shape,
            minimum=0,
            maximum=1,
            dtype=np.int32
        )
        self._spacing = np.linspace(
            act_spec.minimum, act_spec.maximum, bins)

    def step(self, action: base.Action) -> dm_env.TimeStep:
        action = action.argmax(-1)
        action = np.take_along_axis(self._spacing, action[np.newaxis], 0)
        return self._env.step(action.squeeze(0))

    def action_spec(self) -> dm_env.specs.BoundedArray:
        return self._action_spec
