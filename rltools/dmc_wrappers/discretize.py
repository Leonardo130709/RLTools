import numpy as np
from dm_env import specs

from rltools.dmc_wrappers.base import Wrapper


class DiscreteActionWrapper(Wrapper):
    """Discretizing action space."""

    def __init__(self, env, bins: int):
        super().__init__(env)
        act_spec = env.action_spec()
        assert isinstance(act_spec, specs.BoundedArray)
        self._action_spec = specs.BoundedArray(
            shape=act_spec.shape + (bins,),
            minimum=0,
            maximum=1,
            dtype=np.int32
        )
        self._spacing = np.linspace(
            act_spec.minimum, act_spec.maximum, bins)

    def step(self, action):
        action = action.argmax(-1)
        action = np.take_along_axis(self._spacing, action[np.newaxis], 0)
        return self._env.step(action)

    def action_spec(self):
        return self._action_spec
