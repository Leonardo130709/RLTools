import numpy as np
import dm_env.specs

from .base import Wrapper


class DiscreteActionWrapper(Wrapper):
    """Discretize an action space."""
    def __init__(self, env, num_actions: int):
        super().__init__(env)
        act_spec = env.action_spec()
        self._action_spec = dm_env.specs.BoundedArray(
            shape=act_spec.shape + (num_actions,),
            minimum=0,
            maximum=1,
            dtype=int
        )
        self._spacing = np.linspace(act_spec.minimum, act_spec.maximum, num_actions)

    def step(self, action):
        action = action.argmax(-1)
        action = np.take_along_axis(self._spacing, action[np.newaxis], 0)
        return self.env.step(action)

    def action_spec(self):
        return self._action_spec
