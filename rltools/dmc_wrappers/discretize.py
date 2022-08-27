import dm_env.specs
import numpy as np

from .base import Wrapper


class DiscreteActionWrapper(Wrapper):
    """Discretize an action space."""
    def __init__(self, env: dm_env.Environment, num_actions: int, one_hot: bool = True):
        super().__init__(env)
        act_spec = env.action_spec()
        self.one_hot = one_hot
        self._action_spec = dm_env.specs.BoundedArray(
            shape=act_spec.shape, dtype=int,
            minimum=0, maximum=num_actions - 1
        )
        self._spacing = np.linspace(act_spec.minimum, act_spec.maximum, num_actions)

    def step(self, action):
        if self.one_hot:
            action = action.argmax(-1)
        action = np.take_along_axis(self._spacing, action[None], 0)
        return self.env.step(action)

    def action_spec(self):
        return self._action_spec
