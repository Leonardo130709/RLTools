import numpy as np
from .base import Wrapper


class DiscreteActionWrapper(Wrapper):
    def __init__(self, env, num_actions: int, one_hot=True):
        super().__init__(env)
        act_spec = env.action_spec()
        self.one_hot = one_hot
        self._action_spec = specs.BoundedArray(
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
