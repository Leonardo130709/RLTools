from .base import Wrapper
import numpy as np
from collections import defaultdict


class Monitor(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._data = defaultdict(list)

    def reset(self):
        self._data = defaultdict(list)
        return self.env.reset()

    def step(self, action):
        obs, r, d = self.env.step(action)
        state = self.physics.state()
        self._data['states'].append(state)
        self._data['observations'].append(obs)
        return obs, r, d

    @property
    def data(self):
        return {k: np.array(v) for k, v in self._data.items()}
