from collections import defaultdict
import numpy as np
import dm_env
from .base import Wrapper


class Monitor(Wrapper):
    """Store proprietary states with observations coming from the environment."""
    def __init__(self, env: dm_env.Environment):
        super().__init__(env)
        self._data = defaultdict(list)

    def reset(self):
        self._data = defaultdict(list)
        return self.env.reset()

    def step(self, action):
        timestep = self.env.step(action)
        state = self.physics.state()
        self._data['states'].append(state)
        self._data['observations'].append(timestep.observation)
        return timestep

    @property
    def data(self):
        return {k: np.array(v) for k, v in self._data.items()}
