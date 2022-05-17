from .base import Wrapper
from collections import deque
import numpy as np


class FrameStack(Wrapper):
    def __init__(self, env, frame_number: int = 1):
        super().__init__(env)
        self.fn = frame_number
        self._state = None

    def reset(self):
        obs = self.env.reset()
        self._state = deque(self.fn * [obs], maxlen=self.fn)
        return self.observation(None)

    def step(self, action):
        obs, r, d = self.env.step(action)
        self._state.append(obs)
        return self.observation(None), r, d

    def observation(self, timestamp):
        return np.stack(self._state)

    def observation_spec(self):
        spec = self.env.observation_spec()
        return spec.replace(
            shape=(self.fn, *spec.shape),
            name=f'{self.fn}_stacked_{spec.name}'
        )
