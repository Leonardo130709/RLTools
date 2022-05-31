import numpy as np
from collections import deque
from .base import Wrapper


class FrameStack(Wrapper):
    def __init__(self, env, frames_number: int = 1):
        super().__init__(env)
        self.fn = frames_number
        self._state = None

    def reset(self):
        self._state = None
        return super().reset()

    def observation(self, timestep):
        if self._state is None:
            self._state = deque(self.fn * [timestep.observation], maxlen=self.fn)
        else:
            self._state.append(timestep.observation)
        return np.stack(self._state)

    def observation_spec(self):
        spec = self.env.observation_spec()
        return spec.replace(
            shape=(self.fn, *spec.shape),
            name=f'{self.fn}_stacked_{spec.name}'
        )
