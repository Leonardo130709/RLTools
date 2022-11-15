from collections import deque

import tree
import numpy as np

from rltools.dmc_wrappers.base import Wrapper


class FrameStack(Wrapper):
    """Stack previous observations to form a richer state."""
    def __init__(self, env, frames_number: int = 1):
        super().__init__(env)
        self.frames_number = frames_number
        self._buffer = None

    def reset(self):
        self._buffer = None
        return super().reset()

    def _observation_fn(self, timestep):
        if self._buffer is None:
            self._buffer = deque(
                self.frames_number * [timestep.observation],
                maxlen=self.frames_number
            )
        else:
            self._buffer.append(timestep.observation)
        return tree.map_structure(lambda *obs: np.stack(obs), *self._buffer)

    def observation_spec(self):
        spec_dict = self._env.observation_spec()
        return tree.map_structure(
            lambda sp: sp.replace(shape=(self.frames_number,) + sp.shape),
            spec_dict
        )
