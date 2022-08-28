from typing import List, Dict
from collections import deque, OrderedDict

import numpy as np

from .base import Wrapper
from .utils.nested import nested_fn


#TODO: nested stack
class FrameStack(Wrapper):
    """Stack previous observations to form a richer state."""
    def __init__(self, env, frames_number: int = 1):
        super().__init__(env)
        self._fn = frames_number
        self._buffer = None

    def reset(self):
        self._buffer = None
        return super().reset()

    def observation(self, timestep):
        if self._buffer is None:
            self._buffer = deque(self._fn * [timestep.observation], maxlen=self._fn)
        else:
            self._buffer.append(timestep.observation)
        return _stack_dicts(self._buffer)

    def observation_spec(self):
        spec_dict = self._env.observation_spec()
        return nested_fn(lambda spec: spec.replace(shape=(self._fn,) + spec.shape), spec_dict)


def _stack_dicts(frames: List[Dict[str, np.ndarray]]):
    stacked = OrderedDict()
    for key in sorted(frames[0].keys()):
        stacked[key] = np.stack([frame[key] for frame in frames])
    return stacked
    