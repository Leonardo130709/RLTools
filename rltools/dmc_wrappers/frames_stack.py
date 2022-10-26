from typing import Sequence, Dict
from collections import deque, OrderedDict

import numpy as np

from rltools.dmc_wrappers.base import Wrapper
from rltools.dmc_wrappers.utils.nested import nested_fn


class FrameStack(Wrapper):
    """Stack previous observations to form a richer state."""
    def __init__(self, env, frames_number: int = 1):
        super().__init__(env)
        self.frames_number = frames_number
        self._buffer = None

    def reset(self):
        self._buffer = None
        return super().reset()

    def observation(self, timestep):
        if self._buffer is None:
            self._buffer = deque(
                self.frames_number * [timestep.observation],
                maxlen=self.frames_number
            )
        else:
            self._buffer.append(timestep.observation)
        return _stack_dicts(self._buffer)

    def observation_spec(self):
        spec_dict = self._env.observation_spec()
        return nested_fn(
            lambda spec: spec.replace(shape=(self.frames_number,) + spec.shape),
            spec_dict
        )


def _stack_dicts(frames: Sequence[Dict[str, np.ndarray]]
                 ) -> Dict[str, np.ndarray]:
    stacked = OrderedDict()
    for key in sorted(frames[0].keys()):
        stacked[key] = np.stack([frame[key] for frame in frames])
    
    return stacked
    