from collections import deque

import tree
import numpy as np
import dm_env.specs

from rltools.dmc_wrappers import base


class FrameStack(base.Wrapper):
    """Stack previous observations to form a richer state.

    Naive implementation that does not alias consecutive frames.
    """

    def __init__(self,
                 env: dm_env.Environment,
                 frames_number: int = 1
                 ) -> None:
        super().__init__(env)
        self.frames_number = frames_number
        self._buffer = None

    def reset(self) -> dm_env.TimeStep:
        self._buffer = None
        return super().reset()

    def _observation_fn(self, timestep: dm_env.TimeStep) -> base.Observation:
        observation = timestep.observation
        if self._buffer is None:
            self._buffer = deque(
                self.frames_number * [observation],
                maxlen=self.frames_number
            )
        else:
            self._buffer.append(observation)
        return tree.map_structure(lambda *obs: np.stack(obs), *self._buffer)

    def observation_spec(self) -> base.ObservationSpec:
        spec_dict = super().observation_spec()
        return tree.map_structure(
            lambda sp: sp.replace(shape=(self.frames_number,) + sp.shape),
            spec_dict
        )
