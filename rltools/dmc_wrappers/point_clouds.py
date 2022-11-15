from typing import Iterable

import numpy as np
from dm_env import specs
from dm_control.rl.control import Environment

from rltools.dmc_wrappers.base import Wrapper
from rltools.dmc_wrappers.utils.point_cloud_generator import (
    PointCloudGenerator,
    CameraParams
)


class PointCloudWrapper(Wrapper):
    def __init__(self,
                 env: Environment,
                 pn_number: int,
                 render_kwargs: Iterable[CameraParams],
                 stride: int
                 ):
        super().__init__(env)
        self._pcg = PointCloudGenerator(pn_number, render_kwargs, stride)
        self._pn_number = pn_number

    def _observation_fn(self, timestep):
        obs = timestep.observation
        obs.update(point_cloud=self._pcg(self._env.physics))
        return obs

    def observation_spec(self):
        obs_spec = self._env.observation_spec().copy()
        obs_spec.update(
            point_cloud=specs.Array(
                shape=(self._pn_number, 3),
                dtype=np.float32,
                name="point_cloud"
            )
        )
        return obs_spec
