from collections import OrderedDict
from typing import Iterable

import dm_env.specs
import numpy as np

from dmc_wrappers import base
from dmc_wrappers.utils.point_cloud_generator import PointCloudGenerator


class PointCloudWrapper(base.Wrapper):
    def __init__(self,
                 env: base.Environment,
                 pn_number: int,
                 render_kwargs: Iterable[base.CameraParams],
                 stride: int
                 ):
        super().__init__(env)
        self._pcg = PointCloudGenerator(pn_number, render_kwargs, stride)
        self._pn_number = pn_number

    def observation(self, timestep):
        return OrderedDict(point_cloud=self._pcg(self._env.physics))

    def observation_spec(self):
        return OrderedDict(
            point_cloud=dm_env.specs.Array(
                shape=(self._pn_number, 3),
                dtype=np.float32,
                name="point_cloud"
            )
        )
