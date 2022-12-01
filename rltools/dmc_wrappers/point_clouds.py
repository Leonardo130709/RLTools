from typing import Iterable

import dm_env
import numpy as np
from dm_env import specs
from dm_control.rl.control import Environment

from rltools.dmc_wrappers import base
from rltools.dmc_wrappers.utils.point_cloud_generator import (
    PointCloudGenerator,
    CameraParams
)


class PointCloudWrapper(base.Wrapper):
    """Add point cloud from dm_control physics to observations dict."""

    def __init__(self,
                 env: Environment,
                 pn_number: int,
                 render_kwargs: Iterable[CameraParams],
                 stride: int
                 ) -> None:
        super().__init__(env)
        self._pcg = PointCloudGenerator(pn_number, render_kwargs, stride)
        self._pn_number = pn_number

    def _observation_fn(self, timestep: dm_env.TimeStep) -> base.Observation:
        obs = super()._observation_fn(timestep)
        obs.update(point_cloud=self._pcg(self._env.physics))
        return obs

    def observation_spec(self) -> base.ObservationSpec:
        obs_spec = super().observation_spec()
        obs_spec.update(
            point_cloud=specs.Array(
                shape=(self._pn_number, 3),
                dtype=np.float32,
                name="point_cloud"
            )
        )
        return obs_spec
