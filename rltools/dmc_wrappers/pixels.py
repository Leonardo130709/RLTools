from typing import Iterable
from collections import OrderedDict

import numpy as np
import dm_env.specs
from dm_control.rl.control import Environment, Physics

from rltools.dmc_wrappers import base
from rltools.dmc_wrappers.utils.point_cloud_generator import CameraParams

_GRAYSCALE_RGB = np.array([.3, .59, .11]).reshape(3, 1)


#TODO: store rgb in uint to save memory.
class PixelsWrapper(base.Wrapper):
    """Makes environment to return 2D array as an observation.
    It could be one of the following: RGB array, grayscaled image, depth map.
    Combinations of those modalities are also allowed."""
    _channels_shapes = dict(image=3, depth=1, grayscale=1)

    def __init__(self,
                 env: Environment,
                 render_kwargs: Iterable[CameraParams],
                 channels: Iterable[str]
                 ) -> None:
        super().__init__(env)
        self._render_kwargs = sorted(render_kwargs)
        self._channels = sorted(channels)

    def _observation_fn(self, timestep: dm_env.TimeStep) -> base.Observation:
        observation = super()._observation_fn(timestep)
        for camera in self._render_kwargs:
            cam_name = camera["camera_id"]
            cam_obs = cam_observation(self._env.physics, camera, self._channels)
            for key, value in cam_obs.items():
                observation[f"{cam_name}/{key}"] = value
        return observation

    def observation_spec(self) -> base.ObservationSpec:
        obs_spec = super().observation_spec()
        for channel in self._channels:
            for camera in self._render_kwargs:
                shape = (
                    camera["height"],
                    camera["width"],
                    self._channels_shapes[channel],
                )
                obs_name = f"{camera['camera_id']}/{channel}"
                obs_spec[obs_name] = dm_env.specs.Array(
                    shape=shape, dtype=np.float32, name=obs_name)
        return obs_spec


def cam_observation(physics: Physics,
                    render_kwargs: CameraParams,
                    channels: Iterable[str]
                    ) -> base.Observation:
    """Get an observation from the camera."""
    observation = OrderedDict()
    for mode in channels:
        if mode in ("image", "grayscale"):
            rgb = physics.render(**render_kwargs)
            value = (rgb / 255.).astype(np.float32)
            if mode == "grayscale":
                value = value @ _GRAYSCALE_RGB
        elif mode == "depth":
            depth = physics.render(depth=True, **render_kwargs)
            value = np.expand_dims(depth, axis=-1).astype(np.float32)
        else:
            raise ValueError(mode)
        observation[mode] = value
    return observation
