from typing import Literal, Iterable
from collections import OrderedDict

import numpy as np
from dm_env import specs

from . import base

RenderMode = Literal["image", "depth", "grayscale"]
GRAYSCALE_RGB = np.array([.3, .59, .11]).reshape(3, 1)


#TODO: store rgb in uint to save memory.
class PixelsWrapper(base.Wrapper):
    """Makes environment to return 2D array as an observation.
    It could be one of the following: RGB array, grayscaled image, depth map.
    Combinations of those modalities are also allowed."""
    _channels_shapes = dict(image=3, depth=1, grayscale=1)

    def __init__(self, env: base.Environment,
                 render_kwargs: Iterable[base.CameraParams],
                 channels: Iterable[RenderMode]
                 ):
        super().__init__(env)
        self._render_kwargs = sorted(render_kwargs)
        self._channels = sorted(channels)

    def observation(self, timestep):
        observation = OrderedDict()
        for camera in self._render_kwargs:
            cam_name = camera.get("camera_id")
            cam_obs = cam_observation(self._env.physics, camera, self._channels)
            for key, value in cam_obs.items():
                observation[f"cam_{cam_name}_{key}"] = value

        return observation

    def observation_spec(self):
        obs_spec = OrderedDict()
        for channel in self._channels:
            for camera in self._render_kwargs:
                shape = (
                    camera.get("height", 240),
                    camera.get("width", 320),
                    self._channels_shapes[channel],
                )
                obs_name = f"cam_{camera.get('camera_id')}_{channel}"
                obs_spec[obs_name] = specs.Array(shape=shape, dtype=np.float32, name=obs_name)

        return obs_spec


def cam_observation(physics, render_kwargs: base.CameraParams, channels: Iterable[RenderMode]):
    observation = OrderedDict()
    for mode in channels:
        if mode in ("image", "grayscale"):
            rgb = physics.render(**render_kwargs)
            rgb = (rgb / 255).astype(np.float32)
            value = rgb
            if mode == "grayscale":
                value = value @ GRAYSCALE_RGB
        elif mode == "depth":
            depth = physics.render(depth=True, **render_kwargs)
            depth = np.expand_dims(depth, axis=-1).astype(np.float32)
            value = depth
        else:
            raise ValueError(mode)
        observation[mode] = value

    return observation


