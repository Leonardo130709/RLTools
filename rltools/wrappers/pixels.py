from typing import Literal
import numpy as np
from dm_env import specs
from .base import Wrapper
Modalities = Literal['rgb', 'rgbd', 'd', 'g', 'gd']


class PixelsWrapper(Wrapper):
    """Makes environment to return 2D array as an observation.
    It could be one of the following: RGB array, grayscaled image, depth map.
    Combinations of those modalities are also allowed."""
    channels = dict(rgb=3, rgbd=4, d=1, g=1, gd=2)

    def __init__(self, env, render_kwargs=None, mode: Modalities = 'rgb'):
        super().__init__(env)
        self.render_kwargs = render_kwargs or dict(camera_id=0, height=84, width=84)
        self.mode = mode
        self._gs_coef = np.array([0.299, 0.587, 0.114])

    def observation(self, timestep):
        if self.mode != 'd':
            rgb = self.physics.render(**self.render_kwargs).astype(np.float32)
            rgb /= 255.
        obs = ()
        if 'rgb' in self.mode:
            obs += (rgb - .5,)
        if 'd' in self.mode:
            depth = self.physics.render(depth=True, **self.render_kwargs)
            obs += (depth[..., np.newaxis],)
        if 'rgb' not in self.mode and 'g' in self.mode:
            g = rgb @ self._gs_coef
            obs += (g[..., np.newaxis],)
        obs = np.concatenate(obs, -1)
        return obs.transpose((2, 1, 0)).astype(np.float32)

    def observation_spec(self):
        shape = (
            self.channels[self.mode],
            self.render_kwargs.get('height', 240),
            self.render_kwargs.get('width', 320)
        )
        return specs.Array(shape=shape, dtype=np.float32, name=self.mode)
