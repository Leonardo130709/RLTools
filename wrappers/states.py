from .base import Wrapper
import numpy as np
from dm_env import specs


class StatesWrapper(Wrapper):
    """ Converts OrderedDict obs to 1-dim np.ndarray[np.float32]. """
    def __init__(self, env):
        super().__init__(env)
        self._observation_spec = self._infer_obs_specs(env)

    def observation(self, timestamp):
        obs = []
        for v in timestamp.observation.values():
            if v.ndim == 0:
                v = v[None]
            obs.append(v.flatten())
        obs = np.concatenate(obs)
        return obs.astype(np.float32)

    @staticmethod
    def _infer_obs_specs(env) -> specs.Array:
        dim = sum((np.prod(ar.shape) for ar in env.observation_spec().values()))
        return specs.Array(shape=(dim,), dtype=np.float32, name='states')

    def observation_spec(self):
        return self._observation_spec
