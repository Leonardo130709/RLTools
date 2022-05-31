import numpy as np
from dm_env import specs
from .base import Wrapper


class StatesWrapper(Wrapper):
    """ Converts OrderedDict obs to 1-dim np.ndarray[np.float32]. """
    def observation(self, timestamp):
        obs = []
        for v in timestamp.observation.values():
            if v.ndim == 0:
                v = v[None]
            obs.append(v.flatten())
        obs = np.concatenate(obs)
        return obs.astype(np.float32)

    def observation_spec(self):
        dim = sum((np.prod(ar.shape) for ar in env.observation_spec().values()))
        return specs.Array(shape=(dim,), dtype=np.float32, name='states')
