from typing import Optional
import numpy as np

from dmc_wrappers.base import Wrapper
from dmc_wrappers.utils.nested import nested_fn


class TypesCast(Wrapper):
    def __init__(self,
                 env,
                 observation_dtype: Optional[np.dtype] = None,
                 action_dtype: Optional[np.dtype] = None,
                 reward_dtype: Optional[np.dtype] = None,
                 discount_dtype: Optional[np.dtype] = None
                 ):
        super().__init__(env)
        self._observation_dtype = observation_dtype
        self._action_dtype = action_dtype
        self._reward_dtype = reward_dtype
        self._discount_dtype = discount_dtype

    def observation_spec(self):
        return _replace_dtype(self._env.observation_spec(),
                              self._observation_dtype)

    def action_spec(self):
        return _replace_dtype(self._env.action_spec(),
                              self._action_dtype)

    def discount_spec(self):
        return _replace_dtype(self._env.discount_spec(),
                              self._discount_dtype)

    def reward_spec(self):
        return _replace_dtype(self._env.reward_spec(),
                              self._reward_dtype)

    def observation(self, timestep):
        return _cast_type(timestep.observation,
                          self._observation_dtype)

    def reward(self, timestep):
        return _cast_type(timestep.reward,
                          self._reward_dtype)

    def discount(self, timestep):
        return _cast_type(timestep.discount,
                          self._discount_dtype)


def _replace_dtype(spec, dtype):
    if dtype:
        spec = nested_fn(lambda sp: sp.replace(dtype=dtype), spec)
    return spec


def _cast_type(item, dtype):
    if item and dtype:
        item = nested_fn(lambda x: np.asarray(x, dtype=dtype), item)
    return item
