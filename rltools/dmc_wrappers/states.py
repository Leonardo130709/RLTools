import numpy as np
from dm_env import specs
from dm_control.rl.control import flatten_observation, FLAT_OBSERVATION_KEY
from rltools.dmc_wrappers.base import Wrapper


class StatesWrapper(Wrapper):
    """Flatten observations to a single vector."""
    def observation(self, timestamp):
        return flatten_observation(
            timestamp.observation,
            output_key=FLAT_OBSERVATION_KEY
        )

    def observation_spec(self):
        dim = sum(
            np.prod(ar.shape)
            for ar in self._env.observation_spec().values()
        )
        return specs.Array(
            shape=(dim,), dtype=np.float32,
            name=FLAT_OBSERVATION_KEY
        )
