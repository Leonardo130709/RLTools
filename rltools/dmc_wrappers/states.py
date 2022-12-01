import numpy as np
import dm_env.specs
from dm_control.rl.control import flatten_observation, FLAT_OBSERVATION_KEY

from rltools.dmc_wrappers import base


class StatesWrapper(base.Wrapper):
    """Flatten observations to a single vector."""

    def _observation_fn(self, timestep: dm_env.TimeStep) -> base.Observation:
        return flatten_observation(
            timestep.observation,
            output_key=FLAT_OBSERVATION_KEY
        )

    def observation_spec(self) -> base.ObservationSpec:
        dim = sum(
            np.prod(ar.shape)
            for ar in self._env.observation_spec().values()
        )
        return {
            FLAT_OBSERVATION_KEY: dm_env.specs.Array(
                shape=(int(dim),), dtype=np.float32,
                name=FLAT_OBSERVATION_KEY)
        }
