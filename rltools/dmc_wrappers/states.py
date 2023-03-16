import collections.abc

import numpy as np
import dm_env.specs

from rltools.dmc_wrappers import base

FLAT_OBSERVATION_KEY = "observations"


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
            for ar in self.env.observation_spec().values()
        )
        return {
            FLAT_OBSERVATION_KEY: dm_env.specs.Array(
                shape=(int(dim),), dtype=np.float32,
                name=FLAT_OBSERVATION_KEY)
        }


# taken directly from dm_control rep.
def flatten_observation(observation: base.Observation,
                        output_key: str = FLAT_OBSERVATION_KEY
                        ) -> base.Observation:
    """Flattens multiple observation arrays into a single numpy array.

    Args:
    observation: A mutable mapping from observation names to numpy arrays.
    output_key: The key for the flattened observation array in the output.

    Returns:
    A mutable mapping of the same type as `observation`. This will contain a
    single key-value pair consisting of `output_key` and the flattened
    and concatenated observation array.

    Raises:
    ValueError: If `observation` is not a `collections.abc.MutableMapping`.
    """
    if not isinstance(observation, collections.abc.MutableMapping):
        raise ValueError("Can only flatten dict-like observations.")

    if isinstance(observation, collections.OrderedDict):
        keys = observation.keys()
    else:
        # Keep a consistent ordering for other mappings.
        keys = sorted(observation.keys())

    observation_arrays = [observation[key].ravel() for key in keys]
    return type(observation)([(output_key, np.concatenate(observation_arrays))])
