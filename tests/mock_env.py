import tree
import dm_env
from dm_env import specs
import numpy as np

BOUNDED_OBS_SHAPE = (2, 2, 1)
ACT_DIM = 5
REWARD_SPEC = specs.Array(shape=(), dtype=float, name="reward")
DISCOUNT_SPEC = specs.BoundedArray(
        shape=(), dtype=float, minimum=0., maximum=1., name="discount")


class MockEnv(dm_env.Environment):
    """Testing env with a time limit."""

    def __init__(self,
                 obs_spec: specs.Array,
                 act_spec: specs.Array,
                 reward_const: float = 1.,
                 discount_const: float = 1.,
                 time_limit: int = float("inf")
                 ):
        self._obs_spec = obs_spec
        self._act_spec = act_spec
        self._reward_const = reward_const
        self._discount_const = discount_const

        self._time_limit = time_limit
        self._step = None

    def reset(self) -> dm_env.TimeStep:
        self._step = 0
        return dm_env.restart(self._get_obs())

    def step(self, action) -> dm_env.TimeStep:
        self._step += 1
        if self._step < self._time_limit:
            fn = dm_env.transition
        else:
            fn = dm_env.truncation
        return fn(
            self._reward_const,
            self._get_obs(),
            self._discount_const
        )

    def _get_obs(self):
        return tree.map_structure(specs.Array.generate_value,
                                  self.observation_spec())

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._act_spec


class TestEnv(MockEnv):
    def __init__(self, discrete: bool = False):
        obs_spec = dict(
            bounded_obs=specs.BoundedArray(
                BOUNDED_OBS_SHAPE, np.uint8,
                minimum=0, maximum=255, name="bounded_obs"),
            scalar_obs=specs.Array((), np.float32, name="scalar_obs")
        )
        if discrete:
            act_spec = specs.DiscreteArray(ACT_DIM)
        else:
            lim = np.full((ACT_DIM,), 1.)
            act_spec = specs.BoundedArray((ACT_DIM,), float, -lim, lim)
        super().__init__(obs_spec, act_spec)
