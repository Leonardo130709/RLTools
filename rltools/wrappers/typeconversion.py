import dm_env
import collections
from .base import Wrapper


# TODO: needs update
class TypesConvertor(Wrapper):
    def __init__(self,
                 env: dm_env.Environment,
                 observation_dtype=None,
                 reward_dtype=None,
                 discount_dtype=None
                 ):
        super().__init__(env)
        self._observation_dtype = observation_dtype
        self._reward_dtype = reward_dtype
        self._discount_dtype = discount_dtype

    def _wrap_timestep(self, timestep):
        timestep = super()._wrap_timestep(timestep)
        return dm_env.TimeStep(
            observation=self._observation_dtype(timestep.observation) if self._observation_dtype else timestep.observation,
            reward=self._reward_dtype(timestep.reward) if self._reward_dtype else timestep.observation,
            discount=self._discount_dtype(timestep.discount) if self._discount_dtype else timestep.discount,
            step_type=timestep.step_type
        )

    def observation_spec(self):
        return self._replacer(
            self.env.observation_spec(),
            self._observation_dtype
        )

    def reward_spec(self):
        return self._replacer(
            self.env.reward_spec(),
            self._reward_dtype
        )
    
    def discount_spec(self):
        return self._replacer(
            self.env.discount_spec(),
            self._discount_dtype
        )

    @staticmethod
    def _replacer(struct, dtype):
        if dtype:
            if isinstance(struct, collections.abc.Mapping):
                for k, v in struct.items():
                    struct[k] = v.replace(dtype=dtype)
            elif isinstance(struct, dm_env.specs.Array):
                struct = struct.replace(dtype=dtype)
            else:
                raise NotImplementedError
        return struct

