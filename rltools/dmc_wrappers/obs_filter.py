from typing import Union, Callable
import re
from functools import singledispatch

import dm_env.specs

from rltools.dmc_wrappers import base

Predicate = Callable[[str, dm_env.specs.Array], bool]
Pattern = Union[str, re.Pattern]
ReMatch = Callable[[re.Pattern, str], re.Match]


class ObsFilter(base.Wrapper):
    """Allows to ignore obs keys that don't match pattern."""

    def __init__(self,
                 env: dm_env.Environment,
                 *args, **kwargs
                 ) -> None:
        """Constructor calls factory function which returns predicate.
        Predicate must take (key, spec) from ObsSpec and return bool.

        Predicate constructors:
            1) arg: Callable -- Predicate itself.
            2) pattern: str, method = re.search -- RegEx filter on key.
            3) dtype: type -- specs are filtered by data type.
        """
        super().__init__(env)
        predicate = _predicate_factory(*args, **kwargs)
        valid_keys = tuple(
            k for k, spec in env.observation_spec().items()
            if predicate(k, spec)
        )

        def _filter(obj: Union[base.Observation, base.ObservationSpec]):
            return type(obj)(
                {k: v for k, v in obj.items() if k in valid_keys}
            )
        self._filter = _filter

    def _observation_fn(self, timestep: dm_env.TimeStep
                        ) -> base.Observation:
        return self._filter(timestep.observation)

    def observation_spec(self) -> base.ObservationSpec:
        obs_spec = self._env.observation_spec()
        return self._filter(obs_spec)


@singledispatch
def _predicate_factory(arg: Predicate) -> Predicate:
    return arg


@_predicate_factory.register
def _(pattern: str, method: ReMatch = re.search) -> Predicate:
    pattern = re.compile(pattern)
    return lambda key, spec: method(pattern, key) is not None


@_predicate_factory.register
def _(pattern: re.Pattern, method: ReMatch = re.search) -> Predicate:
    return lambda key, spec: method(pattern, key) is not None


@_predicate_factory.register
def _(dtype: type) -> Predicate:
    return lambda key, spec: spec.dtype == dtype
