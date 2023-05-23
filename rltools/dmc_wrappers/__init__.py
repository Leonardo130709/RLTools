"""dm_env.Environment wrappers."""
from rltools.dmc_wrappers.action_repeat import ActionRepeat
from rltools.dmc_wrappers.action_rescale import ActionRescale
from rltools.dmc_wrappers.autoreset import AutoReset
from rltools.dmc_wrappers.discretize import DiscreteActionWrapper
from rltools.dmc_wrappers.frames_stack import FrameStack
from rltools.dmc_wrappers.obs_filter import ObsFilter
from rltools.dmc_wrappers.states import StatesWrapper
from rltools.dmc_wrappers.time_limit import TimeLimit
from rltools.dmc_wrappers.vectorize import AsyncEnv, SequentialEnv

try:
    from rltools.dmc_wrappers.gymnasium_adapter import (
        DmcToGymnasium, GymnasiumToDmc
    )
except ImportError:
    import logging
    logging.info("Skipping gymnasium")

from rltools.dmc_wrappers.base import EnvironmentSpecs
