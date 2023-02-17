"""dm_env.Environment wrappers."""
from rltools.dmc_wrappers.action_repeat import ActionRepeat
from rltools.dmc_wrappers.action_rescale import ActionRescale
from rltools.dmc_wrappers.discretize import DiscreteActionWrapper
from rltools.dmc_wrappers.frames_stack import FrameStack
from rltools.dmc_wrappers.obs_filter import ObsFilter
from rltools.dmc_wrappers.states import StatesWrapper
from rltools.dmc_wrappers.time_limit import TimeLimit
from rltools.dmc_wrappers.types_cast import TypesCast
from rltools.dmc_wrappers.vectorize import Vectorize

try:
    from rltools.dmc_wrappers.gym_adapter import DmcToGym, GymToDmc
except ImportError:
    import logging
    logging.info("Skipping gym")

from rltools.dmc_wrappers.base import EnvironmentSpecs
