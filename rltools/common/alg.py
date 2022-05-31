import abc
import pathlib
from typing import Union
import dm_env
from .base_config import BaseConfig
from ...rltools.wrappers.base import Wrapper


class RLAlg(abc.ABC):
    """Algorithm class declaration."""
    def __init__(self, config: BaseConfig):
        self.config = config

    @abc.abstractmethod
    def learn(self):
        """Runs training loop with corresponding logs."""
        pass

    @abc.abstractmethod
    def make_env(self, config: BaseConfig) -> dm_env.Environment:
        """Creates environment on which agent will operate. Should subclass dm_env.Environment."""
        pass

    @abc.abstractmethod
    def save(self, path: Union[str, pathlib.Path]):
        """Save configs/weights/logs/buffer and everything else to the path."""
        pass

    @abc.abstractmethod
    def load(self, path: Union[str, pathlib.Path]):
        """Restores algorithm state from the saved directory."""
        pass
