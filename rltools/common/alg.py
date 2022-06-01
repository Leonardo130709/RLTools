import abc
from typing import TypeVar
import dm_env
from .base_config import BaseConfig

Config = TypeVar('Config', bound=BaseConfig)


class RLAlg(abc.ABC):
    """Algorithm class declaration."""
    def __init__(self, config: Config):
        self.config = config

    @abc.abstractmethod
    def learn(self):
        """Initiate training loop."""

    @abc.abstractmethod
    def make_env(self, config: Config) -> dm_env.Environment:
        """Creates environment on which agent will operate. Must follow dm_env.Environment."""

    @abc.abstractmethod
    def save(self, path: str):
        """Saves configs/weights/logs/buffer and everything else to the path."""

    @abc.abstractmethod
    def load(self, path: str):
        """Restores algorithm state from the saved directory."""
