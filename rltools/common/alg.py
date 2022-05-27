from .base_config import BaseConfig
import abc
from dm_env import Environment
from ...rltools.wrappers.base import Wrapper
from typing import Union
import pathlib


class RLAlg(abc.ABC):
    def __init__(self, config: BaseConfig):
        self.config = config

    @abc.abstractmethod
    def learn(self):
        pass

    @abc.abstractmethod
    def make_env(self, config: BaseConfig) -> Union[Environment, Wrapper]:
        pass

    @abc.abstractmethod
    def save(self, path: Union[str, pathlib.Path] = None):
        pass

    @abc.abstractmethod
    def load(self, path: Union[str, pathlib.Path]):
        pass

