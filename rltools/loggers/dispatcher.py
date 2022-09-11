from typing import Sequence

from .base import Logger


class Dispatcher(Logger):
    def __init__(self, to: Sequence[Logger]):
        self._to = to

    def write(self, metrics):
        for logger in self._to:
            logger.write(metrics)

    def close(self):
        for logger in self._to:
            logger.close()