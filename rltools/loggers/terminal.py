import numpy as np

from .base import Logger


class TerminalOutput(Logger):
    def __init__(self, precision: int = 3):
        self._precision = precision

    def write(self, metrics):
        with np.printoptions(precision=self._precision):
            print({k: np.asarray(v) for k, v in metrics.items()})

    def close(self):
        pass
