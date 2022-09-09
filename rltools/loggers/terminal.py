import numpy as np

from .base import Logger


class TerminalLogger(Logger):
    def write(self, metrics):
        with np.printoptions(precision=2):
            print({k: np.asarray(v) for k, v in metrics.items()})

    def close(self):
        pass
