import numpy as np

from rltools.loggers import base


class TerminalOutput(base.Logger):
    def __init__(self, precision: int = 3):
        self._precision = precision

    def write(self, metrics: base.Metrics):
        with np.printoptions(precision=self._precision):
            print({k: np.asarray(v) for k, v in metrics.items()})

    def close(self):
        pass
