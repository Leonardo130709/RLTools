import pprint

import numpy as np

from rltools.loggers import base


class TerminalOutput(base.Logger):
    """As simple as pprint."""

    def __init__(self, precision: int = 3) -> None:
        self._precision = precision

    def write(self, metrics: base.Metrics) -> None:
        with np.printoptions(precision=self._precision):
            pprint.pprint({k: np.asarray(v) for k, v in metrics.items()})

    def close(self) -> None:
        pass
