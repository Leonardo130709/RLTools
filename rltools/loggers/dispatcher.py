from typing import Sequence

from rltools.loggers import base


class Dispatcher(base.Logger):
    def __init__(self, to: Sequence[base.Logger]):
        self._to = to

    def write(self, metrics: base.Metrics):
        for logger in self._to:
            logger.write(metrics)

    def close(self):
        for logger in self._to:
            logger.close()
