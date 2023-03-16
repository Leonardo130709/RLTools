from collections.abc import Sequence

from rltools.loggers import base


class Dispatcher(base.Logger):
    """Dispatch metrics to multiple loggers."""

    def __init__(self, to: Sequence[base.Logger]) -> None:
        self._to = to

    def write(self, metrics: base.Metrics) -> None:
        for logger in self._to:
            logger.write(metrics)

    def close(self) -> None:
        for logger in self._to:
            logger.close()
