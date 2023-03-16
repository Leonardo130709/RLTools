import abc
from typing import Dict

import numpy as np

Metrics = Dict[str, np.numeric]


class Logger(abc.ABC):
    """Handle metrics stream."""

    @abc.abstractmethod
    def write(self, metrics: Metrics) -> None:
        """Write metrics to destination."""

    @abc.abstractmethod
    def close(self) -> None:
        """Flush data and close a logger."""
