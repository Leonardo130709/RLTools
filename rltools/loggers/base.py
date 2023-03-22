import abc
from typing import Dict

Metrics = Dict[str, float]


class Logger(abc.ABC):
    """Handle metrics stream."""

    @abc.abstractmethod
    def write(self, metrics: Metrics) -> None:
        """Write metrics to destination."""

    @abc.abstractmethod
    def close(self) -> None:
        """Flush data and close a logger."""
